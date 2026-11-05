"""Predeclared LR postprocessing candidates for unseen-degradation evaluation.

These fixed settings are experimental stress conditions, not a claim that they
faithfully model real degradation. This module trains nothing, selects no
parameters on test results, and receives only an existing RGB LRV.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

PROTOCOL_VERSION = "fixed_lr_postprocess_v1"
MODES = ("identity", "blur", "noise", "compression", "combined")
BLUR_SIGMA = 1.0
BLUR_KERNEL = 7
NOISE_SIGMA = 5.0 / 255.0
CRF = 28
PRESET = "medium"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def fps_fraction(fps):
    """Preserve integer/noninteger rates as bounded FFmpeg rational numbers."""
    if isinstance(fps, bool):
        raise ValueError("FPS must be positive finite, not bool")
    try:
        original = fps if isinstance(fps, Fraction) else Fraction(str(fps))
        number = float(original)
        if not math.isfinite(number) or number <= 0:
            raise ValueError("FPS must be positive finite")
        rational = original if isinstance(fps, Fraction) else original.limit_denominator(1_000_000)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
        raise ValueError("FPS must be positive finite or a valid positive rational") from error
    if rational <= 0 or max(rational.numerator, rational.denominator) > 2_147_483_647:
        raise ValueError("FPS rational is outside FFmpeg's supported positive integer range")
    if not math.isclose(float(rational), number, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("FPS cannot be rationalized without changing its rate")
    return rational


def protocol_spec(mode, *, seed=20261102):
    if mode not in MODES:
        raise ValueError(f"Unknown robustness mode {mode!r}; expected {MODES}")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
        raise ValueError("Noise seed must be an integer in [0,2**63)")
    sequence = {"identity": [], "blur": ["blur"], "noise": ["noise"],
                "compression": ["compression"], "combined": ["blur", "noise", "compression"]}[mode]
    return {"version": PROTOCOL_VERSION, "mode": mode, "sequence": sequence,
            "blur": {"sigma_lr_pixels": BLUR_SIGMA, "kernel": BLUR_KERNEL, "boundary": "reflection",
                     "parameters_shared_across_video": True},
            "noise": {"sigma_rgb_01": NOISE_SIGMA, "seed": seed, "generator": "local CPU torch.Generator",
                      "samples": "independent per frame/pixel/channel; one fixed sigma for the whole video",
                      "clamp_after_noise": True, "global_rng_modified": False},
            "compression": {"codec": "libx264", "pixel_format": "yuv420p", "crf": CRF, "preset": PRESET,
                            "threads": 1, "fps_mode": "passthrough", "rgb_input_quantization": "round to uint8 RGB24",
                            "color_conversion": "pinned FFmpeg default RGB24↔YUV420 conversion; version/args recorded",
                            "resize": False, "require_even_lr_dimensions": True},
            "scope": "fixed candidate protocol; not calibrated for realism and not selected on test metrics"}


def _validate_video(video):
    if not isinstance(video, torch.Tensor) or video.ndim != 5 or tuple(video.shape[:2]) != (1, 3) or min(video.shape) < 1:
        raise ValueError("LR video must be nonempty [1,3,T,H,W] RGB")
    if not video.is_floating_point():
        raise ValueError("LR input dtype must be floating RGB in [0,1]")
    value = video.detach().to(device="cpu")
    if not bool(torch.isfinite(value).all()) or float(value.min()) < 0 or float(value.max()) > 1:
        raise ValueError("LR input must contain finite RGB values in [0,1]")
    return value.to(dtype=torch.float32).contiguous()


def _blur(video):
    h, w = video.shape[-2:]
    radius = BLUR_KERNEL // 2
    if min(h, w) <= radius:
        raise ValueError("Kernel7 reflection blur requires both LR dimensions greater than3; no resizing is allowed")
    coordinates = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel1d = torch.exp(-coordinates.square() / (2 * BLUR_SIGMA**2))
    kernel1d = kernel1d / kernel1d.sum()
    kernel = torch.outer(kernel1d, kernel1d).float().view(1, 1, BLUR_KERNEL, BLUR_KERNEL).expand(3, 1, -1, -1).contiguous()
    frames = video[0].permute(1, 0, 2, 3)
    blurred = F.conv2d(F.pad(frames, (radius, radius, radius, radius), mode="reflect"), kernel, groups=3)
    return blurred.permute(1, 0, 2, 3).unsqueeze(0).contiguous().clamp(0, 1)


def _noise(video, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(video.shape, generator=generator, dtype=torch.float32, device="cpu") * NOISE_SIGMA
    return (video + noise).clamp(0, 1)


def _binary(name):
    resolved = shutil.which(str(name))
    if resolved is None:
        raise FileNotFoundError(f"Required CPU video executable is unavailable: {name}")
    return str(Path(resolved).resolve())


def _run(args, *, timeout):
    try:
        return subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=True, timeout=timeout)
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"CPU video command failed ({error.returncode}): {message}") from error


def _validate_stream(stream, *, frames, height, width, fps):
    if stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p":
        raise ValueError("Compression stream is not H.264 yuv420p")
    if (stream.get("height"), stream.get("width")) != (height, width):
        raise ValueError("Compression changed LR geometry; resizing/truncation is forbidden")
    count = stream.get("nb_read_frames", stream.get("nb_frames"))
    try:
        actual_count = int(count)
    except (TypeError, ValueError) as error:
        raise ValueError("Unable to verify compressed video frame count") from error
    if actual_count != frames:
        raise ValueError("Compression changed frame count; implicit truncation/duplication is forbidden")
    try:
        average_rate, nominal_rate = Fraction(stream["avg_frame_rate"]), Fraction(stream["r_frame_rate"])
    except (KeyError, ValueError, ZeroDivisionError) as error:
        raise ValueError("Unable to verify compressed video FPS") from error
    if average_rate != fps or nominal_rate != fps:
        raise ValueError("Compression changed original rational FPS")


def _compress(video, fps, *, ffmpeg_bin, ffprobe_bin, timeout):
    t, h, w = video.shape[2:]
    if h % 2 or w % 2:
        raise ValueError("H.264 yuv420p requires even LR dimensions; no hidden resize is allowed")
    ffmpeg, ffprobe = _binary(ffmpeg_bin), _binary(ffprobe_bin)
    ffmpeg_version = _run([ffmpeg, "-version"], timeout=timeout).stdout.decode("utf-8", errors="replace")
    ffprobe_version = _run([ffprobe, "-version"], timeout=timeout).stdout.decode("utf-8", errors="replace")
    rate = f"{fps.numerator}/{fps.denominator}"
    raw = video[0].permute(1, 2, 3, 0).clamp(0, 1).mul(255).round().byte().contiguous().numpy().tobytes()
    expected_bytes = t * h * w * 3
    if len(raw) != expected_bytes:
        raise ValueError("RGB input byte count differs from complete video geometry")
    with tempfile.TemporaryDirectory(prefix="vsr_lr_robustness_") as temporary:
        folder = Path(temporary)
        source, encoded = folder / "input.rgb", folder / "compressed.mp4"
        source.write_bytes(raw)
        encode_args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                       "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{w}x{h}",
                       "-framerate", rate, "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
                       "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF), "-pix_fmt", "yuv420p",
                       "-threads", "1", "-fps_mode", "passthrough", "-map_metadata", "-1",
                       "-fflags", "+bitexact", "-flags:v", "+bitexact", str(encoded)]
        _run(encode_args, timeout=timeout)
        probe_args = [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                      "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames",
                      "-of", "json", str(encoded)]
        data = json.loads(_run(probe_args, timeout=timeout).stdout)
        if len(data.get("streams", [])) != 1:
            raise ValueError("Expected exactly one compressed video stream")
        stream = data["streams"][0]
        _validate_stream(stream, frames=t, height=h, width=w, fps=fps)
        decode_args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(encoded),
                       "-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough",
                       "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "pipe:1"]
        decoded = _run(decode_args, timeout=timeout).stdout
        if len(decoded) != expected_bytes:
            raise ValueError("Decoded RGB frame/byte count differs from input; implicit truncation is forbidden")
        output = torch.from_numpy(np.frombuffer(decoded, dtype=np.uint8).copy()).reshape(t, h, w, 3).permute(3, 0, 1, 2).unsqueeze(0).contiguous().float().div_(255)
        info = {"ffmpeg_path": ffmpeg, "ffmpeg_sha256": _file_sha(ffmpeg), "ffmpeg_version": ffmpeg_version,
                "ffprobe_path": ffprobe, "ffprobe_sha256": _file_sha(ffprobe), "ffprobe_version": ffprobe_version,
                "encode_args": encode_args, "probe_args": probe_args, "decode_args": decode_args,
                "stream": stream, "encoded_mp4_sha256": _file_sha(encoded), "encoded_mp4_bytes": encoded.stat().st_size,
                "input_rgb24_sha256": hashlib.sha256(raw).hexdigest(), "decoded_rgb24_sha256": hashlib.sha256(decoded).hexdigest(),
                "encoded_artifact_retained": False, "input_frame_count": t, "output_frame_count": t,
                "geometry_verified": True, "fps_verified": True, "cpu_encoder": True}
    return output, info


def apply_robustness(video, mode="identity", *, fps, seed=20261102,
                     ffmpeg_bin="ffmpeg", ffprobe_bin="ffprobe", timeout_seconds=180):
    """Return (CPU float32 RGB video, provenance) for one fixed candidate mode.

    Blurring uses one LR-scale Gaussian kernel for all frames. Noise samples
    are independent across frames/pixels while sigma and local seed are fixed
    for the entire video. Combined order is blur, noise, whole-clip H.264.
    Frame count, spatial dimensions and input rational FPS must be preserved.
    """
    started = time.perf_counter()
    specification = protocol_spec(mode, seed=seed)
    rate = fps_fraction(fps)
    if not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive finite")
    value = _validate_video(video)
    input_sha = _tensor_sha(value)
    output, compression = value.clone(), None
    for operation in specification["sequence"]:
        if operation == "blur":
            output = _blur(output)
        elif operation == "noise":
            output = _noise(output, seed)
        elif operation == "compression":
            output, compression = _compress(output, rate, ffmpeg_bin=ffmpeg_bin, ffprobe_bin=ffprobe_bin, timeout=timeout_seconds)
    if output.shape != value.shape or not bool(torch.isfinite(output).all()):
        raise ValueError("Robustness postprocessing changed complete video shape or produced nonfinite RGB")
    provenance = {"protocol": specification, "protocol_sha256": _digest(specification),
                  "code_sha256": _file_sha(__file__), "torch_version": str(torch.__version__),
                  "input_shape": list(value.shape), "output_shape": list(output.shape),
                  "input_dtype": str(video.dtype), "output_dtype": str(output.dtype), "output_device": "cpu",
                  "fps": f"{rate.numerator}/{rate.denominator}", "fps_float": float(rate),
                  "canonical_input_rgb_f32_sha256": input_sha, "output_rgb_f32_sha256": _tensor_sha(output),
                  "compression": compression, "elapsed_seconds": time.perf_counter() - started,
                  "parameters_selected_on_test": False, "real_degradation_fidelity_claimed": False,
                  "training_performed": False}
    return output, provenance
