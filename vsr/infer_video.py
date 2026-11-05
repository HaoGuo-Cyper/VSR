"""User-file inference: LRV + caption + optional one supplied HRI -> x4 video.

All original frames and their FPS are retained unless --frames explicitly
selects a prefix. No HR target, teacher, optical flow or training RGB cache is
read. Only checkpoints inside the new VSR runs tree may be restored.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import gc
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".deps"))

import torch

from .audit import dump, file_hash
from .checkpoints import (PRODUCTION_RUNS_ROOT, _check_config, _safe_checkpoint,
                          load_training_checkpoint)
from .data import write_video
from .inference import restore_video
from .reproducibility import configure_reproducibility


def _positive(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint under /home/hugo/Projects/VSR/runs")
    parser.add_argument("--input", required=True, help="User-supplied low-resolution video")
    parser.add_argument("--output", required=True, help="New output .mp4 path")
    parser.add_argument("--caption", default="A high quality video.")
    parser.add_argument("--hri", help="Optional supplied HR reference image, exact x4 target geometry")
    parser.add_argument("--hri-index", type=_nonnegative, help="Known reference index in the original input clip")
    parser.add_argument("--frames", type=_positive, default=None, help="Explicitly select this prefix; default processes ALL input frames")
    parser.add_argument("--seed", type=_nonnegative, default=42)
    parser.add_argument("--steps", type=_positive, default=20)
    parser.add_argument("--tile-height", type=_positive, default=256)
    parser.add_argument("--tile-width", type=_positive, default=448)
    parser.add_argument("--overlap", type=_nonnegative, default=64)
    parser.add_argument("--save-rgb", action="store_true", help="Also save potentially large lossless float32 RGB tensor")
    return parser


def decode_lrv(path, *, frame_limit=None):
    """Sequentially decode to EOF, or to an explicitly requested prefix."""
    import cv2
    import numpy as np
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input LRV does not exist: {path}")
    if frame_limit is not None and (not isinstance(frame_limit, int) or isinstance(frame_limit, bool) or frame_limit < 1):
        raise ValueError("Explicit frame prefix must be a positive integer")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open user LRV")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Input LRV lacks a valid original frame rate")
        reported = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        reported_count = int(round(reported)) if math.isfinite(reported) and reported > 0 else None
        frames, shape = [], None
        while frame_limit is None or len(frames) < frame_limit:
            okay, frame = capture.read()
            if not okay:
                break
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("LRV decoder did not return three-channel frames")
            if shape is not None and frame.shape != shape:
                raise ValueError("Input LRV changes spatial resolution within the clip")
            shape = frame.shape
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not frames:
            raise ValueError("User LRV contains no decodable frames")
        if frame_limit is not None and len(frames) != frame_limit:
            raise ValueError("Explicit prefix exceeds the number of decodable input frames")
        if frame_limit is None and reported_count is not None and len(frames) != reported_count:
            raise ValueError(f"Decoded {len(frames)} frames but input metadata reports {reported_count}; refusing implicit truncation")
    finally:
        capture.release()
    tensor = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0).contiguous().float().div_(255)
    return tensor, {"path": str(path), "fps": fps, "decoded_frames": len(frames),
                    "reported_source_frame_count": reported_count, "full_input_decoded": frame_limit is None,
                    "explicit_prefix_frames": frame_limit, "source_frame_indices": list(range(len(frames))),
                    "rgb_shape": list(tensor.shape), "temporal_resampling": False}


def read_hri(path, target_hw):
    import numpy as np
    from PIL import Image, ImageOps
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Supplied HRI does not exist: {path}")
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if (image.height, image.width) != tuple(target_hw):
            raise ValueError(f"Supplied HRI must have exact x4 target size {tuple(target_hw)}, received {(image.height,image.width)}")
        pixels = np.asarray(image, dtype="float32").copy() / 255
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).contiguous()


def preview_training_config(checkpoint, *, allowed_root=PRODUCTION_RUNS_ROOT):
    """CPU-only safe preview before loading the large UMT5 encoder."""
    path = _safe_checkpoint(checkpoint, allowed_root)
    digest = file_hash(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "vsr_training_v1":
        raise ValueError("Unsupported VSR training checkpoint")
    step = payload.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("Checkpoint requires at least one completed training update")
    config = copy.deepcopy(payload.get("config"))
    _check_config(config)
    variant = path.parent.name
    if variant not in config["training"]["variants"]:
        raise ValueError("Checkpoint parent must identify its recorded training variant")
    if payload.get("variant") is not None and payload["variant"] != variant:
        raise ValueError("Checkpoint variant differs from its directory")
    del payload
    gc.collect()
    return {"path": str(path), "checkpoint_sha256": digest, "config": config, "variant": variant, "step": step}


@torch.no_grad()
def encode_caption(config, caption, *, device="cuda"):
    """Use native Wan T5 cleaning/masking, then release UMT5 before DiT/VAE."""
    from diffusers import WanPipeline
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, UMT5EncoderModel
    device = torch.device(device)
    base = config["model"]["local_dir"]
    tokenizer = AutoTokenizer.from_pretrained(base, subfolder="tokenizer", local_files_only=True)
    encoder = UMT5EncoderModel.from_pretrained(base, subfolder="text_encoder", dtype=torch.bfloat16,
        local_files_only=True, use_safetensors=True).requires_grad_(False).eval().to(device)
    try:
        context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with context:
            # The native method only needs these two components when device is
            # explicit; no diffusion model or VAE is constructed to encode text.
            embedding = WanPipeline._get_t5_prompt_embeds(SimpleNamespace(tokenizer=tokenizer, text_encoder=encoder),
                prompt=caption, max_sequence_length=512, device=device, dtype=torch.bfloat16).detach().cpu()
        audit = {"caption_provided": caption, "caption_after_native_cleaning": prompt_clean(caption),
                 "caption_source": "user CLI argument (default generic only when argument omitted)",
                 "max_sequence_length": 512, "embedding_shape": list(embedding.shape), "encoder_frozen": True,
                 "native_method": "WanPipeline._get_t5_prompt_embeds", "training_caption_semantics_verified": False,
                 "encoder_has_gradient": any(parameter.grad is not None for parameter in encoder.parameters())}
        return embedding, audit
    finally:
        del encoder, tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def load_frozen_vae(config, *, device="cuda"):
    from diffusers import AutoencoderKLWan
    return AutoencoderKLWan.from_pretrained(config["model"]["local_dir"], subfolder="vae", dtype=torch.float32,
        local_files_only=True, use_safetensors=True).requires_grad_(False).eval().to(device)


def verify_output_video(path, *, frames, height, width, fps):
    """Audit encoded container geometry/FPS after the existing MP4 writer."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Written output video cannot be reopened")
        actual = {"frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
                  "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  "fps": float(capture.get(cv2.CAP_PROP_FPS))}
        if (actual["frame_count"], actual["height"], actual["width"]) != (frames, height, width):
            raise ValueError(f"Encoded output geometry changed: {actual}")
        if not math.isfinite(actual["fps"]) or not math.isclose(actual["fps"], fps, rel_tol=1e-4, abs_tol=1e-5):
            raise ValueError("Encoded output frame rate differs from original input FPS")
        return actual
    finally:
        capture.release()


def run_inference(args, *, device="cuda"):
    started = time.perf_counter()
    device = torch.device(device)
    if bool(args.hri) != (args.hri_index is not None):
        raise ValueError("Supply --hri and its known --hri-index together")
    if (args.tile_height % 16 or args.tile_width % 16 or args.overlap % 16
            or args.overlap >= min(args.tile_height, args.tile_width)):
        raise ValueError("HR tiles and overlap must be 16-aligned, with overlap smaller than both tile dimensions")
    output = Path(args.output).expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must be a new .mp4 path")
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    rgb_path = output.with_suffix(output.suffix + ".rgb.pt")
    if output.exists() or metadata_path.exists() or (args.save_rgb and rgb_path.exists()):
        raise FileExistsError("Output or requested sidecar already exists; choose a new output path")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    preview = preview_training_config(args.checkpoint)
    config = preview["config"]
    configure_reproducibility(config)
    if args.hri and not config.get("reference", {}).get("enabled", False):
        raise ValueError("Supplied HRI requires a reference-trained checkpoint")
    lrv, source = decode_lrv(args.input, frame_limit=args.frames)
    if args.hri_index is not None and args.hri_index >= lrv.shape[2]:
        raise ValueError("HRI index lies outside the explicitly selected original input frames")
    reference = read_hri(args.hri, (lrv.shape[-2]*4, lrv.shape[-1]*4)) if args.hri else None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("User-video Wan inference requires a visible CUDA GPU")
        torch.cuda.reset_peak_memory_stats(device)
    # UMT5 must be fully released before loading the DiT and VAE on this 16GB
    # machine. The returned embedding is detached and resident on CPU.
    text, caption_audit = encode_caption(config, args.caption, device=device)
    model = vae = None
    try:
        model, checkpoint_audit = load_training_checkpoint(preview["path"], device=device)
        if checkpoint_audit["checkpoint_sha256"] != preview["checkpoint_sha256"] or checkpoint_audit["config"] != config:
            raise ValueError("Checkpoint changed between CPU preview and model restore")
        vae = load_frozen_vae(config, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        sampling_started = time.perf_counter()
        restored, tiling = restore_video(model, vae, lrv, text, config, steps=args.steps, seed=args.seed,
            tile_hw=(args.tile_height, args.tile_width), overlap=args.overlap,
            reference_image=reference, reference_index=args.hri_index)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        sampling_seconds = time.perf_counter() - sampling_started
        expected = (1, 3, lrv.shape[2], lrv.shape[-2]*4, lrv.shape[-1]*4)
        if tuple(restored.shape) != expected or not bool(torch.isfinite(restored).all()):
            raise ValueError("Restoration changed x4 geometry/frame count or returned nonfinite pixels")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_video(output, restored, source["fps"])
        encoded = verify_output_video(output, frames=expected[2], height=expected[3], width=expected[4], fps=source["fps"])
        if args.save_rgb:
            torch.save(restored.detach().cpu(), rgb_path)
        metadata = {"status": "completed", "input": {**source, "sha256": file_hash(Path(args.input).expanduser().resolve())},
                    "output": {"path": str(output), "shape": list(restored.shape), "requested_fps": source["fps"],
                               "encoded_container": encoded, "lossless_rgb_path": str(rgb_path) if args.save_rgb else None,
                               "audio_preserved": False, "input_audio_inspected": False, "playback_codec": "mp4v"},
                    "caption": caption_audit, "checkpoint": checkpoint_audit, "tiling": tiling,
                    "reference": {"provided": reference is not None, "path": str(Path(args.hri).expanduser().resolve()) if args.hri else None,
                                  "known_original_frame_index": args.hri_index, "geometry_policy": "exact x4 target image; no resize",
                                  "exif_orientation_applied": reference is not None},
                    "resources": {"device": str(device), "restore_seconds": sampling_seconds,
                                  "end_to_end_seconds": time.perf_counter()-started,
                                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                                  "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                                  "load_order": ["CPU checkpoint config", "UMT5 encode and release", "checkpoint DiT", "frozen VAE"]},
                    "source_code_sha256": {"vsr/infer_video.py": file_hash(Path(__file__)),
                                           "vsr/inference.py": file_hash(Path(__file__).with_name("inference.py"))},
                    "inference_inputs": ["user LRV", "user caption"] + (["one supplied HRI", "known HRI index"] if reference is not None else []),
                    "hr_target_accessed": False, "teacher_accessed": False, "flow_accessed": False,
                    "quality_conclusion": "user-file inference output; no ground-truth metric or generalization claim"}
        dump(metadata_path, metadata)
        return metadata
    finally:
        if model is not None:
            model.clear_features()
            if callable(getattr(model, "clear_reference", None)):
                model.clear_reference()
        del model, vae
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    args = build_parser().parse_args()
    metadata = run_inference(args)
    print(f"Saved {metadata['output']['path']} ({metadata['output']['shape'][2]} frames, {metadata['output']['requested_fps']} FPS)")


if __name__ == "__main__":
    main()
