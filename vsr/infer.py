"""Native x4 restoration of a complete LR input using spatial tiles only.

No HR target or teacher argument exists. Temporal padding repeats the final
input frame; no frames are dropped, reordered, or temporally resampled. The
caller retains the original FPS metadata. Full-sequence CPU RGB/noise/output
buffers are allocated, so use inference.max_frames as an explicit budget cap
when appropriate. There is no implicit prefix selection or temporal windowing.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
import math

import torch
import torch.nn.functional as F

from .vae import inverse_normalize as _inverse_normalize, module_device as _module_device
from .vae import normalize_latent


def _field(config, name):
    return config[name] if isinstance(config, Mapping) else getattr(config, name)


def _geometry(vae):
    temporal = 2 ** sum(bool(flag) for flag in _field(vae.config, "temperal_downsample"))
    spatial = 2 ** (len(_field(vae.config, "dim_mult")) - 1)
    channels = int(_field(vae.config, "z_dim"))
    if (temporal, spatial) != (4, 8):
        raise ValueError("This first Wan2.1 tiler requires actual VAE temporal/spatial strides 4/8")
    if channels <= 0 or len(_field(vae.config, "latents_mean")) != channels or len(_field(vae.config, "latents_std")) != channels:
        raise ValueError("VAE latent normalization does not match configured channels")
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in _field(vae.config, "latents_std")):
        raise ValueError("VAE latent standard deviations must be finite and positive")
    return channels, temporal, spatial


def _rgb(video, label):
    if not isinstance(video, torch.Tensor) or video.ndim != 5 or tuple(video.shape[:2]) != (1, 3) or not all(video.shape):
        raise ValueError(f"{label} must be nonempty RGB [1,3,T,H,W]")
    if not video.is_floating_point() or not bool(torch.isfinite(video).all()):
        raise ValueError(f"{label} must contain finite floating pixels")
    if float(video.min()) < 0 or float(video.max()) > 1:
        raise ValueError(f"{label} RGB must lie in [0,1]")


def _positive_integer(value, label, *, zero=False):
    if not isinstance(value, int) or isinstance(value, bool) or value < (0 if zero else 1):
        raise ValueError(f"{label} must be {'nonnegative' if zero else 'positive'} integer")
    return value


def tile_positions(length, tile_size, overlap):
    """16-aligned tile starts, including a possibly more-overlapped last tile."""
    if length % 16 or tile_size % 16 or overlap % 16:
        raise ValueError("Padded dimensions, tiles and overlap must be multiples of 16")
    if min(length, tile_size) <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Invalid tile or overlap dimensions")
    actual = min(length, tile_size)
    final = length - actual
    positions = list(range(0, final + 1, tile_size - overlap))
    if positions[-1] != final:
        positions.append(final)
    return positions, actual


def _blend_window(height, width):
    # Positive samples of a raised cosine. A floor keeps corner weights well
    # conditioned, and division by accumulated weights preserves image borders.
    def axis(length):
        position = (torch.arange(length, dtype=torch.float32) + 0.5) / length
        return torch.sin(math.pi * position).square().clamp_min(1e-3)
    return axis(height)[:, None] * axis(width)[None, :]


def _clear(model):
    if hasattr(model, "clear_features"):
        model.clear_features()
    if hasattr(model, "clear_reference"):
        model.clear_reference()


@torch.no_grad()
def sample_tile_latent(model, low_latent, text, config, *, steps, seed, initial_noise,
                       frame_count, ref_latent=None, ref_index=None, ref_valid=False):
    """Native UniPC from a shared CPU Gaussian crop; no teacher/HR input."""
    if ref_latent is not None or ref_index is not None or ref_valid:
        raise ValueError('Mainline is NoRef; HRI is a historical archived extension')
    from diffusers import UniPCMultistepScheduler
    device = _module_device(model, low_latent.device)
    if initial_noise.shape != low_latent.shape or not bool(torch.isfinite(initial_noise).all()):
        raise ValueError('Initial Gaussian tile noise must match encoded LR latent')
    low_latent, text = low_latent.to(device), text.to(device)
    x = initial_noise.detach().to(device=device, dtype=torch.float32).clone()
    try:
        scheduler = UniPCMultistepScheduler.from_pretrained(
            config['model']['local_dir'], subfolder='scheduler', local_files_only=True)
        scheduler.set_timesteps(steps, device=device)
        for timestep in scheduler.timesteps:
            context = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
            with context:
                velocity, _ = model(x, timestep.expand(x.shape[0]), text, low_latent, enable_lrv=True)
            x = scheduler.step(velocity.float(), timestep, x, return_dict=False)[0]
        return x.detach().cpu()
    finally:
        _clear(model)



@torch.no_grad()
def restore_video(model, vae, lrv, text, config, *, steps=20, seed=42,
                  tile_hw=(256, 448), overlap=64, reference_image=None,
                  reference_index=None, sampler_fn=None):
    """Return (CPU RGB SR, metadata), exactly T,Hlr*4,Wlr*4.

    ``sampler_fn`` is an optional CPU testing hook. Its inputs are encoded LR
    and explicit noise, never HR targets or teacher features. HRI is rejected.
    """
    _rgb(lrv, "LRV")
    _positive_integer(steps, "steps")
    _positive_integer(seed, "seed", zero=True)
    if len(tile_hw) != 2:
        raise ValueError("tile_hw must contain HR tile height and width")
    tile_h, tile_w = (_positive_integer(value, "tile_hw") for value in tile_hw)
    _positive_integer(overlap, "overlap", zero=True)
    if tile_h % 16 or tile_w % 16 or overlap % 16 or overlap >= min(tile_h, tile_w):
        raise ValueError("HR tiles/overlap must be 16-aligned and overlap smaller than both tile dimensions")
    if not isinstance(text, torch.Tensor) or text.ndim != 3 or text.shape[0] != 1 or not bool(torch.isfinite(text).all()):
        raise ValueError("Pass one finite native text embedding tensor [1,N,C]")
    frames, lr_h, lr_w = map(int, lrv.shape[2:])
    budget = config.get("inference", {}).get("max_frames")
    if budget is not None:
        _positive_integer(budget, "inference.max_frames")
        if frames > budget:
            raise ValueError(f"Input has {frames} frames, exceeding explicit max_frames={budget}; choose a prefix explicitly or raise the budget")
    channels, temporal_stride, spatial_stride = _geometry(vae)
    target_h, target_w = lr_h * 4, lr_w * 4
    padded_t = ((frames - 1 + temporal_stride - 1) // temporal_stride) * temporal_stride + 1
    padded_h, padded_w = ((target_h + 15) // 16) * 16, ((target_w + 15) // 16) * 16
    if reference_image is not None or reference_index is not None:
        raise ValueError('Mainline is NoRef; supplied HRI remains an archived extension')
    enabled = False
    y_positions, actual_h = tile_positions(padded_h, tile_h, overlap)
    x_positions, actual_w = tile_positions(padded_w, tile_w, overlap)
    # Exactly one global x4 interpolation, in full image coordinates, before
    # either padding or tiling. No per-tile rescaling changes border pixels.
    original = lrv.detach().float().cpu()
    upsampled = F.interpolate(original[0].permute(1, 0, 2, 3), size=(target_h, target_w),
                             mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    upsampled = upsampled.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    padded = F.pad(upsampled, (0, padded_w-target_w, 0, padded_h-target_h, 0, padded_t-frames), mode="replicate")
    noise_shape = (1, channels, (padded_t-1)//4 + 1, padded_h//8, padded_w//8)
    noise = torch.randn(noise_shape, generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32)
    accumulator = torch.zeros((1, 3, padded_t, padded_h, padded_w), dtype=torch.float32)
    coverage = torch.zeros((padded_h, padded_w), dtype=torch.float32)
    weights = _blend_window(actual_h, actual_w)
    vae_device, model_device = _module_device(vae, "cpu"), _module_device(model, "cpu")
    vae_dtype = next(vae.parameters(), original).dtype
    previous_training = [(module, module.training) for network in (model, vae) for module in network.modules()]
    model.eval()
    vae.eval()
    _clear(model)
    sampler = sample_tile_latent if sampler_fn is None else sampler_fn
    reference, tiles = None, []
    try:
        for top in y_positions:
            for left in x_positions:
                tile = padded[:, :, :, top:top+actual_h, left:left+actual_w]
                raw = vae.encode((tile * 2 - 1).to(device=vae_device, dtype=vae_dtype)).latent_dist.mode()
                low = normalize_latent(raw, vae)
                expected = (1, channels, noise_shape[2], actual_h//8, actual_w//8)
                if tuple(low.shape) != expected or not bool(torch.isfinite(low).all()):
                    raise ValueError("Tile VAE encoding differs from the configured temporal/spatial geometry")
                initial = noise[:, :, :, top//8:(top+actual_h)//8, left//8:(left+actual_w)//8]
                latent = sampler(model, low, text, config, steps=steps, seed=seed, initial_noise=initial,
                    frame_count=frames, ref_latent=reference, ref_index=reference_index if enabled else None, ref_valid=enabled)
                if tuple(latent.shape) != expected or not bool(torch.isfinite(latent).all()):
                    raise ValueError("Sampler returned an invalid tile latent")
                decoded = vae.decode(_inverse_normalize(latent.to(device=vae_device, dtype=vae_dtype), vae)).sample
                if tuple(decoded.shape) != (1, 3, padded_t, actual_h, actual_w) or not bool(torch.isfinite(decoded).all()):
                    raise ValueError("VAE decoding changed tile spatial/temporal geometry or became nonfinite")
                rgb = (decoded.detach().float().cpu()/2 + .5).clamp(0, 1)
                accumulator[:, :, :, top:top+actual_h, left:left+actual_w] += rgb * weights
                coverage[top:top+actual_h, left:left+actual_w] += weights
                tiles.append({"hr_tlhw": [top, left, actual_h, actual_w],
                              "latent_tlhw": [top//8, left//8, actual_h//8, actual_w//8]})
                _clear(model)
        if not bool((coverage > 0).all()):
            raise RuntimeError("Spatial tiling left uncovered output pixels")
        restored = (accumulator / coverage)[:, :, :frames, :target_h, :target_w].contiguous()
        if not bool(torch.isfinite(restored).all()):
            raise RuntimeError("Spatial blending produced nonfinite output")
        metadata = {
            "scale": 4, "input_shape": list(lrv.shape), "output_shape": list(restored.shape),
            "padded_shape": list(padded.shape), "input_frames": frames, "output_frames": frames,
            "fps_policy": "caller retains original FPS unchanged; no temporal resampling",
            "padding": {"mode": "replicate", "time_end": padded_t-frames,
                        "bottom": padded_h-target_h, "right": padded_w-target_w},
            "spatial_resampling": "one whole-input CPU bicubic x4, antialias=True, align_corners=False, clamp[0,1]",
            "temporal_windowing": False, "explicit_max_frames": budget,
            "tile_hw_requested": [tile_h, tile_w], "overlap_requested": overlap,
            "tile_positions_multiple": 16, "tile_count": len(tiles), "tiles": tiles,
            "blend": {"window": "positive raised cosine with 1e-3 floor on each axis",
                      "all_pixels_covered": True, "minimum_weight_sum": float(coverage.min())},
            "sampling": {"steps": steps, "seed": seed, "cfg": 1.0,
                         "sampler": "local official-config UniPC" if sampler_fn is None else "injected_sampler",
                         "global_noise_shape": list(noise_shape), "noise_device": "cpu",
                         "noise_rule": "one Gaussian field per video, crop spatial offsets/8 for every tile"},
            "reference": {"enabled": enabled, "index": reference_index if enabled else None,
                          "original_frame_count": frames, "encode_count": 1 if enabled else 0,
                          "latent_shape": list(reference.shape) if enabled else None,
                          "context": "same whole supplied HRI for every spatial tile" if enabled else "None with explicit false validity"},
            "inference_inputs": ["LRV", "native text embedding"] + (["supplied HRI", "known original index"] if enabled else []),
            "full_sequence_cpu_buffers_bytes": {"upsampled_rgb": upsampled.numel()*upsampled.element_size(),
                "padded_rgb": padded.numel()*padded.element_size(), "gaussian_latent": noise.numel()*noise.element_size(),
                "accumulator_and_weights": accumulator.numel()*accumulator.element_size()+coverage.numel()*coverage.element_size()},
        }
        return restored, metadata
    finally:
        _clear(model)
        for module, training in previous_training:
            module.training = training


# Video CLI: cached text avoids loading a text encoder as a hidden dependency.
import argparse
import copy
import gc
import os
from pathlib import Path
import time
from .checkpoints import (PRODUCTION_RUNS_ROOT, _check_config, _safe_checkpoint,
                          load_training_checkpoint)
from .data import file_hash, write_video, write_json_new, read_json
from .checkpoints import gpu_lease
from .model import configure_reproducibility
from .vae import load_frozen_vae

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


def build_parser():
    parser = argparse.ArgumentParser(description='NoRef x4 VSR: student checkpoint + LR video + cached text.')
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('preview', 'run'):
        sub = commands.add_parser(command)
        sub.add_argument('--checkpoint', required=True, type=Path)
        sub.add_argument('--allowed-checkpoint-root', required=True, type=Path)
        sub.add_argument('--model-dir', type=Path, help='Explicit local official model path override; source checkpoint config is preserved')
        if command == 'run':
            sub.add_argument('--input', required=True, type=Path)
            sub.add_argument('--output', required=True, type=Path)
            sub.add_argument('--text-embedding', required=True, type=Path)
            sub.add_argument('--caption', default='A high quality video.')
            sub.add_argument('--device', default='cpu')
            sub.add_argument('--gpu-lock', type=Path)
            sub.add_argument('--gpu-admission', type=Path)
            sub.add_argument('--frames', type=_positive)
            sub.add_argument('--seed', type=_nonnegative, default=42)
            sub.add_argument('--steps', type=_positive, default=20)
            sub.add_argument('--tile-height', type=_positive, default=256)
            sub.add_argument('--tile-width', type=_positive, default=448)
            sub.add_argument('--overlap', type=_nonnegative, default=64)
            sub.add_argument('--save-rgb', action='store_true')
            sub.add_argument('--execute', action='store_true', help='Explicitly permit model loading and inference')
    return parser


def _cached_text(path, caption):
    value = torch.load(Path(path), map_location='cpu', weights_only=True)
    embedding = value.get(caption) if isinstance(value, dict) else value
    if (not isinstance(embedding, torch.Tensor) or embedding.ndim != 3 or embedding.shape[0] != 1
            or not embedding.is_floating_point() or not bool(torch.isfinite(embedding).all())):
        raise ValueError('Cached text must be finite floating [1,N,C], or a caption-keyed dictionary of that tensor')
    return embedding.detach().cpu()


def _run_inference_unleased(args, *, device=None, gpu_receipt=None):
    """Explicit user-file restoration; no training, metrics or remote access."""
    if not getattr(args, 'execute', False):
        raise ValueError('Inference requires explicit --execute; preview does not load a model')
    device = torch.device(device if device is not None else args.device)
    if (args.tile_height % 16 or args.tile_width % 16 or args.overlap % 16
            or args.overlap >= min(args.tile_height, args.tile_width)):
        raise ValueError('Tiles/overlap must be16-aligned, with overlap smaller than both dimensions')
    output = Path(args.output).expanduser().resolve()
    metadata_path = output.with_suffix(output.suffix + '.metadata.json')
    rgb_path = output.with_suffix(output.suffix + '.rgb.pt')
    if output.suffix.lower() != '.mp4':
        raise ValueError('Output must be a new .mp4 path')
    if output.exists() or metadata_path.exists() or (args.save_rgb and rgb_path.exists()):
        raise FileExistsError('Output or requested sidecar already exists')
    started = time.perf_counter()
    preview = preview_training_config(args.checkpoint, allowed_root=args.allowed_checkpoint_root)
    config = preview['config']
    configure_reproducibility(config)
    text = _cached_text(args.text_embedding, args.caption)
    lrv, source = decode_lrv(args.input, frame_limit=args.frames)
    model = vae = None
    try:
        model, audit = load_training_checkpoint(preview['path'], device=device, allowed_root=args.allowed_checkpoint_root,
                                                model_directory=getattr(args, 'model_dir', None))
        if audit['checkpoint_sha256'] != preview['checkpoint_sha256'] or audit['config'] != config:
            raise ValueError('Checkpoint changed between preview and restore')
        runtime_config = copy.deepcopy(config)
        if getattr(args, 'model_dir', None) is not None:
            runtime_config['model']['local_dir'] = str(args.model_dir.expanduser().resolve())
        vae = load_frozen_vae(runtime_config, device=device)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        sampling_started = time.perf_counter()
        restored, tiling = restore_video(model, vae, lrv, text, runtime_config,
            steps=args.steps, seed=args.seed, tile_hw=(args.tile_height,args.tile_width), overlap=args.overlap)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_video(output, restored, source['fps'])
        encoded = verify_output_video(output, frames=lrv.shape[2], height=lrv.shape[-2]*4,
                                      width=lrv.shape[-1]*4, fps=source['fps'])
        if args.save_rgb: torch.save(restored.detach().cpu(), rgb_path)
        metadata = {'status':'completed','input':{**source,'sha256':file_hash(args.input)},
            'output':{'path':str(output),'shape':list(restored.shape),'requested_fps':source['fps'],
                      'encoded_container':encoded,'audio_preserved':False},
            'checkpoint':audit,'tiling':tiling,'text':{'caption':args.caption,
                'path':str(args.text_embedding),'sha256':file_hash(args.text_embedding),
                'shape':list(text.shape),'text_encoder_loaded':False},
            'resources':{'device':str(device),'restore_seconds':time.perf_counter()-sampling_started,
                'end_to_end_seconds':time.perf_counter()-started},
            'source_code_sha256':{'vsr/infer.py':file_hash(Path(__file__))},
            'gpu_lease':gpu_receipt,
            'inference_inputs':['LRV','cached text','student checkpoint'],
            'hr_target_accessed':False,'teacher_accessed':False,'training_updates':0}
        write_json_new(metadata_path, metadata)
        return metadata
    finally:
        if model is not None: _clear(model)
        del model,vae
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()



def run_inference(args, *, device=None):
    """Keep the shared GPU lease through model/VAE cleanup when CUDA is used."""
    if not getattr(args, 'execute', False):
        raise ValueError('Inference requires explicit --execute; preview does not load a model')
    selected = torch.device(device if device is not None else args.device)
    if selected.type != 'cuda':
        return _run_inference_unleased(args, device=selected)
    if getattr(args, 'gpu_lock', None) is None or getattr(args, 'gpu_admission', None) is None:
        raise ValueError('CUDA inference requires --gpu-lock and --gpu-admission')
    with gpu_lease(args.gpu_lock, selected, read_json(args.gpu_admission)) as receipt:
        # _run_inference_unleased owns a finally block that clears all models.
        # The outer lease is released only after that cleanup has completed.
        return _run_inference_unleased(args, device=selected, gpu_receipt=receipt)

def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'preview':
        import json
        print(json.dumps(preview_training_config(args.checkpoint, allowed_root=args.allowed_checkpoint_root),
                         ensure_ascii=False, indent=2))
    else:
        metadata = run_inference(args)
        print('Saved '+metadata['output']['path'])


if __name__ == '__main__':
    main()
