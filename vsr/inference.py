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

from .evaluation import _inverse_normalize, _module_device
from .preflight import normalize_latent


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
    """Native UniPC sampling from an explicitly supplied global-noise crop."""
    from diffusers import UniPCMultistepScheduler
    device = _module_device(model, low_latent.device)
    if initial_noise.shape != low_latent.shape or not bool(torch.isfinite(initial_noise).all()):
        raise ValueError("Initial Gaussian tile noise must match the encoded LR latent")
    reference_capable = hasattr(model, "clear_reference")
    if ref_valid and not reference_capable:
        raise ValueError("Supplied HRI requires a reference-trained model")
    low_latent, text = low_latent.to(device), text.to(device)
    x = initial_noise.detach().to(device=device, dtype=torch.float32).clone()
    reference = ref_latent.to(device) if ref_valid else None
    try:
        scheduler = UniPCMultistepScheduler.from_pretrained(
            config["model"]["local_dir"], subfolder="scheduler", local_files_only=True)
        scheduler.set_timesteps(steps, device=device)
        for timestep in scheduler.timesteps:
            reference_kwargs = {}
            if reference_capable:
                reference_kwargs = {"ref_latent": reference, "ref_index": ref_index if ref_valid else None,
                    "frame_count": frame_count,
                    "ref_valid": torch.full((x.shape[0],), ref_valid, device=device, dtype=torch.bool)}
            context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with context:
                velocity, _ = model(x, timestep.expand(x.shape[0]), text, low_latent,
                                     enable_lrv=True, **reference_kwargs)
            x = scheduler.step(velocity.float(), timestep, x, return_dict=False)[0]
        return x.detach().cpu()
    finally:
        _clear(model)


@torch.no_grad()
def restore_video(model, vae, lrv, text, config, *, steps=20, seed=42,
                  tile_hw=(256, 448), overlap=64, reference_image=None,
                  reference_index=None, sampler_fn=None):
    """Return (CPU RGB SR, metadata), exactly T,Hlr*4,Wlr*4.

    HRI, when present, is one supplied RGB frame at the complete target spatial
    resolution, plus its known index in the original unpadded video. It is VAE
    encoded as a whole single-frame video once and shared with every tile.
    ``sampler_fn`` is an optional testing hook with sample_tile_latent's exact
    signature; it receives no pixels other than encoded LR and optional HRI.
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
    enabled = reference_image is not None
    if enabled:
        _rgb(reference_image, "HRI")
        if tuple(reference_image.shape[2:]) != (1, target_h, target_w):
            raise ValueError("HRI must be one whole supplied frame at the exact x4 target spatial size")
        _positive_integer(reference_index, "reference_index", zero=True)
        if reference_index >= frames:
            raise ValueError("Known reference index lies outside the original input video")
        if not hasattr(model, "clear_reference"):
            raise ValueError("HRI input requires a reference-trained model")
    elif reference_index is not None:
        raise ValueError("A reference_index requires an actually supplied HRI")
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
        if enabled:
            provided = F.pad(reference_image.detach().float().cpu(),
                             (0, padded_w-target_w, 0, padded_h-target_h, 0, 0), mode="replicate")
            raw = vae.encode((provided * 2 - 1).to(device=vae_device, dtype=vae_dtype)).latent_dist.mode()
            reference = normalize_latent(raw, vae).detach().to(model_device)
            if tuple(reference.shape) != (1, channels, 1, padded_h//8, padded_w//8):
                raise ValueError("Whole HRI VAE encoding changed expected single-frame geometry")
            if not bool(torch.isfinite(reference).all()):
                raise ValueError("Whole HRI encoding is nonfinite")
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
