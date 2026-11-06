"""Oracle-noised-HR training diagnostics, never restoration inference metrics.

The denoising probe intentionally constructs x_sigma from the clean target.
It can localize failure to noise levels but cannot establish LRV-only recovery.
No probe output is substituted for sampled-video evaluation.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .audit import dump


def _device(model, fallback):
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device(fallback)


def _noise_seed(seed, record_id, sigma):
    key = f"fixed-denoising-probe-v1:{int(seed)}:{record_id}:{float(sigma):.12g}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)


def _mean(rows):
    return {key: sum(row[key] for row in rows) / len(rows)
            for key in ("fm_mse", "x0hat_latent_mse", "zero_velocity_fm_mse", "zero_velocity_x0_latent_mse")}


@torch.no_grad()
def evaluate_denoising_probe(model, encoded_pairs, clips, text_embeddings, config, output_dir,
                             modes=("correct", "disabled", "shuffled"), sigmas=(.1, .5, .9, .99),
                             seeds=(73,), sample_indices=None):
    """Measure fixed-noise FM errors at explicitly fixed sigma values.

    Clean normalized HR latents are used *only* to make this oracle training
    diagnostic's noised input and target. Wrong-LR controls use the next clip
    circularly; text, x_sigma and target remain identical across modes.
    Native Wan flow convention is velocity=noise-clean and timestep=1000*sigma.
    """
    modes, sigmas, seeds = tuple(modes), tuple(float(s) for s in sigmas), tuple(int(s) for s in seeds)
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"correct", "disabled", "shuffled"}:
        raise ValueError("Modes must be distinct correct/disabled/shuffled values")
    if not sigmas or len(set(sigmas)) != len(sigmas) or any(not math.isfinite(s) or s <= 0 or s > 1 for s in sigmas):
        raise ValueError("Probe sigmas must be distinct finite values in (0,1]")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Probe seeds must be distinct nonnegative integers")
    if not clips or len(clips) != len(encoded_pairs):
        raise ValueError("Each probe clip must have a clean/low latent pair")
    if "shuffled" in modes and len(clips) < 2:
        raise ValueError("Shuffled LR probe requires at least two clips")
    indices = list(range(len(clips))) if sample_indices is None else list(sample_indices)
    if not indices or len(set(indices)) != len(indices) or any(not isinstance(i, int) or i < 0 or i >= len(clips) for i in indices):
        raise ValueError("Invalid probe sample indices")
    device = _device(model, encoded_pairs[indices[0]]["clean"].device)
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    model.clear_features()
    per_video = []
    try:
        for index in indices:
            metadata = clips[index]["metadata"]
            record_id = metadata.get("record_id", str(index))
            clean = encoded_pairs[index]["clean"].to(device=device, dtype=torch.float32)
            low = encoded_pairs[index]["low"]
            if clean.ndim != 5 or clean.shape != low.shape or not bool(torch.isfinite(clean).all()):
                raise ValueError("Probe needs matching finite [B,C,T,H,W] clean/low latents")
            caption = metadata.get("caption", "")
            text = text_embeddings[caption].to(device)
            video = {"clip_index": index, "record_id": record_id, "source_id": metadata.get("source_id"),
                     "split": metadata.get("split"), "by_sigma": {}}
            for sigma in sigmas:
                key = f"{sigma:.12g}"
                sigma_results = {mode: [] for mode in modes}
                for seed in seeds:
                    actual_seed = _noise_seed(seed, record_id, sigma)
                    rng = torch.Generator(device=device).manual_seed(actual_seed)
                    noise = torch.randn(clean.shape, generator=rng, device=device, dtype=torch.float32)
                    noisy = (1 - sigma) * clean + sigma * noise
                    target = noise - clean
                    timestep = torch.full((clean.shape[0],), sigma * 1000, device=device, dtype=torch.float32)
                    baseline_fm = float(target.square().mean())
                    baseline_x0 = float((noisy - clean).square().mean())
                    for mode in modes:
                        donor = (index + 1) % len(clips) if mode == "shuffled" else index
                        condition = encoded_pairs[donor]["low"].to(device)
                        if condition.shape != clean.shape or not bool(torch.isfinite(condition).all()):
                            raise ValueError("Probe controls require matching finite LR latent geometry")
                        context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
                        try:
                            with context:
                                prediction, _ = model(noisy, timestep, text, condition, enable_lrv=mode != "disabled")
                            if prediction.shape != target.shape or not bool(torch.isfinite(prediction).all()):
                                raise ValueError("Probe model returned invalid velocity")
                            velocity = prediction.float()
                            x0hat = noisy - sigma * velocity
                            sigma_results[mode].append({
                                "seed": seed, "noise_seed": actual_seed, "sigma": sigma, "timestep": sigma * 1000,
                                "fm_mse": float(F.mse_loss(velocity, target)),
                                "x0hat_latent_mse": float(F.mse_loss(x0hat, clean)),
                                "zero_velocity_fm_mse": baseline_fm,
                                "zero_velocity_x0_latent_mse": baseline_x0,
                                "lrv_enabled": mode != "disabled",
                                "conditioning_record_id": clips[donor]["metadata"].get("record_id", str(donor))})
                        finally:
                            model.clear_features()
                video["by_sigma"][key] = {
                    mode: {"seeds": rows, "mean": _mean(rows)} for mode, rows in sigma_results.items()}
            per_video.append(video)
        aggregate = {f"{sigma:.12g}": {mode: _mean([video["by_sigma"][f"{sigma:.12g}"][mode]["mean"] for video in per_video])
                                                for mode in modes} for sigma in sigmas}
        summary = {"status": "completed", "kind": "oracle_noised_hr_training_diagnostic",
                   "restoration_inference": False, "generalization_claim": False,
                   "protocol": {"sigmas": list(sigmas), "seeds": list(seeds), "modes": list(modes),
                                "noise": "independent SHA256(seed,record_id,sigma) generator; identical across controls",
                                "interpolation": "x_sigma=(1-sigma)*clean+sigma*noise",
                                "native_timestep": "1000*sigma", "target": "noise-clean",
                                "x0hat": "x_sigma-sigma*predicted_velocity",
                                "clean_target_access": "explicitly used to construct noised input and diagnostic targets",
                                "metric_dependency": "x0hat latent MSE equals sigma^2*FM MSE up to floating point",
                                "aggregation": "mean seeds per video, then equal video weights",
                                "warning": "HR-dependent denoising probes cannot replace LRV-only sampled-video evaluation"},
                   "official_revision": config.get("model", {}).get("revision"),
                   "per_video": per_video, "aggregate_by_sigma": aggregate}
        dump(Path(output_dir) / "denoising_probe.json", summary)
        return summary
    finally:
        model.clear_features()
        for module, was_training in flags:
            module.training = was_training


@torch.no_grad()
def lr_reprojection_error(predicted_hr, input_lr):
    """Custom LR data-consistency diagnostic using the protocol's bicubic kernel.

    This rewards agreement with supplied LR pixels. It is not an established
    perceptual/temporal metric and does not establish HR detail correctness.
    """
    if predicted_hr.ndim != 5 or input_lr.ndim != 5 or predicted_hr.shape[:3] != input_lr.shape[:3] or predicted_hr.shape[:2] != (1, 3):
        raise ValueError("Expected matching RGB [1,3,T,H,W] with only spatial size differing")
    if any(size <= 0 for size in (*predicted_hr.shape, *input_lr.shape)):
        raise ValueError("Video dimensions must be positive")
    h, w = predicted_hr.shape[-2:]
    lh, lw = input_lr.shape[-2:]
    if h % lh or w % lw or h // lh != w // lw:
        raise ValueError("HR and supplied LR must share one integer spatial scale")
    for video in (predicted_hr, input_lr):
        if not bool(torch.isfinite(video).all()) or float(video.min()) < -1e-6 or float(video.max()) > 1 + 1e-6:
            raise ValueError("LR reprojection expects finite RGB values in [0,1]")
    hr_frames = predicted_hr.detach().float().cpu()[0].permute(1, 0, 2, 3)
    lr_frames = input_lr.detach().float().cpu()[0].permute(1, 0, 2, 3)
    projected = F.interpolate(hr_frames, size=(lh, lw), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    per_frame = (projected - lr_frames).square().flatten(1).mean(1)
    return {"lr_reprojection_mse_rgb_01": float(per_frame.mean()), "per_frame": per_frame.tolist(),
            "scale": h // lh, "definition": "bicubic antialias, align_corners=False, clamp[0,1]; MSE against supplied LR",
            "interpretation": "custom LR consistency diagnostic, not HR restoration quality"}
