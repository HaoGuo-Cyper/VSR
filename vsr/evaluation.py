"""Paired restoration and LR-condition controls on bounded debug clips.

Only the encoded LRV, caption embedding and Gaussian noise enter the sampler.
HR tensors are consumed after decoding for metrics. Results on the overfit
clips diagnose the training loop; they do not establish generalization.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
import re
import time

import torch

from .audit import dump
from .data import write_video


METRIC_KEYS = ("psnr_rgb_db", "ssim_rgb", "mse_rgb_01")
PSNR_MSE_FLOOR = 1e-12


def _validate_video(video):
    if video.ndim != 5 or video.shape[:2] != (1, 3) or not all(video.shape):
        raise ValueError("Expected nonempty RGB [1,3,T,H,W] video")
    if not bool(torch.isfinite(video).all()):
        raise ValueError("Metrics require finite RGB pixels")
    if float(video.min()) < -1e-6 or float(video.max()) > 1 + 1e-6:
        raise ValueError("Metrics require RGB values in [0,1]")


def rgb_metrics(prediction, target):
    """Frame-mean RGB PSNR and skimage SSIM; no border shave or Y transform.

    PSNR's MSE floor is 1e-12 (120 dB maximum) so exact matches remain valid
    finite JSON. The number of exact frames and the floor are recorded.
    SSIM uses skimage defaults: 7x7 uniform window, sample covariance and
    channel_axis=-1, data_range=1. Small windows are explicitly rejected.
    """
    from skimage.metrics import structural_similarity
    _validate_video(prediction)
    _validate_video(target)
    if prediction.shape != target.shape:
        raise ValueError("Metric inputs must have exactly equal frame/spatial shapes")
    if min(prediction.shape[-2:]) < 7:
        raise ValueError("SSIM requires at least 7 pixels in each spatial dimension")
    pred = prediction.detach().float().cpu()[0].permute(1, 2, 3, 0).numpy()
    truth = target.detach().float().cpu()[0].permute(1, 2, 3, 0).numpy()
    per_frame = []
    for p, t in zip(pred, truth):
        mse = float(((p.astype("float64") - t.astype("float64")) ** 2).mean())
        per_frame.append({"mse_rgb_01": mse,
                          "psnr_rgb_db": -10 * math.log10(max(mse, PSNR_MSE_FLOOR)),
                          "ssim_rgb": float(structural_similarity(p, t, channel_axis=-1, data_range=1.0))})
    return {**{key: sum(frame[key] for frame in per_frame) / len(per_frame) for key in METRIC_KEYS},
            "frame_count": len(per_frame), "exact_match_frames": sum(x["mse_rgb_01"] == 0 for x in per_frame),
            "per_frame": per_frame}


def _mean_metrics(rows):
    if not rows:
        raise ValueError("Cannot aggregate an empty metric group")
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in METRIC_KEYS}


def aggregate_per_video(per_video, modes):
    """Give every video equal weight after averaging its sampling seeds."""
    if not per_video:
        raise ValueError("No videos to aggregate")
    return {"video_count": len(per_video),
            "bicubic": _mean_metrics([row["bicubic"] for row in per_video]),
            "modes": {mode: _mean_metrics([row["modes"][mode]["mean"] for row in per_video]) for mode in modes}}


def _module_device(module, fallback):
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    return buffer.device if buffer is not None else torch.device(fallback)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def sample_conditioned_latent(model, low_latent, text, config, *, steps, seed, enable_lrv=True):
    """Sample with local native UniPC; the interface cannot accept HR/teacher.

    A new generator initialized to the same seed is used for each condition
    mode. Matched latent shape therefore gives identical starting noise.
    The caller controls eval mode; no training flags are changed here.
    """
    from diffusers import UniPCMultistepScheduler
    device = _module_device(model, low_latent.device)
    low_latent, text = low_latent.to(device), text.to(device)
    scheduler = UniPCMultistepScheduler.from_pretrained(
        config["model"]["local_dir"], subfolder="scheduler", local_files_only=True
    )
    scheduler.set_timesteps(steps, device=device)
    rng = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn(low_latent.shape, generator=rng, device=device, dtype=torch.float32)
    try:
        for timestep in scheduler.timesteps:
            context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with context:
                velocity, _ = model(x, timestep.expand(x.shape[0]), text, low_latent, enable_lrv=enable_lrv)
            x = scheduler.step(velocity.float(), timestep, x, return_dict=False)[0]
        return x.detach().cpu()
    finally:
        model.clear_features()


def _inverse_normalize(latent, vae):
    mean = latent.new_tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = latent.new_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    return latent * std + mean


@torch.no_grad()
def evaluate_restoration(model, vae, encoded_pairs, clips, text_embeddings, config, output_dir,
                         modes=("correct", "disabled", "shuffled"), seeds=(42, 43), steps=20,
                         train_sample_indices=None):
    """Save every clip/seed/control output and return per-video + macro metrics.

    ``encoded_pairs[i]['low']`` is the VAE encoding of upsampled LR. The
    ``'clean'`` target latent, if supplied, is never read by this function.
    Shuffled LR is a deterministic circular shift to the next input clip while
    caption and sampling seed are held fixed. At least two clips with equal
    latent geometry are required for that control.
    """
    modes, seeds = tuple(modes), tuple(int(seed) for seed in seeds)
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"correct", "disabled", "shuffled"}:
        raise ValueError("Modes must be unique members of correct/disabled/shuffled")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Provide distinct nonnegative sampling seeds")
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("Sampling steps must be positive")
    if len(clips) != len(encoded_pairs) or not clips:
        raise ValueError("Each clip needs one encoded LR pair")
    indices = list(range(len(clips))) if train_sample_indices is None else list(train_sample_indices)
    if not indices or len(set(indices)) != len(indices) or any(not isinstance(i, int) or i < 0 or i >= len(clips) for i in indices):
        raise ValueError("Invalid evaluation clip indices")
    if "shuffled" in modes and len(clips) < 2:
        raise ValueError("Shuffled conditioning requires at least two different clips")
    for clip in clips:
        _validate_video(clip["hr"])
        _validate_video(clip["lr_up"])
        if clip["hr"].shape != clip["lr_up"].shape:
            raise ValueError("Bicubic and HR videos must have identical frame/spatial shapes")
        if not math.isfinite(float(clip["fps"])) or clip["fps"] <= 0:
            raise ValueError("Every clip must retain a valid original frame rate")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    original_training = [(module, module.training) for net in (model, vae) for module in net.modules()]
    model.eval()
    vae.eval()
    model.clear_features()
    model_device = _module_device(model, "cpu")
    vae_device = _module_device(vae, "cpu")
    per_video = []
    try:
        for index in indices:
            clip = clips[index]
            metadata = clip["metadata"]
            record_id = metadata.get("record_id", str(index))
            safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(record_id))[:100]
            directory = output_dir / f"clip_{index:03d}_{safe_id}"
            directory.mkdir(parents=True, exist_ok=True)
            caption = metadata.get("caption", "")
            if caption not in text_embeddings:
                raise KeyError(f"No caption embedding for clip {record_id}")
            low = encoded_pairs[index]["low"]
            bicubic = rgb_metrics(clip["lr_up"], clip["hr"])
            torch.save(clip["lr_up"].detach().cpu(), directory / "bicubic_rgb.pt")
            write_video(directory / "bicubic.mp4", clip["lr_up"], clip["fps"])
            row = {"clip_index": index, "record_id": record_id, "source_id": metadata.get("source_id"),
                   "split": metadata.get("split"), "frame_count": int(clip["hr"].shape[2]),
                   "fps": float(clip["fps"]), "bicubic": bicubic, "modes": {}, "condition_differences": {}}
            decoded = {}
            for mode in modes:
                donor = (index + 1) % len(clips) if mode == "shuffled" else index
                condition = encoded_pairs[donor]["low"]
                if condition.shape != low.shape:
                    raise ValueError("Condition controls require matching encoded LR geometry")
                mode_rows = []
                for seed in seeds:
                    _sync(model_device)
                    start = time.perf_counter()
                    latent = sample_conditioned_latent(model, condition, text_embeddings[caption], config,
                                                        steps=steps, seed=seed, enable_lrv=mode != "disabled")
                    _sync(model_device)
                    sampling_seconds = time.perf_counter() - start
                    _sync(vae_device)
                    decode_start = time.perf_counter()
                    # VAE decoding uses its native dtype, keeping the float32
                    # precision selected by the runner. There is no HR input.
                    raw = _inverse_normalize(latent.to(device=vae_device, dtype=next(vae.parameters(), latent).dtype), vae)
                    video = (vae.decode(raw).sample.float() / 2 + .5).clamp(0, 1).cpu()
                    _sync(vae_device)
                    decode_seconds = time.perf_counter() - decode_start
                    if video.shape != clip["hr"].shape:
                        raise ValueError("Generated video changed frame count or spatial geometry")
                    metrics = rgb_metrics(video, clip["hr"])
                    mode_dir = directory / mode
                    mode_dir.mkdir(exist_ok=True)
                    tensor_path = mode_dir / f"seed_{seed}_rgb.pt"
                    video_path = mode_dir / f"seed_{seed}.mp4"
                    torch.save(video, tensor_path)
                    write_video(video_path, video, clip["fps"])
                    mode_rows.append({"seed": seed, "noise_seed": seed, "metrics": metrics,
                                      "conditioning_record_id": clips[donor]["metadata"].get("record_id", str(donor)),
                                      "lrv_enabled": mode != "disabled", "sampling_seconds": sampling_seconds,
                                      "decode_seconds": decode_seconds, "rgb_path": str(tensor_path), "video_path": str(video_path),
                                      "frame_count": row["frame_count"], "fps": row["fps"]})
                    decoded[(mode, seed)] = video
                row["modes"][mode] = {"seeds": mode_rows, "mean": _mean_metrics([x["metrics"] for x in mode_rows])}
            if "correct" in modes:
                for mode in ("disabled", "shuffled"):
                    if mode not in modes:
                        continue
                    differences = []
                    for k, seed in enumerate(seeds):
                        correct_metrics = row["modes"]["correct"]["seeds"][k]["metrics"]
                        control_metrics = row["modes"][mode]["seeds"][k]["metrics"]
                        differences.append({"seed": seed,
                            "correct_minus_control_psnr_db": correct_metrics["psnr_rgb_db"] - control_metrics["psnr_rgb_db"],
                            "correct_minus_control_ssim": correct_metrics["ssim_rgb"] - control_metrics["ssim_rgb"],
                            "output_rgb_mae": float((decoded[("correct", seed)] - decoded[(mode, seed)]).abs().mean())})
                    row["condition_differences"][mode] = {"seeds": differences,
                        "mean": {key: sum(item[key] for item in differences) / len(differences)
                                 for key in ("correct_minus_control_psnr_db", "correct_minus_control_ssim", "output_rgb_mae")}}
                row["correct_minus_bicubic"] = {key: row["modes"]["correct"]["mean"][key] - bicubic[key]
                                                   for key in ("psnr_rgb_db", "ssim_rgb")}
            dump(directory / "metrics.json", row)
            per_video.append(row)
        aggregate = aggregate_per_video(per_video, modes)
        if "correct" in modes:
            aggregate["correct_minus_bicubic"] = {key: aggregate["modes"]["correct"][key] - aggregate["bicubic"][key]
                                                    for key in ("psnr_rgb_db", "ssim_rgb")}
            aggregate["condition_differences"] = {
                mode: {key: sum(row["condition_differences"][mode]["mean"][key] for row in per_video) / len(per_video)
                       for key in ("correct_minus_control_psnr_db", "correct_minus_control_ssim", "output_rgb_mae")}
                for mode in ("disabled", "shuffled") if mode in modes}
        summary = {"status": "completed", "scope": config.get("evaluation", {}).get("scope", "overfit_debug_clips_only_not_generalization"),
                   "quality_hypotheses_verified": False, "per_video": per_video, "aggregate": aggregate,
                   "sampling": {"seeds": list(seeds), "steps": steps, "cfg": 1.0, "initial_state": "standard_gaussian",
                                "matched_noise_across_modes": True, "inference_inputs": ["LRV latent", "caption embedding"],
                                "hri": False, "shuffled_rule": "next input clip circularly; same text and noise"},
                   "metric_protocol": {"rgb_range": [0, 1], "border_shave": 0, "frame_aggregation": "arithmetic mean per video",
                                       "seed_aggregation": "all seeds arithmetic mean per video; no best-seed selection",
                                       "video_aggregation": "equal video weights", "psnr_mse_floor": PSNR_MSE_FLOOR,
                                       "psnr_max_db": 120.0, "ssim_implementation": "skimage.metrics.structural_similarity",
                                       "ssim_options": {"channel_axis": -1, "data_range": 1.0, "win_size": 7,
                                                        "gaussian_weights": False, "use_sample_covariance": True},
                                       "measurement_source": "lossless float32 RGB tensors; MP4 only for playback"}}
        dump(output_dir / "metrics.json", summary)
        return summary
    finally:
        model.clear_features()
        # Restore exact flags, including intentionally frozen eval-only children.
        for module, was_training in original_training:
            module.training = was_training
