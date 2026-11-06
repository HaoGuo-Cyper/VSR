"""Matched supplied-HRI / missing-HRI evaluation for reference-trained models.

The sampler accepts LR, text, and one supplied reference latent/index only.
HR target pixels are used after sampling to measure quality. A provided frame
is excluded from both modes' primary score, even when HRI is switched off.
First-frame and middle-frame reference cases are summarized independently.
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
from .evaluation import (PSNR_MSE_FLOOR, _inverse_normalize,
                         _mean_metrics, _module_device, _sync, _validate_video,
                         rgb_metrics)


MODES = ("hri_on", "hri_off")
GAIN_KEYS = ("psnr_gain_db", "ssim_gain", "mse_reduction")


def _mean_gains(rows):
    if not rows:
        raise ValueError("Cannot aggregate an empty HRI comparison")
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in GAIN_KEYS}


def _gain(on, off):
    return {"psnr_gain_db": on["psnr_rgb_db"] - off["psnr_rgb_db"],
            "ssim_gain": on["ssim_rgb"] - off["ssim_rgb"],
            "mse_reduction": off["mse_rgb_01"] - on["mse_rgb_01"]}


def _integer_choices(values, label):
    values = tuple(values)
    if (not values or len(set(values)) != len(values)
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values)):
        raise ValueError(f"{label} must contain distinct nonnegative integers")
    return values


@torch.no_grad()
def sample_reference_latent(model, low_latent, text, config, *, steps, seed,
                            frame_count, ref_latent=None, ref_index=None, ref_valid=False):
    """Sample without an HR target argument; reference state is set every step.

    Missing reference means None latent/index and an explicit all-false vector.
    That path never inspects or transfers a supplied placeholder object.
    """
    from diffusers import UniPCMultistepScheduler
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("Sampling steps must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Sampling seed must be a nonnegative integer")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count <= 0:
        raise ValueError("frame_count must be the positive original clip length")
    if not isinstance(ref_valid, bool):
        raise ValueError("Single-video reference control must be an explicit boolean")
    if not isinstance(low_latent, torch.Tensor) or low_latent.ndim != 5 or not all(low_latent.shape):
        raise ValueError("Encoded LR must be a nonempty B,C,Tlatent,Hlatent,Wlatent tensor")
    if not bool(torch.isfinite(low_latent).all()):
        raise ValueError("Encoded LR must be finite")
    device = _module_device(model, low_latent.device)
    low_latent, text = low_latent.to(device), text.to(device)
    batch, channels, _, height, width = low_latent.shape
    if ref_valid:
        if not isinstance(ref_latent, torch.Tensor) or tuple(ref_latent.shape) != (batch, channels, 1, height, width):
            raise ValueError("Supplied HRI must be one normalized VAE frame with matching channels/spatial grid")
        if not bool(torch.isfinite(ref_latent).all()):
            raise ValueError("Supplied HRI latent must be finite")
        if not isinstance(ref_index, int) or isinstance(ref_index, bool) or not 0 <= ref_index < frame_count:
            raise ValueError("Supplied HRI needs a known original clip frame index")
        reference, index = ref_latent.to(device), ref_index
    else:
        reference, index = None, None
    valid = torch.full((batch,), ref_valid, device=device, dtype=torch.bool)
    try:
        scheduler = UniPCMultistepScheduler.from_pretrained(
            config["model"]["local_dir"], subfolder="scheduler", local_files_only=True)
        scheduler.set_timesteps(steps, device=device)
        rng = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(low_latent.shape, generator=rng, device=device, dtype=torch.float32)
        for timestep in scheduler.timesteps:
            context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with context:
                velocity, _ = model(x, timestep.expand(batch), text, low_latent, enable_lrv=True,
                                     ref_latent=reference, ref_index=index, frame_count=frame_count, ref_valid=valid)
            x = scheduler.step(velocity.float(), timestep, x, return_dict=False)[0]
        return x.detach().cpu()
    finally:
        model.clear_features()
        model.clear_reference()


def _distance_gains(seed_rows, frame_indices, reference_index, fps):
    per_frame = []
    for position, frame_index in enumerate(frame_indices):
        gains = [_gain(row["hri_on"]["per_frame"][position], row["hri_off"]["per_frame"][position])
                 for row in seed_rows]
        distance = abs(frame_index - reference_index)
        per_frame.append({"frame_index": frame_index, "distance_frames": distance,
                          "distance_seconds": distance / fps, **_mean_gains(gains)})
    distances = sorted({row["distance_frames"] for row in per_frame})
    by_distance = {str(distance): {"frame_count": sum(row["distance_frames"] == distance for row in per_frame),
                    **_mean_gains([row for row in per_frame if row["distance_frames"] == distance])}
                   for distance in distances}
    return per_frame, by_distance


@torch.no_grad()
def evaluate_reference_restoration(model, vae, encoded_pairs, clips, text_embeddings,
                                   reference_latents, config, output_dir,
                                   reference_indices=(0, 4), seeds=(42, 43), steps=20,
                                   sample_indices=None):
    """Evaluate one reference-trained checkpoint with HRI enabled and disabled.

    ``reference_latents[i][original_frame_index]`` contains exactly the provided
    HRI encoded as a single-frame VAE video and normalized. No frame is encoded
    here. ``encoded_pairs[i]['clean']`` is never read. Frame indices refer to the
    complete input clip, including when the primary metric pool excludes HRI.
    """
    reference_indices = _integer_choices(reference_indices, "reference_indices")
    seeds = _integer_choices(seeds, "seeds")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("Sampling steps must be positive")
    if not clips or len(clips) != len(encoded_pairs) or len(clips) != len(reference_latents):
        raise ValueError("Each clip needs encoded LR and a supplied-reference mapping")
    selected = list(range(len(clips))) if sample_indices is None else list(sample_indices)
    selected = _integer_choices(selected, "sample_indices")
    if any(index >= len(clips) for index in selected):
        raise ValueError("sample_indices outside selected clip collection")
    for index in selected:
        clip = clips[index]
        _validate_video(clip["hr"])
        _validate_video(clip["lr_up"])
        if clip["hr"].shape != clip["lr_up"].shape or min(clip["hr"].shape[-2:]) < 7:
            raise ValueError("Paired RGB requires matching shapes and SSIM-compatible spatial dimensions")
        if clip["hr"].shape[2] < 2 or any(ref >= clip["hr"].shape[2] for ref in reference_indices):
            raise ValueError("Reference must be in range with at least one non-reference evaluation frame")
        if not math.isfinite(float(clip["fps"])) or float(clip["fps"]) <= 0:
            raise ValueError("Every clip must retain its positive original FPS")
        if clip["metadata"].get("caption", "") not in text_embeddings:
            raise KeyError("Missing native caption embedding")
        for ref in reference_indices:
            if ref not in reference_latents[index]:
                raise KeyError(f"Missing supplied HRI at original frame {ref} for clip {index}")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    original_training = [(module, module.training) for network in (model, vae) for module in network.modules()]
    model.eval()
    vae.eval()
    model.clear_features()
    model.clear_reference()
    model_device, vae_device = _module_device(model, "cpu"), _module_device(vae, "cpu")
    cases = {}
    try:
        for ref in reference_indices:
            case_directory = output_dir / f"reference_{ref:03d}"
            per_video = []
            for index in selected:
                clip, low = clips[index], encoded_pairs[index]["low"]
                metadata, frame_count = clip["metadata"], int(clip["hr"].shape[2])
                record_id = metadata.get("record_id", str(index))
                safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(record_id))[:100]
                directory = case_directory / f"clip_{index:03d}_{safe_id}"
                directory.mkdir(parents=True, exist_ok=True)
                caption = metadata.get("caption", "")
                frames = [frame for frame in range(frame_count) if frame != ref]
                role = "first_frame_main" if ref == 0 else ("middle_frame_adaptation" if ref == frame_count // 2 else "known_time_additional")
                row = {"clip_index": index, "record_id": record_id, "source_id": metadata.get("source_id"),
                       "split": metadata.get("split"), "reference_index": ref, "reference_role": role,
                       "frame_count": frame_count, "fps": float(clip["fps"]), "evaluation_frame_indices": frames,
                       "bicubic": rgb_metrics(clip["lr_up"][:, :, frames], clip["hr"][:, :, frames]),
                       "modes": {}}
                seed_rows = [{"seed": seed} for seed in seeds]
                for mode in MODES:
                    enabled = mode == "hri_on"
                    mode_rows = []
                    for seed_position, seed in enumerate(seeds):
                        _sync(model_device)
                        started = time.perf_counter()
                        # The off branch does not index or transfer the reference
                        # mapping; no black image represents a missing condition.
                        supplied = reference_latents[index][ref] if enabled else None
                        latent = sample_reference_latent(model, low, text_embeddings[caption], config,
                            steps=steps, seed=seed, frame_count=frame_count, ref_latent=supplied,
                            ref_index=ref if enabled else None, ref_valid=enabled)
                        _sync(model_device)
                        sampling_seconds = time.perf_counter() - started
                        _sync(vae_device)
                        decode_start = time.perf_counter()
                        dtype = next(vae.parameters(), latent).dtype
                        raw = _inverse_normalize(latent.to(device=vae_device, dtype=dtype), vae)
                        video = (vae.decode(raw).sample.float() / 2 + .5).clamp(0, 1).cpu()
                        _sync(vae_device)
                        decode_seconds = time.perf_counter() - decode_start
                        if video.shape != clip["hr"].shape:
                            raise ValueError("Reference sampling changed full frame count or spatial geometry")
                        primary = rgb_metrics(video[:, :, frames], clip["hr"][:, :, frames])
                        primary["frame_indices"] = frames
                        reference_score = rgb_metrics(video[:, :, ref:ref+1], clip["hr"][:, :, ref:ref+1])
                        mode_directory = directory / mode
                        mode_directory.mkdir(exist_ok=True)
                        rgb_path, video_path = mode_directory / f"seed_{seed}_rgb.pt", mode_directory / f"seed_{seed}.mp4"
                        torch.save(video, rgb_path)
                        write_video(video_path, video, clip["fps"])
                        entry = {"seed": seed, "noise_seed": seed, "hri_valid": enabled,
                                 "supplied_reference_index": ref if enabled else None,
                                 "metrics": primary, "provided_reference_frame_metrics": reference_score,
                                 "sampling_seconds": sampling_seconds, "decode_seconds": decode_seconds,
                                 "rgb_path": str(rgb_path), "video_path": str(video_path),
                                 "frame_count": frame_count, "fps": float(clip["fps"])}
                        mode_rows.append(entry)
                        seed_rows[seed_position][mode] = primary
                    row["modes"][mode] = {"seeds": mode_rows, "mean": _mean_metrics([entry["metrics"] for entry in mode_rows]),
                        "provided_reference_frame_mean": _mean_metrics([entry["provided_reference_frame_metrics"] for entry in mode_rows])}
                comparisons = [{"seed": entry["seed"], **_gain(entry["hri_on"], entry["hri_off"])} for entry in seed_rows]
                row["nonreference_gain"] = {"seeds": comparisons, "mean": _mean_gains(comparisons)}
                row["provided_reference_frame_gain"] = _gain(row["modes"]["hri_on"]["provided_reference_frame_mean"],
                                                             row["modes"]["hri_off"]["provided_reference_frame_mean"])
                row["per_frame_gain"], row["gain_by_distance_frames"] = _distance_gains(seed_rows, frames, ref, float(clip["fps"]))
                dump(directory / "metrics.json", row)
                per_video.append(row)
            distances = sorted({int(distance) for row in per_video for distance in row["gain_by_distance_frames"]})
            aggregate = {"video_count": len(per_video), "bicubic": _mean_metrics([row["bicubic"] for row in per_video]),
                "modes": {mode: _mean_metrics([row["modes"][mode]["mean"] for row in per_video]) for mode in MODES},
                "provided_reference_frame": {mode: _mean_metrics([row["modes"][mode]["provided_reference_frame_mean"] for row in per_video]) for mode in MODES},
                "nonreference_gain": _mean_gains([row["nonreference_gain"]["mean"] for row in per_video]),
                "gain_by_distance_frames": {str(distance): {
                    "video_count": sum(str(distance) in row["gain_by_distance_frames"] for row in per_video),
                    **_mean_gains([row["gain_by_distance_frames"][str(distance)] for row in per_video if str(distance) in row["gain_by_distance_frames"]])}
                    for distance in distances}}
            cases[str(ref)] = {"reference_index": ref,
                "reference_roles": sorted({row["reference_role"] for row in per_video}),
                "per_video": per_video, "aggregate": aggregate}
            dump(case_directory / "metrics.json", cases[str(ref)])
        summary = {"status": "completed", "scope": config.get("evaluation", {}).get("scope", "reference_model_diagnostics"),
            "model_setting": "reference_trained_with_dropout", "comparison": "same_reference_trained_checkpoint_hri_on_vs_explicit_hri_off",
            "quality_hypotheses_verified": False, "reference_cases": cases,
            "sampling": {"seeds": list(seeds), "steps": steps, "cfg": 1.0, "matched_noise_across_modes": True,
                "initial_state": "standard_gaussian", "inference_inputs": ["LRV latent", "caption embedding", "optional supplied HRI latent and known index"],
                "frame_count_and_fps": "complete input preserved", "hri_off": "None latent/index with explicit false validity vector"},
            "metric_protocol": {"primary": "same non-reference frames in hri_on and hri_off; provided frame excluded from both",
                "provided_reference_frame": "reported separately for each mode", "distance_gain": "abs(original_frame_index-reference_index); distance zero excluded",
                "first_and_middle_reference": "separate cases; never pool reference choices",
                "measurement_source": "lossless float32 RGB; full videos saved; MP4 playback only",
                "rgb_range": [0, 1], "border_shave": 0, "psnr_mse_floor": PSNR_MSE_FLOOR,
                "ssim": "existing rgb_metrics; skimage SSIM defaults, RGB data_range=1",
                "aggregation": "all seeds mean per video, then equal video weights"}}
        dump(output_dir / "metrics.json", summary)
        return summary
    finally:
        model.clear_features()
        model.clear_reference()
        for module, was_training in original_training:
            module.training = was_training
