"""Measure new HRI-run lossless samples, retaining original temporal pairs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".deps"))

import torch

from .audit import dump, file_hash
from .cache import PairedClipSequence
from .metric_pools import PAIRED_METRIC_KEYS, POOL_NAMES, paired_metric_pools
from .paired_statistics import paired_summary
from .perceptual_metrics import PerceptualMetrics

MODES = ("hri_on", "hri_off")


def mean_pools(values):
    """Preserve unavailable temporal values; never interpret None as zero."""
    if not values:
        raise ValueError("No samples to average")
    return {pool: {key: sum(float(row[pool][key]) for row in values) / len(values)
                   if all(row[pool][key] is not None for row in values) else None
                   for key in PAIRED_METRIC_KEYS} for pool in POOL_NAMES}


def score_cases(metrics, clips, perceptual, load_rgb):
    """CPU-testable scorer; load_rgb accepts exactly the supplied sample path."""
    if metrics["status"] != "completed" or metrics["model_setting"] != "reference_trained_with_dropout":
        raise ValueError("Requires completed matched evaluation of a reference-trained model")
    if metrics["sampling"].get("matched_noise_across_modes") is not True:
        raise ValueError("HRI on/off comparison requires actual matched noise")
    declared_seeds = metrics["sampling"]["seeds"]
    if not declared_seeds or len(set(declared_seeds)) != len(declared_seeds):
        raise ValueError("Distinct preset seeds are required")
    if not metrics["reference_cases"]:
        raise ValueError("No reference evaluation cases")
    cases = {}
    for case_id, case in metrics["reference_cases"].items():
        ref = case["reference_index"]
        if str(ref) != case_id:
            raise ValueError("Reference case identity differs")
        videos, observations, seen = [], {mode: [] for mode in MODES}, set()
        for row in case["per_video"]:
            record = row["record_id"]
            if record in seen:
                raise ValueError("Duplicate video in reference case")
            seen.add(record)
            clip = clips[row["clip_index"]]
            metadata, target = clip["metadata"], clip["hr"]
            if (record, row["source_id"], row["split"]) != (metadata["record_id"], metadata["source_id"], metadata["split"]):
                raise ValueError("Reference metric identity differs from bound RGB target")
            frames = [i for i in range(target.shape[2]) if i != ref]
            if row["reference_index"] != ref or row["evaluation_frame_indices"] != frames:
                raise ValueError("Reference case did not retain the expected identical nonreference pool")
            if not math.isfinite(row["fps"]) or row["fps"] <= 0 or row["fps"] != clip["fps"]:
                raise ValueError("Reference metric FPS differs from its input")
            baseline = paired_metric_pools(clip["lr_up"], target, perceptual, reference_index=ref)
            video = {"record_id": record, "source_id": row["source_id"], "split": row["split"],
                     "reference_index": ref, "fps": row["fps"], "frame_count": int(target.shape[2]),
                     "bicubic": baseline, "modes": {}}
            for mode in MODES:
                samples = row["modes"][mode]["seeds"]
                if len(samples) != len(declared_seeds) or {s["seed"] for s in samples} != set(declared_seeds):
                    raise ValueError("Reference mode lacks complete matched sampling seeds")
                scored = []
                for sample in samples:
                    if sample.get("noise_seed") != sample["seed"]:
                        raise ValueError("Stored actual noise seed differs from its sampling label")
                    if sample["hri_valid"] != (mode == "hri_on"):
                        raise ValueError("Stored sample reference validity disagrees with its mode")
                    if sample["supplied_reference_index"] != (ref if mode == "hri_on" else None):
                        raise ValueError("Stored sample supplied reference index disagrees with its case")
                    prediction = load_rgb(sample["rgb_path"])
                    value = paired_metric_pools(prediction, target, perceptual, reference_index=ref)
                    scored.append({"seed": sample["seed"], "metrics": value})
                    observations[mode].append({"record_id": record, "source_id": row["source_id"], "seed": sample["seed"],
                        "metrics": {key: value["matched_nonreference"][key] for key in PAIRED_METRIC_KEYS}})
                video["modes"][mode] = {"seeds": scored, "mean": mean_pools([s["metrics"] for s in scored])}
            videos.append(video)
        if not videos:
            raise ValueError("Empty reference case")
        available = [key for key in PAIRED_METRIC_KEYS if all(row["metrics"][key] is not None
                    for rows in observations.values() for row in rows)]
        groups = {mode: [{**row, "metrics": {key: row["metrics"][key] for key in available}} for row in rows]
                  for mode, rows in observations.items()}
        comparison = paired_summary(groups, "hri_off")
        distances = {}
        for video in videos:
            on = {s["seed"]: s["metrics"] for s in video["modes"]["hri_on"]["seeds"]}
            off = {s["seed"]: s["metrics"] for s in video["modes"]["hri_off"]["seeds"]}
            values = []
            for frame in range(video["frame_count"]):
                if frame == ref:
                    continue
                seed_values = []
                for seed in declared_seeds:
                    pool_on, pool_off = on[seed]["all_frames"], off[seed]["all_frames"]
                    seed_values.append({"psnr_gain_db": pool_on["per_frame"][frame]["psnr_rgb_db"] - pool_off["per_frame"][frame]["psnr_rgb_db"],
                        "ssim_gain": pool_on["per_frame"][frame]["ssim_rgb"] - pool_off["per_frame"][frame]["ssim_rgb"],
                        "lpips_on_minus_off": pool_on["lpips_per_frame"][frame] - pool_off["lpips_per_frame"][frame]})
                values.append({"frame_index": frame, "distance_frames": abs(frame-ref),
                    "distance_seconds": abs(frame-ref)/video["fps"],
                    **{key: sum(row[key] for row in seed_values)/len(seed_values) for key in seed_values[0]}})
            distances[video["record_id"]] = values
        cases[case_id] = {"reference_index": ref, "per_video": videos,
            "aggregate": {"bicubic": mean_pools([v["bicubic"] for v in videos]),
                          "modes": {mode: mean_pools([v["modes"][mode]["mean"] for v in videos]) for mode in MODES}},
            "matched_nonreference_comparison": comparison,
            "unavailable_comparison_metrics": sorted(set(PAIRED_METRIC_KEYS)-set(available)),
            "gain_by_original_frame_and_distance": distances}
    return {"status": "completed", "scope": metrics["scope"], "reference_cases": cases,
        "sampling": metrics["sampling"], "protocol": {
            "measurement_source": "own saved lossless float RGB; playback MP4 is never decoded for metrics",
            "primary": "identical nonreference frame pool in on/off; supplied reference measured separately",
            "temporal": "evaluate full sequence once; remove existing original pairs incident to reference; never delete/reindex frames",
            "reference_cases": "first and middle separately, no pooling of reference choices",
            "distance": "per-video all-seed means at original frame index; time uses that video's sampled FPS",
            "unavailable": "empty temporal pools are null; omitted from paired comparison, never zero",
            "delta": "on minus off; LPIPS/tLP/tOF lower is better", "quality_hypotheses_verified": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--variant", required=True, choices=("Base", "Align", "Relation", "Both"))
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run = Path(args.run).resolve()
    if not run.is_relative_to(ROOT / "runs"):
        raise ValueError("Only this independent project's saved reference samples may be scored")
    directory = run / args.variant / f"step_{args.step:06d}"
    output = directory / "perceptual_reference_metrics.json"
    if output.exists():
        raise FileExistsError("Scoring record exists; preserve it and use the recorded result")
    source = run
    if (run / "cache_reuse.json").exists():
        source = Path(json.loads((run / "cache_reuse.json").read_text())["source_run"]).resolve()
        if not source.is_relative_to(ROOT / "runs"):
            raise ValueError("RGB cache must originate in this independent project")
    clips = PairedClipSequence(source)
    evaluator = PerceptualMetrics(ROOT / "weights/alexnet-owt-7be5be79.pth", ROOT / "weights/lpips_v0.1_alex.pth", device=args.device)
    sampled_artifacts = {}
    def load_rgb(path):
        path = Path(path).resolve()
        if not path.is_relative_to(run):
            raise ValueError("Generated sample escapes selected reference experiment")
        before = file_hash(path)
        value = torch.load(path, map_location="cpu", weights_only=True)
        if file_hash(path) != before:
            raise ValueError("Generated HRI sample changed while being loaded")
        sampled_artifacts[str(path)] = before
        return value
    started = time.perf_counter()
    metrics_path = directory / "metrics.json"
    report = score_cases(json.loads(metrics_path.read_text()), clips, evaluator, load_rgb)
    target_artifacts = {}
    for case in json.loads(metrics_path.read_text())["reference_cases"].values():
        for row in case["per_video"]:
            for filename in ("metadata.json", "paired_rgb.pt"):
                path = source / f"clip_{row['clip_index']}" / filename
                target_artifacts[str(path)] = file_hash(path)
    report.update(audit=evaluator.audit, wall_seconds=time.perf_counter()-started, device=args.device,
                  sampled_artifact_sha256=sampled_artifacts, target_artifact_sha256=target_artifacts,
                  rgb_metrics_sha256=file_hash(metrics_path), scorer_sha256=file_hash(Path(__file__)))
    dump(output, report)
    print(json.dumps({"output": str(output), "cases": list(report["reference_cases"])}))


if __name__ == "__main__":
    main()
