"""Report paired own-run validation results, without decoding or sampling.

Example: python -m vsr.summarize_saved --result Base=runs/.../Base/step_008192
--result Align=runs/.../Align/step_008192 --reference Base --output reports/matrix
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .audit import dump, file_hash
from .paired_statistics import paired_summary

ROOT = Path(__file__).resolve().parent.parent
RGB = ("psnr_rgb_db", "ssim_rgb")
LPIPS = "lpips_alex01_rgb_allframes"
TLP = "tLP100_alex_v0_1_TecoGAN_crop"
TOF = "tOF_Farneback_TecoGAN_crop"
METRICS = (*RGB, LPIPS, TLP, TOF)


def _unique(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError(f"Duplicate {key}: {value}")
        result[value] = row
    return result


def _perceptual(score):
    return {LPIPS: score[LPIPS], TLP: score["tecogan"][TLP], TOF: score["tecogan"][TOF]}


def load_saved_result(directory, *, split="val", mode="correct"):
    """Join RGB and perceptual per-seed values; fail on incomplete comparisons."""
    directory = Path(directory).resolve()
    run = directory.parent.parent
    rgb_path, perceptual_path = directory / "metrics.json", directory / "perceptual_metrics_v2.json"
    rgb = json.loads(rgb_path.read_text())
    perceptual = json.loads(perceptual_path.read_text())
    if rgb["status"] != "completed":
        raise ValueError("RGB evaluation is not completed")
    if rgb["sampling"].get("matched_noise_across_modes") is not True:
        raise ValueError("Actual noise must be matched across modes")
    if (perceptual.get("format") != "vsr_saved_perceptual_v2" or perceptual.get("status") != "completed"
            or perceptual.get("rgb_metrics_sha256") != file_hash(rgb_path)):
        raise ValueError("Perceptual score lacks a completed binding to the exact RGB metrics; rerun score_saved")
    if rgb["sampling"]["hri"]:
        raise ValueError("Reference evaluation has distinct frame pools; use its reference report")
    source = run
    reuse = run / "cache_reuse.json"
    if reuse.exists():
        source = Path(json.loads(reuse.read_text())["source_run"]).resolve()
    rgb_rows = _unique(rgb["per_video"], "record_id")
    perceptual_rows = _unique(perceptual["per_video"], "record_id")
    if set(rgb_rows) != set(perceptual_rows):
        raise ValueError("RGB/perceptual video sets differ")
    observations, baselines, controls, bindings = [], [], [], {}
    for record, row in rgb_rows.items():
        if row["split"] != split:
            continue
        other = perceptual_rows[record]
        if (other["split"], other.get("source_id")) != (row["split"], row["source_id"]):
            raise ValueError("RGB/perceptual source or split identity differs")
        metadata_path = source / f"clip_{row['clip_index']}" / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        if (other.get("source_metadata_sha256") != file_hash(metadata_path)
                or other.get("source_rgb_sha256") != file_hash(source / f"clip_{row['clip_index']}" / "paired_rgb.pt")):
            raise ValueError("Perceptual score target/crop binding differs from its RGB cache")
        if (metadata["record_id"], metadata["source_id"], metadata["split"]) != (record, row["source_id"], split):
            raise ValueError("Metric video identity differs from frozen RGB metadata")
        bindings[record] = {key: metadata["binding"][key] for key in ("data_cache_key", "manifest_sha256", "official_revision")}
        seeds = _unique(row["modes"][mode]["seeds"], "seed")
        perceptual_seeds = _unique(other["modes"][mode]["seeds"], "seed")
        if set(seeds) != set(perceptual_seeds) or set(seeds) != set(rgb["sampling"]["seeds"]):
            raise ValueError("RGB/perceptual/declared sampling seed sets differ")
        for seed, sample in seeds.items():
            perceptual_sample = perceptual_seeds[seed]
            tensor_path = Path(sample["rgb_path"]).resolve()
            if not tensor_path.is_relative_to(run):
                raise ValueError("Sample escapes its own VSR experiment")
            if (sample.get("noise_seed") != seed or perceptual_sample.get("noise_seed") != seed
                    or Path(perceptual_sample["rgb_path"]).resolve() != tensor_path
                    or perceptual_sample.get("rgb_sha256") != file_hash(tensor_path)):
                raise ValueError("RGB/perceptual sample bytes or actual noise seed differ")
            observation = {"record_id": record, "source_id": row["source_id"], "seed": seed}
            observations.append({**observation, "metrics": {**{key: sample["metrics"][key] for key in RGB},
                                **_perceptual(perceptual_seeds[seed]["metrics"])}})
            baselines.append({**observation, "metrics": {**{key: row["bicubic"][key] for key in RGB},
                             **_perceptual(other["bicubic"])}})
            for control in ("disabled", "shuffled"):
                if control not in row["modes"]:
                    continue
                control_seeds = _unique(row["modes"][control]["seeds"], "seed")
                if set(control_seeds) != set(seeds):
                    raise ValueError("LR condition control seed set differs")
                if control_seeds[seed].get("noise_seed") != seed:
                    raise ValueError("LR control used a different actual noise seed")
                controls.append({**observation, "control": control,
                    "correct_minus_control_psnr_db": sample["metrics"]["psnr_rgb_db"] - control_seeds[seed]["metrics"]["psnr_rgb_db"],
                    "correct_minus_control_ssim": sample["metrics"]["ssim_rgb"] - control_seeds[seed]["metrics"]["ssim_rgb"]})
    if not observations:
        raise ValueError(f"No {split} videos in supplied evaluation")
    config_path = run / "config.json"
    config = json.loads(config_path.read_text())
    summary_path = directory.parent / "summary.json"
    costs = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    return {"observations": observations, "bicubic": baselines, "bindings": bindings,
        "compatibility": {"sampling_seeds": rgb["sampling"]["seeds"], "steps": rgb["sampling"]["steps"],
                          "cfg": rgb["sampling"]["cfg"], "metric_protocol": rgb["metric_protocol"],
                          "perceptual_audit": perceptual["audit"]},
        "provenance": {"directory": str(directory), "run": str(run), "variant": directory.parent.name,
            "step": directory.name, "split": split, "mode": mode, "scope": rgb["scope"],
            "data_seed": config["seed"], "training_seed": config["training"].get("seed", config["seed"]),
            "reproducibility": config.get("reproducibility", {"mode": "legacy_seeded_not_strict"}),
            "input_sha256": {str(path): file_hash(path) for path in (rgb_path, perceptual_path, config_path)},
            "training_gpu_hours": costs.get("training_gpu_hours"), "phase_wall_seconds": costs.get("phase_wall_seconds"),
            "peak_allocated_bytes": costs.get("peak_allocated_bytes")},
        "condition_controls": controls}


def summarize_results(results, reference, *, replicates=10000, seed=20261102):
    if not results or reference not in results or "Bicubic" in results:
        raise ValueError("Supply a reference and reserve the Bicubic label")
    first = results[reference]
    for label, result in results.items():
        if result["compatibility"] != first["compatibility"]:
            raise ValueError(f"{label}: sampling or metric protocol differs")
        if result["bindings"] != first["bindings"]:
            raise ValueError(f"{label}: evaluated raw samples, crops or degradation differ")
        baseline = {(r["record_id"], r["seed"]): r["metrics"] for r in result["bicubic"]}
        expected = {(r["record_id"], r["seed"]): r["metrics"] for r in first["bicubic"]}
        if baseline.keys() != expected.keys() or any(
                not math.isclose(value, expected[key][metric], rel_tol=1e-8, abs_tol=1e-9)
                for key, values in baseline.items() for metric, value in values.items()):
            raise ValueError(f"{label}: bicubic baseline differs")
    groups = {label: result["observations"] for label, result in results.items()}
    groups["Bicubic"] = first["bicubic"]
    output = paired_summary(groups, reference, replicates=replicates, seed=seed)
    output["provenance"] = {label: result["provenance"] for label, result in results.items()}
    output["condition_controls"] = {}
    for label, result in results.items():
        output["condition_controls"][label] = {}
        for control in sorted({row["control"] for row in result["condition_controls"]}):
            selected = [r for r in result["condition_controls"] if r["control"] == control]
            output["condition_controls"][label][control] = {
                "sample_count": len(selected), "minimum_correct_minus_control_psnr_db": min(r["correct_minus_control_psnr_db"] for r in selected),
                "all_video_seed_pairs_psnr_positive": all(r["correct_minus_control_psnr_db"] > 0 for r in selected),
                "per_video_seed": selected}
    return output


def text_report(report):
    lines = ["VSR matched saved-sample results", "", f"{report['video_count']} videos / {report['source_count']} source groups; sampling seeds {report['sampling_seeds']}.",
             "Seeds are averaged within each video. Primary results use equal video weights.",
             "PSNR/SSIM: higher is better. LPIPS/tLP100/tOF: lower is better.", "", "Method\tPSNR\tSSIM\tLPIPS\ttLP100\ttOF"]
    for label, row in report["experiments"].items():
        lines.append(label + "\t" + "\t".join(f"{row['video_macro'][key]:.6f}" for key in METRICS))
    lines += ["", f"Paired differences: candidate minus {report['reference']}; source-cluster bootstrap 95% intervals."]
    for label, values in report["comparisons"].items():
        for key in METRICS:
            value = values[key]
            lines.append(f"{label} {key}: {value['delta_video_macro']:+.6f}; CI {value['ci95_source_cluster_video_macro']}")
    lines += ["", "These pointwise intervals do not include uncertainty across training seeds and are not a confirmed module-benefit claim.",
              "Source-macro supplementary results, every video delta, matched controls, source hashes and costs are in the JSON."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, help="LABEL=run/variant/step directory")
    parser.add_argument("--reference", default="Base")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", required=True, help="New report prefix; creates .json and .txt")
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    results = {}
    for item in args.result:
        label, directory = item.split("=", 1)
        directory = Path(directory).resolve()
        if not label or label in results or not directory.is_relative_to(ROOT / "runs"):
            raise ValueError("Labels must be unique and every input must be this project's own run")
        results[label] = load_saved_result(directory, split=args.split)
    prefix = Path(args.output)
    paths = [prefix.with_suffix(".json"), prefix.with_suffix(".txt")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Report exists; use a new prefix to preserve experiment history")
    report = summarize_results(results, args.reference, replicates=args.replicates)
    dump(paths[0], report)
    paths[1].write_text(text_report(report))
    print(text_report(report))


if __name__ == "__main__":
    main()
