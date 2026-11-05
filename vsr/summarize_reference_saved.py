"""Bound, source-paired comparisons of new saved HRI validation samples.

Reference choices remain separate. All summaries use per-seed measurements;
sampling repeats never stand in for independent training repeats.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from .audit import file_hash
from .metric_pools import PAIRED_METRIC_KEYS, POOL_NAMES
from .paired_statistics import paired_summary

ROOT = Path(__file__).resolve().parents[1]
MODES = ("hri_on", "hri_off")
TEMPORAL = tuple(key for key in PAIRED_METRIC_KEYS if key.startswith(("tLP", "tOF")))
RGB = ("psnr_rgb_db", "ssim_rgb", "mse_rgb_01")


def _inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Saved reference artifact escapes its independent VSR run")
    return path


def _json(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if hashlib.sha256(raw).hexdigest() != file_hash(path):
        raise ValueError("Saved reference record changed while being read")
    return value


def _unique(rows, field):
    values = {}
    for row in rows:
        key = row[field]
        if key in values:
            raise ValueError(f"Duplicate saved-reference {field}")
        values[key] = row
    return values


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _same_number(a, b):
    return _number(a) and _number(b) and math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-9)


def _verified_map(mapping, root):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Saved reference scorer lacks complete artifact digest maps")
    result = {}
    for name, expected in mapping.items():
        path = _inside(name, root)
        if str(path) != name or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Artifact paths/digests must be canonical and complete")
        if file_hash(path) != expected:
            raise ValueError("Saved reference artifact SHA256 mismatch")
        result[name] = expected
    return result


def _validate_pools(value, frames, reference):
    frame_sets = {"all_frames": list(range(frames)),
        "matched_nonreference": [i for i in range(frames) if i != reference],
        "reference_frame_separate": [reference]}
    original_pairs = [[i, i + 1] for i in range(2, frames - 3)]
    for name in POOL_NAMES:
        pool = value[name]
        indices = frame_sets[name]
        pairs = [pair for pair in original_pairs if all(index in indices for index in pair)]
        if (pool.get("frame_indices") != indices or pool.get("frame_count") != len(indices)
                or pool.get("temporal_pair_indices") != pairs or pool.get("temporal_pair_count") != len(pairs)
                or len(pool.get("per_frame", [])) != len(indices)
                or [row.get("original_frame_index") for row in pool["per_frame"]] != indices
                or [row.get("indices") for row in pool.get("per_temporal_pair", [])] != pairs):
            raise ValueError("Saved HRI pool changed original frame/temporal-pair indices")
        for key in PAIRED_METRIC_KEYS:
            metric = pool.get(key)
            if key in TEMPORAL and not pairs:
                if metric is not None:
                    raise ValueError("Empty temporal pool must remain null")
            elif not _number(metric):
                raise ValueError("Available HRI metrics must be finite numbers")
        for key in TEMPORAL:
            if pairs:
                values = [row.get(key) for row in pool["per_temporal_pair"]]
                if not all(_number(item) for item in values) or not _same_number(pool[key], sum(values) / len(values)):
                    raise ValueError("Temporal score does not average the retained original pairs")
        for key in RGB:
            values = [row.get(key) for row in pool["per_frame"]]
            if not all(_number(item) for item in values) or not _same_number(pool[key], sum(values) / len(values)):
                raise ValueError("RGB score does not average the retained original frames")
        lpips = pool.get("lpips_per_frame", [])
        if (len(lpips) != len(indices) or not all(_number(item) for item in lpips)
                or not _same_number(pool["lpips_alex_v0_1"], sum(lpips) / len(lpips))):
            raise ValueError("LPIPS score does not average the retained original frames")
        complete = value["all_frames"]
        for position, index in enumerate(indices):
            if (any(not _same_number(pool["per_frame"][position][key], complete["per_frame"][index][key]) for key in RGB)
                    or not _same_number(lpips[position], complete["lpips_per_frame"][index])):
                raise ValueError("HRI pool changed original full-sequence frame measurements")
        full_pairs = {tuple(row["indices"]): row for row in complete["per_temporal_pair"]}
        for pair in pool["per_temporal_pair"]:
            if any(not _same_number(pair[key], full_pairs[tuple(pair["indices"])][key]) for key in TEMPORAL):
                raise ValueError("HRI pool changed original full-sequence temporal measurements")
    return {pool: {key: value[pool][key] for key in PAIRED_METRIC_KEYS} for pool in POOL_NAMES}


def load_reference_result(directory, *, project_root=ROOT):
    """Verify exact score/RGB/cache/sample joins without loading task weights."""
    root = Path(project_root).resolve()
    directory = _inside(directory, root / "runs")
    run = directory.parent.parent
    if directory.parent.name not in ("Base", "Align", "Relation", "Both") or not re.fullmatch(r"step_\d{6}", directory.name):
        raise ValueError("Use an own variant/step directory")
    rgb_path, score_path = directory / "metrics.json", directory / "perceptual_reference_metrics.json"
    rgb, score = _json(rgb_path), _json(score_path)
    if (rgb.get("status") != "completed" or score.get("status") != "completed"
            or rgb.get("model_setting") != "reference_trained_with_dropout"
            or score.get("rgb_metrics_sha256") != file_hash(rgb_path)
            or rgb.get("sampling", {}).get("matched_noise_across_modes") is not True
            or score.get("sampling") != rgb.get("sampling") or score.get("scope") != rgb.get("scope")
            or not isinstance(score.get("audit"), dict) or not score["audit"]
            or not isinstance(score.get("protocol"), dict) or not score["protocol"]):
        raise ValueError("Requires completed, SHA-bound, matched-noise HRI measurements")
    if not re.fullmatch(r"[0-9a-f]{64}", score.get("scorer_sha256", "")):
        raise ValueError("HRI scorer version SHA256 is required")
    seeds = rgb["sampling"]["seeds"]
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds)):
        raise ValueError("Require distinct nonnegative preset sampling seeds")
    config_path, init_path = run / "config.json", directory.parent / "initialization.json"
    config, initialization = _json(config_path), _json(init_path)
    if (config.get("reference", {}).get("enabled") is not True
            or config["training"].get("resume") is not False
            or config["training"].get("official_initialization_per_variant") is not True
            or initialization.get("fresh_official_initialization") is not True
            or initialization.get("legacy_state_loaded") is not False
            or initialization.get("optimizer_fresh") is not True):
        raise ValueError("HRI comparison requires recorded independent official fresh initialization")
    training_seed = config["training"].get("seed", config["seed"])
    if (initialization.get("training_seed", initialization.get("seed")) != training_seed
            or initialization.get("data_seed") != config["seed"]):
        raise ValueError("Recorded official initialization seeds differ from this training/data configuration")
    initial_groups = initialization.get("trainable_initialization_hashes", {}).get("groups", {})
    initial_hashes = {}
    for name in ("lrv", "lora", "hri", "projector"):
        group = initial_groups.get(name, {})
        if (not re.fullmatch(r"[0-9a-f]{64}", group.get("sha256", ""))
                or not isinstance(group.get("numel"), int) or group["numel"] < 0
                or not isinstance(group.get("tensor_count"), int) or group["tensor_count"] < 0):
            raise ValueError("Official fresh initialization requires complete trainable group hashes")
        initial_hashes[name] = {key: group[key] for key in ("sha256", "numel", "tensor_count")}
    if ((directory.parent.name == "Base") != (initial_hashes["projector"]["numel"] == 0)
            or any(initial_hashes[name]["numel"] <= 0 for name in ("lrv", "lora", "hri"))):
        raise ValueError("Recorded fresh HRI/LoRA/LRV/projection groups differ from the declared variant")
    selection = _json(run / "selection.json")
    source = run
    reuse_path = run / "cache_reuse.json"
    reuse = _json(reuse_path) if reuse_path.exists() else None
    if reuse:
        source = _inside(reuse["source_run"], root / "runs")
        if reuse.get("legacy_state_read") is not False:
            raise ValueError("Saved HRI inputs may not originate from legacy task state")
    source_selection = _json(source / "selection.json")
    for key in ("manifest_sha256", "training_ids", "validation_ids", "test_accessed"):
        if source_selection.get(key) != selection.get(key):
            raise ValueError("HRI selection differs from the bound new-data cache")
    if selection.get("test_accessed") is not False or not selection.get("validation_ids"):
        raise ValueError("This report requires a predeclared non-test validation selection")
    seal_path = source / "cache_asset_seal.json"
    asset_seal = None
    if reuse and not seal_path.is_file():
        raise ValueError("Reused HRI data requires its existing prepared asset seal")
    if seal_path.exists():
        asset_seal = _json(seal_path)
        if reuse and file_hash(seal_path) != reuse.get("source_asset_seal_sha256"):
            raise ValueError("Bound cache asset seal changed")
    samples = _verified_map(score.get("sampled_artifact_sha256"), directory)
    targets = _verified_map(score.get("target_artifact_sha256"), source)
    expected_samples, expected_targets, bindings, cases = set(), set(), {}, {}
    if not rgb.get("reference_cases") or set(rgb["reference_cases"]) != set(score.get("reference_cases", {})):
        raise ValueError("Scored/RGB reference case sets differ")
    clip_ids = selection["training_ids"] + selection["validation_ids"]
    for case_id, original_case in rgb["reference_cases"].items():
        ref = original_case["reference_index"]
        measured_case = score["reference_cases"][case_id]
        if (not isinstance(ref, int) or isinstance(ref, bool) or str(ref) != case_id
                or measured_case.get("reference_index") != ref):
            raise ValueError("Reference case identity differs")
        original = _unique(original_case["per_video"], "record_id")
        measured = _unique(measured_case["per_video"], "record_id")
        if set(original) != set(measured) or set(original) != set(selection["validation_ids"]):
            raise ValueError("All reference cases must retain the complete fixed validation set")
        observations = {mode: [] for mode in MODES}
        baselines = []
        for record, row in original.items():
            other = measured[record]
            index = row["clip_index"]
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(clip_ids) or clip_ids[index] != record:
                raise ValueError("Reference clip index differs from cache order")
            metadata_path, target_path = source / f"clip_{index}/metadata.json", source / f"clip_{index}/paired_rgb.pt"
            metadata = _json(metadata_path)
            for path in (metadata_path, target_path):
                expected_targets.add(str(path))
                if str(path) not in targets or (asset_seal and asset_seal["artifact_sha256"].get(str(path.relative_to(source))) != targets[str(path)]):
                    raise ValueError("HRI target is not bound to its prepared asset seal")
            frames, fps = row["frame_count"], row["fps"]
            if (not isinstance(frames, int) or isinstance(frames, bool) or frames < 6 or not 0 <= ref < frames
                    or not _number(fps) or fps <= 0 or metadata.get("output_fps") != fps
                    or frames != config["data"]["num_frames"]):
                raise ValueError("HRI frame count/FPS differs from the prepared input")
            identity = (record, row["source_id"], row["split"])
            if (identity != (metadata.get("record_id"), metadata.get("source_id"), metadata.get("split"))
                    or identity != (other.get("record_id"), other.get("source_id"), other.get("split"))
                    or row["split"] != "val" or other.get("reference_index") != ref
                    or other.get("frame_count") != frames or other.get("fps") != fps
                    or row.get("reference_index") != ref
                    or row.get("evaluation_frame_indices") != [i for i in range(frames) if i != ref]):
                raise ValueError("HRI target/source/reference-frame identity differs")
            binding = {key: metadata["binding"][key] for key in ("data_cache_key", "manifest_sha256", "official_revision")}
            if binding["manifest_sha256"] != selection["manifest_sha256"] or binding["official_revision"] != config["model"]["revision"]:
                raise ValueError("HRI prepared input manifest/official revision binding differs")
            binding.update(metadata_sha256=targets[str(metadata_path)], target_rgb_sha256=targets[str(target_path)],
                source_id=row["source_id"], frames=frames, fps=fps)
            if record in bindings and bindings[record] != binding:
                raise ValueError("Reference cases changed their target/crop bindings")
            bindings[record] = binding
            baseline = _validate_pools(other["bicubic"], frames, ref)
            if set(row.get("modes", {})) != set(MODES) or set(other.get("modes", {})) != set(MODES):
                raise ValueError("HRI report must retain exactly paired on/off modes")
            for key in RGB:
                if not _same_number(baseline["matched_nonreference"][key], row["bicubic"][key]):
                    raise ValueError("Scored bicubic pool differs from its original RGB record")
            for mode in MODES:
                a = _unique(row["modes"][mode]["seeds"], "seed")
                b = _unique(other["modes"][mode]["seeds"], "seed")
                if set(a) != set(b) or set(a) != set(seeds):
                    raise ValueError("Every HRI mode/video requires all declared sampling seeds")
                for seed in seeds:
                    sample = a[seed]
                    path = _inside(sample["rgb_path"], directory)
                    expected_samples.add(str(path))
                    if (str(path) not in samples or sample.get("noise_seed") != seed
                            or sample.get("hri_valid") is not (mode == "hri_on")
                            or sample.get("supplied_reference_index") != (ref if mode == "hri_on" else None)
                            or (mode == "hri_on" and type(sample.get("supplied_reference_index")) is not int)
                            or sample.get("frame_count") != frames or sample.get("fps") != fps
                            or sample["metrics"].get("frame_indices") != [i for i in range(frames) if i != ref]):
                        raise ValueError("Actual saved sample path/noise/HRI/frame contract differs")
                    values = _validate_pools(b[seed]["metrics"], frames, ref)
                    for key in RGB:
                        if (not _same_number(values["matched_nonreference"][key], sample["metrics"][key])
                                or not _same_number(values["reference_frame_separate"][key], sample["provided_reference_frame_metrics"][key])):
                            raise ValueError("Scored sample RGB metrics differ from their original record")
                    observations[mode].append({"record_id": record, "source_id": row["source_id"], "seed": seed,
                        "pools": values})
            for seed in seeds:
                baselines.append({"record_id": record, "source_id": row["source_id"], "seed": seed, "pools": baseline})
        cases[case_id] = {"reference_index": ref, "observations": observations, "bicubic": baselines}
    if expected_samples != set(samples) or expected_targets != set(targets):
        raise ValueError("Saved reference artifact digest maps do not match the complete original path sets")
    costs_path = directory.parent / "summary.json"
    numerical_training = {key: value for key, value in config["training"].items()
        if key not in ("variants", "log_every", "gradient_log_every", "evaluate_at", "checkpoint_at")}
    return {"cases": cases, "bindings": bindings,
        "compatibility": {"sampling": rgb["sampling"], "rgb_metric_protocol": rgb.get("metric_protocol"),
            "perceptual_protocol": score.get("protocol"), "perceptual_audit": score.get("audit"),
            "scorer_sha256": score["scorer_sha256"], "data_seed": config["seed"], "training_seed": training_seed,
            "step": directory.name, "updates_budget": config["training"]["updates"],
            "model": config["model"], "reference": config["reference"], "reproducibility": config.get("reproducibility"),
            "numerical_training": numerical_training, "conditioning": config.get("conditioning"),
            "teacher": config.get("teacher"), "flow": config.get("flow"), "weight_selection": config.get("weight_selection"),
            "common_trainable_initialization": {name: initial_hashes[name] for name in ("lrv", "lora", "hri")}},
        "projection_initialization": initial_hashes["projector"],
        "provenance": {"directory": str(directory), "run": str(run), "variant": directory.parent.name,
            "data_seed": config["seed"], "training_seed": training_seed, "step": directory.name,
            "input_sha256": {str(path): file_hash(path) for path in (rgb_path, score_path, config_path, init_path)},
            "sampled_artifact_sha256": samples, "target_artifact_sha256": targets,
            "costs": _json(costs_path) if costs_path.exists() else None,
            "task_weights_or_optimizer_loaded_by_reporter": False}}


def _metric_groups(rows_by_label, pool):
    available = [key for key in PAIRED_METRIC_KEYS if all(row["pools"][pool][key] is not None
        for rows in rows_by_label.values() for row in rows)]
    if not available:
        raise ValueError("No available matched HRI metrics")
    groups = {label: [{key: row[key] for key in ("record_id", "source_id", "seed")} |
        {"metrics": {key: row["pools"][pool][key] for key in available}} for row in rows]
        for label, rows in rows_by_label.items()}
    return groups, sorted(set(PAIRED_METRIC_KEYS) - set(available))


def summarize_reference_results(results, reference="Base", *, replicates=10000, seed=20261102):
    if not results or reference not in results or "Bicubic" in results:
        raise ValueError("Require Base/reference and reserve the Bicubic label")
    first = results[reference]
    projected = [value["projection_initialization"] for value in results.values()
        if value["projection_initialization"]["numel"] > 0]
    if projected and any(value != projected[0] for value in projected):
        raise ValueError("Auxiliary variants require the same independent projection initialization")
    for label, result in results.items():
        if result["compatibility"] != first["compatibility"] or result["bindings"] != first["bindings"]:
            raise ValueError(f"{label}: sampling/training/metric protocol or raw target/crop binding differs")
        if set(result["cases"]) != set(first["cases"]):
            raise ValueError("Reference case choices cannot be pooled or silently dropped")
    report = {"schema": "vsr_matched_saved_reference_summary_v1", "reference": reference, "reference_cases": {},
        "prespecified_primary_reference_index": 0, "primary_reference_case_present": "0" in first["cases"],
        "compatibility": first["compatibility"], "provenance": {name: value["provenance"] for name, value in results.items()},
        "quality_hypotheses_verified": False, "training_repeat_uncertainty_included": False,
        "protocol": "reference choices separate; per-seed measured pools, equal video means and source-cluster paired bootstrap; pointwise descriptive intervals, no multiplicity correction"}
    for case_id, base_case in first["cases"].items():
        case_report = {"reference_index": base_case["reference_index"],
            "reference_case_role": "prespecified_primary_ref0" if case_id == "0" else "prespecified_secondary_case",
            "cross_method": {}, "within_method_on_minus_off": {},
            "interaction_on_minus_off_vs_reference": {}, "reference_frame_separate": {}, "unavailable_metrics": {}}
        for label, result in results.items():
            candidate = result["cases"][case_id]
            if candidate["reference_index"] != base_case["reference_index"] or candidate["bicubic"] != base_case["bicubic"]:
                raise ValueError("Reference index/bicubic measurements differ across HRI methods")
            for pool in ("matched_nonreference", "reference_frame_separate"):
                groups, missing = _metric_groups(candidate["observations"], pool)
                value = paired_summary(groups, "hri_off", replicates=replicates, seed=seed)
                target = "within_method_on_minus_off" if pool == "matched_nonreference" else "reference_frame_separate"
                case_report[target][label] = value
                case_report["unavailable_metrics"][f"{label}:{pool}"] = missing
        gain_rows = {}
        for mode in MODES:
            rows = {label: value["cases"][case_id]["observations"][mode] for label, value in results.items()}
            rows["Bicubic"] = base_case["bicubic"]
            groups, missing = _metric_groups(rows, "matched_nonreference")
            case_report["cross_method"][mode] = paired_summary(groups, reference, replicates=replicates, seed=seed)
            case_report["unavailable_metrics"][f"cross_method:{mode}"] = missing
        for label, value in results.items():
            observations = value["cases"][case_id]["observations"]
            groups, _ = _metric_groups(observations, "matched_nonreference")
            on = {(row["record_id"], row["seed"]): row for row in groups["hri_on"]}
            off = {(row["record_id"], row["seed"]): row for row in groups["hri_off"]}
            if set(on) != set(off):
                raise ValueError("On/off gains require the identical complete video/seed set")
            gain_rows[label] = [{key: row[key] for key in ("record_id", "source_id", "seed")} |
                {"metrics": {metric: metric_value - off[pair]["metrics"][metric]
                    for metric, metric_value in row["metrics"].items()}} for pair, row in on.items()]
        case_report["interaction_on_minus_off_vs_reference"] = paired_summary(gain_rows, reference, replicates=replicates, seed=seed)
        case_report["interaction_interpretation"] = "(candidate on-off) minus (reference on-off), same original nonreference pool; metric-specific direction remains unchanged"
        report["reference_cases"][case_id] = case_report
    return report


def text_report(report):
    lines = ["VSR matched saved HRI validation results", "Reference cases remain separate; supplied frame is reported separately.",
        "PSNR/SSIM higher is better; LPIPS/tLP/tOF/MSE lower is better.",
        "Source-cluster intervals are descriptive and exclude training-repeat uncertainty."]
    for case_id, case in report["reference_cases"].items():
        lines += ["", f"Reference index {case_id} ({case['reference_case_role']}); identical nonreference frames in on/off."]
        for mode, value in case["cross_method"].items():
            lines.append(f"{mode}: {value['video_count']} videos / {value['source_count']} sources; seeds {value['sampling_seeds']}")
            for label, experiment in value["experiments"].items():
                lines.append(label + " " + ", ".join(f"{key}={number:.6f}" for key, number in experiment["video_macro"].items()))
        lines.append("On-minus-off within each method:")
        for label, value in case["within_method_on_minus_off"].items():
            for metric, change in value["comparisons"]["hri_on"].items():
                lines.append(f"{label} {metric}: {change['delta_video_macro']:+.6f}; CI {change['ci95_source_cluster_video_macro']}")
    lines += ["", "JSON retains per-video/source paired differences, cross-method interactions, provided-frame measurements, unavailable-null metric lists and all artifact SHA bindings.",
        "No quality hypothesis or generalization benefit is established by this report alone."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, metavar="LABEL=VARIANT_STEP_DIRECTORY")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--reference", default="Base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    paths = [args.output.with_suffix(suffix) for suffix in (".json", ".txt")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Reference report exists; preserve it and use a new output prefix")
    results = {}
    for item in args.result:
        label, separator, path = item.partition("=")
        if not separator or not label or label in results:
            raise ValueError("Use distinct LABEL=variant/step result mappings")
        results[label] = load_reference_result(path, project_root=args.project_root)
    report = summarize_reference_results(results, args.reference, replicates=args.replicates)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves prior results even if another reporter races.
    with paths[0].open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with paths[1].open("x") as stream:
        stream.write(text_report(report))
    print(json.dumps({"json": str(paths[0]), "text": str(paths[1]), "reference_cases": list(report["reference_cases"])}))


if __name__ == "__main__":
    main()
