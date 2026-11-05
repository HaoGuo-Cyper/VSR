"""Describe matched Base/Align training repeats without pseudo-replication.

Saved metric loaders establish the measurement bindings. Training seeds are
the units of the across-run description; sampling seeds are first averaged
inside each video. There is deliberately no across-training variance estimate,
confidence interval, p-value, best-seed selection or benefit classifier.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re

from .audit import file_hash
from .paired_statistics import paired_summary
from .summarize_saved import load_saved_result
from .summarize_reference_saved import load_reference_result

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("noref", "reference")
VARIANTS = ("Base", "Align")
KNOWN_VARIANTS = {"Base", "Align", "Relation", "Both"}
DATA_SEED = 20261102
SAMPLING_SEEDS = [42, 43]
# These select runs/assets or logging boundaries, rather than optimizer
# equations. The actual teacher/flow/model settings and prepared source are
# checked separately; record the preparation list in each run's provenance.
_NON_OPTIMIZATION_OPTIONS = {"variants", "preparation_variants", "log_every", "gradient_log_every",
    "evaluate_at", "checkpoint_at", "seed"}
_METADATA_FIELDS = ("data_version", "record_id", "source_id", "source_sha256", "split", "seed",
    "window_seed", "source_indices", "temporal_stride", "original_fps", "output_fps", "fps_policy",
    "source_hw", "resize_hw", "crop_tlhw", "degradation", "caption", "caption_status", "valid_region")


def _inside(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Training-repeat input escapes the independent project's runs")
    return path


def _read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()


def _integer(value, label):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _reference_config(value, training_seed):
    value = copy.deepcopy(value)
    if value.get("initialization_seed") == training_seed + 100:
        value["initialization_seed"] = "training_seed+100"
    return value


def _metadata_contract(metadata, row, config):
    if any(field not in metadata for field in _METADATA_FIELDS):
        raise ValueError("Complete source/window/FPS/degradation metadata is required")
    data = config["data"]
    frames, stride = data["num_frames"], data.get("temporal_stride", 1)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
           for value in (frames, stride, data["height"], data["width"], data["scale"])):
        raise ValueError("Frame/stride/crop/scale configuration must use positive integers")
    indices = metadata["source_indices"]
    if (metadata["split"] != "val" or metadata["seed"] != DATA_SEED
            or metadata["temporal_stride"] != stride or len(indices) != frames
            or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in indices)
            or any(b-a != stride for a, b in zip(indices, indices[1:]))
            or row.get("frame_count") != frames or row.get("fps") != metadata["output_fps"]):
        raise ValueError("Saved validation frame/window/FPS differs from its frozen data configuration")
    original, output = metadata["original_fps"], metadata["output_fps"]
    if (not isinstance(original, (int, float)) or not isinstance(output, (int, float))
            or isinstance(original, bool) or isinstance(output, bool)
            or not math.isfinite(original) or not math.isfinite(output) or min(original, output) <= 0
            or not math.isclose(output, original/stride, rel_tol=1e-8, abs_tol=1e-9)):
        raise ValueError("Sampled FPS must retain the original FPS divided by the fixed stride")
    crop = metadata["crop_tlhw"]
    if (len(crop) != 4 or crop[2:] != [data["height"], data["width"]]
            or metadata["degradation"].get("scale") != data["scale"]):
        raise ValueError("Source crop or degradation differs from the data configuration")
    return {field: metadata[field] for field in _METADATA_FIELDS}


def _envelope(directory, family, root):
    directory = _inside(directory, root / "runs")
    if directory.name != "step_008192" or directory.parent.name not in VARIANTS:
        raise ValueError("Training repeats require completed own Base/Align step_008192 directories")
    variant, run = directory.parent.name, directory.parent.parent
    config, init = _read(run/"config.json"), _read(directory.parent/"initialization.json")
    costs = _read(directory.parent/"summary.json")
    options = config["training"]
    trained = options.get("variants")
    prepared = options.get("preparation_variants", trained)
    # Match prepare_auxiliary's input validation before treating these lists as
    # non-numerical selection metadata. A malformed or inapplicable declaration
    # must not disappear when constructing the normalized comparison protocol.
    for field, names in (("variants", trained), ("preparation_variants", prepared)):
        if (not isinstance(names, (list, tuple)) or not names
                or any(not isinstance(name, str) or name not in KNOWN_VARIANTS for name in names)
                or len(set(names)) != len(names)):
            raise ValueError(f"training.{field} must contain distinct known variants")
    if variant not in trained:
        raise ValueError("Saved directory variant is absent from training.variants")
    if not set(trained).issubset(prepared):
        raise ValueError("training.preparation_variants must include every training variant")
    training_seed = _integer(options.get("seed", config["seed"]), "training seed")
    if (config["seed"] != DATA_SEED or options.get("updates") != 8192
            or costs.get("status") != "completed" or costs.get("updates") != 8192
            or options.get("resume") is not False or options.get("official_initialization_per_variant") is not True
            or init.get("fresh_official_initialization") is not True or init.get("legacy_state_loaded") is not False
            or init.get("optimizer_fresh") is not True or init.get("data_seed") != DATA_SEED
            or init.get("training_seed", init.get("seed")) != training_seed):
        raise ValueError("Completed8192, fixed data seed and independent official fresh initialization are required")
    revision = config.get("model", {}).get("revision")
    if not isinstance(revision, str) or not revision or init.get("official_revision") != revision:
        raise ValueError("Official initialization revision must match the model configuration")
    reference = bool(config.get("reference", {}).get("enabled", False))
    if reference != (family == "reference"):
        raise ValueError("Mixed no-reference/reference training contexts are forbidden")
    groups = init.get("trainable_initialization_hashes", {}).get("groups", {})
    initial = {}
    for name in ("lrv", "lora", "hri", "projector"):
        group = groups.get(name, {})
        if (not re.fullmatch(r"[0-9a-f]{64}", group.get("sha256", ""))
                or any(not isinstance(group.get(k), int) or isinstance(group[k], bool) or group[k] < 0
                       for k in ("numel", "tensor_count"))):
            raise ValueError("Complete fresh trainable-group initialization hashes are required")
        initial[name] = {k: group[k] for k in ("sha256", "numel", "tensor_count")}
    if (any(initial[name]["numel"] <= 0 for name in ("lrv", "lora"))
            or (initial["hri"]["numel"] > 0) != reference
            or (initial["projector"]["numel"] == 0) != (variant == "Base")):
        raise ValueError("Fresh initialization group roles differ from the recorded model family/variant")
    selection = _read(run/"selection.json")
    source, reuse = run, None
    if (run/"cache_reuse.json").exists():
        reuse = _read(run/"cache_reuse.json")
        source = _inside(reuse["source_run"], root/"runs")
        if reuse.get("legacy_state_read") is not False:
            raise ValueError("Data-cache reuse may not read legacy task state")
    origin = _read(source/"selection.json")
    selection_fields = ("manifest_sha256", "training_ids", "validation_ids", "test_accessed")
    if (selection.get("test_accessed") is not False or not selection.get("validation_ids")
            or any(selection.get(k) != origin.get(k) for k in selection_fields)):
        raise ValueError("Training/validation selection differs from the sealed non-test source")
    seal_path = source/"cache_asset_seal.json"
    if not seal_path.exists():
        raise ValueError("Repeat data requires its existing prepared asset seal")
    seal = _read(seal_path)
    if (reuse and reuse.get("source_asset_seal_sha256") != file_hash(seal_path)) or not seal.get("artifact_sha256"):
        raise ValueError("Prepared data asset seal changed or is incomplete")
    if any(seal.get(k, False) for k in ("contains_task_initialization", "task_checkpoint_read", "optimizer_state_read")):
        raise ValueError("Prepared source seal includes forbidden task initialization state")
    # The full map identifies the shared frozen training source. Rehash the
    # actual evaluation targets below and the shared numerical inputs here.
    for filename in ("fresh_training_cache.pt", "fresh_text_embeddings.pt"):
        if filename in seal["artifact_sha256"]:
            if file_hash(source/filename) != seal["artifact_sha256"][filename]:
                raise ValueError("Prepared numerical training input changed")
    rgb = _read(directory/"metrics.json")
    if rgb.get("sampling", {}).get("seeds") != SAMPLING_SEEDS:
        raise ValueError("Every repeat requires the complete preset sampling seeds42/43")
    raw_rows = (rgb["per_video"] if family == "noref" else
        [row for case in rgb["reference_cases"].values() for row in case["per_video"]])
    ids = selection["training_ids"] + selection["validation_ids"]
    targets = {}
    for row in raw_rows:
        if row["split"] != "val":
            if row["split"] == "test":
                raise ValueError("Training-repeat selection cannot use test scores")
            continue
        record, index = row["record_id"], row["clip_index"]
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(ids) or ids[index] != record:
            raise ValueError("Validation clip index differs from the frozen selection order")
        metadata_path, target_path = source/f"clip_{index}/metadata.json", source/f"clip_{index}/paired_rgb.pt"
        metadata = _read(metadata_path)
        if (metadata["record_id"], metadata["source_id"], metadata["split"]) != (record, row["source_id"], "val"):
            raise ValueError("Source identity differs from validation measurement")
        values = _metadata_contract(metadata, row, config)
        binding = metadata.get("binding", {})
        if (not isinstance(binding, dict) or binding.get("manifest_sha256") != selection["manifest_sha256"]
                or binding.get("official_revision") != revision):
            raise ValueError("Prepared input manifest/official revision binding differs from this run")
        for path in (metadata_path, target_path):
            if seal["artifact_sha256"].get(str(path.relative_to(source))) != file_hash(path):
                raise ValueError("Actual validation target/metadata differs from its prepared seal")
        if family == "noref":
            for sample in row["modes"]["correct"]["seeds"]:
                if sample.get("frame_count") != row["frame_count"] or sample.get("fps") != row["fps"]:
                    raise ValueError("Saved sample frame count/FPS differs from its fixed target")
        item = {"metadata": values, "target_rgb_sha256": file_hash(target_path),
            "binding": {k: binding[k] for k in ("data_cache_key", "manifest_sha256", "official_revision")}}
        if record in targets and targets[record] != item:
            raise ValueError("Reference choices use different physical targets")
        targets[record] = item
    if set(targets) != set(selection["validation_ids"]):
        raise ValueError("Every repeat must evaluate the entire fixed validation set")
    numerical = {k: v for k, v in options.items() if k not in _NON_OPTIMIZATION_OPTIONS}
    common_structure = {name: {k: initial[name][k] for k in ("numel", "tensor_count")} for name in ("lrv", "lora", "hri")}
    protocol = {"family": family, "data_seed": DATA_SEED,
        "data": {k: v for k, v in config["data"].items() if k not in ("cache_from", "save_input_videos")},
        "training": numerical, "model": config["model"], "conditioning": config.get("conditioning"),
        "reference": _reference_config(config.get("reference", {}), training_seed),
        "teacher": config.get("teacher"), "flow": config.get("flow"),
        "weight_selection": config.get("weight_selection"), "reproducibility": config.get("reproducibility"),
        "caption_policy": config.get("caption_policy"), "common_initialization_structure": common_structure}
    data_contract = {"data_seed": DATA_SEED, "selection": {k: selection[k] for k in selection_fields},
        "prepared_asset_map_sha256": _digest(seal["artifact_sha256"]), "targets": targets}
    provenance = {"directory": str(directory), "run": str(run), "source_run": str(source),
        "official_revision": revision,
        "training_seed_source": "config.training.seed" if "seed" in options else "config.seed fallback",
        "preparation_variants": options.get("preparation_variants", options.get("variants")),
        "preparation_variants_audit": {
            "raw_training_variants": options.get("variants"),
            "preparation_variants_field_present": "preparation_variants" in options,
            "raw_preparation_variants": options.get("preparation_variants"),
            "effective_preparation_variants": options.get("preparation_variants", options.get("variants")),
            "normalization_reason": "run/asset preparation selection does not enter optimizer equations; actual model, loss weights, teacher/flow settings and frozen data remain separately exact-matched"},
        "input_sha256": {str(path): file_hash(path) for path in (run/"config.json", directory.parent/"initialization.json",
            directory.parent/"summary.json", run/"selection.json", seal_path)},
        "cost": {k: costs.get(k) for k in ("training_gpu_hours", "phase_wall_seconds", "peak_allocated_bytes")},
        "runtime_audit_artifact_sha256": {str(path): file_hash(path) for path in
            (run/"environment.json", directory.parent/"training.jsonl") if path.is_file()},
        "runtime_audit_scope": "environment/training-log hashes are provenance links only; executing-source equality and stepwise training-draw matching require separate independent run audits",
        "task_weights_or_optimizer_loaded_by_reporter": False}
    return directory, config, rgb, initial, training_seed, protocol, data_contract, provenance


def load_repeat_result(directory, *, family="noref", project_root=ROOT):
    if family not in FAMILIES:
        raise ValueError("Choose noref or reference repeat context")
    root = Path(project_root).resolve()
    directory, config, rgb, initial, training_seed, protocol, data_contract, provenance = _envelope(directory, family, root)
    contexts, context_metadata = {}, {}
    if family == "noref":
        loaded = load_saved_result(directory, split="val", mode="correct")
        protocol["measurement"] = loaded["compatibility"]
        contexts["all_frames"] = loaded["observations"]
        context_metadata["all_frames"] = {"family": family, "mode": "correct", "pool": "all_frames"}
    else:
        loaded = load_reference_result(directory, project_root=root)
        measured = copy.deepcopy(loaded["compatibility"])
        for key in ("training_seed", "common_trainable_initialization", "numerical_training"):
            measured.pop(key, None)
        measured["reference"] = _reference_config(measured.get("reference", {}), training_seed)
        protocol["measurement"] = measured
        for case_id, case in loaded["cases"].items():
            modes = {}
            for mode in ("hri_on", "hri_off"):
                rows = case["observations"][mode]
                available = [key for key in rows[0]["pools"]["matched_nonreference"]
                    if all(row["pools"]["matched_nonreference"][key] is not None for row in rows)]
                modes[mode] = [{key: row[key] for key in ("record_id", "source_id", "seed")} |
                    {"metrics": {key: row["pools"]["matched_nonreference"][key] for key in available}} for row in rows]
                name = f"ref{case_id}/{mode}"
                contexts[name] = modes[mode]
                context_metadata[name] = {"family": family, "reference_index": case["reference_index"],
                    "reference_role": "prespecified_primary_ref0" if case_id == "0" else "prespecified_secondary",
                    "mode": mode, "pool": "matched_nonreference"}
            off = {(row["record_id"], row["seed"]): row for row in modes["hri_off"]}
            name = f"ref{case_id}/on_minus_off"
            contexts[name] = [{key: row[key] for key in ("record_id", "source_id", "seed")} |
                {"metrics": {key: value-off[(row["record_id"], row["seed"])]["metrics"][key]
                    for key, value in row["metrics"].items()}} for row in modes["hri_on"]]
            context_metadata[name] = {**context_metadata[f"ref{case_id}/hri_on"], "mode": "on_minus_off",
                "interpretation": "Base/Align means are within-model reference gains; Align-minus-Base is gain interaction"}
        if set(loaded["cases"]) != {"0", "4"}:
            raise ValueError("Reference repeats require both predeclared ref0 and ref4 cases")
    provenance["bound_measurement"] = copy.deepcopy(loaded["provenance"])
    if family == "reference" and isinstance(provenance["bound_measurement"].get("costs"), dict):
        costs = provenance["bound_measurement"]["costs"]
        provenance["bound_measurement"]["costs"] = {key: costs.get(key) for key in
            ("status", "updates", "training_gpu_hours", "phase_wall_seconds", "peak_allocated_bytes")}
    return {"family": family, "variant": directory.parent.name, "training_seed": training_seed,
        "data_seed": DATA_SEED, "protocol": protocol, "data_contract": data_contract,
        "common_initialization": {name: initial[name] for name in ("lrv", "lora", "hri")},
        "projection_initialization": initial["projector"], "contexts": contexts,
        "context_metadata": context_metadata, "provenance": provenance, "fresh_verified": True}


def _training_stats(values):
    return {"mean": sum(values.values())/len(values), "min": min(values.values()), "max": max(values.values()),
        "values_by_training_seed": values}


def _source_rows(report):
    values = {}
    for variant in VARIANTS:
        for row in report["experiments"][variant]["per_video"]:
            source = values.setdefault(row["source_id"], {"record_ids": [], "Base": [], "Align": []})
            if row["record_id"] not in source["record_ids"]:
                source["record_ids"].append(row["record_id"])
            source[variant].append(row["metrics"])
    output = {}
    for source_id, source in values.items():
        means = {name: {key: sum(row[key] for row in source[name])/len(source[name])
                       for key in source[name][0]} for name in VARIANTS}
        output[source_id] = {"record_ids": sorted(source["record_ids"]), **means,
            "Align_minus_Base": {key: means["Align"][key]-means["Base"][key] for key in means["Base"]}}
    return output


def summarize_repeat_results(pairs, *, family="noref", replicates=10000, seed=DATA_SEED):
    if family not in FAMILIES or not pairs:
        raise ValueError("Provide at least one matched training pair and a single family")
    seeds = sorted(_integer(value, "training seed") for value in pairs)
    first = pairs[seeds[0]]["Base"] if set(pairs[seeds[0]]) == set(VARIANTS) else None
    if first is None:
        raise ValueError("Each training seed requires exactly Base and Align")
    expected_contexts = set(first["contexts"])
    declared_contexts = ({"all_frames"} if family == "noref" else
        {f"ref{index}/{mode}" for index in (0, 4) for mode in ("hri_on", "hri_off", "on_minus_off")})
    if expected_contexts != declared_contexts:
        raise ValueError("Keep exactly the full declared family/reference contexts; never select a best reference")
    observation_contracts = {context: sorted((row["record_id"], row["source_id"], row["seed"],
        tuple(sorted(row["metrics"]))) for row in rows) for context, rows in first["contexts"].items()}
    projection_structure = None
    for training_seed in seeds:
        pair = pairs[training_seed]
        if set(pair) != set(VARIANTS):
            raise ValueError("Each training seed requires exactly Base and Align")
        for variant, result in pair.items():
            if (result["family"] != family or result["variant"] != variant
                    or result["training_seed"] != training_seed or result["data_seed"] != DATA_SEED
                    or result.get("fresh_verified") is not True):
                raise ValueError("Training-seed label, variant, family or fresh-init evidence differs")
            if (result["protocol"] != first["protocol"] or result["data_contract"] != first["data_contract"]
                    or result["context_metadata"] != first["context_metadata"]
                    or set(result["contexts"]) != expected_contexts):
                raise ValueError("Training pairs differ in data/targets/windows/FPS/protocol/reference contexts")
            for context, observations in result["contexts"].items():
                contract = sorted((row["record_id"], row["source_id"], row["seed"],
                    tuple(sorted(row["metrics"]))) for row in observations)
                if contract != observation_contracts[context]:
                    raise ValueError("Training pairs differ in actual video/source/sampling/metric sets")
                by_record = {}
                for row in observations:
                    by_record.setdefault(row["record_id"], []).append(row["seed"])
                if not by_record or any(sorted(values) != SAMPLING_SEEDS for values in by_record.values()):
                    raise ValueError("Every video requires sampling seeds42/43 exactly once")
        if pair["Base"]["common_initialization"] != pair["Align"]["common_initialization"]:
            raise ValueError("Base/Align must share the recorded fresh common initialization within each training seed")
        structure = {key: pair["Align"]["projection_initialization"][key] for key in ("numel", "tensor_count")}
        if projection_structure is None:
            projection_structure = structure
        elif structure != projection_structure:
            raise ValueError("Projection architecture changed between training repeats")
    contexts = {}
    for context in sorted(expected_contexts):
        per_training, sources = {}, {}
        for training_seed in seeds:
            report = paired_summary({variant: pairs[training_seed][variant]["contexts"][context]
                for variant in VARIANTS}, "Base", replicates=replicates, seed=seed)
            per_training[str(training_seed)] = report
            for source_id, values in _source_rows(report).items():
                sources.setdefault(source_id, {})[str(training_seed)] = values
        keys = list(next(iter(per_training.values()))["experiments"]["Base"]["video_macro"])
        across = {variant: {key: _training_stats({str(train): per_training[str(train)]["experiments"][variant]["video_macro"][key]
            for train in seeds}) for key in keys} for variant in VARIANTS}
        across["Align_minus_Base"] = {}
        for key in keys:
            values = {str(train): per_training[str(train)]["comparisons"]["Align"][key]["delta_video_macro"] for train in seeds}
            across["Align_minus_Base"][key] = {**_training_stats(values),
                "positive_training_seeds": sum(value > 0 for value in values.values()),
                "negative_training_seeds": sum(value < 0 for value in values.values()),
                "zero_training_seeds": sum(value == 0 for value in values.values())}
        contexts[context] = {"context": first["context_metadata"][context], "per_training_seed": per_training,
            "across_training_seeds": across, "per_source_per_training_seed": sources,
            "across_training_unit": "one completed matched training-seed pair, equal weight per training seed"}
    count = len(seeds)
    return {"schema": "vsr_matched_training_repeats_v1", "family": family,
        "training_seed_count": count, "training_seeds": seeds, "data_seed": DATA_SEED,
        "sampling_seeds": SAMPLING_SEEDS, "contexts": contexts,
        "protocol": first["protocol"], "data_contract": first["data_contract"],
        "training_pair_provenance": {str(train): {variant: pairs[train][variant]["provenance"] for variant in VARIANTS} for train in seeds},
        "replication_note": (f"N={count} completed training pairs; " + ("partial N=1/2 is insufficient for stable-training claims. " if count < 3 else "N remains small. ")
            + "Across-training output is mean/range/sign counts only, without precise variance, p-value or interval; sampling repeats/source bootstraps are not training replicas."),
        "source_intervals_scope": "per-training-pair pointwise descriptive source-cluster95%; excludes training-repeat uncertainty and multiplicity correction",
        "validation_scope": "configuration/fresh-init/completed-budget and bound data/measurement checks; actual executing-source and stepwise training-draw verification is supplied by separate independent run audits, not inferred from hashes here",
        "sign_counts_scope": "arithmetic sign of Align-minus-Base; favorable direction depends on metric",
        "selection_policy": "all explicitly declared completed seed pairs retained; no best seed, reference choice or checkpoint step selection"}


def aggregate_training_repeats(pairs, *, family="noref", project_root=ROOT, replicates=10000, seed=DATA_SEED):
    loaded = {training_seed: {variant: load_repeat_result(path, family=family, project_root=project_root)
        for variant, path in pair.items()} for training_seed, pair in pairs.items()}
    return summarize_repeat_results(loaded, family=family, replicates=replicates, seed=seed)


def parse_pairs(items):
    pairs = {}
    for item in items:
        number, colon, rest = item.partition(":")
        variant, equals, path = rest.partition("=")
        if not colon or not number.isdigit() or not equals or variant not in VARIANTS or not path:
            raise ValueError("Use TRAININGSEED:Base=directory and TRAININGSEED:Align=directory")
        training_seed = int(number)
        pair = pairs.setdefault(training_seed, {})
        if variant in pair:
            raise ValueError("Duplicate training seed/variant entry; never count a seed twice")
        pair[variant] = path
    if not pairs or any(set(pair) != set(VARIANTS) for pair in pairs.values()):
        raise ValueError("Each explicit training seed requires both Base and Align")
    return pairs


def text_report(report):
    lines = ["VSR matched Base/Align training-repeat description", report["replication_note"],
        "Data seed is fixed; training seeds are distinct; sampling42/43 are averaged inside each video.",
        "Source bootstrap intervals apply within each training pair only; no across-training inference or best-seed choice.",
        report["source_intervals_scope"], report["sign_counts_scope"], report["validation_scope"]]
    for context, result in report["contexts"].items():
        lines += ["", context + ": " + json.dumps(result["context"], ensure_ascii=False)]
        for train, pair in result["per_training_seed"].items():
            lines.append("Training seed " + train + f"; {pair['video_count']} videos / {pair['source_count']} sources")
            for variant in VARIANTS:
                lines.append(variant + " " + ", ".join(f"{key}={value:.8f}" for key, value in pair["experiments"][variant]["video_macro"].items()))
            for key, value in pair["comparisons"]["Align"].items():
                lines.append(f"Align-Base {key}: {value['delta_video_macro']:+.8f}; descriptive source CI {value['ci95_source_cluster_video_macro']}")
        lines.append("Across training seeds (equal seed weights; no variance/p-value):")
        for variant in VARIANTS:
            for key, value in result["across_training_seeds"][variant].items():
                lines.append(f"{variant} {key}: mean {value['mean']:.8f}, min {value['min']:.8f}, max {value['max']:.8f}")
        for key, value in result["across_training_seeds"]["Align_minus_Base"].items():
            lines.append(f"{key}: mean {value['mean']:+.8f}, min {value['min']:+.8f}, max {value['max']:+.8f}; positive/negative/zero training seeds {value['positive_training_seeds']}/{value['negative_training_seeds']}/{value['zero_training_seeds']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", required=True, metavar="TRAININGSEED:VARIANT=DIRECTORY")
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    paths = [args.output.with_suffix(ext) for ext in (".json", ".txt")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Preserve an existing repeat report; use a new output prefix")
    report = aggregate_training_repeats(parse_pairs(args.pair), family=args.family,
        project_root=args.project_root, replicates=args.replicates)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    with paths[0].open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with paths[1].open("x") as stream:
        stream.write(text_report(report))
    print(json.dumps({"json": str(paths[0]), "text": str(paths[1]), "training_seed_count": report["training_seed_count"]}))


if __name__ == "__main__":
    main()
