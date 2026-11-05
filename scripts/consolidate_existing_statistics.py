"""Consolidate saved HQ metrics without importing torch or reading pixel payloads.

The validation repeats and frozen internal test remain separate protocols.
Only JSON inputs are read. Reported hashes bind JSON records, not the underlying
checkpoints, RGB tensors, PNGs, cache tensors, or source videos.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SEEDS = (20261102, 20261103, 20261104)
SAMPLE_SEEDS = (42, 43)
VARIANTS = ("Base", "Align", "Relation", "Both")
METRICS = ("psnr_rgb_db", "ssim_rgb", "lpips_alex_v0_1",
           "tLP100_alex_v0_1_TecoGAN_crop", "tOF_Farneback_TecoGAN_crop")
RUNS = {
    ("NoRef", 20261102, "Base"): "formal_hq_v2_noref_20261102T142214.588571Z",
    ("NoRef", 20261102, "aux"): "formal_hq_v2_remaining_noref_20261102T153108.569879Z",
    ("NoRef", 20261103, "all"): "formal_hq_v2_noref_repeat_seed20261103_20261102T194730.270735Z",
    ("NoRef", 20261104, "all"): "formal_hq_v2_noref_repeat_seed20261104_20261102T204607.424405Z",
    ("HRI", 20261102, "all"): "formal_hq_v2_reference_20261102T171545.936007Z",
    ("HRI", 20261103, "all"): "formal_hq_v2_reference_repeat_seed20261103_20261102T214631.884279Z",
    ("HRI", 20261104, "all"): "formal_hq_v2_reference_repeat_seed20261104_20261102T225815.432804Z",
}
FROZEN = "HQ_test_frozen_T1_policy_20261103T022832.538880Z/hq_test"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        path = Path(path)
        require(path.suffix == ".json", f"Non-JSON input forbidden: {path}")
        raw = path.read_bytes()
        self.hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)


def selected_metrics(values):
    result = dict(values)
    if "lpips_alex_v0_1" not in result:
        result["lpips_alex_v0_1"] = result.get("lpips_alex01_rgb_allframes")
    answer = {key: result.get(key) for key in METRICS}
    require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in answer.values()), "A required metric is missing or non-finite")
    return answer


def mean_samples(samples):
    require(sorted(samples) == list(SAMPLE_SEEDS), "Sampling seeds must be exactly 42/43")
    return {key: float(np.mean([samples[s][key] for s in SAMPLE_SEEDS])) for key in METRICS}


def rows_by_id(rows):
    result = {row["record_id"]: row for row in rows}
    require(len(result) == len(rows), "Duplicate record ID")
    return result


def by_seed(rows):
    result = {row["seed"]: row for row in rows}
    require(len(result) == len(rows) and sorted(result) == list(SAMPLE_SEEDS),
            "Duplicated/incomplete sampling seeds")
    return result


def assert_close(a, b, message):
    require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), message)


def run_path(root, family, seed, variant):
    role = "all" if family == "HRI" or seed != SEEDS[0] else ("Base" if variant == "Base" else "aux")
    return root / "runs" / RUNS[(family, seed, role)]


def load_validation(root, inputs):
    """Return context -> variant -> training seed -> record -> clip mean."""
    groups = defaultdict(lambda: defaultdict(dict))
    contracts = {}
    selection_ids = None
    input_target_hashes = {}
    sampling_contracts = {}
    counts = {}
    for family in ("NoRef", "HRI"):
        for variant in VARIANTS:
            for seed in (SEEDS if variant in ("Base", "Align") else SEEDS[:1]):
                run = run_path(root, family, seed, variant)
                step = run / variant / "step_008192"
                config = inputs.read(run / "config.json")
                summary = inputs.read(run / variant / "summary.json")
                selection = inputs.read(run / "selection.json")
                require(config["training"].get("seed", config["seed"]) == seed,
                        "Training-seed label differs from config")
                require(config["seed"] == SEEDS[0] and summary["status"] == "completed"
                        and summary["updates"] == 8192, "Incomplete or differently seeded training")
                require(config["data"]["num_frames"] == 9 and config["data"].get("temporal_stride", 1) == 4
                        and (config["data"]["height"], config["data"]["width"]) == (128, 224),
                        "Unexpected validation geometry")
                require(selection["test_accessed"] is False, "Validation selection accessed test")
                ids = selection["validation_ids"]
                require(len(ids) == 32 and len(set(ids)) == 32, "Require entire val32")
                if selection_ids is None:
                    selection_ids = ids
                require(ids == selection_ids, "Validation records differ between models")
                origin = run
                if (run / "cache_reuse.json").exists():
                    origin = Path(inputs.read(run / "cache_reuse.json")["source_run"])
                seal = inputs.read(origin / "cache_asset_seal.json")
                rgb_path = step / "metrics.json"
                rgb = inputs.read(rgb_path)
                perceptual = inputs.read(step / ("perceptual_metrics_v2.json" if family == "NoRef"
                                                else "perceptual_reference_metrics.json"))
                require(rgb["status"] == perceptual["status"] == "completed", "Incomplete saved metrics")
                require(perceptual["rgb_metrics_sha256"] == inputs.hashes[str(rgb_path)],
                        "Perceptual JSON binding to RGB metric JSON differs")
                require(rgb["sampling"]["seeds"] == list(SAMPLE_SEEDS)
                        and rgb["sampling"]["steps"] == 20 and rgb["sampling"]["cfg"] == 1.0,
                        "Validation inference protocol changed")
                require(rgb["sampling"] == perceptual["sampling"], "RGB/perceptual sampling differs")
                if family not in sampling_contracts:
                    sampling_contracts[family] = {"sampling": rgb["sampling"],
                                                  "metric_protocol": rgb["metric_protocol"],
                                                  "perceptual_audit": perceptual["audit"]}
                require(sampling_contracts[family] == {"sampling": rgb["sampling"],
                          "metric_protocol": rgb["metric_protocol"], "perceptual_audit": perceptual["audit"]},
                        "Validation measurement contract differs across variants/training seeds")
                cases = [("all_frames", rgb["per_video"], perceptual["per_video"], "correct")] if family == "NoRef" else [
                    (f"ref{ref}/{mode}", rgb["reference_cases"][str(ref)]["per_video"],
                     perceptual["reference_cases"][str(ref)]["per_video"], mode)
                    for ref in (0, 4) for mode in ("hri_on", "hri_off")]
                for case, raw_rgb, raw_perceptual, mode in cases:
                    context = f"HQ_val32/{family}/{case}"
                    rr, pp = rows_by_id(raw_rgb), rows_by_id(raw_perceptual)
                    require(set(rr) == set(pp) == set(ids), "Saved rows do not cover exactly val32")
                    values = {}
                    for record in ids:
                        a, b = rr[record], pp[record]
                        require(a["source_id"] == b["source_id"] and a["split"] == b["split"] == "val",
                                "Validation source/split mismatch")
                        metadata_path = origin / f"clip_{a['clip_index']}" / "metadata.json"
                        metadata = inputs.read(metadata_path)
                        target_name = f"clip_{a['clip_index']}/paired_rgb.pt"
                        metadata_name = f"clip_{a['clip_index']}/metadata.json"
                        require(inputs.hashes[str(metadata_path)] == seal["artifact_sha256"][metadata_name],
                                "Actual metadata JSON differs from historical seal")
                        require(metadata["record_id"] == record and metadata["source_id"] == a["source_id"]
                                and metadata["temporal_stride"] == 4 and len(metadata["source_indices"]) == 9,
                                "Validation window differs from recorded metadata")
                        target_hash = seal["artifact_sha256"][target_name]
                        if record not in input_target_hashes:
                            input_target_hashes[record] = target_hash
                        require(input_target_hashes[record] == target_hash, "Validation target digest differs")
                        aa, bb = by_seed(a["modes"][mode]["seeds"]), by_seed(b["modes"][mode]["seeds"])
                        samples = {}
                        for sample_seed in SAMPLE_SEEDS:
                            av, bv = aa[sample_seed], bb[sample_seed]
                            require(av.get("noise_seed", sample_seed) == sample_seed,
                                    "Actual validation noise seed differs")
                            if family == "NoRef":
                                require(av["rgb_path"] == bv["rgb_path"] and bv["noise_seed"] == sample_seed,
                                        "NoRef saved RGB path/noise binding mismatch")
                                require(b["source_rgb_sha256"] == target_hash,
                                        "NoRef target digest binding mismatch")
                                metrics = selected_metrics({**av["metrics"], **bv["metrics"],
                                                            **bv["metrics"]["tecogan"]})
                            else:
                                require(perceptual["target_artifact_sha256"][str(origin / target_name)] == target_hash,
                                        "HRI target digest binding mismatch")
                                require(av["rgb_path"] in perceptual["sampled_artifact_sha256"],
                                        "HRI sampled RGB digest binding absent")
                                pool = bv["metrics"]["matched_nonreference"]
                                ref = int(case[3])
                                require(pool["frame_indices"] == [i for i in range(9) if i != ref],
                                        "HRI reference exclusion pool changed")
                                require(pool["temporal_pair_indices"] == [[i, i+1] for i in range(2, 6)
                                        if ref not in (i, i+1)], "HRI temporal pair pool changed")
                                metrics = selected_metrics(pool)
                                for key in METRICS[:2]:
                                    assert_close(metrics[key], av["metrics"][key], "HRI spatial JSON scores differ")
                            samples[sample_seed] = metrics
                        contract = {k: metadata[k] for k in ("source_id", "source_sha256", "source_indices",
                                    "temporal_stride", "crop_tlhw", "degradation", "original_fps", "output_fps")}
                        contract["target_rgb_stored_sha256"] = target_hash
                        key = (context, record)
                        require(key not in contracts or contracts[key] == contract,
                                "Validation source/window/crop/degradation contract differs")
                        contracts[key] = contract
                        values[record] = {"source_id": a["source_id"], "metrics": mean_samples(samples)}
                    groups[context][variant][seed] = values
                    counts[context] = {"clips": len(values), "sources": len({v['source_id'] for v in values.values()})}
    return groups, {"coverage": counts, "sampling_and_metric_contracts": sampling_contracts,
                    "target_stored_digest_map_sha256": digest(input_target_hashes),
                    "validation_ids_sha256": digest(selection_ids),
                    "underlying_target_payload_digests_recomputed": False}


def load_frozen(root, inputs):
    base = root / "runs" / FROZEN
    protocol = inputs.read(base / "protocol.json")
    summary = inputs.read(base / "summary.json")
    require(summary["status"] == "completed" and protocol["seeds"] == list(SAMPLE_SEEDS)
            and protocol["frames"] == 9 and protocol["steps"] == 20, "Unexpected frozen HQ protocol")
    require(protocol["hq_configuration"]["split"] == "test"
            and protocol["hq_configuration"]["temporal_stride"] == 4, "Frozen split/stride changed")
    expected_labels = {f"{family}_{variant}_seed20261102_step8192" for family in ("NoRef", "HRI") for variant in VARIANTS}
    require(set(protocol["checkpoints"]) == set(summary["results"]) == expected_labels,
            "Frozen checkpoint cohort differs from the declared eight seed02 models")
    groups = defaultdict(lambda: defaultdict(dict))
    contracts, count = {}, 0
    for label in sorted(expected_labels):
        family, variant = label.split("_")[:2]
        for dataset, ids in sorted(protocol["datasets"].items()):
            require(len(ids) == len(set(ids)) == 211, "Frozen test must cover all 211 records")
            directory = base / "results" / label / dataset
            actual = {p.parent.name for p in directory.glob("*/metrics.json")}
            require(actual == set(ids), "Frozen metric coverage differs from the entire test211")
            models = defaultdict(dict)
            for record in sorted(ids):
                path = directory / record / "metrics.json"
                data = inputs.read(path)
                complete = inputs.read(path.parent / "complete.json")
                require(complete["metrics_sha256"] == inputs.hashes[str(path)]
                        and complete["contract_sha256"] == data["contract_sha256"],
                        "Frozen complete/metrics JSON binding mismatch")
                require(data["status"] == "completed" and data["label"] == label
                        and data["dataset"] == dataset and data["sequence_id"] == record,
                        "Frozen metric label or completion mismatch")
                require(data["checkpoint_sha256"] == protocol["checkpoints"][label]["checkpoint_sha256"],
                        "Frozen checkpoint digest label differs")
                require(data["source"]["split"] == "test" and data["source"]["temporal_stride"] == 4,
                        "Frozen metadata split/stride mismatch")
                stable = data["stable_input_contract"]
                require(digest(stable) == data["stable_input_contract_sha256"],
                        "Frozen stable input contract JSON digest mismatch")
                key = dataset, record
                require(key not in contracts or contracts[key] == stable,
                        "Frozen candidates used different actual target/condition contracts")
                contracts[key] = stable
                modes = ("no_reference",) if family == "NoRef" else ("hri_on", "hri_off")
                require(set(data["modes"]) == set(modes), "Unexpected frozen reference modes")
                for mode in modes:
                    samples = by_seed(data["modes"][mode]["seeds"])
                    pool = "all_frames" if family == "NoRef" else "matched_nonreference"
                    metrics = {}
                    for s, sample in samples.items():
                        require(sample["noise_seed"] == sample["requested_noise_seed"] == s
                                and sample["noise_seed_verified_by_sampler_metadata"] is True,
                                "Frozen actual sampling noise seed mismatch")
                        require(sample["restoration"]["sampling"]["steps"] == 20
                                and sample["restoration"]["sampling"]["cfg"] == 1.0,
                                "Frozen sampling settings changed")
                        require(sample["reference_index"] == (0 if family == "HRI" and mode == "hri_on" else None),
                                "Frozen supplied-reference index changed")
                        metrics[s] = selected_metrics(sample["metrics"][pool])
                    models[mode][record] = {"source_id": data["source_id"], "metrics": mean_samples(metrics)}
                count += 1
            for mode, rows in models.items():
                context = f"HQ_test211/{family}/{dataset}/{mode}"
                groups[context][variant][SEEDS[0]] = rows
                expected = summary["primary_report"][label][dataset][mode]["metrics"]
                for metric in METRICS:
                    assert_close(float(np.mean([row["metrics"][metric] for row in rows.values()])),
                                 expected[metric], "Recomputed frozen mean differs from existing summary")
    small_protocol = {k: protocol[k] for k in ("frames", "start", "seeds", "steps", "tile_hw", "overlap",
                       "quantization", "reference", "primary_paired_pool", "temporal", "hq_configuration")}
    return groups, {"metric_files_validated": count, "training_seeds": [SEEDS[0]],
                    "protocol": small_protocol, "actual_input_contracts_sha256": digest(
                        {f"{ds}/{record}": value for (ds, record), value in contracts.items()}),
                    "underlying_checkpoint_png_and_tensor_hashes_recomputed": False}


def bootstrap(delta, sources, replicates, seed):
    unique = sorted(set(sources))
    members = [np.flatnonzero(np.array(sources) == source) for source in unique]
    sums = np.stack([delta[idx].sum(axis=0) for idx in members])
    counts = np.array([len(idx) for idx in members], dtype=float)
    draws = np.random.default_rng(seed).integers(0, len(members), size=(replicates, len(members)))
    values = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
    source_values = (sums / counts[:, None])[draws].mean(axis=1)
    return np.quantile(values, [.025, .975], axis=0), np.quantile(source_values, [.025, .975], axis=0)


def describe(values):
    return {"mean": float(np.mean(values)), "sample_std_ddof1": float(np.std(values, ddof=1)) if len(values)>1 else None,
            "training_seed_count": len(values), "min": float(min(values)), "max": float(max(values))}


def summarize(groups, replicates, bootstrap_seed):
    reports = {}
    for context, variants in sorted(groups.items()):
        first = variants["Base"][SEEDS[0]]
        records = sorted(first)
        sources = [first[record]["source_id"] for record in records]
        arrays = {}
        report = {"clip_count": len(records), "source_count": len(set(sources)), "sampling_seeds": list(SAMPLE_SEEDS),
                  "primary_context": context == "HQ_val32/NoRef/all_frames" or context == "HQ_val32/HRI/ref0/hri_on"
                    or "/HQ-test-identity/" in context,
                  "variants": {}, "comparisons_minus_Base": {}}
        for variant, trainings in variants.items():
            arrays[variant] = {}
            means = {}
            for seed, rows in sorted(trainings.items()):
                require(set(rows) == set(records) and all(rows[r]["source_id"] == first[r]["source_id"] for r in records),
                        "Clip/source pairing mismatch in summary")
                array = np.array([[rows[r]["metrics"][metric] for metric in METRICS] for r in records], dtype=float)
                arrays[variant][seed] = array
                means[seed] = array.mean(axis=0)
            report["variants"][variant] = {"training_seeds": sorted(trainings),
                "label": "single training seed, indicative" if len(trainings)==1 else "three matched training seeds; descriptive",
                "per_training_seed": {str(seed): dict(zip(METRICS, values.tolist())) for seed, values in means.items()},
                "metrics": {metric: describe([means[seed][i] for seed in means]) for i, metric in enumerate(METRICS)}}
        for variant in VARIANTS[1:]:
            trainings = sorted(arrays[variant])
            deltas = {seed: arrays[variant][seed] - arrays["Base"][seed] for seed in trainings}
            averaged = np.mean(list(deltas.values()), axis=0)
            ci, source_ci = bootstrap(averaged, sources, replicates, bootstrap_seed)
            per_seed = {}
            for seed, delta in deltas.items():
                seed_ci, _ = bootstrap(delta, sources, replicates, bootstrap_seed)
                per_seed[str(seed)] = {metric: {"delta": float(delta[:, i].mean()), "ci95_source_cluster": seed_ci[:,i].tolist()}
                                      for i, metric in enumerate(METRICS)}
            report["comparisons_minus_Base"][variant] = {"paired_training_seeds": trainings,
                "bootstrap_conditioned_on_fixed_training_seeds": True,
                "per_training_seed": per_seed,
                "metrics": {metric: {**describe([float(deltas[s][:,i].mean()) for s in trainings]),
                    "ci95_source_cluster_fixed_training_seeds": ci[:,i].tolist(),
                    "ci95_equal_source_macro_fixed_training_seeds": source_ci[:,i].tolist(),
                    "favorable_training_seed_count": int(sum((deltas[s][:,i].mean()>0) if i<2 else (deltas[s][:,i].mean()<0) for s in trainings))}
                    for i, metric in enumerate(METRICS)}}
        reports[context] = report
    return reports


def write_outputs(output, result):
    output.mkdir(parents=True, exist_ok=True)
    (output / "statistics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    fields = ["context", "variant", "training_seed_count", "clip_count", "source_count", "metric", "mean", "sample_std_ddof1",
              "paired_delta_minus_Base", "paired_delta_sample_std_ddof1", "ci95_source_cluster_low", "ci95_source_cluster_high", "label"]
    with (output / "statistics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for context, report in result["contexts"].items():
            for variant, values in report["variants"].items():
                for metric in METRICS:
                    stats = values["metrics"][metric]
                    row = {"context": context, "variant": variant, "clip_count": report["clip_count"], "source_count": report["source_count"],
                           "metric": metric, "label": values["label"], **{k:stats[k] for k in ("mean", "sample_std_ddof1", "training_seed_count")}}
                    if variant != "Base":
                        comparison = report["comparisons_minus_Base"][variant]["metrics"][metric]
                        row.update(paired_delta_minus_Base=comparison["mean"], paired_delta_sample_std_ddof1=comparison["sample_std_ddof1"],
                            ci95_source_cluster_low=comparison["ci95_source_cluster_fixed_training_seeds"][0],
                            ci95_source_cluster_high=comparison["ci95_source_cluster_fixed_training_seeds"][1])
                    writer.writerow(row)
    text = ["# Stage 0: existing internal HQ statistics (2026-11-05)", "",
            "Status: partial T0.4. External evaluation coverage is incomplete; these tables do not close Stage 0.", "",
            "Only existing JSON scores were read. No checkpoint, RGB tensor, video or PNG payload was loaded, and no metric was rescored. JSON bindings and candidate input-contract fields were checked; underlying stored payload digests were not revalidated.", "",
            "Validation: fixed source-held-out val32, 32 sources, nine frames at stride 4, 128×224, ×4 bicubic, 20-step UniPC/CFG1, sampling seeds 42/43 averaged within each clip. Base/Align have three matched training seeds (20261102/03/04). HRI ref0/on is the predeclared primary reference context; ref4 and off remain separate supplementary contexts. HRI excludes the supplied reference frame and all temporal pairs touching it; NoRef uses all frames. Do not compare these unequal family pools directly.", "",
            "Frozen internal test: all 211 held-out clips, 180 sources, only training seed 20261102 for all eight models. It uses saved uint8 PNG/255 measurement and independently frozen inference, whereas validation uses saved float32 RGB. Identity/blur/noise/compression/combined are kept separate. Test results are single training seed, indicative; there are no test211 repeated-seed statistics.", "",
            "Mean ± SD uses equal clip means within each training seed, then equal training-seed weights; SD is sample SD (ddof=1), descriptive with N=3. Delta is candidate minus Base. Paired 95% percentile intervals resample source groups, retaining their clips and matched candidates. For three-seed comparisons each clip's three paired deltas is averaged before resampling sources: the intervals condition on those fixed trained instances and exclude training-seed uncertainty. "
            f"There are {result['bootstrap']['replicates']:,} draws with seed {result['bootstrap']['seed']}, no multiplicity correction and no p-values. In val32 each source has one clip, so source-cluster bootstrap is also paired clip bootstrap.", "",
            "PSNR/SSIM: higher is better. LPIPS/tLP100/tOF: lower is better. tLP100 is the TecoGAN temporal LPIPS scale, not E*warp or a newly measured temporal metric.", ""]
    for context, report in result["contexts"].items():
        text += [f"## {context}", "", f"{report['clip_count']} clips / {report['source_count']} source groups. No protocol or reference-pool averaging.", "",
                 "| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for variant in VARIANTS:
            value=report["variants"][variant]
            cells=[]
            for metric in METRICS:
                stats=value["metrics"][metric]
                cells.append(f"{stats['mean']:.6f}" + (f" ± {stats['sample_std_ddof1']:.6f}" if stats['sample_std_ddof1'] is not None else ""))
            text.append(f"| {variant} | {len(value['training_seeds'])}"+(" (single seed, indicative)" if len(value['training_seeds'])==1 else "")+" | "+" | ".join(cells)+" |")
        text += ["", "| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |",
                 "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for variant in VARIANTS[1:]:
            compare=report["comparisons_minus_Base"][variant]
            cells=[]
            for metric in METRICS:
                stats=compare["metrics"][metric];lo,hi=stats["ci95_source_cluster_fixed_training_seeds"]
                cells.append(f"{stats['mean']:+.6f} [{lo:+.6f}, {hi:+.6f}]")
            text.append(f"| {variant} (N={len(compare['paired_training_seeds'])}) | "+" | ".join(cells)+" |")
        text += [""]
    text += ["## Interpretation and limits", "",
             "The matched val32 repeats show mixed effects. NoRef Align has slightly higher mean PSNR/SSIM but worse mean LPIPS and tOF; HRI ref0/on has nearly unchanged mean PSNR with favorable mean LPIPS/tLP100/tOF, but the direction changes between training seeds. Those effects do not establish a stable overall auxiliary-loss gain. Relation/Both and every frozen-test model remain single training seed, indicative. Report all metrics and contexts; do not select the best seed, reference position, corruption condition or metric.", "",
             "The CSV is a compact metric-wise publication table. JSON additionally contains per-training-seed means/deltas, source-bootstrap intervals, coverage and input audit fingerprints. T0.4 remains partial until the external cohort is completed under a declared compatible protocol.", ""]
    (output / "statistics.md").write_text("\n".join(text))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/home/hugo/Projects/VSR"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261102)
    args=parser.parse_args()
    require(args.bootstrap_replicates>=100, "At least 100 bootstrap replicates required")
    require(not any((args.output / n).exists() for n in ("statistics.json", "statistics.csv", "statistics.md")),
            "Refusing to overwrite an existing statistics report; choose a new output directory")
    inputs=Inputs()
    val, val_audit=load_validation(args.source_root, inputs)
    print("Validation JSON contracts passed; reading frozen HQ test JSON records.", flush=True)
    frozen, frozen_audit=load_frozen(args.source_root, inputs)
    reports=summarize({**val, **frozen}, args.bootstrap_replicates, args.bootstrap_seed)
    result={"schema_version":1,"date":"2026-11-05","status":"partial_T0.4_external_incomplete",
            "source_root":str(args.source_root), "protocols_not_pooled":["HQ_val32_float32", "HQ_test211_saved_uint8"],
            "metrics":list(METRICS), "bootstrap":{"method":"paired source-cluster percentile bootstrap of clip-macro deltas",
                "replicates":args.bootstrap_replicates,"seed":args.bootstrap_seed,"confidence":0.95,
                "training_instances":"fixed; average paired training-seed deltas within clip before source resampling",
                "training_seed_uncertainty_included":False,"p_values_computed":False,"multiplicity_correction":False},
            "std":{"definition":"sample standard deviation across training-seed clip-macro means; ddof=1", "single_training_seed_std":None},
            "validation_audit":val_audit,"frozen_test_audit":frozen_audit,
            "input_audit":{"json_file_count":len(inputs.hashes),"sorted_path_and_sha256_map_fingerprint":digest(inputs.hashes),
                "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "payloads_read":False,"underlying_stored_payload_digests_recomputed":False},
            "contexts":reports}
    write_outputs(args.output, result)
    print(json.dumps({"contexts":len(reports),"json_inputs":len(inputs.hashes),"output":str(args.output)}))


if __name__ == "__main__":
    main()
