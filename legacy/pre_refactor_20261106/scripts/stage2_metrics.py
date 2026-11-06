"""Pure-stdlib Stage2 pixel-mask pools and predeclared three-seed G2 summary.

Consumes existing per-frame/per-original-pair scores, never loads a metric
model, pixels, checkpoints or CUDA. A later evaluator must bind those inputs.
"""
from __future__ import annotations
import math
import re
import statistics

TLP = "tLP100_alex_v0_1_TecoGAN_crop"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def score_pools(rgb, perceptual, mask):
    need(len(mask) == 33 and all(type(x) is bool for x in mask) and sum(mask) == 8,
         "Require33 boolean mask entries with8 selected frames")
    selected = [i for i, flag in enumerate(mask) if flag]
    need(selected == list(range(selected[0], selected[0]+8)) and selected[0] in (4, 8, 12, 16, 20),
         "Require the frozen interior8-frame burst")
    neighbors = [selected[0]-1, selected[-1]+1]
    frames = rgb["per_frame"]
    lpips = perceptual["lpips_per_frame"]
    need(len(frames) == len(lpips) == 33 and all(finite(x) for x in lpips), "Incomplete/nonfinite per-frame metrics")
    for row in frames:
        need(all(finite(row[k]) for k in ("psnr_rgb_db", "ssim_rgb")), "Nonfinite frame fidelity metric")
    pairs = perceptual["tecogan"]["per_pair"]
    pairmap = {tuple(row["indices"]): row for row in pairs}
    expected = {(i, i+1) for i in range(2, 30)}
    need(len(pairmap) == len(pairs) == 28 and set(pairmap) == expected
         and all(finite(row[TLP]) for row in pairs), "Expected28 complete original TecoGAN adjacent pairs")
    touch = sorted(p for p in expected if any(mask[i] for i in p))
    boundary = [p for p in touch if mask[p[0]] != mask[p[1]]]
    need(len(touch) == 9 and len(boundary) == 2, "Unexpected temporal mask support")
    pools = {}
    for name, indices in (("masked", selected), ("neighbors", neighbors), ("all_frames", list(range(33)))):
        pools[name] = {"indices": indices, "psnr_rgb_db": statistics.mean(frames[i]["psnr_rgb_db"] for i in indices),
                       "ssim_rgb": statistics.mean(frames[i]["ssim_rgb"] for i in indices),
                       "lpips_alex_v0_1": statistics.mean(lpips[i] for i in indices)}
    return {"mask": mask, "pools": pools,
            "raw_harm_gap": {"psnr_rgb_db": pools["neighbors"]["psnr_rgb_db"]-pools["masked"]["psnr_rgb_db"],
                             "lpips_alex_v0_1": pools["masked"]["lpips_alex_v0_1"]-pools["neighbors"]["lpips_alex_v0_1"]},
            "temporal": {"mask_touch_original_pairs": [list(p) for p in touch],
                "boundary_original_pairs": [list(p) for p in boundary],
                "mask_touch_tLP100": statistics.mean(pairmap[p][TLP] for p in touch),
                "all28pairs_tLP100": statistics.mean(row[TLP] for row in pairs)}}


def paired_excess(clean, corrupt):
    need(clean["mask"] == corrupt["mask"], "Clean/corrupt masks differ")
    return {metric: {"clean_gap": clean["raw_harm_gap"][metric],
                    "corrupt_gap": corrupt["raw_harm_gap"][metric],
                    "paired_excess_gap": corrupt["raw_harm_gap"][metric]-clean["raw_harm_gap"][metric]}
            for metric in ("psnr_rgb_db", "lpips_alex_v0_1")}


def summarize_g2(rows, seal):
    cfg = seal["config"]
    cohort = {r["record"]["id"]: r for r in seal["cohort"]}
    seeds, samples, conditions = cfg["training_seeds"], cfg["inference"]["sampling_seeds"], cfg["conditions"]
    need(len(seeds) == 3 and len(samples) == 2 and len(conditions) == 5 and len(cohort) == 32,
         "G2 requires3 matched training seeds,2 sampler seeds,all5 types and val32")
    seen, checkpoint_hashes = {}, {}
    for row in rows:
        key = row["training_seed"], row["record_id"], row["sampling_seed"], row["condition"]
        need(key not in seen and key[0] in seeds and key[1] in cohort and key[2] in samples
             and key[3] in ["clean", *conditions], "Duplicate/out-of-cohort score observation")
        need(row["source_id"] == cohort[key[1]]["record"]["source_id"]
             and row["cohort_seal_digest"] == seal["seal_digest"]
             and row["score"]["mask"] == cohort[key[1]]["mask"], "Source/seal/mask score binding mismatch")
        need(isinstance(row["checkpoint_sha256"], str) and re.fullmatch('[0-9a-f]{64}',row["checkpoint_sha256"]),
             "A checkpoint digest declaration is required")
        checkpoint_hashes.setdefault(key[0], row["checkpoint_sha256"])
        need(checkpoint_hashes[key[0]] == row["checkpoint_sha256"], "Checkpoint changed within training seed")
        seen[key] = row["score"]
    need(len(seen) == 3*32*2*6, "Missing score rows; cannot drop clips/seeds/types")
    appendix, per_seed = [], {}
    metrics = ("psnr_rgb_db", "lpips_alex_v0_1")
    for seed in seeds:
        type_rows = {}
        for condition in conditions:
            clips = []
            for record in cohort:
                comparisons = [paired_excess(seen[seed, record, sample, "clean"], seen[seed, record, sample, condition])
                               for sample in samples]
                values = {metric: {field: statistics.mean(c[metric][field] for c in comparisons)
                                   for field in ("clean_gap", "corrupt_gap", "paired_excess_gap")} for metric in metrics}
                clips.append(values)
                appendix.append({"training_seed": seed, "record_id": record, "condition": condition, "metrics": values})
            type_rows[condition] = {metric: {field: statistics.mean(c[metric][field] for c in clips)
                                            for field in ("clean_gap", "corrupt_gap", "paired_excess_gap")} for metric in metrics}
        per_seed[str(seed)] = {"per_type": type_rows, "equal_type_macro": {
            metric: {field: statistics.mean(type_rows[c][metric][field] for c in conditions)
                     for field in ("clean_gap", "corrupt_gap", "paired_excess_gap")} for metric in metrics}}
    gate = {}
    for metric in metrics:
        clean = [per_seed[str(s)]["equal_type_macro"][metric]["clean_gap"] for s in seeds]
        excess = [per_seed[str(s)]["equal_type_macro"][metric]["paired_excess_gap"] for s in seeds]
        spread, mean = statistics.stdev(clean), statistics.mean(excess)
        need(all(finite(x) for x in [*clean, *excess]), "Nonfinite G2 values")
        gate[metric] = {"clean_gap_by_training_seed": dict(zip(map(str, seeds), clean)),
                        "paired_excess_gap_by_training_seed": dict(zip(map(str, seeds), excess)),
                        "mean_paired_excess_gap": mean, "paired_excess_gap_sample_std": statistics.stdev(excess),
                        "clean_gap_sample_std_ddof1_threshold": spread, "passes": mean > spread}
    return {"status": "passed_G2" if all(x["passes"] for x in gate.values()) else "failed_G2_stop",
            "gate_metrics": gate, "per_training_seed_and_type": per_seed, "per_clip_appendix": appendix,
            "cohort_seal_digest": seal["seal_digest"], "checkpoint_digest_declarations": checkpoint_hashes,
            "observations": len(rows), "p_values_computed": False, "Stage5_significance_decision": False,
            "underlying_prediction_payloads_revalidated_by_this_summarizer": False,
            "failure_policy": "stop/report; no in-place severity or cohort change"}
