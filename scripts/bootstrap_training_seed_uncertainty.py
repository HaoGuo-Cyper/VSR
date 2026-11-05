"""Exploratory crossed training-seed/source bootstrap of existing val32 scores.

No torch, checkpoint, RGB, PNG or video payload is read. This report is a
supplement to the conditional-over-source intervals in statistics.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def loader_module():
    # A private in-memory module lets the existing strict JSON loader read only
    # Base/Align, without changing its source or scanning Relation/Both/test211.
    sys.dont_write_bytecode = True
    path = Path(__file__).with_name("consolidate_existing_statistics.py")
    spec = importlib.util.spec_from_file_location("_val32_seed_uncertainty_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.VARIANTS = ("Base", "Align")
    return module


def crossed_interval(delta, sources, seed_draws, source_draws):
    """delta: training seed × clip × metric; preserve both paired axes."""
    unique = sorted(set(sources))
    require = lambda ok: None if ok else (_ for _ in ()).throw(ValueError("Invalid crossed bootstrap shape"))
    require(delta.ndim == 3 and delta.shape[1] == len(sources))
    require(seed_draws.shape[0] == source_draws.shape[0]
            and seed_draws.shape[1] == delta.shape[0] and source_draws.shape[1] == len(unique))
    membership = [np.flatnonzero(np.array(sources) == value) for value in unique]
    counts = np.array([len(indices) for indices in membership], dtype=float)
    sums = np.stack([delta[:, indices, :].sum(axis=1) for indices in membership], axis=1)
    sampled = sums[seed_draws[:, :, None], source_draws[:, None, :], :]
    values = sampled.sum(axis=(1, 2)) / (seed_draws.shape[1] * counts[source_draws].sum(axis=1))[:, None]
    return np.quantile(values, [.025, .975], axis=0)


def synthetic_checks():
    rng = np.random.default_rng(20261105)
    train = rng.integers(0, 3, size=(10000, 3))
    source = rng.integers(0, 3, size=(10000, 3))
    sources = ["a", "a", "b", "c"]
    constant = crossed_interval(np.full((3, 4, 2), 2.0), sources, train, source)
    assert np.all(constant == 2.0), "Constant deltas must have a point interval"
    flip = np.repeat(np.array([-1.0, 0.0, 1.0])[:, None, None], 4, axis=1)
    interval = crossed_interval(flip, sources, train, source)
    assert interval[0, 0] < 0 < interval[1, 0], "Sign-flipping seeds must span zero"
    assert float(flip.mean()) == 0.0 and float(np.std(flip.mean(axis=1).ravel(), ddof=1)) == 1.0
    # Conditioning on all three seed instances averages this synthetic delta
    # to zero in every clip; crossing the training-seed axis restores its spread.
    assert np.all(flip.mean(axis=0) == 0.0)
    return {"constant_delta_ci95": constant[:, 0].tolist(),
            "sign_flip_training_seed_deltas": [-1.0, 0.0, 1.0],
            "sign_flip_crossed_ci95": interval[:, 0].tolist(),
            "sign_flip_conditional_over_sources_ci95": [0.0, 0.0],
            "sample_std_ddof1_check": 1.0, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/home/hugo/Projects/VSR"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    loader = loader_module()
    loader.require(args.replicates >= 100, "At least 100 bootstrap replicates required")
    paths = [args.output_dir / f"seed_uncertainty.{suffix}" for suffix in ("json", "md")]
    loader.require(not any(path.exists() for path in paths), "Refusing to overwrite a seed-uncertainty supplement")
    checks = synthetic_checks()
    inputs = loader.Inputs()
    groups, audit = loader.load_validation(args.source_root, inputs)
    conditional_path = args.output_dir / "statistics.json"
    conditional = inputs.read(conditional_path)
    expected = {"HQ_val32/NoRef/all_frames", *{f"HQ_val32/HRI/ref{ref}/{mode}"
                for ref in (0, 4) for mode in ("hri_on", "hri_off")}}
    loader.require(set(groups) == expected and conditional["source_root"] == str(args.source_root),
                   "Require all five predeclared validation contexts and the same source root")
    first = groups["HQ_val32/NoRef/all_frames"]["Base"][loader.SEEDS[0]]
    records = sorted(first)
    sources = [first[record]["source_id"] for record in records]
    loader.require(len(records) == len(set(sources)) == 32, "Require fixed val32/32 sources")
    rng = np.random.default_rng(args.seed)
    seed_draws = rng.integers(0, 3, size=(args.replicates, 3))
    source_draws = rng.integers(0, 32, size=(args.replicates, 32))
    reports = {}
    for context, variants in sorted(groups.items()):
        loader.require(set(variants) == {"Base", "Align"}, "Only paired Base/Align allowed")
        arrays = {}
        for variant in ("Base", "Align"):
            loader.require(sorted(variants[variant]) == list(loader.SEEDS), "Require all three matched training seeds")
            arrays[variant] = []
            for seed in loader.SEEDS:
                rows = variants[variant][seed]
                loader.require(set(rows) == set(records) and all(rows[r]["source_id"] == first[r]["source_id"] for r in records),
                               "Crossed contexts differ in paired clip/source identities")
                arrays[variant].append([[rows[r]["metrics"][metric] for metric in loader.METRICS] for r in records])
            arrays[variant] = np.array(arrays[variant], dtype=float)
        delta = arrays["Align"] - arrays["Base"]
        per_seed = delta.mean(axis=1)
        interval = crossed_interval(delta, sources, seed_draws, source_draws)
        metrics = {}
        old = conditional["contexts"][context]["comparisons_minus_Base"]["Align"]
        for i, metric in enumerate(loader.METRICS):
            mean, std = float(per_seed[:, i].mean()), float(per_seed[:, i].std(ddof=1))
            loader.assert_close(mean, old["metrics"][metric]["mean"], "Mean differs from original conditional report")
            loader.assert_close(std, old["metrics"][metric]["sample_std_ddof1"], "Training SD differs from original report")
            for j, seed in enumerate(loader.SEEDS):
                loader.assert_close(float(per_seed[j, i]), old["per_training_seed"][str(seed)][metric]["delta"],
                                    "A training-seed paired delta differs from the original report")
            metrics[metric] = {
                "delta": "Align minus Base", "favorable_direction": "positive" if i < 2 else "negative",
                "per_training_seed_delta": {str(seed): float(per_seed[j, i]) for j, seed in enumerate(loader.SEEDS)},
                "mean_delta": mean, "sample_std_ddof1": std,
                "crossed_training_seed_and_source_ci95": interval[:, i].tolist(),
                "original_conditional_source_cluster_ci95": old["metrics"][metric]["ci95_source_cluster_fixed_training_seeds"],
                "positive_training_seed_count": int((per_seed[:, i] > 0).sum()),
                "negative_training_seed_count": int((per_seed[:, i] < 0).sum()),
                "zero_training_seed_count": int((per_seed[:, i] == 0).sum()),
                "observed_sign_flip_between_training_seeds": bool(per_seed[:, i].min() < 0 < per_seed[:, i].max()),
            }
        reports[context] = {"primary_reference_context": context in ("HQ_val32/NoRef/all_frames", "HQ_val32/HRI/ref0/hri_on"),
                            "metrics": metrics}
    result = {"schema_version": 1, "date": "2026-11-05", "status": "exploratory_internal_validation_supplement",
              "source_root": str(args.source_root), "scope": "Base/Align only; val32 only; all five predeclared contexts",
              "training_seeds": list(loader.SEEDS), "training_seed_count": 3,
              "clip_count": 32, "source_group_count": 32, "sampling_seeds_averaged_within_clip": list(loader.SAMPLE_SEEDS),
              "bootstrap": {"replicates": args.replicates, "rng_seed": args.seed,
                  "method": "crossed paired training-seed/source-group percentile bootstrap",
                  "draw_order": "draw all training-seed indices, then all source-group indices with one NumPy default_rng",
                  "per_replicate": "draw 3 training seeds with replacement and 32 source groups with replacement; use their Cartesian product, keep Base/Align and all metrics/contexts paired",
                  "source_group_weighting": "retain every clip in each drawn source; equal clip weights, then equal drawn training-seed weights",
                  "confidence": 0.95, "numpy_version": np.__version__, "p_values_computed": False,
                  "multiplicity_correction": False, "Stage5_significance_decision": False,
                  "limitation": "Only three observed training-seed pairs; exploratory descriptive intervals are coarse and do not reliably characterize a training-seed population."},
              "original_conditional_interval": {"path": str(conditional_path),
                  "definition": "Original statistics report resampled source groups after averaging each clip's three fixed training-seed deltas; it excludes training-seed uncertainty.",
                  "bootstrap": conditional["bootstrap"]},
              "audit": {"validation": audit, "json_file_count": len(inputs.hashes),
                  "sorted_json_path_and_sha256_map_fingerprint": loader.digest(inputs.hashes),
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "loader_sha256": hashlib.sha256(Path(loader.__file__).read_bytes()).hexdigest(),
                  "checkpoint_rgb_png_video_payloads_read": False, "torch_imported": "torch" in sys.modules,
                  "frozen_test_records_read": False, "synthetic_checks": checks},
              "contexts": reports}
    loader.require(not result["audit"]["torch_imported"], "Torch must not be imported")
    text = ["# Training-seed uncertainty supplement (2026-11-05)", "",
            "Exploratory, descriptive val32 analysis only. This does not close T0.4, test Stage 5 significance, or provide a p-value.", "",
            "Three matched Base/Align training-seed pairs (20261102/03/04), the same 32 validation clips/32 source groups, and sampling seeds 42/43 averaged within each clip. NoRef correct/all-frames and HRI ref0/ref4 × on/off keep their original measurement pools and remain separate. Ref0/on is HRI's predeclared primary reference context. No frozen test211 or external results enter this supplement.", "",
            f"Each of {args.replicates:,} replicates independently draws three training seeds and 32 source groups with replacement (NumPy default_rng seed {args.seed}); it evaluates their Cartesian product. All metrics, models and contexts use the same paired draws. The new 95% percentile intervals include resampling of the observed training-seed pairs and source groups. The previous conditional intervals resample sources after averaging the three fixed seeds. The previous files remain unchanged.", "",
            "There are only three training-seed pairs. Their empirical distribution is sparse, so the new intervals are exploratory and coarse; neither crossing nor excluding zero is a confirmatory benefit or equivalence decision. Mean ± SD uses sample SD across the three per-training-seed paired deltas (ddof=1). Positive Δ favors Align for PSNR/SSIM; negative Δ favors Align for LPIPS/tLP100/tOF. No multiplicity correction or p-values are used.", ""]
    for context, report in reports.items():
        text += [f"## {context}", "", "| Metric | seed02 Δ | seed03 Δ | seed04 Δ | Mean Δ ± SD | Crossed 95% CI | Previous conditional 95% CI |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for metric, value in report["metrics"].items():
            seed_values = " | ".join(f"{value['per_training_seed_delta'][str(seed)]:+.6f}" for seed in loader.SEEDS)
            lo, hi = value["crossed_training_seed_and_source_ci95"]
            old_lo, old_hi = value["original_conditional_source_cluster_ci95"]
            text.append(f"| {metric} | {seed_values} | {value['mean_delta']:+.6f} ± {value['sample_std_ddof1']:.6f} | [{lo:+.6f}, {hi:+.6f}] | [{old_lo:+.6f}, {old_hi:+.6f}] |")
        text += [""]
    text += ["## Interpretation", "",
             "NoRef Align's mean PSNR/SSIM improvement changes sign across training seeds; its LPIPS and tOF deltas are unfavorable in all three observed seeds. HRI ref0/on changes sign across seeds for every reported metric. The observed seed spread and mixed quality tradeoffs support keeping the current description as 'no stable overall auxiliary-loss benefit established'. They do not prove that the true effect is zero or that every metric worsens. Relation/Both were not included because their training-seed uncertainty cannot be estimated from one run.", "",
             "The loader checked the existing validation JSON identities, windows, sampling seeds, reference exclusions and stored digest bindings. Underlying checkpoints, targets, RGB tensors, source videos and PNGs were not read or rehashed. Constant-delta and sign-flipping-seed synthetic cases passed; the latter has a zero-width conditional interval but a crossed interval spanning both signs.", ""]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    paths[1].write_text("\n".join(text))
    print(json.dumps({"contexts": len(reports), "json_inputs": len(inputs.hashes), "outputs": [str(path) for path in paths]}))


if __name__ == "__main__":
    main()
