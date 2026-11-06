"""Paired comparisons with video means and source-cluster uncertainty.

Sampling seeds are repeated measurements of one video, never independent
videos. Source clusters keep adjacent clips from one upload together. These
intervals describe the measured videos, not uncertainty over training runs.
"""
from __future__ import annotations

import math
import numpy as np


def _observations(rows):
    result = {}
    for row in rows:
        record, source, seed = row["record_id"], row["source_id"], row["seed"]
        if not isinstance(record, str) or not record or not isinstance(source, str) or not source:
            raise ValueError("Every video requires an explicit record_id and source_id")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("Sampling seeds must be nonnegative integers")
        key = record, seed
        if key in result:
            raise ValueError(f"Duplicate video/seed: {key}")
        values = row["metrics"]
        if not values or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                             for v in values.values()):
            raise ValueError("Metric values must be finite numbers")
        result[key] = {"source_id": source, "metrics": {k: float(v) for k, v in values.items()}}
    if not result:
        raise ValueError("No observations")
    metrics = set(next(iter(result.values()))["metrics"])
    records = {record for record, _ in result}
    first = next(iter(records))
    seeds = {seed for record, seed in result if record == first}
    for record in records:
        if {seed for r, seed in result if r == record} != seeds:
            raise ValueError("Every video must include the same complete sampling seed set")
        if len({row["source_id"] for (r, _), row in result.items() if r == record}) != 1:
            raise ValueError("Source identity changed across sampling seeds")
    if any(set(row["metrics"]) != metrics for row in result.values()):
        raise ValueError("Metric keys differ between observations")
    return result


def paired_summary(groups, reference, *, replicates=10000, seed=20261102):
    """Compare matched experiments; delta always means candidate minus reference.

    The primary mean gives every video equal weight. For its percentile 95%
    interval, resample source IDs, retaining all videos of each drawn source.
    Equal-source means are a disclosed supplementary estimand. Confidence
    intervals are pointwise descriptive intervals, without multiplicity
    correction, training-repeat uncertainty, or a significance claim.
    """
    if reference not in groups or not groups:
        raise ValueError("Reference experiment missing")
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    observed = {label: _observations(rows) for label, rows in groups.items()}
    base = observed[reference]
    keys = set(base)
    metric_keys = sorted(next(iter(base.values()))["metrics"])
    for label, rows in observed.items():
        if set(rows) != keys:
            raise ValueError(f"{label}: video/seed sets differ; paired comparisons cannot drop observations")
        for key in keys:
            if rows[key]["source_id"] != base[key]["source_id"]:
                raise ValueError(f"{label}: source identities differ")
            if sorted(rows[key]["metrics"]) != metric_keys:
                raise ValueError(f"{label}: metric sets differ")
    records = sorted({record for record, _ in keys})
    seeds = sorted({s for _, s in keys})
    sources = [base[(record, seeds[0])]["source_id"] for record in records]
    unique_sources = sorted(set(sources))
    membership = [np.array([i for i, value in enumerate(sources) if value == source], dtype=int)
                  for source in unique_sources]
    source_counts = np.array([len(indices) for indices in membership], dtype=float)
    rng = np.random.default_rng(seed)
    # Common cluster draws preserve pairing and make comparisons order invariant.
    draws = rng.integers(0, len(membership), size=(replicates, len(membership)))
    result = {"protocol": {
        "mean": "all sampling seeds per video, then equal video weights",
        "interval": "paired source-cluster percentile bootstrap, 95%, pointwise descriptive",
        "supplementary": "clips averaged within source, then equal source weights",
        "delta": "candidate minus reference; direction depends on metric",
        "bootstrap_replicates": replicates, "bootstrap_seed": seed,
        "training_repeat_uncertainty_included": False, "multiplicity_correction": False,
        "single_source_interval_available": False,
        "limitation": "sampling repeats and source bootstrap do not establish reproducibility across training seeds"},
        "reference": reference, "video_count": len(records), "source_count": len(membership),
        "sampling_seeds": seeds, "experiments": {}, "comparisons": {}}
    arrays = {}
    for label, rows in observed.items():
        array = np.array([[np.mean([rows[(record, s)]["metrics"][metric] for s in seeds])
                           for metric in metric_keys] for record in records])
        source_array = np.stack([array[indices].mean(axis=0) for indices in membership])
        arrays[label] = array
        result["experiments"][label] = {
            "video_macro": dict(zip(metric_keys, array.mean(axis=0).tolist())),
            "source_macro": dict(zip(metric_keys, source_array.mean(axis=0).tolist())),
            "per_video": [{"record_id": record, "source_id": source,
                           "metrics": dict(zip(metric_keys, values.tolist()))}
                          for record, source, values in zip(records, sources, array)]}
    for label, array in arrays.items():
        if label == reference:
            continue
        delta = array - arrays[reference]
        source_sums = np.stack([delta[indices].sum(axis=0) for indices in membership])
        source_means = source_sums / source_counts[:, None]
        if len(membership) > 1:
            samples_video = source_sums[draws].sum(axis=1) / source_counts[draws].sum(axis=1)[:, None]
            samples_source = source_means[draws].mean(axis=1)
            video_interval = np.quantile(samples_video, [.025, .975], axis=0)
            source_interval = np.quantile(samples_source, [.025, .975], axis=0)
        result["comparisons"][label] = {
            metric: {"delta_video_macro": float(delta[:, m].mean()),
                     "ci95_source_cluster_video_macro": video_interval[:, m].tolist() if len(membership) > 1 else None,
                     "delta_source_macro": float(source_means[:, m].mean()),
                     "ci95_source_macro": source_interval[:, m].tolist() if len(membership) > 1 else None,
                     "video_deltas": dict(zip(records, delta[:, m].tolist())),
                     "positive_videos": int((delta[:, m] > 0).sum()),
                     "negative_videos": int((delta[:, m] < 0).sum()),
                     "zero_videos": int((delta[:, m] == 0).sum())}
            for m, metric in enumerate(metric_keys)}
    return result
