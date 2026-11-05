#!/usr/bin/env python3
"""Summarize existing final-run JSON/JSONL losses, without loading checkpoints.

Only Python's standard library is used. Historical inputs are read-only. Existing
output files are never overwritten; use a new dated output directory for a rerun.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import statistics


VARIANTS = {"Base", "Align", "Relation", "Both"}
LOSS_FIELDS = {
    "flow": ("fm", "flow", "loss_fm"),
    "alignment": ("alignment", "align", "loss_align"),
    "relation": ("relation", "loss_relation"),
}


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def read_rows(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    steps = [row.get("step") for row in rows]
    if not rows or any(type(step) is not int or step < 1 for step in steps):
        raise ValueError(f"Missing/invalid step values: {path}")
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError(f"Duplicate or unordered steps: {path}")
    return rows, hashlib.sha256(raw).hexdigest()


def window_summary(points):
    values = [value for _, value in points]
    return {
        "sample_count": len(points), "step_first": points[0][0],
        "step_last": points[-1][0], "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "first_value": values[0], "last_value": values[-1],
    }


def summarize_loss(rows, aliases, active, window_records):
    points, fields = [], Counter()
    for row in rows:
        present = [name for name in aliases if name in row]
        if not present:
            continue
        value = row[present[0]]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid loss at step {row['step']}: {present[0]}")
        if any(row[name] != value for name in present[1:]):
            raise ValueError(f"Conflicting loss aliases at step {row['step']}: {present}")
        fields[present[0]] += 1
        points.append((row["step"], value))
    base = {"active_objective": active, "source_fields": dict(fields),
            "sample_count": len(points), "missing_record_count": len(rows) - len(points)}
    if len(points) < 2:
        return {**base, "status": "insufficient_observations"}
    # Use observations, not an invented dense update series. Keep windows disjoint.
    n = min(window_records, len(points) // 2)
    early, late = window_summary(points[:n]), window_summary(points[-n:])
    delta = late["mean"] - early["mean"]
    relative = delta / abs(early["mean"]) * 100 if early["mean"] else None
    bins = [window_summary(points[i:i + window_records]) for i in range(0, len(points), window_records)]
    return {**base, "status": "observed" if active else "inactive_logged_placeholder",
            "window_records_actual": n, "first": window_summary(points[:1]),
            "last": window_summary(points[-1:]), "overall": window_summary(points),
            "early": early, "late": late, "late_minus_early_mean": delta,
            "relative_change_percent": relative,
            "early_to_late_decrease": delta < 0 if active else None,
            "all_logged_values_zero": all(value == 0 for _, value in points),
            "binned_observations": bins}


def analyze_variant(folder, runs_root, window_records, expected_updates):
    summary_path, init_path, log_path = (folder / name for name in
                                       ("summary.json", "initialization.json", "training.jsonl"))
    summary, summary_hash = read_json(summary_path)
    init, init_hash = read_json(init_path)
    rows, log_hash = read_rows(log_path)
    variant = folder.name
    if summary.get("variant") != variant or summary.get("updates") != expected_updates:
        raise ValueError(f"Unexpected final summary variant/updates: {summary_path}")
    if summary.get("status") != "completed":
        raise ValueError(f"Final checkpoint has non-completed summary: {summary_path}")
    if rows[-1]["step"] != expected_updates:
        raise ValueError(f"Final summary and last logged update differ: {log_path}")
    seed = init.get("training_seed", init.get("seed"))
    if type(seed) is not int:
        raise ValueError(f"Missing training seed: {init_path}")
    family = "HRI" if init.get("reference_enabled", summary.get("reference_trained", False)) else "NoRef"
    steps = [row["step"] for row in rows]
    losses = {name: summarize_loss(rows, aliases,
              name == "flow" or (name == "alignment" and variant in {"Align", "Both"})
              or (name == "relation" and variant in {"Relation", "Both"}), window_records)
              for name, aliases in LOSS_FIELDS.items()}
    if any(loss["active_objective"] and loss["status"] != "observed" for loss in losses.values()):
        raise ValueError(f"Active objective has insufficient observations: {log_path}")
    inputs = {str(path): digest for path, digest in
              ((summary_path, summary_hash), (init_path, init_hash), (log_path, log_hash))}
    for path, expected_hash in inputs.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Historical input changed during analysis: {path}")
    return {
        "family": family, "variant": variant, "training_seed": seed,
        "run": folder.parent.name, "variant_relative_path": str(folder.relative_to(runs_root)),
        "checkpoint_path": str(folder / f"checkpoint_{expected_updates:06d}.pt"),
        "checkpoint_read": False, "summary_status": summary["status"],
        "summary_updates": summary["updates"], "log_records": len(rows),
        "log_step_first": steps[0], "log_step_last": steps[-1],
        "log_covers_every_update": steps == list(range(1, expected_updates + 1)),
        "logged_step_gaps": [[a, b] for a, b in zip(steps, steps[1:]) if b != a + 1],
        "input_sha256": inputs, "losses": losses,
    }


def aggregate_seeds(records):
    groups = {}
    for record in records:
        for name, loss in record["losses"].items():
            if loss["active_objective"]:
                groups.setdefault((record["family"], record["variant"], name), []).append((record, loss))
    results = []
    for (family, variant, name), members in sorted(groups.items()):
        fields = {}
        for key, getter in {
            "early_mean": lambda loss: loss["early"]["mean"],
            "late_mean": lambda loss: loss["late"]["mean"],
            "relative_change_percent": lambda loss: loss["relative_change_percent"],
        }.items():
            values = [getter(loss) for _, loss in members]
            fields[key] = {"mean": statistics.fmean(values),
                           "sample_std": statistics.stdev(values) if len(values) > 1 else None}
        results.append({"family": family, "variant": variant, "loss": name,
                        "training_seeds": [record["training_seed"] for record, _ in members],
                        "seed_count": len(members),
                        "interpretation": "descriptive seed statistics" if len(members) > 1 else "single seed, indicative",
                        **fields})
    return results


def format_number(value):
    return "n/a" if value is None else f"{value:.6g}"


def markdown(report):
    records, n = report["records"], report["window_records_requested"]
    dense = sum(record["log_covers_every_update"] for record in records)
    ranges = sorted({(loss["early"]["step_first"], loss["early"]["step_last"],
                      loss["late"]["step_first"], loss["late"]["step_last"])
                     for record in records for loss in record["losses"].values()
                     if loss["active_objective"]})
    range_text = "; ".join(f"early {a}–{b}, late {c}–{d}" for a, b, c, d in ranges)
    placeholders_zero = all(loss.get("all_logged_values_zero", False)
                            for record in records for loss in record["losses"].values()
                            if not loss["active_objective"])
    lines = ["# Existing training loss diagnostics", "", f"Analysis date: {report['analysis_date']}.", "",
             f"Read-only JSON/JSONL analysis of {len(records)} final-checkpoint runs. The comparison uses the first and last **{n} recorded loss observations**, fixed before calculating the results. JSON records the actual count and step range for each window, source hashes, final values, inactive placeholders and binned observations. No checkpoint payload, GPU, model forward pass or optimizer was used.", "",
             f"Coverage: {dense}/{len(records)} selected logs contain every update from 1 through {report['expected_updates']}. Observed window step ranges: {range_text}. This coverage was checked from the logs, not inferred from checkpoint names. Inactive auxiliary fields are excluded from decrease counts; all their logged values are zero: {placeholders_zero}.", "",
             "## Log coverage", "", "| Family | Variant | Training seed | Records | Logged steps | Run |",
             "| --- | --- | --- | ---: | --- | --- |"]
    for record in records:
        lines.append(f"| {record['family']} | {record['variant']} | {record['training_seed']} | {record['log_records']} | {record['log_step_first']}–{record['log_step_last']} | `{record['run']}` |")
    for name in LOSS_FIELDS:
        lines += ["", f"## {name.capitalize()} loss", "",
                  "Raw objective values; auxiliary coefficients are not applied in this table.", "",
                  "| Family | Variant | Training seed | Early mean | Late mean | Change % | Final observed value | Window samples (early / late) |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for record in records:
            loss = record["losses"][name]
            if not loss["active_objective"]:
                continue
            lines.append(f"| {record['family']} | {record['variant']} | {record['training_seed']} | {format_number(loss['early']['mean'])} | {format_number(loss['late']['mean'])} | {loss['relative_change_percent']:+.2f}% | {format_number(loss['last']['last_value'])} | {loss['early']['sample_count']} / {loss['late']['sample_count']} |")
    lines += ["", "## Seed summaries", "", "Sample standard deviation across training seeds; these summarize loss trajectories, not evaluation quality or independent per-update replicates.", "",
              "| Family | Variant | Loss | Seeds | Early mean ± SD | Late mean ± SD | Change % mean ± SD |",
              "| --- | --- | --- | ---: | --- | --- | --- |"]
    for group in report["seed_summaries"]:
        cells = [f"{format_number(group[key]['mean'])} ± {format_number(group[key]['sample_std'])}"
                 for key in ("early_mean", "late_mean", "relative_change_percent")]
        seed_label = str(group["seed_count"]) if group["seed_count"] > 1 else "1 (indicative)"
        lines.append(f"| {group['family']} | {group['variant']} | {group['loss']} | {seed_label} | " + " | ".join(cells) + " |")
    lines += ["", "## Exclusions and interpretation", ""]
    for excluded in report["excluded_logs_without_final_checkpoint"]:
        lines.append(f"- Excluded `{excluded['log_path']}`: {excluded['records']} recorded updates, steps {excluded['step_first']}–{excluded['step_last']}; no matching final checkpoint. Excluded attempts are not concatenated with the selected completed logs.")
    lines += ["", report["conclusion"], "",
              "The records sample different training clips, noise and timesteps; these are descriptive first/last-window comparisons, not paired held-out tests, statistical significance tests, or evidence of monotonic loss decrease. The final single observation can differ from the late-window mean. Lower alignment/relation training loss does not establish better restoration, temporal consistency, or gains over the augmentation-only C1 control. No feature-similarity diagnostic was run.", "",
              "Reproduce into a fresh output directory (the script refuses to overwrite results):", "", "```bash",
              f"python scripts/analyze_existing_losses.py --runs-root {report['runs_root']} --window-records {n} --analysis-date {report['analysis_date']} --output-dir docs/stage0_analysis_20261105_rerun", "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("/home/hugo/Projects/VSR/runs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-records", type=int, default=512)
    parser.add_argument("--updates", type=int, default=8192)
    parser.add_argument("--expected-checkpoints", type=int, default=16)
    parser.add_argument("--analysis-date", default=date.today().isoformat())
    args = parser.parse_args()
    if args.window_records < 1 or args.updates < 2:
        parser.error("window-records must be positive and updates at least two")
    source, output = args.runs_root.resolve(), args.output_dir.resolve()
    if output.is_relative_to(source.parent):
        raise ValueError("Output must be outside the historical source project")
    targets = [output / name for name in ("loss_diagnostics.json", "loss_diagnostics.md")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite existing loss diagnostics; choose a new directory")
    folders = sorted(p.parent for p in source.glob(f"formal*/*/checkpoint_{args.updates:06d}.pt")
                     if p.parent.name in VARIANTS)
    if len(folders) != args.expected_checkpoints:
        raise ValueError(f"Expected {args.expected_checkpoints} final checkpoint paths, found {len(folders)}")
    records = [analyze_variant(folder, source, args.window_records, args.updates) for folder in folders]
    records.sort(key=lambda r: (r["family"], ("Base", "Align", "Relation", "Both").index(r["variant"]), r["training_seed"]))
    identities = [(r["family"], r["variant"], r["training_seed"]) for r in records]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate family/variant/training-seed identities")
    excluded = []
    for path in sorted(source.glob("formal*/*/training.jsonl")):
        if path.parent.name in VARIANTS and path.parent not in folders:
            rows, digest = read_rows(path)
            excluded.append({"log_path": str(path), "records": len(rows),
                             "step_first": rows[0]["step"], "step_last": rows[-1]["step"],
                             "sha256": digest, "reason": "no corresponding final checkpoint"})
    counts = {name: {"decreased": sum(r["losses"][name].get("early_to_late_decrease") is True for r in records),
                     "active_runs": sum(r["losses"][name]["active_objective"] for r in records)} for name in LOSS_FIELDS}
    conclusion = (f"The first/last-window mean decreased for {counts['flow']['decreased']}/{counts['flow']['active_runs']} flow objectives, "
                  f"{counts['alignment']['decreased']}/{counts['alignment']['active_runs']} active alignment objectives, and "
                  f"{counts['relation']['decreased']}/{counts['relation']['active_runs']} active relation objectives. "
                  "The auxiliary objectives therefore show optimization progress in these logged windows; this does not establish any quality gain.")
    report = {"schema_version": 1, "analysis_date": args.analysis_date, "runs_root": str(source),
              "expected_updates": args.updates,
              "scope": "descriptive training-loss diagnostics, not evaluation-quality inference",
              "window_records_requested": args.window_records,
              "window_policy": "first/last min(requested, available//2) observed values, disjoint; never interpolate absent updates",
              "historical_input_hashes_verified_unchanged_during_read": True,
              "checkpoint_payloads_read": False, "gpu_used": False,
              "records": records, "seed_summaries": aggregate_seeds(records),
              "decrease_counts": counts, "excluded_logs_without_final_checkpoint": excluded,
              "conclusion": conclusion}
    output.mkdir(parents=True, exist_ok=True)
    with targets[0].open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with targets[1].open("x") as stream:
        stream.write(markdown(report))
    print(json.dumps({"output_dir": str(output), "runs": len(records), "decrease_counts": counts}))


if __name__ == "__main__":
    main()
