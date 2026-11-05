#!/usr/bin/env python3
"""Read-only coverage/cost plan for the proposed 16-checkpoint external cohort.

Reads JSON and checks marker/metrics-path presence only. Does not hash or open
checkpoint/PNG/video payloads, load tensor libraries, launch a GPU, or queue work.
Historical completion markers are not new strict acceptance decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import statistics
import sys


def need(ok, message):
    if not ok:
        raise ValueError(message)


class JSONInputs:
    def __init__(self):
        self.paths = set()

    def read(self, path):
        path = Path(path)
        need(path.suffix == ".json", "Only JSON input reads are permitted")
        self.paths.add(str(path))
        return json.loads(path.read_text())


def build(args):
    inputs = JSONInputs()
    assets = inputs.read(args.assets)
    outer_path = args.old_root / "runs/external_paired_frozen_20261103T125240.867596Z/execution_policy.json"
    child_root = outer_path.parent / "external"
    outer = inputs.read(outer_path)
    child = inputs.read(child_root / "protocol.json")
    native = outer["native_metadata"]
    scope = outer["root_protocol"]["external_native_prefix"]
    need(native["phase"] == "paired" and native["sequence_count"] == 44
         and len(native["records"]) == 44 and native["sum_tiles"] == 1228,
         "Require exact frozen 44-sequence native capacity metadata")
    need(child["frames"] == scope["frames"] == 17 and child["start"] == scope["start"] == 0
         and child["steps"] == scope["steps"] == 20 and child["seeds"] == scope["sampling_seeds"] == [42, 43]
         and child["tile_hw"] == scope["tile_hw"] == [256, 448] and child["overlap"] == scope["overlap"] == 64
         and scope["cfg"] == 1, "Historical 17-frame prefix protocol changed")
    geometries = {(x["dataset"], x["sequence_id"]): x for x in native["records"]}
    expected_sequences = {("REDS4", x) for x in ("000", "011", "015", "020")} | {("YouHQ40", f"{i:03d}") for i in range(40)}
    need(set(geometries) == expected_sequences, "Duplicate/missing/substituted external sequence")
    checkpoints = {}
    for cp in assets["checkpoints"]:
        family, variant, seed = cp["family"], cp["variant"], cp["training_seed"]
        need(family in ("NoRef", "HRI") and variant in ("Base", "Align", "Relation", "Both")
             and cp["updates"] == 8192, "Unexpected checkpoint role")
        path = Path(cp["path"])
        need(path.is_absolute() and path.name == "checkpoint_008192.pt" and path.parent.name == variant
             and str(path.parent.parent) == cp["run"], "Checkpoint path/run/variant metadata disagrees")
        config = inputs.read(cp["run_config"])
        need(config["training"].get("seed", config["seed"]) == seed and config["seed"] == cp["data_seed"] == 20261102,
             "Training seed differs from originating JSON config")
        label = f"{family}_{variant}_seed{seed}_step8192"
        need(label not in checkpoints, "Duplicate checkpoint identity")
        checkpoints[label] = {"checkpoint_label": label, "family": family, "variant": variant,
            "training_seed": seed, "data_seed": cp["data_seed"], "updates": 8192,
            "checkpoint_path": cp["path"], "training_run": cp["run"], "origin_config": cp["run_config"]}
    need(len(checkpoints) == 16, "Require exactly 16 checkpoint metadata entries")
    for family in ("NoRef", "HRI"):
        counts = Counter(x["variant"] for x in checkpoints.values() if x["family"] == family)
        need(counts == {"Base": 3, "Align": 3, "Relation": 1, "Both": 1}, "Incomplete family checkpoint cohort")
    need(len(child["checkpoints"]) == 8, "Historical worker must remain identified as its original eight-checkpoint cohort")
    for label, spec in child["checkpoints"].items():
        need(label in checkpoints and checkpoints[label]["checkpoint_path"] == spec["path"]
             and checkpoints[label]["training_seed"] == 20261102, "Historical checkpoint identity differs from 16-cohort mapping")
    markers = {}
    for path in sorted((child_root / "results").glob("*/*/*/complete.json")):
        label, dataset, sequence = path.relative_to(child_root / "results").parts[:3]
        need(label in child["checkpoints"] and (dataset, sequence) in geometries,
             "Historical marker outside its declared checkpoint/sequence cohort")
        marker = inputs.read(path)
        need(all(re.fullmatch("[0-9a-f]{64}", marker.get(k, "")) for k in ("contract_sha256", "metrics_sha256")),
             "Malformed historical completion marker")
        markers[label, dataset, sequence] = {"path": str(path), "declared_contract_sha256": marker["contract_sha256"],
            "declared_metrics_sha256": marker["metrics_sha256"], "metrics_json_exists": path.with_name("metrics.json").is_file(),
            "acceptance_status": "marker_only_needs_separate_strict_review", "hashes_recomputed": False}
    need(len(markers) == 116, "Historical marker count differs from the recorded 116-row pause snapshot")
    repro = inputs.read(args.reproduction_summary)
    base_label = "NoRef_Base_seed20261102_step8192"
    need(repro["status"] == "passed" and repro["dataset"] == "REDS4" and repro["sequence_id"] == "000"
         and repro["seeds"] == [42, 43] and repro["PNG_files_byte_identical"] == 34
         and repro["all_pool_scalar_metrics_passed"] is True
         and repro["checkpoint"] == checkpoints[base_label]["checkpoint_path"],
         "Cost calibration requires the separately passed REDS4/000 Base reproduction summary")
    roles = repro["policy_role_counters"]
    need(roles["restores"] == 2 and roles["original_delegate"] == roles["events"] == 32,
         "Calibration must be two 16-tile restores")
    seconds = [repro["timings"][f"seed_{s}_restore"]["wall_seconds"] for s in (42, 43)]
    rates = [value / 16 for value in seconds]
    rate = statistics.mean(rates)
    rows = []
    for label, cp in sorted(checkpoints.items()):
        for (dataset, sequence), geometry in sorted(geometries.items()):
            restores = 2 if cp["family"] == "NoRef" else 4
            tile_calls = restores * geometry["tile_count"]
            marker = markers.get((label, dataset, sequence))
            pending = 0 if marker else restores
            pending_tiles = 0 if marker else tile_calls
            rows.append({**cp, "dataset": dataset, "sequence_id": sequence,
                "prefix_frames": 17, "temporal_stride": 1, "start": 0,
                "native_LR_hw": geometry["LR_hw"], "native_HR_hw": geometry["HR_hw"],
                "padded_HR_hw": geometry["padded_HR_hw"], "tile_count_per_restore": geometry["tile_count"],
                "restores_required": restores, "original_delegate_tile_calls_required": tile_calls,
                "T1_reference_encodes_required": 2 if cp["family"] == "HRI" else 0,
                "historical_complete_marker": marker,
                "historical_marker_present": marker is not None,
                "new_16cohort_acceptance": "not_accepted_new_protocol_and_strict_review_required",
                "optimistic_pending_restores_after_marker_review": pending,
                "optimistic_pending_tile_calls_after_marker_review": pending_tiles,
                "optimistic_restore_component_hours_at_NoRef_rate": pending_tiles * rate / 3600,
                "calibration_row_separately_reproduced": label == base_label and dataset == "REDS4" and sequence == "000"})
    need(len(rows) == 704, "Expected 16 checkpoints × 44 sequences")
    totals = {"model_sequence_rows": len(rows), "required_restores": sum(x["restores_required"] for x in rows),
        "required_original_delegate_tile_calls": sum(x["original_delegate_tile_calls_required"] for x in rows),
        "required_T1_reference_encodes": sum(x["T1_reference_encodes_required"] for x in rows),
        "historical_marker_rows": len(markers),
        "historical_marker_restore_equivalents_not_newly_accepted": sum(x["restores_required"] for x in rows if x["historical_marker_present"]),
        "historical_marker_tile_calls_not_newly_accepted": sum(x["original_delegate_tile_calls_required"] for x in rows if x["historical_marker_present"]),
        "new_16cohort_rows_accepted": 0,
        "optimistic_pending_rows_after_marker_review": sum(not x["historical_marker_present"] for x in rows),
        "optimistic_minimum_pending_restores_after_marker_review": sum(x["optimistic_pending_restores_after_marker_review"] for x in rows),
        "optimistic_minimum_pending_tile_calls_after_marker_review": sum(x["optimistic_pending_tile_calls_after_marker_review"] for x in rows)}
    need((totals["required_restores"], totals["historical_marker_restore_equivalents_not_newly_accepted"],
          totals["optimistic_minimum_pending_restores_after_marker_review"]) == (2112, 232, 1880), "Restore accounting mismatch")
    need(totals["required_original_delegate_tile_calls"] == 58944 and totals["optimistic_minimum_pending_tile_calls_after_marker_review"] == 52488,
         "Native tile accounting mismatch")
    def estimate(tile_calls):
        return {"mean_rate_restore_component_hours": tile_calls * rate / 3600,
                "two_calibration_rates_hours_range_not_forecast_CI": [tile_calls * min(rates)/3600, tile_calls * max(rates)/3600]}
    by_family = {}
    for family in ("NoRef", "HRI"):
        family_rows = [x for x in rows if x["family"] == family]
        pending_tiles = sum(x["optimistic_pending_tile_calls_after_marker_review"] for x in family_rows)
        by_family[family] = {"rows": len(family_rows), "required_restores": sum(x["restores_required"] for x in family_rows),
            "historical_marker_rows": sum(x["historical_marker_present"] for x in family_rows),
            "optimistic_pending_restores": sum(x["optimistic_pending_restores_after_marker_review"] for x in family_rows),
            "optimistic_pending_tile_calls": pending_tiles, **estimate(pending_tiles)}
    return {"schema_version": 1, "date": "2026-11-05", "status": "read_only_plan_T0.3_incomplete",
        "audit": {"JSON_inputs": sorted(inputs.paths), "file_hashes_recomputed": False,
            "checkpoint_PNG_video_tensor_payloads_opened": False, "torch_imported": "torch" in sys.modules,
            "GPU_launched": False, "GPU_queue_implemented_or_started": False, "old_results_modified": False},
        "source": {"asset_inventory": str(args.assets), "historical_outer_protocol": str(outer_path),
                   "historical_child_run": str(child_root), "old_complete_marker_count": 116},
        "retained_external_scope": {"frames": 17, "start": 0, "temporal_stride": 1, "sampler": "official-config UniPC",
            "steps": 20, "cfg": 1.0, "sampling_seeds": [42, 43], "tile_hw": [256, 448], "overlap": 64,
            "sequences": 44, "native_tile_sum_one_restore_each_sequence": 1228,
            "NoRef": "two sampling seeds, no supplied reference",
            "HRI": "two sampling seeds × hri_on/hri_off; on supplies GT frame0; same excluded-reference pool for both",
            "limitation": "17 consecutive-frame prefixes, not full-source-sequence evaluation; training used nine frames at stride4"},
        "calibration": {"summary": str(args.reproduction_summary), "stored_status": repro["status"],
            "stored_protocol_sha256_not_rehashed": repro["protocol_sha256"], "restore_seconds": seconds,
            "tiles_per_restore": 16, "seconds_per_tile_by_sampling_seed": dict(zip(("42", "43"), rates)),
            "mean_NoRef_seconds_per_tile": rate, "stored_total_wall_seconds": repro["wall_seconds"],
            "stored_wall_minus_restore_seconds": repro["wall_seconds"] - sum(seconds),
            "separate_reproduction_is_not_auto_acceptance_of_new16cohort": True,
            "note": "Only the existing passed summary is read; its payloads/digests are not revalidated here."},
        "totals": totals, "by_family": by_family,
        "cost": {"optimistic_minimum_after_all_historical_markers_pass_review": estimate(totals["optimistic_minimum_pending_tile_calls_after_marker_review"]),
            "if_every_row_requires_new_restore": estimate(totals["required_original_delegate_tile_calls"]),
            "interpretation": "optimistic restore-component lower cost extrapolation at one NoRef REDS4/000 per-tile rate; not a rigorous wall-time bound, forecast interval or completion-date promise",
            "excluded_unknowns": ["Sequence-dependent tile/attention/encode throughput", "HRI adapter forward overhead", "HRI T1 reference encoding (704 calls)",
                "CPU baseline and per-seed metrics", "PNG encoding and disk I/O", "Cold model/VAE load and cache state", "Strict acceptance review and possible reruns", "Failures, contention and pauses"]},
        "required_before_any_execution": [
            "Review all original116 markers separately: contracts, source/target/checkpoint/code and complete per-seed PNG/metric bindings; marker presence alone does not pass acceptance.",
            "Freeze a new explicit16-checkpoint cohort protocol while preserving this labeled17-frame prefix comparison. Do not copy/extend the original8 sealed worker and claim its old seal covers16 roles.",
            "Use fresh outputs in a reviewed new-project run directory. Never append to or overwrite the paused historical run.",
            "Declare which reviewed historical rows can be accepted under the new cohort protocol before scheduling; invalid/unknown rows require separate fresh work.",
            "Review resource and cost evidence before any GPU queue. This script supplies a plan only and launches no work."],
        "rows": rows}


def write_plan(output, plan):
    output.mkdir(parents=True, exist_ok=True)
    (output / "external_completion_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2)+"\n")
    fields = ["checkpoint_label", "family", "variant", "training_seed", "data_seed", "checkpoint_path", "training_run",
              "dataset", "sequence_id", "prefix_frames", "temporal_stride", "tile_count_per_restore", "restores_required",
              "original_delegate_tile_calls_required", "T1_reference_encodes_required", "historical_marker_present", "historical_complete_marker_path",
              "new_16cohort_acceptance", "optimistic_pending_restores_after_marker_review", "optimistic_pending_tile_calls_after_marker_review",
              "optimistic_restore_component_hours_at_NoRef_rate", "calibration_row_separately_reproduced"]
    with (output / "external_completion_plan.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in plan["rows"]:
            item = {key: row.get(key) for key in fields}
            item["historical_complete_marker_path"] = row["historical_complete_marker"]["path"] if row["historical_complete_marker"] else ""
            writer.writerow(item)
    totals, cost = plan["totals"], plan["cost"]
    text = ["# T0.3 external completion coverage/cost plan (2026-11-05)", "",
        "Read-only plan; no GPU queue was implemented or started. T0.3 remains incomplete. The 704-row CSV/JSON maps all16 checkpoints (NoRef/HRI Base3 + Align3 + Relation1 + Both1) to the44 frozen REDS4/YouHQ40 sequence prefixes.", "",
        "The old116 complete.json files are markers requiring separate strict acceptance review. They are not116 newly accepted rows. Base44 + Align44 + Relation28 belong only to NoRef training seed20261102; HRI and the other training seeds have no historical markers here. The separate successful REDS4/000 reproduction is a calibration/validation of one original row, not automatic acceptance of an expanded16-model cohort.", "",
        "| Family | Model-sequence rows | Required restores | Old markers (unaccepted) | Optimistic minimum pending restores | Pending tile-restores | NoRef-rate restore-only hours |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for family, row in plan["by_family"].items():
        text.append(f"| {family} | {row['rows']} | {row['required_restores']} | {row['historical_marker_rows']} | {row['optimistic_pending_restores']} | {row['optimistic_pending_tile_calls']} | {row['mean_rate_restore_component_hours']:.3f} |")
    text += [f"| Total | 704 | 2112 | 116 | 1880 | 52488 | {cost['optimistic_minimum_after_all_historical_markers_pass_review']['mean_rate_restore_component_hours']:.3f} |", "",
        "NoRef requires two restores per row (sampling seeds42/43). HRI requires four (both seeds × reference on/off) and two full-reference T1 encodes per row. Native capacity metadata sums to1228 spatial tiles for one restore across all44 sequences. Total scope is58,944 tile-restores; old markers cover6,456 tile-restores/232 restores without acceptance. At least588 rows/1880 restores remain if all old markers eventually pass review; unaccepted markers may increase the work.", "",
        "## Restore-component extrapolation", "",
        f"The passed NoRef REDS4/000 reproduction recorded restores of {plan['calibration']['restore_seconds'][0]:.6f}s and {plan['calibration']['restore_seconds'][1]:.6f}s, each16 tiles. Their mean rate is {plan['calibration']['mean_NoRef_seconds_per_tile']:.9f}s per tile.", "",
        f"Applying that NoRef rate to52,488 minimum pending tile-restores gives {cost['optimistic_minimum_after_all_historical_markers_pass_review']['mean_rate_restore_component_hours']:.3f} hours of restore computation alone. If all704 rows need fresh restores, the same extrapolation is {cost['if_every_row_requires_new_restore']['mean_rate_restore_component_hours']:.3f} hours. These are optimistic component estimates, not rigorous wall-time bounds or completion-date promises. The two calibration-rate range is not a confidence or forecast interval.", "",
        f"The calibration row's stored wall time was {plan['calibration']['stored_total_wall_seconds']:.3f}s, including {plan['calibration']['stored_wall_minus_restore_seconds']:.3f}s outside the two restore phases. That overhead is disclosed, not multiplied across different sequences. Sequence-dependent throughput, HRI forward/T1 encoding, all CPU metrics, PNG/disk I/O, cold starts, review/reruns and failures remain unpriced. HRI uses a NoRef rate only as a lower-component proxy; its actual runtime has not been measured for this plan.", "",
        "## Protocol and acceptance requirements", "",
        "Preserve the existing labeled17-frame prefix, start0/stride1, native ×4 geometry,20-step official-config UniPC/CFG1, sampler seeds42/43,256×448 HR tiles with64 overlap. This comparison does not become full-source-sequence testing, and it does not resolve the training stride4 versus external stride1 mismatch.", "",
        "Before execution, independently review the original116 rows and freeze a new explicit16-checkpoint cohort protocol. The old8-model sealed worker and its role budget do not cover an expanded16-model run. Decide acceptance/reuse under the new protocol explicitly; use fresh reviewed output directories and never append to or overwrite the paused historical run. No completion status or queue is created here.", "",
        "Only existing JSON and marker/metrics-path presence were inspected. No checkpoints, PNGs, source videos or tensor payloads were opened; no file hashes were recomputed. Stored marker digests and reproduction status are provenance declarations pending the stated acceptance review.", ""]
    (output / "external_completion_plan.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=Path("/home/hugo/Projects/VSR"))
    parser.add_argument("--assets", type=Path, default=Path(__file__).resolve().parents[1] / "docs/local-assets-20261105.json")
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    need(not any((args.output_dir / f"external_completion_plan.{suffix}").exists() for suffix in ("json", "md", "csv")),
         "Refusing to overwrite an existing external-completion plan")
    plan = build(args)
    need(not plan["audit"]["torch_imported"], "Read-only planner must not import torch")
    write_plan(args.output_dir, plan)
    print(json.dumps({"status": plan["status"], "totals": plan["totals"], "cost": plan["cost"]["optimistic_minimum_after_all_historical_markers_pass_review"]}))


if __name__ == "__main__":
    main()
