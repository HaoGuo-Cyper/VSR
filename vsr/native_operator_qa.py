"""Predeclared native-pixel validation QA, without training or model selection.

One shared frozen-evaluation run per spatial case retains the existing exact
PNG/metric/input/checkpoint seals. The outer protocol additionally binds both
cases and their structural observations before any QA pixel decode.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

from .frozen_evaluation import ROOT, _digest, _hash_file, _inside, _json, run_frozen_evaluation
from .infer_video import preview_training_config, verify_output_video


def _read_plan(value):
    plan = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
    return json.loads(json.dumps(plan, ensure_ascii=False, allow_nan=False))


def _verify_playback_and_audits(case_run):
    """Reopen actual playback containers; compare observed seeded-noise hashes."""
    rows, observations, noise_hashes = [], [], set()
    for path in sorted((case_run / "results").glob("*/*/*/metrics.json")):
        row = json.loads(path.read_text())
        rows.append({"path": str(path.relative_to(case_run)), "sha256": _hash_file(path)})
        for mode, values in row["modes"].items():
            for entry in values["seeds"]:
                audit = entry["restoration"].get("native_operator_audit", {})
                noise_sha = audit.get("full_noise_sha256")
                if (audit.get("status") != "passed" or not isinstance(noise_sha, str)
                        or len(noise_sha) != 64 or any(ch not in "0123456789abcdef" for ch in noise_sha)):
                    raise ValueError("Native operator QA requires observed successful restore/noise audits")
                noise_hashes.add(noise_sha)
                preview = entry.get("preview")
                if not preview or preview.get("downscaled_for_playback_only") is not False:
                    raise ValueError("QA requires a complete native-size playback verification")
                artifact = _inside(case_run / preview["artifact"]["path"], case_run)
                if _hash_file(artifact) != preview["artifact"]["sha256"]:
                    raise ValueError("Playback artifact changed before container verification")
                height, width = preview["source_hw"]
                observed = verify_output_video(artifact, frames=row["evaluated_frames"],
                    height=height, width=width, fps=row["source_fps"])
                observations.append({"label": row["label"], "mode": mode, "seed": entry["seed"],
                    "artifact": preview["artifact"], "observed_container": observed,
                    "full_noise_sha256": noise_sha, "restore_audit_status": audit["status"]})
    if not rows or len(noise_hashes) != 1 or not observations:
        raise ValueError("All QA model/on-off modes must use the same observed global noise field")
    # Exactly one no-reference Base plus one reference Base on/off, seed42.
    if sorted(item["mode"] for item in observations) != ["hri_off", "hri_on", "no_reference"]:
        raise ValueError("QA did not preserve all three predeclared inference modes")
    return {"status": "passed", "row_metrics": rows, "playback": observations,
        "shared_global_noise_sha256": next(iter(noise_hashes)),
        "quality_scores_used_for_acceptance": False}


def run_native_operator_qa(project_root, checkpoints, plans, *, run_dir=None,
                           device="cuda", metric_device="cpu", factories=None):
    from .native_qa_reader import NativeQAReader
    from .native_qa_restore import AuditedNativeRestore
    from .reference_diagnostics import assert_gpu_idle

    started = time.perf_counter()
    root = Path(project_root).resolve()
    uses_cuda = any(str(value).startswith("cuda") for value in (device, metric_device))
    guard = assert_gpu_idle(root) if uses_cuda else {"device": "cpu", "GPU_not_requested": True}
    plans = [_read_plan(value) for value in plans]
    if len(plans) != 2:
        raise ValueError("Predeclare both native-pixel crop and original whole-frame QA")
    for plan in plans:
        if (plan.get("selection", {}).get("split") != "val"
                or plan.get("input_preparation", {}).get("source_frame_indices") != list(range(17))
                or plan["input_preparation"].get("num_frames") != 17):
            raise ValueError("QA accepts only the predeclared contiguous17-frame validation prefix")
        restoration = plan.get("restoration", {})
        if (restoration.get("steps"), restoration.get("sampling_seed"), restoration.get("HR_tile_hw"),
                restoration.get("overlap")) != (20, 42, [256, 448], 64):
            raise ValueError("QA sampling/tiling differs from the fixed predeclaration")
    if len(checkpoints) != 2:
        raise ValueError("QA requires one no-reference Base and one reference-trained Base")
    overrides = dict(factories or {})
    if set(overrides) & {"reader", "restore"}:
        raise ValueError("QA reader and audited restore cannot be replaced")
    preview_factory = overrides.get("preview", preview_training_config)
    previews = {}
    for label, checkpoint in checkpoints.items():
        path = _inside(checkpoint, root / "runs")
        spec = preview_factory(path, allowed_root=root / "runs")
        if spec["checkpoint_sha256"] != _hash_file(path) or spec["variant"] != "Base" or spec["step"] != 8192:
            raise ValueError("Infrastructure QA uses exactly completed own8192 Base checkpoints")
        previews[label] = {**spec, "path": str(path)}
    reference = [label for label, spec in previews.items() if spec["config"].get("reference", {}).get("enabled", False)]
    if len(reference) != 1:
        raise ValueError("One QA checkpoint must be reference-trained and the other no-reference")
    no_reference = next(spec for label, spec in previews.items() if label not in reference)
    for plan in plans:
        if no_reference["checkpoint_sha256"] != plan["checkpoint_policy"]["no_reference_SHA256"]:
            raise ValueError("No-reference Base differs from the prospectively specified checkpoint SHA")
    # Catalogue construction verifies source identities and reads metadata only.
    reader = NativeQAReader(root, plans=plans)
    try:
        datasets = list(reader.records)
        if len(datasets) != 2 or set(datasets) != {"HQ-val-nativecrop", "HQ-val-nativefull"}:
            raise ValueError("QA must retain distinct crop and original full-frame cases")
        policies = {dataset: reader.policy(dataset) for dataset in datasets}
        catalogue = reader.metadata
    finally:
        reader.close()
    run = _inside(run_dir if run_dir is not None else root / "runs" / (
        "native_operator_qa_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")), root / "runs")
    if run.exists() and any(run.iterdir()) and not (run / "qa_protocol.json").is_file():
        raise ValueError("Refusing nonempty QA directory without a matching outer protocol")
    run.mkdir(parents=True, exist_ok=True)
    protocol = {"schema": "vsr_native_operator_qa_v1", "plans": plans, "catalogue": catalogue,
        "checkpoints": previews, "datasets": datasets, "training_updates": 0,
        "quality_scores_used_for_acceptance": False, "test_or_external_data_used": False,
        "code_sha256": {file.name: _hash_file(file) for file in sorted(Path(__file__).parent.glob("*.py"))},
        "scope": "two fixed validation-prefix structural operator cases; not training, model selection or whole-source-video evaluation"}
    contract = _digest(protocol)
    protocol_path = run / "qa_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing QA protocol changed; retain it and declare a separate version")
    if not protocol_path.exists():
        _json(protocol_path, protocol)
    # Keep failed/interrupted attempt cost and observations on an explicit
    # identical-protocol resume, as well as any earlier completed summary.
    if (run / "qa_status.json").exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        history = run / "qa_attempt_history"
        history.mkdir(exist_ok=True)
        for name in ("qa_status.json", "qa_summary.json"):
            source = run / name
            if source.exists():
                with (history / (stamp + "_" + name)).open("xb") as stream:
                    stream.write(source.read_bytes())
    _json(run / "qa_status.json", {"status": "running", "protocol_sha256": contract, "GPU_guard": guard})
    outcomes = {}
    try:
        for dataset in datasets:
            case_run = run / dataset.rsplit("-", 1)[1]
            height, width = policies[dataset].reader_options["height"], policies[dataset].reader_options["width"]
            case_factories = {**overrides, "reader": NativeQAReader, "restore": AuditedNativeRestore(plans)}
            result = run_frozen_evaluation(root, checkpoints, datasets=(dataset,), frames=17,
                seeds=(42,), steps=20, tile_hw=(256, 448), overlap=64,
                device=device, metric_device=metric_device, run_dir=case_run,
                caption="A high quality video.", preview_max_side=max(height, width),
                selection_note="predeclared8192Base checkpoints for infrastructure only; no score ranking",
                factories=case_factories, policy=policies[dataset])
            audit = _verify_playback_and_audits(case_run)
            _json(case_run / "native_operator_case_audit.json", audit)
            outcomes[dataset] = {"run": str(case_run), "protocol_sha256": result["protocol_sha256"],
                "summary_sha256": _hash_file(case_run / "summary.json"),
                "structural_audit_sha256": _hash_file(case_run / "native_operator_case_audit.json")}
        summary = {"status": "passed", "run": str(run), "protocol_sha256": contract,
            "cases": outcomes, "training_updates": 0, "test_or_external_data_used": False,
            "quality_hypotheses_verified": False, "wall_seconds": time.perf_counter() - started,
            "GPU_guard": guard, "cost_scope": "whole QA wrapper including guard/CPU checkpoint preview, model/text/VAE loads, native restoration, metrics and artifact verification; not CUDA-kernel active time"}
        _json(run / "qa_summary.json", summary)
        _json(run / "qa_status.json", {"status": "passed", "protocol_sha256": contract,
            "summary_sha256": _hash_file(run / "qa_summary.json")})
        return summary
    except Exception as error:
        observations = getattr(error, "audit", None)
        if observations is None and "case_factories" in locals():
            observations = getattr(case_factories["restore"], "last_audit", None)
        failure_observations = None
        if observations is not None:
            try:
                sidecar = run / "native_restore_failure_observations.json"
                _json(sidecar, {"error": str(error), "last_observed_restore": observations})
                failure_observations = {"path": sidecar.name, "sha256": _hash_file(sidecar)}
            except Exception as audit_error:
                failure_observations = {"serialization_error": str(audit_error)}
        _json(run / "qa_status.json", {"status": "failed", "error": str(error),
            "traceback": traceback.format_exc(), "protocol_sha256": contract,
            "wall_seconds": time.perf_counter() - started, "completed_cases": outcomes,
            "no_silent_fallback": True, "training_updates": 0,
            "restore_observations": failure_observations})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--plan", type=Path, action="append", required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--metric-device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    checkpoints = {}
    for item in args.checkpoint:
        label, separator, path = item.partition("=")
        if not separator or label in checkpoints:
            raise ValueError("Use distinct --checkpoint LABEL=PATH entries")
        checkpoints[label] = path
    result = run_native_operator_qa(args.project_root, checkpoints, args.plan,
        run_dir=args.run_dir, metric_device=args.metric_device)
    print(json.dumps({key: result[key] for key in ("status", "run", "protocol_sha256")}))


if __name__ == "__main__":
    main()
