"""Short synthetic resource checks before frozen native-resolution evaluation.

This runs three UniPC steps at fixed maximum header-derived geometries. It
never opens dataset pixels, scores restoration quality, or selects a model.
The separate validation operator QA and actual evaluation remain twenty-step
protocols. Run this entry point in an independent service after GPU exclusion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch

from .frozen_evaluation import (ROOT, _default_caption, _default_model, _default_musiq,
    _default_preview, _default_vae, _digest, _hash_file, _inside, _json)
from .inference import tile_positions
from .native_qa_restore import AuditedNativeRestore, _tensor_sha
from .reference_diagnostics import assert_gpu_idle


def fixed_plans():
    """Metadata arithmetic only; no tensor, decoder, model, or GPU allocation."""
    plans = []
    for label, lr_h, lr_w in (("maximum_native", 576, 720), ("maximum_paired", 270, 480)):
        h, w = lr_h * 4, lr_w * 4
        hp, wp = (h + 15)//16*16, (w + 15)//16*16
        ys, ah = tile_positions(hp, 256, 64)
        xs, aw = tile_positions(wp, 448, 64)
        tiles = [[y, x, ah, aw] for y in ys for x in xs]
        plans.append({"schema": "vsr_synthetic_resource_geometry_v1", "label": label,
            "input_preparation": {"LR_shape": [1, 3, 17, lr_h, lr_w],
                "HR_shape": [1, 3, 17, h, w], "source": "synthetic constant RGB0.5; no source pixels"},
            "restoration": {"steps": 3, "sampling_seed": 42, "HR_tile_hw": [256, 448],
                "overlap": 64, "padded_HR_T_HW": [17, hp, wp],
                "tile_hr_tlhw": tiles, "tile_latent_tlhw": [[v//8 for v in box] for box in tiles],
                "tile_count": len(tiles), "global_CPU_Gaussian_noise_shape": [1, 16, 5, hp//8, wp//8]}})
    return plans


def _device(value):
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Resource probe accepts only CPU or CUDA devices")
    return torch.device("cuda", device.index or 0) if device.type == "cuda" else device


def _host_memory():
    try:
        values = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmHWM:", "VmRSS:")):
                name, value = line.split(":", 1)
                values[name + "_bytes"] = int(value.split()[0])*1024
        return {**values, "VmHWM_scope": "whole process lifetime high-water; not resettable or a per-phase peak"}
    except (OSError, ValueError) as error:
        return {"observation_error": str(error), "VmHWM_bytes": None, "VmRSS_bytes": None}


def _gpu_memory(devices, *, peaks=False):
    values = {}
    for device in devices:
        # Failure logging must not create a CUDA context after a rejected
        # second guard or a pre-CUDA failure.
        if not torch.cuda.is_initialized():
            values[str(device)] = {"CUDA_not_initialized": True,
                "allocated_bytes": None, "reserved_bytes": None}
            continue
        try:
            free, total = torch.cuda.mem_get_info(device)
            row = {"allocated_bytes": torch.cuda.memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device),
                "driver_free_bytes": free, "driver_total_bytes": total}
            if peaks:
                row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            values[str(device)] = row
        except Exception as error:
            values[str(device)] = {"observation_error": f"{type(error).__name__}: {error}"}
    return values


def _synthetic(shape):
    tensor = torch.full(tuple(shape), .5, dtype=torch.float32, device="cpu")
    return tensor, {"shape": list(shape), "dtype": "torch.float32", "device": "cpu",
        "constant_rgb": .5, "source": "synthetic allocation only", "sha256": _tensor_sha(tensor)}


def _verify_musiq_assets(root):
    """CPU hashes only; bind the real metric before any model/GPU loading."""
    from .no_reference import (KONIQ_CHECKPOINT, KONIQ_SHA256, SOURCE_COMMIT,
                               verify_musiq_sources)
    source = verify_musiq_sources(root)
    weight = Path(root)/"weights"/KONIQ_CHECKPOINT
    if _hash_file(weight) != KONIQ_SHA256:
        raise ValueError("Resource probe MUSIQ weight differs from pinned KonIQ SHA256")
    return {"verification": "pinned sources and full weight SHA256 verified on CPU before protocol seal",
        "source_commit": SOURCE_COMMIT, "source_repo": source.get("source_repo"),
        "provenance_manifest_sha256": _hash_file(Path(root)/"docs/musiq-provenance.json"),
        "weights_sha256": KONIQ_SHA256, "weights_path": str(weight.resolve()),
        "vendor_file_sha256": {name: item["vendored_sha256"] for name,item in source["files"].items()}}


class _Measurements:
    def __init__(self, run, devices):
        self.run, self.devices, self.phases = run, devices, {}

    def call(self, name, action):
        started = time.perf_counter()
        row = {"status": "running", "host_before": _host_memory(), "started_utc": datetime.now(timezone.utc).isoformat()}
        self.phases[name] = row
        _json(self.run/"phase_observations.json", self.phases)
        try:
            for device in self.devices:
                torch.cuda.synchronize(device)
            row["cuda_baseline"] = _gpu_memory(self.devices)
            for device in self.devices:
                torch.cuda.reset_peak_memory_stats(device)
            result = action()
            for device in self.devices:
                torch.cuda.synchronize(device)
            row["status"] = "completed"
            return result
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            row.update(wall_seconds=time.perf_counter()-started, host_after=_host_memory(),
                       cuda_end_and_peaks=_gpu_memory(self.devices, peaks=True))
            _json(self.run/"phase_observations.json", self.phases)


def run_resource_probe(project_root, no_reference_checkpoint, reference_checkpoint, *,
                       device="cuda", metric_devices=("cpu",), run_dir=None,
                       capacity_audit=None, factories=None):
    started = time.perf_counter()
    root = Path(project_root).resolve()
    device = _device(device)
    if not isinstance(metric_devices, (tuple, list)) or not metric_devices:
        raise ValueError("Declare one or both metric devices explicitly")
    metric_devices = tuple(_device(value) for value in metric_devices)
    if len({str(value) for value in metric_devices}) != len(metric_devices):
        raise ValueError("Metric devices must be distinct")
    cuda_devices = sorted({value for value in (device, *metric_devices) if value.type == "cuda"}, key=str)
    guard = assert_gpu_idle(root) if cuda_devices else {"GPU_not_requested": True}
    defaults = {"preview": _default_preview, "caption": _default_caption, "model": _default_model,
        "vae": _default_vae, "musiq": _default_musiq, "restore": None, "synthetic": _synthetic}
    supplied = factories or {}
    if set(supplied)-set(defaults):
        raise ValueError("Unknown resource probe factory override")
    f = {**defaults, **supplied}
    checkpoints = {}
    for role, path in (("no_reference", no_reference_checkpoint), ("reference", reference_checkpoint)):
        path = _inside(path, root/"runs")
        spec = f["preview"](path, allowed_root=root/"runs")
        if (spec["checkpoint_sha256"] != _hash_file(path) or spec["step"] != 8192
                or spec["variant"] != "Both" or bool(spec["config"].get("reference", {}).get("enabled", False)) != (role == "reference")):
            raise ValueError("Resource probe requires explicit completed own Both8192 checkpoints with the correct reference roles")
        checkpoints[role] = {**spec, "path": str(path)}
    if checkpoints["no_reference"]["config"]["model"] != checkpoints["reference"]["config"]["model"]:
        raise ValueError("Both resource checkpoints must use the same pinned official model")
    metric_assets = ({"verification": "synthetic fixture override; real MUSIQ assets not verified",
                      "synthetic_fixture": True} if "musiq" in supplied else _verify_musiq_assets(root))
    capacity_binding = None
    if capacity_audit is not None:
        capacity_path = Path(capacity_audit).expanduser().resolve()
        capacity_record = json.loads(capacity_path.read_text())
        capacity_binding = {"path": str(capacity_path), "sha256": _hash_file(capacity_path),
            "source_manifest_sha256": capacity_record.get("source_manifest_sha256"),
            "source_stream_report_sha256": capacity_record.get("source_stream_report_sha256")}
        for field, relative in (("source_manifest_sha256", "manifests/external.json"),
                                ("source_stream_report_sha256", "docs/external-streams.json")):
            if capacity_binding[field] is not None and _hash_file(root/relative) != capacity_binding[field]:
                raise ValueError("Capacity audit metadata changed before resource protocol sealing")
    plans = fixed_plans()
    cases = [
        {"id": "no_reference_maximum_native_off", "role": "no_reference", "plan": 0, "hri": False},
        {"id": "reference_maximum_paired_on", "role": "reference", "plan": 1, "hri": True},
        {"id": "reference_maximum_native_off", "role": "reference", "plan": 0, "hri": False}]
    run = _inside(run_dir if run_dir is not None else root/"runs"/(
        "native_resource_probe_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")), root/"runs")
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Preserve each resource attempt; use a new run directory instead of overwriting or retrying")
    run.mkdir(parents=True, exist_ok=True)
    protocol = {"schema": "vsr_native_resource_probe_v1", "checkpoints": checkpoints,
        "inference_device": str(device), "metric_devices": list(map(str, metric_devices)),
        "plans": plans, "cases": cases, "synthetic_rgb": .5, "caption": "A high quality video.",
        "MUSIQ_input_shape": [1, 3, 1, 2304, 2880], "MUSIQ_batch_size": 1,
        "MUSIQ_assets": metric_assets, "capacity_audit": capacity_binding,
        "MUSIQ_model_co_residency": "reference Both model and VAE remain resident on the declared inference device",
        "factory_overrides": sorted(supplied), "training_updates": 0, "dataset_pixels_opened": False,
        "quality_scores_retained": False, "quality_hypotheses_verified": False,
        "sampling_scope": "resource-only3steps; does not replace20stepvalidationQA or frozen evaluation",
        "memory_scope": "instrumented probe includes state/noise/output hashing; PyTorch CUDA allocator peaks exclude driver allocations; VmHWM is lifetime; no end-to-end PNG/bicubic/source-buffer peak or capacity upperbound",
        "failure_scope": "handled Python exceptions retained; abrupt OS exit requires service/journal reconciliation; no fallback",
        "code_sha256": {path.name: _hash_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))}}
    contract = _digest(protocol)
    _json(run/"protocol.json", protocol)
    _json(run/"status.json", {"status": "running", "protocol_sha256": contract, "GPU_guard": guard})
    measurements = _Measurements(run, cuda_devices)
    restore = f["restore"] if f["restore"] is not None else AuditedNativeRestore(plans)
    observations, metric_observations = {}, {}
    model = vae = text = metric = None
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    try:
        # Inventory can change between CPU preview/sealing and first GPU load.
        if cuda_devices:
            # Do not initialize CUDA for measurement before this second guard.
            _json(run/"GPU_guard_before_load.json", assert_gpu_idle(root))
        for role, spec in checkpoints.items():
            path = Path(spec["path"])
            text, caption_audit = measurements.call(role+"_caption", lambda:
                f["caption"](spec["config"], protocol["caption"], path, root, device=device))
            if text.ndim != 3 or text.shape[0] != 1 or not bool(torch.isfinite(text).all()):
                raise ValueError("Resource caption loader returned an invalid native embedding")
            _json(run/(role+"_caption.json"), {"audit": caption_audit, "embedding_sha256": _tensor_sha(text)})
            model, loaded = measurements.call(role+"_model_load", lambda:
                f["model"](path, device=device, allowed_root=root/"runs"))
            if loaded["checkpoint_sha256"] != spec["checkpoint_sha256"] or loaded["config"] != spec["config"]:
                raise ValueError("Own checkpoint changed between protocol preview and model load")
            vae = measurements.call(role+"_VAE_load", lambda: f["vae"](spec["config"], device=device))
            model.eval(); vae.requires_grad_(False).eval()
            for case in (case for case in cases if case["role"] == role):
                plan = plans[case["plan"]]
                def native_action():
                    lrv, input_audit = f["synthetic"](plan["input_preparation"]["LR_shape"])
                    hri = None
                    hri_audit = None
                    if case["hri"]:
                        hri_shape = [1, 3, 1, *plan["input_preparation"]["HR_shape"][-2:]]
                        hri, hri_audit = f["synthetic"](hri_shape)
                    _json(run/(case["id"]+"_input.json"), {"LRV": input_audit, "HRI": hri_audit,
                        "observed_LRV_shape": list(lrv.shape), "observed_HRI_shape": list(hri.shape) if hri is not None else None})
                    if list(lrv.shape) != plan["input_preparation"]["LR_shape"] or (hri is not None and list(hri.shape) != hri_shape):
                        raise ValueError("Synthetic factory changed the fixed resource geometry")
                    video, metadata = restore(model, vae, lrv, text, spec["config"], steps=3, seed=42,
                        tile_hw=(256,448), overlap=64, reference_image=hri,
                        reference_index=0 if case["hri"] else None)
                    audit = metadata.get("native_operator_audit", {})
                    if list(video.shape) != plan["input_preparation"]["HR_shape"] or audit.get("status") != "passed":
                        raise ValueError("Resource restore must pass the observed geometry/noise/state audit")
                    result = {"status": "completed", "role": role, "LRV": input_audit, "HRI": hri_audit,
                        "output_shape": list(video.shape), "restoration": metadata,
                        "output_RGB_not_saved_or_scored": True}
                    del video, lrv, hri
                    return result
                observations[case["id"]] = measurements.call(case["id"], native_action)
                _json(run/(case["id"]+".json"), observations[case["id"]])
            if role == "reference":
                for metric_device in metric_devices:
                    label = "MUSIQ_"+str(metric_device).replace(":", "_")
                    metric = measurements.call(label+"_load", lambda:
                        f["musiq"](root, device=metric_device))
                    def metric_action():
                        frame, synthetic_audit = f["synthetic"](protocol["MUSIQ_input_shape"])
                        if list(frame.shape) != protocol["MUSIQ_input_shape"]:
                            raise ValueError("MUSIQ synthetic geometry changed")
                        value = metric.evaluate_video(frame, batch_size=1)
                        scores = value.get("per_frame_scores", [])
                        if value.get("frame_count") != 1 or len(scores) != 1 or not all(math.isfinite(float(score)) for score in scores):
                            raise ValueError("MUSIQ resource forward did not return one finite synthetic scalar")
                        # Deliberately do not serialize synthetic MOS/mean values.
                        result = {"status": "completed", "input": synthetic_audit, "device": str(metric_device),
                            "batch_size": 1, "forward_completed": True, "synthetic_quality_value_discarded": True,
                            "SR_model_and_VAE_co_resident": True, "inference_device": str(device),
                            "metric_provenance": value.get("provenance")}
                        del value, scores, frame
                        return result
                    metric_observations[str(metric_device)] = measurements.call(label+"_forward", metric_action)
                    _json(run/(label+".json"), metric_observations[str(metric_device)])
                    del metric; metric = None; gc.collect()
            del model, vae, text
            model = vae = text = None
            gc.collect()
            for cuda_device in cuda_devices:
                torch.cuda.empty_cache()
        summary = {"status": "passed", "run": str(run), "protocol_sha256": contract,
            "native_cases": observations, "MUSIQ_resource_forwards": metric_observations,
            "phase_observations_sha256": _hash_file(run/"phase_observations.json"),
            "wall_seconds": time.perf_counter()-started, "host_lifetime_memory": _host_memory(),
            "GPU_guard": guard, "quality_scores_retained": False, "quality_hypotheses_verified": False,
            "training_updates": 0, "dataset_pixels_opened": False, "capacity_upperbound_claimed": False,
            "wall_scope": "run_resource_probe entry through summary observation; excludes import startup and final JSON serialization/cleanup",
            "scope": protocol["sampling_scope"]+"; "+protocol["memory_scope"]}
        _json(run/"summary.json", summary)
        _json(run/"status.json", {"status": "passed", "protocol_sha256": contract,
            "summary_sha256": _hash_file(run/"summary.json")})
        return summary
    except Exception as error:
        partial = getattr(error, "audit", None) or getattr(restore, "last_audit", None)
        _json(run/"failure_observations.json", {"error": str(error), "partial_restore_audit": partial,
            "completed_native_cases": observations, "completed_MUSIQ_devices": metric_observations,
            "host_lifetime_memory": _host_memory(), "cuda_memory": _gpu_memory(cuda_devices, peaks=True)})
        _json(run/"status.json", {"status": "failed", "protocol_sha256": contract,
            "error": str(error), "traceback": traceback.format_exc(), "wall_seconds": time.perf_counter()-started,
            "failure_observations_sha256": _hash_file(run/"failure_observations.json"),
            "no_silent_fallback": True, "quality_scores_retained": False,
            "wall_scope": "run_resource_probe entry through failure observation; excludes import startup and final serialization/cleanup"})
        raise
    finally:
        del model, vae, text, metric
        gc.collect()
        for cuda_device in cuda_devices:
            try:
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-reference-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--capacity-audit", type=Path, help="optional metadata-only capacity report to SHA-bind")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--metric-devices", choices=("cpu", "cuda"), nargs="+", default=["cpu"])
    args = parser.parse_args()
    result = run_resource_probe(args.project_root, args.no_reference_checkpoint, args.reference_checkpoint,
        device=args.device, metric_devices=args.metric_devices, run_dir=args.run_dir, capacity_audit=args.capacity_audit)
    print({key: result[key] for key in ("status", "run", "protocol_sha256")})


if __name__ == "__main__":
    main()
