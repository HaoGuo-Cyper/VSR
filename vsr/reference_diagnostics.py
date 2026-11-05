"""Zero-update HRI mechanism diagnostics on this project's own checkpoints.

Clean HR latents deliberately construct the oracle-noised input. Teacher/flow
targets are used only for diagnostic losses. These are not LR-only restoration
scores and cannot establish useful reference propagation or a shared universal
space. The CLI must run only after the GPU training service is idle.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".deps"))

import numpy as np
import torch

from .alignment import latent_geometry, point_alignment
from .audit import dump, environment, file_hash, verify_official
from .cache import PairedClipSequence, _within, validate_cache
from .checkpoints import load_training_checkpoint
from .data import validate_manifest
from .feature_diagnostics import feature_statistics, same_anchor_patch_identity_retrieval
from .model import gradient_norms
from .reproducibility import configure_reproducibility, reproducibility_flags
from .train import compute_relation_objective

MODES = ("hri_on", "hri_off")
OBJECTIVES = {"Base": (), "Align": ("A",), "Relation": ("R",), "Both": ("A", "R")}
GENERATOR_GROUPS = ("lrv_stem", "lrv_residuals", "lora")
SUPERVISION_ASSETS = ("config.json", "environment.json", "selection.json", "correspondence.json", "cache_reuse.json",
    "teacher_cache_contract.json", "fresh_teacher_features.pt", "flow_provenance.json", "fresh_anchor_flows.pt",
    "reference_cache/reference_cache_seal.json")


def _supervision_assets(count):
    return (*SUPERVISION_ASSETS, *(f"reference_cache/clip_{index}/{name}" for index in range(count)
                                   for name in ("reference_latents.pt", "reference_binding.json")))


def _file_stat(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def seal_reference_supervision(reference_run, config, *, project_root=ROOT):
    """Producer hook: call after T=1 caches, before any variant is created.

    This immutable seal makes later mechanism diagnostics use the exact
    teacher/flow payloads present before training. It cannot be reconstructed
    retrospectively after a variant directory or completed status appears.
    It reads bytes only, creates no model and changes no RNG or train status.
    """
    root = Path(project_root).resolve(); run = _within(reference_run, root/"runs")
    if run == root/"runs" or not config.get("reference", {}).get("enabled") or _json(run/"config.json") != config:
        raise ValueError("Supervision sealing requires this own fresh reference run")
    def before_training():
        if any((run/variant).exists() for variant in OBJECTIVES):
            raise ValueError("Supervision seal must be produced before any variant training")
        status = run/"status.json"
        if status.exists() and _json(status).get("status") not in ("preparing",):
            raise ValueError("Cannot reconstruct supervision seal after training")
    before_training()
    paths = {name: _within(run/name, run) for name in SUPERVISION_ASSETS}
    stats = {name: _file_stat(path) for name, path in paths.items()}
    selected = _json(paths["selection.json"])
    geometry = _json(paths["correspondence.json"])
    if selected.get("test_accessed") is not False or not selected.get("training_ids") or not selected.get("validation_ids"):
        raise ValueError("Supervision preparation requires train/val selection without test access")
    refs = _json(paths["reference_cache/reference_cache_seal.json"])
    count = len(selected["training_ids"])+len(selected["validation_ids"])
    ordered_ids = selected["training_ids"]+selected["validation_ids"]
    if (refs.get("clip_count") != count or refs.get("single_frame_only") is not True
            or refs.get("video_latent_slicing") is not False or refs.get("new_cache") is not True
            or refs.get("legacy_state_loaded") is not False):
        raise ValueError("Wait for complete separate single-frame reference preparation")
    records = refs.get("per_clip", [])
    if len(records) != count:
        raise ValueError("Reference preparation has incomplete per-clip seals")
    for index, row in enumerate(records):
        if row.get("clip_index") != index or row.get("record_id") != ordered_ids[index] or row.get("frame_count") != config["data"]["num_frames"]:
            raise ValueError("Reference cache order/identity differs from its supervision selection")
        tensor_name = f"reference_cache/clip_{index}/reference_latents.pt"
        binding_name = f"reference_cache/clip_{index}/reference_binding.json"
        for name in (tensor_name, binding_name):
            paths[name] = _within(run/name, run); stats[name] = _file_stat(paths[name])
        binding = _json(paths[binding_name])
        if (file_hash(paths[tensor_name]) != row.get("sha256")
                or binding.get("reference_latents_sha256") != row.get("sha256")
                or binding.get("binding", {}).get("clip_metadata", {}).get("record_id") != row["record_id"]
                or binding["binding"].get("config_sha256") != config.get("_sha256")
                or binding["binding"].get("encoder_source_sha256") != refs.get("encoder_source_sha256")):
            raise ValueError("Reference cache payload or binding changed before supervision sealing")
    seal = {"schema": "vsr_reference_supervision_seal_v1", "phase": "prepared_before_any_variant",
        "optimizer_updates_before_seal": 0, "reference_run": str(run), "source_config_sha256": config.get("_sha256"),
        "manifest_sha256": selected["manifest_sha256"], "ordered_record_ids": ordered_ids,
        "geometry": geometry, "artifact_sha256": {name: file_hash(path) for name, path in paths.items()},
        "producer_module_sha256": file_hash(Path(__file__)), "task_checkpoint_or_optimizer_read": False}
    if any(_file_stat(path) != stats[name] for name, path in paths.items()):
        raise ValueError("Supervision preparation changed while being sealed")
    before_training()
    target = run/"reference_supervision_seal.json"; temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=run, prefix=".reference_supervision_", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(seal, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
            handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if _json(target) != seal:
                raise ValueError("Existing reference supervision seal differs; refusing overwrite") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return seal


def _verify_supervision_seal(run, config):
    path = _within(run/"reference_supervision_seal.json", run)
    if not path.is_file():
        raise ValueError("Missing pre-training reference supervision seal; do not reconstruct it post hoc")
    seal = _json(path)
    selection = _json(run/"selection.json"); geometry = _json(run/"correspondence.json")
    count = len(selection["training_ids"])+len(selection["validation_ids"])
    if (seal.get("schema") != "vsr_reference_supervision_seal_v1" or seal.get("phase") != "prepared_before_any_variant"
            or seal.get("optimizer_updates_before_seal") != 0 or seal.get("reference_run") != str(run)
            or seal.get("source_config_sha256") != config.get("_sha256")
            or seal.get("manifest_sha256") != selection.get("manifest_sha256")
            or seal.get("ordered_record_ids") != selection["training_ids"]+selection["validation_ids"]
            or seal.get("geometry") != geometry or set(seal.get("artifact_sha256", {})) != set(_supervision_assets(count))):
        raise ValueError("Reference supervision seal identity/order/geometry differs")
    for name, expected in seal["artifact_sha256"].items():
        if file_hash(_within(run/name, run)) != expected:
            raise ValueError(f"Pre-training supervision asset changed: {name}")
    return path, seal


def _json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _state_hash(model):
    """Stream all parameters and buffers, including nonpersistent buffers."""
    digest = hashlib.sha256()
    entries = [("parameter:"+name, value) for name, value in model.named_parameters()]
    entries += [("buffer:"+name, value) for name, value in model.named_buffers()]
    for name, value in sorted(entries, key=lambda item: item[0]):
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big")); digest.update(header)
        digest.update(len(raw).to_bytes(8, "big")); digest.update(raw)
    return digest.hexdigest()


def _rng_snapshot():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state().clone(),
            "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None}


def _rng_equal(first, second):
    a, b = first["numpy"], second["numpy"]
    same_numpy = a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]
    same_cuda = (first["cuda"] is None and second["cuda"] is None) or (
        first["cuda"] is not None and second["cuda"] is not None and len(first["cuda"]) == len(second["cuda"])
        and all(torch.equal(x, y) for x, y in zip(first["cuda"], second["cuda"])))
    return first["python"] == second["python"] and same_numpy and torch.equal(first["torch"], second["torch"]) and same_cuda


def _restore_rng(snapshot):
    random.setstate(snapshot["python"]); np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch"])
    if snapshot["cuda"] is not None:
        torch.cuda.set_rng_state_all(snapshot["cuda"])


def _clear(model):
    model.zero_grad(set_to_none=True)
    model.clear_features(); model.clear_reference()


def _noise_seed(seed, record_id):
    value = json.dumps(["reference-oracle-diagnostic-v1", seed, record_id, .5], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**63)


def _finite(value, label, shape=None):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point() or not bool(torch.isfinite(value).all())
            or (shape is not None and tuple(value.shape) != tuple(shape))):
        raise ValueError(f"Invalid finite diagnostic tensor: {label}")
    return value.detach()


def _sample(sample, geometry, config, expected_split):
    metadata = sample["metadata"]
    if metadata.get("split") != expected_split or not metadata.get("record_id") or not metadata.get("source_id"):
        raise ValueError("Diagnostic sample identity or train/val role is invalid")
    frame_count = config["data"]["num_frames"]
    indices = metadata.get("source_indices", [])
    stride = config["data"].get("temporal_stride", 1)
    if (len(indices) != frame_count or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in indices)
            or indices != [indices[0]+stride*i for i in range(frame_count)]):
        raise ValueError("Original sampled-frame identity is missing")
    clean = _finite(sample["clean"], "clean")
    if clean.ndim != 5 or clean.shape[0] != 1 or list(clean.shape[2:]) != list(geometry["latent_grid"]):
        raise ValueError("Diagnostic clean latent geometry differs")
    _finite(sample["low"], "low", clean.shape)
    _finite(sample["text"], "text")
    teacher = _finite(sample["teacher"], "teacher")
    if teacher.ndim != 3 or teacher.shape[:2] != (1, math.prod(geometry["token_grid"])):
        raise ValueError("Diagnostic teacher grid differs")
    for key in ("forward", "backward"):
        flow = _finite(sample["flows"][key], key)
        if tuple(flow.shape) != (1, geometry["token_grid"][0]-1, 2, config["data"]["height"], config["data"]["width"]):
            raise ValueError("Diagnostic direct-anchor flow geometry differs")


def _forward_inputs(sample, device, config):
    clean = sample["clean"].detach().to(device=device, dtype=torch.float32)
    actual_seed = _noise_seed(config["seed"], sample["metadata"]["record_id"])
    # One CPU generator makes the same noise field independent of mode,
    # objective, reference choice, preceding sampling or the global RNG.
    noise = torch.randn(clean.shape, generator=torch.Generator(device="cpu").manual_seed(actual_seed), dtype=torch.float32).to(device)
    return (.5*clean+.5*noise, torch.full((1,), 500., device=device),
            sample["text"].detach().to(device), sample["low"].detach().to(device), actual_seed)


def _reference(sample, mode, ref, frame_count, device):
    if mode == "hri_off":
        return {"ref_latent": None, "ref_index": None, "frame_count": frame_count,
                "ref_valid": torch.zeros(1, dtype=torch.bool, device=device)}
    clean = sample["clean"]
    reference = _finite(sample["reference_latents"][ref], "single-frame HRI",
                        (1, clean.shape[1], 1, clean.shape[3], clean.shape[4]))
    return {"ref_latent": reference.to(device), "ref_index": ref, "frame_count": frame_count,
            "ref_valid": torch.ones(1, dtype=torch.bool, device=device)}


def collect_reference_diagnostics(model, inputs, config, variant, *, reference_indices=(0, 4), modes=MODES):
    """Measure isolated A/R gradients on train0 and fixed-val feature statistics.

    No optimizer is accepted or created. Call only with an idle model having
    no outstanding forward graph or parameter gradients. The public CPU API
    accepts tiny random models/targets for tests; the CLI loads only own final
    reference checkpoints and verified T=1 reference caches.
    """
    if variant not in OBJECTIVES or not callable(getattr(model, "clear_reference", None)):
        raise ValueError("Requires a known reference-trained model variant")
    if config.get("reproducibility", {}).get("mode") != "strict":
        raise ValueError("Reference diagnostics require the strict relation path")
    refs, modes = tuple(reference_indices), tuple(modes)
    frames = config["data"]["num_frames"]
    if (not refs or len(set(refs)) != len(refs) or any(not isinstance(ref, int) or isinstance(ref, bool) or not 0 <= ref < frames for ref in refs)
            or not modes or len(set(modes)) != len(modes) or set(modes)-set(MODES)):
        raise ValueError("Invalid reference cases or modes")
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Clear pre-existing parameter gradients before diagnostics")
    if any(getattr(model, name, None) is not None for name in ("_reference_context", "_condition_tokens", "_captured")):
        raise ValueError("Diagnostics cannot interleave an outstanding forward graph")
    train, validation, geometry = inputs["training"], inputs["validation"], inputs["geometry"]
    _sample(train, geometry, config, "train")
    if not validation:
        raise ValueError("A fixed validation selection is required")
    for sample in validation:
        _sample(sample, geometry, config, "val")
    if len({s["metadata"]["record_id"] for s in validation}) != len(validation):
        raise ValueError("Duplicate validation record")
    if any(sample["metadata"]["source_id"] == train["metadata"]["source_id"] for sample in validation):
        raise ValueError("Training and validation source identity overlap")
    flags = [(module, module.training) for module in model.modules()]
    requirements = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    device = next(model.parameters()).device
    started = time.perf_counter()
    state_before, rng_before = _state_hash(model), _rng_snapshot()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result = {"schema": "vsr_reference_mechanism_diagnostics_v1", "variant": variant,
        "status": "completed", "scope": "oracle_noised_HR_mechanism_diagnostic_not_LRV_only_restoration",
        "runtime_flags": reproducibility_flags(),
        "protocol": {"sigma": .5, "timestep": 500., "noisy": ".5*cleanHR+.5*localCPUnoise",
            "noise": "SHA256(data seed,record ID,.5), identical for A/R, on/off and reference choices",
            "gradient_sample": "fixed training clip index0", "backward": "one unweighted A-only or R-only loss per applicable objective; no FM",
            "validation": "fixed originating selection order; per-video then equal-video scalar means",
            "reference_cases": "separate known-index choices; first/middle/additional roles, no pooling", "optimizer_updates": 0,
            "teacher_flow_latents": "detached targets and inputs; no target/input gradients",
            "restoration_quality_evidence": False, "strict_relation_loss_device": "cpu",
            "caveat": "Alignment, relations, variance, rank or retrieval cannot establish recovery benefit or information equivalence"},
        "applicable_objectives": list(OBJECTIVES[variant]), "reference_cases": {}}
    try:
        if not OBJECTIVES[variant]:
            result["not_applicable_reason"] = "Base has no student projector or A/R training objective"
        else:
            for ref in refs:
                case = {"reference_index": ref, "reference_role": "first_frame_main" if ref == 0 else (
                            "middle_frame_adaptation" if ref == frames//2 else "known_time_additional"),
                        "modes": {}}
                for mode in modes:
                    gradient_rows = []
                    for objective in OBJECTIVES[variant]:
                        _clear(model); model.train()
                        noisy, timestep, text, low, noise_seed = _forward_inputs(train, device, config)
                        teacher = train["teacher"].detach().to(device)
                        mask = torch.ones(teacher.shape[:2], dtype=torch.bool, device=device)
                        with torch.enable_grad():
                            context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
                            with context:
                                _, student = model(noisy, timestep, text, low, **_reference(train, mode, ref, frames, device))
                            if objective == "A":
                                loss, relation_details = point_alignment(student, teacher, mask), None
                            else:
                                loss, relation_details = compute_relation_objective(student, teacher, geometry["token_grid"],
                                    train["flows"]["forward"].detach(), train["flows"]["backward"].detach(), mask, config)
                            if not bool(torch.isfinite(loss)):
                                raise RuntimeError("Nonfinite isolated auxiliary loss")
                            loss.backward()
                        groups = gradient_norms(model)
                        if not all(value["finite"] and math.isfinite(value["l2"]) for value in groups.values()) or groups["frozen"]["elements_with_grad"]:
                            raise RuntimeError("Nonfinite or frozen-parameter auxiliary gradients")
                        if mode == "hri_off" and groups["hri"]["elements_with_grad"]:
                            raise RuntimeError("Missing HRI unexpectedly acquired gradients")
                        reached = {key: value["finite"] and value["elements_with_grad"] > 0 and value["l2"] > 0 for key, value in groups.items()}
                        weight = config["training"]["lambda_align" if objective == "A" else "lambda_relation"]
                        gradient_rows.append({"objective": objective, "loss": float(loss.detach()), "training_weight": weight,
                            "backward_loss": "unweighted auxiliary only", "FM_in_backward": False, "noise_seed": noise_seed,
                            "record_id": train["metadata"]["record_id"], "source_id": train["metadata"]["source_id"],
                            "reference_sampled_frame_index": ref if mode == "hri_on" else None,
                            "reference_source_frame_index": train["metadata"]["source_indices"][ref] if mode == "hri_on" else None,
                            "sampled_input_fps": train["metadata"].get("output_fps"),
                            "gradient_groups": groups, "reached": reached,
                            "all_generator_groups_reached": all(reached[key] for key in GENERATOR_GROUPS),
                            "relation_diagnostics": relation_details})
                        _clear(model)
                    model.eval(); rows = []
                    with torch.no_grad():
                        for sample in validation:
                            noisy, timestep, text, low, noise_seed = _forward_inputs(sample, device, config)
                            teacher = sample["teacher"].detach().to(device)
                            mask = torch.ones(teacher.shape[:2], dtype=torch.bool, device=device)
                            context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
                            with context:
                                _, student = model(noisy, timestep, text, low, **_reference(sample, mode, ref, frames, device))
                            relation, details = compute_relation_objective(student, teacher, geometry["token_grid"],
                                sample["flows"]["forward"].detach(), sample["flows"]["backward"].detach(), mask, config)
                            statistics = feature_statistics(student, teacher, mask)
                            retrieval = same_anchor_patch_identity_retrieval(student, teacher, mask, geometry["token_grid"])
                            rows.append({"record_id": sample["metadata"]["record_id"], "source_id": sample["metadata"]["source_id"],
                                "reference_sampled_frame_index": ref if mode == "hri_on" else None,
                                "reference_source_frame_index": sample["metadata"]["source_indices"][ref] if mode == "hri_on" else None,
                                "sampled_input_fps": sample["metadata"].get("output_fps"),
                                "split": "val", "noise_seed": noise_seed, "alignment_error": float(point_alignment(student, teacher, mask)),
                                "relation_error": float(relation), "relation_diagnostics": details,
                                "statistics": statistics, "same_anchor_patch_retrieval": retrieval})
                            model.clear_features(); model.clear_reference()
                    scalar_keys = ("valid_point_cosine_error", "student_mean_channel_variance", "teacher_mean_channel_variance",
                                   "student_effective_rank", "teacher_effective_rank")
                    case["modes"][mode] = {"auxiliary_only_gradients": gradient_rows, "validation": rows,
                        "validation_macro": {"video_count": len(rows),
                            "alignment_error": sum(row["alignment_error"] for row in rows)/len(rows),
                            "relation_error": sum(row["relation_error"] for row in rows)/len(rows),
                            "same_anchor_spatial_patch_top1": sum(row["same_anchor_patch_retrieval"]["same_anchor_spatial_patch_top1_identity_rate"] for row in rows)/len(rows),
                            **{key: sum(row["statistics"][key] for row in rows)/len(rows) for key in scalar_keys}}}
                result["reference_cases"][str(ref)] = case
    finally:
        _clear(model)
        for module, training in flags:
            module.training = training
        state_after, rng_after = _state_hash(model), _rng_snapshot()
        _restore_rng(rng_before)
        checks = {"all_model_parameters_buffers_unchanged": state_before == state_after,
            "global_rng_not_consumed": _rng_equal(rng_before, rng_after), "global_rng_restored": _rng_equal(rng_before, _rng_snapshot()),
            "module_flags_restored": all(module.training == training for module, training in flags),
            "requires_grad_unchanged": all(parameter.requires_grad == required for parameter, required in requirements),
            "all_parameter_gradients_cleared": all(parameter.grad is None for parameter in model.parameters()),
            "reference_context_cleared": model._reference_context is None,
            "condition_and_feature_context_cleared": all(getattr(model, name, None) is None for name in ("_condition_tokens", "_captured"))}
        if not all(checks.values()):
            raise RuntimeError(f"Reference diagnostic cleanup/state audit failed: {checks}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result.update(state_sha256_before=state_before, state_sha256_after=state_after, cleanup=checks,
                  wall_seconds=time.perf_counter()-started,
                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
    return result


def assert_gpu_idle(project_root=ROOT, *, runner=subprocess.run, own_pid=None):
    """Reject another training PID, permitting this CLI's own service only.

    A stale service-only running marker may be reported after the unit exits;
    actual inactive-unit and GPU checks establish that case. Running main or
    reference-training markers still require root reconciliation first.
    """
    root, own_pid = Path(project_root).resolve(), os.getpid() if own_pid is None else own_pid
    units = {"vsr-training.service"}; markers = {}
    for name in ("active_training_service.json", "active_reference_training.json", "active_training.json"):
        path = root/"runs"/name
        if path.exists():
            row = _json(path); markers[name] = row
            if row.get("unit"):
                units.add(row["unit"])
            if row.get("status") == "running" and (name != "active_training_service.json" or not row.get("unit")):
                raise RuntimeError(f"Active {name}; root must verify/clear training before diagnostics")
    states, owned_units = {}, []
    for unit in sorted(units):
        command = runner(["systemctl", "--user", "show", unit, "-p", "LoadState", "-p", "ActiveState", "-p", "MainPID"],
                         capture_output=True, text=True)
        props = dict(line.split("=", 1) for line in command.stdout.splitlines() if "=" in line)
        states[unit] = props
        if props.get("ActiveState") in {"active", "activating", "deactivating", "reloading"}:
            if props.get("ActiveState") == "active" and props.get("MainPID") == str(own_pid):
                owned_units.append(unit)
            else:
                raise RuntimeError(f"Known GPU service is active: {unit}")
        if command.returncode and props.get("LoadState") != "not-found":
            raise RuntimeError("Cannot verify known GPU service is idle")
    query = runner(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], capture_output=True, text=True)
    if query.returncode:
        raise RuntimeError("Cannot verify GPU compute process inventory")
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        pid, process = line.split(",", 1)
        if int(pid.strip()) != own_pid and Path(process.strip()).name != "gjs":
            raise RuntimeError("Another GPU compute process is active")
    return {"training_markers_checked": sorted(markers), "known_units_checked": sorted(units),
            "unit_states": states, "own_diagnostic_service_units": owned_units,
            "service_marker_running_but_unit_inactive": bool(markers.get("active_training_service.json", {}).get("status") == "running"
                and markers["active_training_service.json"]["unit"] not in owned_units),
            "GPU_idle_or_only_own_process_verified": True, "GPU_idle_verified": True}


def _load(path, *, mmap=False):
    before = path.stat()
    value = torch.load(path, map_location="cpu", weights_only=True, **({"mmap": True} if mmap else {}))
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Frozen diagnostic asset changed during loading")
    return value


def load_reference_inputs(reference_run, config, *, project_root=ROOT, expected_validation_count=32, reference_indices=(0, 4)):
    """Read train0/val32 and bound, separately T=1 encoded own HRI cache."""
    root = Path(project_root).resolve(); allowed = root/"runs"
    run = _within(reference_run, allowed)
    if not config.get("reference", {}).get("enabled") or _json(run/"config.json") != config:
        raise ValueError("Reference checkpoint/config origin mismatch")
    status = _json(run/"status.json")
    if status.get("status") != "completed" or status.get("reference_trained") is not True:
        raise ValueError("Wait for the complete reference experiment before diagnostics")
    supervision_path, supervision_seal = _verify_supervision_seal(run, config)
    reuse_path = run/"cache_reuse.json"; reuse = _json(reuse_path)
    source = _within(reuse["source_run"], allowed)
    if source != Path(config["data"]["cache_from"]).resolve():
        raise ValueError("Reference run RGB source differs from its originating configuration")
    manifest = _json(root/config["data"]["manifest"]); validate_manifest(manifest)
    validate_cache(source, config, manifest["manifest_sha256"],
                   {name: file_hash(root/name) for name in ("vsr/data.py", "vsr/preflight.py")},
                   allowed_root=allowed, expected_manifest=manifest)
    seal_path = source/"cache_asset_seal.json"; seal = _json(seal_path)
    if file_hash(seal_path) != reuse.get("source_asset_seal_sha256"):
        raise ValueError("Source asset seal differs from the reference training reuse audit")
    selection = _json(run/"selection.json"); source_selection = _json(source/"selection.json")
    if any(selection.get(key) != source_selection.get(key) for key in ("manifest_sha256", "training_ids", "validation_ids", "test_accessed")) or selection.get("test_accessed") is not False:
        raise ValueError("Reference selection differs from its frozen source selection")
    training_ids, validation_ids = selection["training_ids"], selection["validation_ids"]
    if not training_ids or len(validation_ids) != expected_validation_count:
        raise ValueError("Expected exactly train0 plus the fixed originating validation32")
    indices = [0, *range(len(training_ids), len(training_ids)+len(validation_ids))]
    clips = PairedClipSequence(source, allowed_root=allowed, max_cached=1)
    geometry = _json(run/"correspondence.json")
    base = Path(config["model"]["local_dir"])
    actual = latent_geometry(_json(base/"vae/config.json"), _json(base/"transformer/config.json"),
                            config["data"]["num_frames"], config["data"]["height"], config["data"]["width"])
    if geometry != actual or geometry != _json(source/"correspondence.json"):
        raise ValueError("Reference diagnostic geometry differs from the actual official configurations")
    teacher_contract = _json(run/"teacher_cache_contract.json")
    env = _json(run/"environment.json")
    if supervision_seal.get("producer_module_sha256") != env["code_sha256"].get("vsr/reference_diagnostics.py"):
        raise ValueError("Supervision seal producer differs from the recorded pre-training code version")
    if (teacher_contract.get("weights_sha256") != config["teacher"]["weights_sha256"]
            or teacher_contract.get("source_revision") != config["teacher"]["revision"]
            or not isinstance(teacher_contract.get("alignment_source_sha256"), str)
            or len(teacher_contract["alignment_source_sha256"]) != 64
            or teacher_contract.get("alignment_source_sha256") != env["code_sha256"].get("vsr/alignment.py")
            or teacher_contract.get("frozen") is not True):
        raise ValueError("Reference teacher target provenance differs")
    flow_provenance = _json(run/"flow_provenance.json")
    if (flow_provenance.get("anchors") != geometry["anchors"] or flow_provenance.get("clip_count") != len(clips)
            or flow_provenance.get("checkpoint_sha256") != config["flow"]["weights_sha256"]
            or flow_provenance.get("num_flow_updates") != config["flow"]["updates"] or flow_provenance.get("frozen") is not True):
        raise ValueError("Reference flow target identity differs")
    reference_seal_path = run/"reference_cache/reference_cache_seal.json"; reference_seal = _json(reference_seal_path)
    if (reference_seal.get("single_frame_only") is not True or reference_seal.get("video_latent_slicing") is not False
            or reference_seal.get("clip_count") != len(clips) or reference_seal.get("new_cache") is not True
            or reference_seal.get("legacy_state_loaded") is not False):
        raise ValueError("Reference cache is not a complete independent T=1 encoding")
    encoder_hashes = reference_seal["encoder_source_sha256"]
    if (set(encoder_hashes) != {"vsr/reference_training.py", "vsr/preflight.py"}
            or any(not isinstance(value, str) or len(value) != 64 or env["code_sha256"].get(name) != value for name, value in encoder_hashes.items())):
        raise ValueError("Reference cache encoder code differs from its recorded training version")
    records = {row["clip_index"]: row for row in reference_seal["per_clip"]}
    if len(records) != len(clips):
        raise ValueError("Reference cache has missing/duplicate clip seals")
    paths = [supervision_path, reuse_path, seal_path, run/"selection.json", reference_seal_path, run/"fresh_teacher_features.pt", run/"fresh_anchor_flows.pt"]
    target_features = _load(_within(run/"fresh_teacher_features.pt", run), mmap=True)
    target_flows = _load(_within(run/"fresh_anchor_flows.pt", run), mmap=True)
    if len(target_features) != len(clips) or len(target_flows) != len(clips):
        raise ValueError("Reference targets differ from the fixed clip selection")
    texts_path = _within(source/"fresh_text_embeddings.pt", source)
    if file_hash(texts_path) != seal["artifact_sha256"].get("fresh_text_embeddings.pt"):
        raise ValueError("Frozen official text cache changed")
    texts = _load(texts_path); paths.append(texts_path)
    samples = []
    for index in indices:
        clip = clips[index]; metadata = clip["metadata"]
        for name in ("metadata.json", "paired_rgb.pt", "fresh_latents.pt"):
            path = _within(source/f"clip_{index}"/name, source)
            if file_hash(path) != seal["artifact_sha256"].get(f"clip_{index}/{name}"):
                raise ValueError("Selected frozen RGB/latent asset differs from its seal")
            paths.append(path)
        latents = _load(_within(source/f"clip_{index}/fresh_latents.pt", source))
        if latents.get("binding") != clip["binding"]:
            raise ValueError("Normalized latent binding differs from its RGB target")
        directory = _within(run/f"reference_cache/clip_{index}", run)
        path, binding_path = _within(directory/"reference_latents.pt", run), _within(directory/"reference_binding.json", run)
        binding_record = _json(binding_path); expected = records[index]
        if expected.get("record_id") != metadata["record_id"] or expected.get("frame_count") != config["data"]["num_frames"] or file_hash(path) != expected.get("sha256") or file_hash(path) != binding_record.get("reference_latents_sha256"):
            raise ValueError("Independent reference tensor seal/identity changed")
        payload = _load(path); binding = payload["binding"]
        if (binding != binding_record["binding"] or binding.get("source_binding") != clip["binding"] or binding.get("clip_metadata") != metadata
                or binding.get("config_sha256") != config.get("_sha256") or binding.get("official_revision") != config["model"]["revision"]
                or binding.get("encoder_source_sha256") != encoder_hashes or binding.get("vae_config_sha256") != file_hash(base/"vae/config.json")
                or binding.get("encoding") != "one separate T=1 VAE encode per original frame" or binding.get("posterior") != "mode"
                or binding.get("normalization") != "official (latent-mean)/std" or binding.get("vae_frozen") is not True
                or binding.get("frame_indices") != list(range(config["data"]["num_frames"]))):
            raise ValueError("HRI cache is not bound to the same raw clip and single-frame encoder")
        clean = latents["clean"]
        refs = payload["reference_latents"]
        if set(refs) != set(range(config["data"]["num_frames"])):
            raise ValueError("Missing original single-frame reference")
        for ref, tensor in refs.items():
            _finite(tensor, "cached T=1 reference", (1, clean.shape[1], 1, clean.shape[3], clean.shape[4]))
        sample = {"metadata": metadata, "clean": clean.detach().clone(), "low": latents["low"].detach().clone(),
            "text": texts[metadata["caption"]].detach().clone(), "teacher": target_features[index].detach().clone(),
            "flows": {key: target_flows[index][key].detach().clone() for key in ("forward", "backward")},
            "reference_latents": {ref: refs[ref].detach().clone() for ref in reference_indices}}
        _sample(sample, geometry, config, "train" if index == 0 else "val")
        samples.append(sample); paths.extend((path, binding_path))
    _verify_supervision_seal(run, config)
    audit = {"reference_run": str(run), "RGB_source_run": str(source), "training_clip_index": 0,
        "training_record_id": samples[0]["metadata"]["record_id"], "validation_ids_in_origin_order": validation_ids,
        "validation_count": len(validation_ids), "test_accessed": False, "reference_source": "same reference run's separate T=1 caches",
        "video_latent_sliced_for_HRI": False, "pre_training_supervision_seal_verified": True,
        "producer_supervision_seal_sha256": file_hash(supervision_path),
        "asset_sha256": {str(path): file_hash(path) for path in paths}}
    return {"training": samples[0], "validation": samples[1:], "geometry": geometry, "source_audit": audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, help="New own-run diagnostic directory; default timestamped run")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    guard = assert_gpu_idle(ROOT)
    from .infer_video import preview_training_config
    preview = preview_training_config(args.checkpoint, allowed_root=ROOT/"runs")
    config, variant = preview["config"], preview["variant"]
    if not config.get("reference", {}).get("enabled") or preview["step"] != config["training"]["updates"]:
        raise ValueError("Only a final own reference checkpoint may be diagnosed")
    origin = Path(args.checkpoint).resolve().parent.parent
    output = _within(args.output or ROOT/"runs"/("reference_diagnostics_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")), ROOT/"runs")
    output.mkdir(parents=True, exist_ok=False)
    dump(output/"status.json", {"status": "running", "kind": "reference_mechanism_diagnostics", "optimizer_updates": 0})
    try:
        inputs = load_reference_inputs(origin, config)
        dump(output/"source_inputs.json", inputs["source_audit"])
        dump(output/"official_artifacts.json", verify_official(ROOT, config))
        dump(output/"reproducibility.json", configure_reproducibility(config))
        guard_before_model = assert_gpu_idle(ROOT)
        model, audit = load_training_checkpoint(args.checkpoint, device=args.device, allowed_root=ROOT/"runs")
        if audit["checkpoint_sha256"] != preview["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during diagnostic setup")
        dump(output/"checkpoint_restore.json", audit)
        dump(output/"environment.json", environment(ROOT))
        report = collect_reference_diagnostics(model, inputs, config, variant)
        report.update(checkpoint=audit, source_audit=inputs["source_audit"], idle_guard={"initial": guard, "before_model_load": guard_before_model},
                      module_sha256=file_hash(Path(__file__)))
        dump(output/"diagnostics.json", report)
        dump(output/"status.json", {"status": "completed", "optimizer_updates": 0, "restoration_quality_evidence": False})
        print(json.dumps({"status": "completed", "output": str(output), "variant": variant, "optimizer_updates": 0}))
    except Exception as error:
        dump(output/"status.json", {"status": "failed", "error": str(error), "traceback": traceback.format_exc(), "optimizer_updates": 0})
        raise


if __name__ == "__main__":
    main()
