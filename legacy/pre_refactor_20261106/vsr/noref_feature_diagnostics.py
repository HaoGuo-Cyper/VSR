"""Zero-update no-reference feature diagnostics with freshly extracted targets.

This is an oracle-noised-HR mechanism check, never LR-only restoration or a
quality-benefit claim. Only the fixed source-held-out val32 is used. Old saved
teacher/flow payloads and the top-level cache (which may embed them) are never
loaded. Official DINO/RAFT targets are freshly computed from bound val HR,
transferred to CPU, and released before the own task checkpoint is restored.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".deps"))

import torch
import torch.nn.functional as F

from .alignment import DinoTeacher, latent_geometry, point_alignment
from .audit import dump, environment, file_hash, verify_official
from .cache import PairedClipSequence, _within, validate_cache
from .checkpoints import load_training_checkpoint
from .data import validate_manifest
from .feature_diagnostics import feature_statistics, same_anchor_patch_identity_retrieval
from .flow import FlowTeacher
from .reference_diagnostics import (_load, _rng_equal, _rng_snapshot, _restore_rng,
                                    _state_hash, assert_gpu_idle)
from .relation import temporal_correspondence
from .reproducibility import configure_reproducibility
from .train import compute_relation_objective, select_records

VARIANTS = ("Base", "Align", "Relation", "Both")
NUMERIC_MODULES = ("vsr/model.py", "vsr/data.py", "vsr/preflight.py", "vsr/cache.py", "vsr/alignment.py",
                   "vsr/flow.py", "vsr/relation.py", "vsr/train.py", "vsr/feature_diagnostics.py", "vsr/reproducibility.py")


def _json(path):
    result = json.loads(Path(path).read_text())
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def _finite(tensor, shape, label, *, dtype=None):
    if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(shape) or not tensor.is_floating_point()
            or not bool(torch.isfinite(tensor).all()) or (dtype is not None and tensor.dtype != dtype)):
        raise ValueError(f"Invalid {label} tensor shape/dtype/finiteness")


def _tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    header = json.dumps([str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
    return hashlib.sha256(header+b"\0"+value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _release(device):
    gc.collect()
    if torch.device(device).type == "cuda" and torch.cuda.is_initialized():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def load_noref_validation(origin, config, *, project_root=ROOT, expected_validation_count=32):
    """Load only SHA-bound val RGB/per-clip VAE/text, preserving source indices."""
    root = Path(project_root).resolve(); allowed = root/"runs"; run = _within(origin, allowed)
    if run == allowed or config.get("reference", {}).get("enabled", False) or _json(run/"config.json") != config:
        raise ValueError("Requires this own no-reference run and exact checkpoint configuration")
    if config.get("reproducibility", {}).get("mode") != "strict":
        raise ValueError("Only the strict formal preparation may be diagnosed")
    if not isinstance(expected_validation_count, int) or isinstance(expected_validation_count, bool) or expected_validation_count < 1:
        raise ValueError("Expected validation count must be positive")
    status = _json(run/"status.json")
    if status.get("status") not in {"completed", "interrupted"}:
        raise ValueError("Wait for the originating training process to finish")
    source = run; reuse_path = run/"cache_reuse.json"
    if config["data"].get("cache_from"):
        reuse = _json(reuse_path); source = _within(reuse["source_run"], allowed)
        if source != Path(config["data"]["cache_from"]).resolve():
            raise ValueError("Diagnostic RGB source differs from training cache_from")
    elif reuse_path.exists():
        raise ValueError("Unexpected indirect cache for a fresh-preparation configuration")
    source_status = _json(source/"status.json")
    seal_path = _within(source/"cache_asset_seal.json", source); seal = _json(seal_path)
    seal_sha_before = file_hash(seal_path)
    if seal.get("contains_task_initialization") is not False:
        raise ValueError("Source seal must exclude task initialization")
    if source_status.get("status") == "interrupted":
        if (seal.get("format") != "vsr_prepared_cache_seal_v1" or seal.get("preparation_completed") is not True
                or seal.get("origin_training_status") != "interrupted" or seal.get("source_run") != str(source)
                or seal.get("origin_status_sha256") != file_hash(source/"status.json")):
            raise ValueError("Interrupted source lacks a verified complete-preparation seal")
    elif source_status.get("status") != "completed":
        raise ValueError("Frozen source preparation is not complete")
    if source != run and seal_sha_before != reuse.get("source_asset_seal_sha256"):
        raise ValueError("Source seal SHA differs from originating training reuse audit")
    manifest_path = _within(root/config["data"]["manifest"], root)
    manifest = _json(manifest_path); validate_manifest(manifest)
    validate_cache(source, config, manifest["manifest_sha256"],
        {name: file_hash(root/name) for name in ("vsr/data.py", "vsr/preflight.py")},
        allowed_root=allowed, expected_manifest=manifest)
    selection = _json(run/"selection.json"); source_selection = _json(source/"selection.json")
    if any(selection.get(key) != source_selection.get(key) for key in ("manifest_sha256", "training_ids", "validation_ids", "test_accessed")) or selection.get("test_accessed") is not False:
        raise ValueError("Run/source selection or test-use identity differs")
    train_records, val_records = select_records(manifest, config)
    if selection["training_ids"] != [row["id"] for row in train_records] or selection["validation_ids"] != [row["id"] for row in val_records]:
        raise ValueError("Validation selection is not the original metric-independent ordering")
    if len(val_records) != expected_validation_count or not train_records:
        raise ValueError("Expected the complete fixed originating validation32")
    assets = seal.get("artifact_sha256", {})
    paths = [seal_path, source/"status.json", run/"selection.json", run/"config.json", run/"correspondence.json", manifest_path]
    expected_bytes = {}
    def verified(name):
        path = _within(source/name, source)
        if file_hash(path) != assets.get(name):
            raise ValueError(f"Frozen diagnostic input SHA mismatch: {name}")
        expected_bytes[str(path)] = assets[name]
        paths.append(path)
        return path
    for name in ("config.json", "environment.json", "selection.json", "correspondence.json"):
        verified(name)
    if source != run:
        paths.append(reuse_path)
    geometry = _json(run/"correspondence.json"); base = Path(config["model"]["local_dir"])
    vae_config = _json(base/"vae/config.json"); dit_config = _json(base/"transformer/config.json")
    actual = latent_geometry(vae_config, dit_config, config["data"]["num_frames"], config["data"]["height"], config["data"]["width"])
    if geometry != actual or geometry != _json(source/"correspondence.json"):
        raise ValueError("Diagnostic geometry differs from frozen official-model geometry")
    channels = vae_config["z_dim"]
    if dit_config["in_channels"] != channels:
        raise ValueError("Official VAE/DiT latent channels differ")
    text_dim = _json(base/"text_encoder/config.json")["d_model"]
    text_path = verified("fresh_text_embeddings.pt"); texts = _load(text_path)
    clips = PairedClipSequence(source, allowed_root=allowed, max_cached=1)
    samples = []
    shape = (1, channels, *geometry["latent_grid"])
    for index, record in enumerate(val_records, start=len(train_records)):
        for name in ("metadata.json", "paired_rgb.pt", "fresh_latents.pt"):
            verified(f"clip_{index}/{name}")
        clip = clips[index]; metadata = clip["metadata"]
        if (metadata["record_id"], metadata["source_id"], metadata["split"]) != (record["id"], record["source_id"], "val") or metadata.get("caption") != record.get("caption"):
            raise ValueError("Diagnostic validation identity/caption differs from source catalogue")
        payload = _load(source/f"clip_{index}/fresh_latents.pt")
        if payload.get("binding") != clip["binding"]:
            raise ValueError("Normalized latent binding differs from the selected RGB")
        for key in ("clean", "low"):
            _finite(payload[key], shape, key, dtype=torch.float32)
        text = texts[metadata["caption"]]; _finite(text, (1, 512, text_dim), "frozen text")
        samples.append({"clip_index": index, "metadata": metadata, "binding": clip["binding"],
            "hr": clip["hr"].detach().cpu().clone(), "clean": payload["clean"].detach().cpu().clone(),
            "low": payload["low"].detach().cpu().clone(), "text": text.detach().cpu().clone()})
    if (any(file_hash(path) != expected for path, expected in expected_bytes.items()) or file_hash(seal_path) != seal_sha_before
            or _json(seal_path) != seal or _json(source/"status.json") != source_status
            or _json(run/"config.json") != config or _json(run/"selection.json") != selection
            or _json(source/"selection.json") != source_selection or _json(manifest_path) != manifest
            or (source != run and _json(reuse_path) != reuse)):
        raise ValueError("Frozen diagnostic inputs changed during loading")
    audit = {"originating_run": str(run), "RGB_source_run": str(source), "validation_count": len(samples),
        "validation_ids_in_origin_order": [row["id"] for row in val_records], "source_clip_indices": [row["clip_index"] for row in samples],
        "test_accessed": False, "teacher_or_flow_payload_read": False, "top_level_training_cache_read": False,
        "source_cache_seal_sha256": file_hash(seal_path), "source_training_status": source_status["status"],
        "input_asset_sha256": {str(path): file_hash(path) for path in paths}}
    return {"validation": samples, "geometry": geometry, "source_audit": audit}


@torch.no_grad()
def describe_hr(hr, flows, geometry, config):
    """Model-independent descriptors; no thresholds/classification/selection."""
    forward, backward = (flows[key].detach().float().cpu() for key in ("forward", "backward"))
    _, pairs, _, height, width = forward.shape
    correspondence = temporal_correspondence(forward.reshape(-1, 2, height, width),
        backward.reshape(-1, 2, height, width), geometry["token_grid"][-2:],
        config["flow"]["fb_alpha"], config["flow"]["fb_beta"])
    def magnitude(value):
        norm = value.square().sum(dim=2).sqrt().reshape(-1)
        return {"mean": float(norm.mean()), "p95": float(torch.quantile(norm, .95))}
    dest_hr, grid = correspondence["destination_hr"], correspondence["grid"]
    token_h, token_w = geometry["token_grid"][-2:]
    dest_token = (grid+1)*grid.new_tensor([token_w, token_h])/2-.5
    eps = 1e-5
    boundary = ((dest_hr[..., 0] >= -eps) & (dest_hr[..., 0] <= width-1+eps)
        & (dest_hr[..., 1] >= -eps) & (dest_hr[..., 1] <= height-1+eps)
        & (dest_token[..., 0] >= -eps) & (dest_token[..., 0] <= token_w-1+eps)
        & (dest_token[..., 1] >= -eps) & (dest_token[..., 1] <= token_h-1+eps))
    fb = correspondence["fb_error_sq"] <= correspondence["fb_threshold_sq"]
    anchor_rgb = hr.detach().double().cpu()[0, :, geometry["anchors"]].permute(1, 0, 2, 3)
    gray = (anchor_rgb*anchor_rgb.new_tensor([.299, .587, .114]).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    kernel = gray.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    laplacian = F.conv2d(gray, kernel)
    return {"forward_flow_norm_hr_pixels": magnitude(forward), "backward_flow_norm_hr_pixels": magnitude(backward),
        "token_boundary_valid_fraction": float(boundary.float().mean()),
        "token_forward_backward_consistent_fraction": float(fb.float().mean()),
        "token_motion_visibility_valid_fraction": float(correspondence["valid"].float().mean()),
        "token_motion_reliability_weight_mean": float(correspondence["weight"].mean()),
        "HR_anchor_luma_laplacian_energy": float(laplacian.square().mean()),
        "protocol": {"scope": "descriptive only, independent of candidate model; no split/threshold/example selection",
            "flow_norm": "all dense HR pixels of direct anchor pairs; displacement in HR pixels, not pixels/second",
            "visibility": "existing temporal_correspondence at student-token centers; bounds + squared-HR-pixel forward/backward check; no padding",
            "fb_alpha": config["flow"]["fb_alpha"], "fb_beta_squared_hr_pixels": config["flow"]["fb_beta"],
            "laplacian": "mean squared four-neighbour discrete Laplacian [[0,1,0],[1,-4,1],[0,1,0]] on anchor luma, valid interior only, pixel spacing=1 HR pixel",
            "luma": "0.299R+0.587G+0.114B from decoded RGB[0,1], float64; not radiometric luminance",
            "energy_units": "squared normalized-luma discrete second differences; no intensity rescaling or padded border",
            "anchors": geometry["anchors"], "anchor_pair_count": pairs}}


@torch.no_grad()
def extract_fresh_targets(inputs, config, *, project_root=ROOT, device="cuda", teacher_factory=None, flow_factory=None):
    """DINO then RAFT, CPU outputs and release, before any task model exists."""
    root, device = Path(project_root), torch.device(device); geometry = inputs["geometry"]
    samples = inputs["validation"]
    if not samples or any(sample["metadata"]["split"] != "val" for sample in samples):
        raise ValueError("Fresh diagnostic targets accept validation only")
    teacher_dim = config["conditioning"]["teacher_dim"]
    features, motion = [], []
    target_started = time.perf_counter()
    teacher = (teacher_factory(config) if teacher_factory else DinoTeacher(root/config["teacher"]["source_dir"], root/config["teacher"]["weights"]))
    teacher.requires_grad_(False).eval().to(device)
    teacher_started = time.perf_counter()
    teacher_load_seconds = teacher_started-target_started
    try:
        for sample in samples:
            value = teacher(sample["hr"].detach().to(device), geometry["anchors"], geometry["token_grid"])
            _finite(value, (1, math.prod(geometry["token_grid"]), teacher_dim), "fresh DINO target")
            features.append(value.detach().cpu().clone())
    finally:
        del teacher; _release(device)
    teacher_seconds = time.perf_counter()-teacher_started
    flow_started = time.perf_counter()
    estimator = (flow_factory(config) if flow_factory else FlowTeacher(root/config["flow"]["weights"],
        num_flow_updates=config["flow"]["updates"], expected_sha256=config["flow"]["weights_sha256"]))
    estimator.requires_grad_(False).eval().to(device)
    flow_provenance = getattr(estimator, "provenance", {})
    try:
        for sample in samples:
            forward, backward = estimator.estimate_anchor_flows(sample["hr"].detach().to(device), geometry["anchors"])
            shape = (1, len(geometry["anchors"])-1, 2, *sample["hr"].shape[-2:])
            for value in (forward, backward):
                _finite(value, shape, "fresh direct HR-anchor flow")
            motion.append({"forward": forward.detach().cpu().clone(), "backward": backward.detach().cpu().clone()})
    finally:
        del estimator; _release(device)
    flow_seconds = time.perf_counter()-flow_started
    prepared, hashes = [], []
    for sample, teacher, flows in zip(samples, features, motion):
        descriptors = describe_hr(sample["hr"], flows, geometry, config)
        prepared.append({**sample, "teacher": teacher, "flows": flows, "HR_descriptors": descriptors})
        hashes.append({"record_id": sample["metadata"]["record_id"], "clip_index": sample["clip_index"],
            "teacher_sha256": _tensor_hash(teacher), "forward_sha256": _tensor_hash(flows["forward"]),
            "backward_sha256": _tensor_hash(flows["backward"])})
    return {**inputs, "validation": prepared, "fresh_target_audit": {"freshly_extracted": True,
        "saved_training_teacher_flow_payload_read": False, "teacher_and_flow_released_before_task_model": True,
        "teacher": {"source_revision": config["teacher"]["revision"], "weights_sha256": config["teacher"]["weights_sha256"]},
        "flow": flow_provenance, "factory_overrides": [name for name, value in (("teacher", teacher_factory), ("flow", flow_factory)) if value],
        "per_video_target_sha256": hashes, "teacher_compute_transfer_release_seconds": teacher_seconds,
        "teacher_load_seconds": teacher_load_seconds,
        "flow_load_compute_transfer_release_seconds": flow_seconds, "entire_target_wall_seconds": time.perf_counter()-target_started}}


@torch.no_grad()
def collect_noref_feature_diagnostics(model, inputs, config, variant):
    """Fixed sigma=.5/t=500 forwards, with no optimizer or backward calls."""
    if variant not in VARIANTS or config.get("reference", {}).get("enabled", False):
        raise ValueError("Expected a no-reference ablation variant")
    if variant == "Base":
        return {"status": "not_applicable", "variant": "Base", "reason": "Base has no projected student feature space", "optimizer_updates": 0}
    if hasattr(model, "reference_adapter"):
        raise ValueError("No-reference diagnostic cannot use a reference-trained model")
    geometry, samples = inputs["geometry"], inputs["validation"]
    if config.get("reproducibility", {}).get("mode") != "strict" or not samples:
        raise ValueError("Requires strict fixed validation diagnostics")
    if any(sample["metadata"]["split"] != "val" for sample in samples) or len({s["metadata"]["record_id"] for s in samples}) != len(samples):
        raise ValueError("Only distinct fixed validation samples may be diagnosed")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Diagnostic model must begin without parameter gradients")
    device = next(model.parameters()).device
    flags = [(module, module.training) for module in model.modules()]
    requirements = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    state_before, rng_before = _state_hash(model), _rng_snapshot()
    rows = []; started = time.perf_counter()
    try:
        model.eval(); model.clear_features()
        for sample in samples:
            teacher_dim = config["conditioning"]["teacher_dim"]
            _finite(sample["teacher"], (1, math.prod(geometry["token_grid"]), teacher_dim), "fresh teacher")
            for key in ("forward", "backward"):
                _finite(sample["flows"][key], (1, len(geometry["anchors"])-1, 2, config["data"]["height"], config["data"]["width"]), "fresh HR flow")
            clean, low, text = (sample[key].detach().to(device) for key in ("clean", "low", "text"))
            if clean.shape != low.shape or tuple(clean.shape[2:]) != tuple(geometry["latent_grid"]):
                raise ValueError("Oracle noised-HR/LR latent geometry mismatch")
            if not isinstance(sample["clip_index"], int) or isinstance(sample["clip_index"], bool) or sample["clip_index"] < 0:
                raise ValueError("Full source cache clip_index is required for original diagnostic noise")
            seed = config["seed"]+91+sample["clip_index"]
            rng = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(clean.shape, generator=rng, device=device, dtype=torch.float32)
            noisy = .5*clean+.5*noise
            timestep = torch.tensor([500.], device=device)
            context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with context:
                _, student = model(noisy, timestep, text, low)
            if student is None:
                raise ValueError("Projected no-reference checkpoint returned no student features")
            teacher = sample["teacher"].detach().to(device)
            mask = torch.ones(student.shape[:2], device=device, dtype=torch.bool)
            alignment = point_alignment(student, teacher, mask)
            relation, details = compute_relation_objective(student, teacher, geometry["token_grid"],
                sample["flows"]["forward"].detach(), sample["flows"]["backward"].detach(), mask, config)
            if not bool(torch.isfinite(alignment)) or not bool(torch.isfinite(relation)):
                raise ValueError("Nonfinite validation feature objective")
            rows.append({"record_id": sample["metadata"]["record_id"], "source_id": sample["metadata"]["source_id"],
                "split": "val", "source_clip_index": sample["clip_index"], "noise_seed": seed,
                "source_frame_indices": sample["metadata"].get("source_indices"),
                "sampled_fps": sample["metadata"].get("output_fps"),
                "alignment_error": float(alignment), "relation_objectives": {
                    "spatial": details["spatial_loss"] if details["spatial_edges"] else None,
                    "temporal": details["temporal_loss"] if details["temporal_edges"] else None,
                    "combined": float(relation) if details["active_edge_types"] else None},
                "relation_diagnostics": details, "statistics": feature_statistics(student, teacher, mask),
                "same_anchor_patch_retrieval": same_anchor_patch_identity_retrieval(student, teacher, mask, geometry["token_grid"]),
                "HR_descriptors": sample["HR_descriptors"]})
            model.clear_features()
    finally:
        model.zero_grad(set_to_none=True); model.clear_features()
        for module, training in flags:
            module.training = training
        state_after, rng_after = _state_hash(model), _rng_snapshot(); _restore_rng(rng_before)
        cleanup = {"parameters_buffers_unchanged": state_before == state_after,
            "global_rng_not_consumed": _rng_equal(rng_before, rng_after), "global_rng_restored": _rng_equal(rng_before, _rng_snapshot()),
            "module_flags_restored": all(module.training == value for module, value in flags),
            "requires_grad_unchanged": all(p.requires_grad == value for p, value in requirements),
            "parameter_gradients_cleared": all(p.grad is None for p in model.parameters()),
            "condition_and_feature_context_cleared": all(getattr(model, key, None) is None for key in ("_condition_tokens", "_captured"))}
        if not all(cleanup.values()):
            raise RuntimeError(f"No-reference diagnostic state cleanup failed: {cleanup}")
    scalar_keys = ("valid_point_cosine_error", "student_mean_channel_variance", "teacher_mean_channel_variance", "student_effective_rank", "teacher_effective_rank")
    relation_macro = {}
    for key in ("spatial", "temporal", "combined"):
        valid = [row["relation_objectives"][key] for row in rows if row["relation_objectives"][key] is not None]
        relation_macro[key] = {"video_count_with_edges": len(valid), "video_mean": sum(valid)/len(valid) if valid else None}
    return {"status": "completed", "variant": variant, "protocol": {"scope": "oracle_noised_HR mechanism diagnostic, not LR-only restoration",
        "sigma": .5, "timestep": 500., "noise": "FP32 local Generator on model device; seed=config.seed+91+full source cache clip_index; matches original noref auxiliary_audit",
        "validation_order": "complete originating fixed val32 in original selection order", "optimizer_updates": 0, "backward_calls": 0,
        "target_source": "fresh official DINO/RAFT from same bound val HR; never old unsealed target payloads",
        "relation": "each edge type separately normalized, then mean of nonempty types; absent components null in diagnostics",
        "restoration_quality_evidence": False, "caveat": "alignment, relations, rank, variance, retrieval or descriptors do not establish recovery benefit or universal information equivalence"},
        "per_video": rows, "aggregate": {"video_count": len(rows),
            "alignment_error": sum(row["alignment_error"] for row in rows)/len(rows), "relation_objectives": relation_macro,
            "same_anchor_spatial_patch_top1": sum(row["same_anchor_patch_retrieval"]["same_anchor_spatial_patch_top1_identity_rate"] for row in rows)/len(rows),
            **{key: sum(row["statistics"][key] for row in rows)/len(rows) for key in scalar_keys}},
        "state_sha256_before": state_before, "state_sha256_after": state_after, "cleanup": cleanup,
        "wall_seconds": time.perf_counter()-started}


def verify_numeric_runtime(origin, current_environment, *, project_root=ROOT):
    root = Path(project_root); previous = _json(Path(origin)/"environment.json")
    matched = {}
    for name in NUMERIC_MODULES:
        actual = file_hash(root/name)
        if previous.get("code_sha256", {}).get(name) != actual:
            raise ValueError(f"Numeric source differs from the trained checkpoint: {name}")
        matched[name] = actual
    for name in ("torch", "torchvision", "diffusers", "transformers", "numpy"):
        version = previous.get("packages", {}).get(name)
        if not isinstance(version, str) or not version or version != current_environment.get("packages", {}).get(name):
            raise ValueError(f"Numeric dependency version changed: {name}")
    sources = previous.get("dependency_source_sha256")
    if not isinstance(sources, dict) or not sources or sources != current_environment.get("dependency_source_sha256"):
        raise ValueError("Native architecture/scheduler source changed")
    return {"numeric_module_sha256": matched, "versions_and_native_sources_match_training": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    guard = assert_gpu_idle(ROOT)
    from .infer_video import preview_training_config
    preview = preview_training_config(args.checkpoint, allowed_root=ROOT/"runs")
    config, variant = preview["config"], preview["variant"]
    origin = Path(args.checkpoint).resolve().parent.parent
    summary = _json(origin/variant/"summary.json")
    if (variant not in VARIANTS or config.get("reference", {}).get("enabled", False)
            or preview["step"] != config["training"]["updates"] or summary.get("status") != "completed"
            or summary.get("updates") != preview["step"] or summary.get("variant", variant) != variant
            or (variant != "Base" and config.get("reproducibility", {}).get("mode") != "strict")):
        raise ValueError("Only a final completed own no-reference group may be diagnosed")
    output = _within(args.output or ROOT/"runs"/("noref_feature_diagnostics_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")), ROOT/"runs")
    output.mkdir(parents=True, exist_ok=False)
    dump(output/"status.json", {"status": "running", "optimizer_updates": 0, "kind": "noref_feature_diagnostics"})
    model = None
    try:
        if variant == "Base":
            report = {"status": "not_applicable", "variant": "Base", "reason": "Base has no student projector", "optimizer_updates": 0, "checkpoint_preview": preview}
        else:
            dump(output/"reproducibility.json", configure_reproducibility(config))
            dump(output/"official_artifacts.json", verify_official(ROOT, config))
            current_environment = environment(ROOT); dump(output/"environment.json", current_environment)
            dump(output/"numeric_runtime.json", verify_numeric_runtime(origin, current_environment, project_root=ROOT))
            inputs = load_noref_validation(origin, config, project_root=ROOT)
            dump(output/"source_inputs.json", inputs["source_audit"])
            inputs = extract_fresh_targets(inputs, config, project_root=ROOT, device=args.device)
            dump(output/"fresh_target_audit.json", inputs["fresh_target_audit"])
            after_targets_guard = assert_gpu_idle(ROOT)
            model, audit = load_training_checkpoint(args.checkpoint, device=args.device, allowed_root=ROOT/"runs")
            if (audit["checkpoint_sha256"] != preview["checkpoint_sha256"] or audit["config"] != config
                    or audit["variant"] != variant or audit["step"] != preview["step"]
                    or audit["optimizer_state_restored"] or audit["rng_state_restored"]):
                raise ValueError("Checkpoint changed or non-task state was restored during setup")
            dump(output/"checkpoint_restore.json", audit)
            report = collect_noref_feature_diagnostics(model, inputs, config, variant)
            report.update(checkpoint=audit, source_audit=inputs["source_audit"], fresh_target_audit=inputs["fresh_target_audit"], post_target_idle_guard=after_targets_guard)
        report.update(idle_guard=guard, module_sha256=file_hash(Path(__file__)))
        dump(output/"diagnostics.json", report)
        dump(output/"status.json", {"status": "completed", "optimizer_updates": 0, "restoration_quality_evidence": False})
        print(json.dumps({"status": "completed", "variant": variant, "output": str(output), "optimizer_updates": 0}))
    except Exception as error:
        dump(output/"status.json", {"status": "failed", "error": str(error), "traceback": traceback.format_exc(), "optimizer_updates": 0})
        raise
    finally:
        if model is not None:
            model.zero_grad(set_to_none=True); model.clear_features(); del model
        _release(args.device)


if __name__ == "__main__":
    main()
