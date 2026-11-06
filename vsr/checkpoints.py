"""Restore task weights from new VSR runs onto a fresh official Wan base.

This loader is for evaluation/inference and never resumes optimizer, scheduler
or RNG state. Training comparisons must still initialize independently using
make_model; loading a task checkpoint must be disclosed as its own origin.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from .reproducibility import configure_reproducibility
from .restoration_architecture import validate_restoration_architecture, METHOD_ARMS

PRODUCTION_RUNS_ROOT = Path("/home/hugo/Projects/VSR/runs")
LEGACY_PROJECT_ROOT = Path("/home/hugo/Projects/ReprVSR-pretrained")
OFFICIAL_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
OFFICIAL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_checkpoint(path, allowed_root):
    path = Path(path).resolve()
    if path.is_relative_to(LEGACY_PROJECT_ROOT):
        raise ValueError("Legacy project checkpoints are forbidden in this experiment")
    if not path.is_relative_to(Path(allowed_root).resolve()):
        raise ValueError("Checkpoint path escapes this experiment's allowed run root")
    if not path.is_file():
        raise FileNotFoundError(f"New VSR checkpoint missing: {path}")
    return path


def _check_config(config):
    if not isinstance(config, dict):
        raise ValueError("Training checkpoint config must be a dictionary")
    model = config.get("model", {})
    if model.get("repo_id") != OFFICIAL_REPO_ID or model.get("revision") != OFFICIAL_REVISION:
        raise ValueError("Checkpoint official Wan repo/revision differs from pinned 1.3B")
    source_hash = config.get("_sha256")
    if config.get("restoration_architecture") is not None and source_hash is None:
        raise ValueError("Method architecture requires canonical config _sha256")
    if source_hash is not None:
        value = {key: item for key, item in config.items() if key != "_sha256"}
        canonical = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
        actual = hashlib.sha256(canonical).hexdigest()
        if source_hash != actual:
            raise ValueError("Checkpoint source config SHA256 mismatch")
    variants = config.get("training", {}).get("variants")
    if not isinstance(variants, list) or not variants or any(not isinstance(v, str) or not v for v in variants) or len(variants) != len(set(variants)):
        raise ValueError("Checkpoint config must identify distinct training variants")
    for variant in variants:
        validate_restoration_architecture(config,variant)
    return source_hash


def load_training_checkpoint(path, *, device="cuda", allowed_root=PRODUCTION_RUNS_ROOT,
                             model_factory=None, variant=None):
    """Return (model,meta), restoring only the exact trainable parameter set.

    A custom model_factory(config,variant) must return (model,initialization)
    and construct a fresh official model. The default imports train.make_model
    lazily. No task state is copied until every key/shape/dtype is checked.
    Floating checkpoint tensors are copied into each destination float dtype;
    source and destination dtype sets are included in the returned audit.
    """
    path = _safe_checkpoint(path, allowed_root)
    checkpoint_sha = _sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "vsr_training_v1":
        raise ValueError("Unsupported VSR training checkpoint format")
    step = payload.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("Training checkpoint step must be an integer >=1")
    config = payload.get("config")
    source_config_sha = _check_config(config)
    selected_variant = path.parent.name if variant is None else variant
    if selected_variant not in config["training"]["variants"]:
        raise ValueError("Requested checkpoint variant is not in its training config")
    if payload.get("variant") is not None and payload["variant"] != selected_variant:
        raise ValueError("Checkpoint payload variant differs from requested variant")
    # Check the neighboring originating run config when it is present. An
    # explicit variant override still permits reviewed copies within own runs.
    run_config_path = path.parent.parent / "config.json"
    if run_config_path.is_file():
        if not run_config_path.resolve().is_relative_to(Path(allowed_root).resolve()):
            raise ValueError("Originating config path escapes allowed run root")
        source_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if source_config != config:
            raise ValueError("Checkpoint config differs from originating run config")
    state = payload.get("trainable_state")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint trainable_state must be a nonempty dictionary")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError("Task checkpoint state requires named floating tensors")
        if selected_variant in METHOD_ARMS and not (key.startswith(("lrv_stem.","lrv_residuals.")) or ".lora_" in key):
            raise ValueError("Method trainable_state must contain only restoration student LoRA/LR: "+key)
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Task checkpoint contains nonfinite tensor: {key}")
    if model_factory is None:
        runtime=configure_reproducibility(config)
        from .train import make_model
        result=make_model(config,selected_variant,device=device)
    else:
        runtime=None
        result = model_factory(config, selected_variant)
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("model_factory must return (fresh_model, initialization)")
    model, fresh_initialization = result
    if not isinstance(fresh_initialization, dict):
        raise ValueError("Fresh model factory must return initialization audit metadata")
    if fresh_initialization.get("official_revision") != OFFICIAL_REVISION or fresh_initialization.get("fresh_official_initialization") is not True or fresh_initialization.get("legacy_state_loaded") is not False:
        raise ValueError("Model factory did not attest fresh pinned official initialization")
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state) != set(parameters):
        missing = sorted(set(parameters) - set(state))
        unexpected = sorted(set(state) - set(parameters))
        raise ValueError(f"Task checkpoint trainable key mismatch; missing={missing}; unexpected={unexpected}")
    converted_state = {}
    for key, parameter in parameters.items():
        if tuple(state[key].shape) != tuple(parameter.shape):
            raise ValueError(f"Task checkpoint tensor shape mismatch: {key}")
        if not parameter.is_floating_point():
            raise ValueError(f"Destination task parameter is not floating point: {key}")
        converted = state[key].to(dtype=parameter.dtype)
        if not bool(torch.isfinite(converted).all()):
            raise ValueError(f"Task tensor overflows destination float dtype: {key}")
        converted_state[key] = converted
    # All validation precedes mutation. Frozen official parameters are excluded.
    with torch.no_grad():
        for key, parameter in parameters.items():
            parameter.copy_(converted_state[key].to(device=parameter.device))
    model = model.to(device)
    model.eval()
    metadata = {
        "checkpoint_path": str(path), "checkpoint_sha256": checkpoint_sha,
        "format": payload["format"], "step": step, "variant": selected_variant,
        "config": config, "config_sha256": source_config_sha,
        "official_repo_id": OFFICIAL_REPO_ID, "official_revision": OFFICIAL_REVISION,
        "fresh_base_initialization": fresh_initialization,
        "task_initialization_origin": "new VSR run checkpoint",
        "restored_parameter_count": sum(p.numel() for p in parameters.values()),
        "restored_parameter_names": sorted(parameters),
        "checkpoint_tensor_dtypes": sorted({str(value.dtype) for value in state.values()}),
        "destination_tensor_dtypes": sorted({str(value.dtype) for value in parameters.values()}),
        "optimizer_state_restored": False, "scheduler_state_restored": False,
        "rng_state_restored": False, "frozen_official_parameters_restored_from_task_state": False,
        "device": str(device), "training_comparison_initialization": False,
        "reproducibility":runtime,
        "restoration_architecture":config.get("restoration_architecture"),
        "inference_weights":"student",
        "training_only_state_restored":False,
        "EMA_weights_selected":False,
        "training_only_fields_present":sorted(map(str,payload.get("training_only",{}))) if isinstance(payload.get("training_only"),dict) else [],
    }
    return model, metadata
