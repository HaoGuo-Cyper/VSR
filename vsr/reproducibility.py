"""Auditable reproducibility configuration and trainable initialization hashes.

Seeded mode records settings without changing them. Strict mode must be called
before CUDA initialization and before model construction. Configuration alone
is not evidence of bitwise repeatability; verify the actual forward/backward
path separately, including optional relation and reference supervision.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os

import torch

WORKSPACE_CONFIG = ":4096:8"
GROUPS = ("lrv", "lora", "projector", "hri", "other_trainable")


def reproducibility_flags():
    """Inspect flags without initializing CUDA or changing any runtime state."""
    return {
        "torch_version": str(torch.__version__), "torch_cuda_version": torch.version.cuda,
        "cuda_initialized": torch.cuda.is_initialized(),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "matmul_allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "sdp_flash": torch.backends.cuda.flash_sdp_enabled(),
        "sdp_memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "sdp_cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        "sdp_math": torch.backends.cuda.math_sdp_enabled(),
        "sdp_math_allow_fp16_bf16_reduction": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        "cpu_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads(),
    }


def configure_reproducibility(config):
    """Configure full config['reproducibility']['mode'] as seeded or strict.

    Default seeded mode leaves backend flags, environment and RNG state intact;
    the training entry point remains responsible for its explicit random seeds.
    Strict does not silently reset an already initialized wrong cuBLAS setup.
    """
    if not isinstance(config, dict):
        raise ValueError("Reproducibility configuration must be a dictionary")
    options = config.get("reproducibility", {})
    if not isinstance(options, dict):
        raise ValueError("config.reproducibility must be a dictionary")
    mode = options.get("mode", "seeded")
    if mode not in ("seeded", "strict"):
        raise ValueError("Reproducibility mode must be 'seeded' or 'strict'")
    before = reproducibility_flags()
    if mode == "strict":
        if before["cuda_initialized"] and before["CUBLAS_WORKSPACE_CONFIG"] != WORKSPACE_CONFIG:
            raise RuntimeError("Strict reproducibility requires CUBLAS_WORKSPACE_CONFIG=':4096:8' before CUDA initialization; start a fresh process")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = WORKSPACE_CONFIG
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    return {
        "mode": mode, "before": before, "flags": reproducibility_flags(),
        "behavior_changed": mode == "strict", "rng_state_modified": False,
        "bitwise_reproducibility_verified": False,
        "strict_relation_requirement": "CUDA grid_sample backward used by relation supervision is unsupported under strict determinism; route relation loss/grid sampling to a verified CPU path",
        "interpretation": "Flags are configuration evidence only; actual same-state forward/backward and complete training repeatability must be tested",
    }


def _group(name):
    parts = name.split(".")
    if any(part in ("reference_adapter", "hri_adapter") for part in parts):
        return "hri"
    if any(part.startswith("lrv_") for part in parts):
        return "lrv"
    if any(part == "lora" or part.startswith("lora_") for part in parts):
        return "lora"
    if "projector" in parts:
        return "projector"
    return "other_trainable"


def tensor_state_hashes(model):
    """Hash only trainable parameters, once immediately after initialization.

    Names, dtype, shape and raw bytes participate in each hash. Frozen official
    tensors are counted but never copied to CPU or included in these hashes.
    Callers should save this audit before evaluation, any optimizer update or
    task-checkpoint restoration, and label later snapshots as later state.
    """
    grouped = defaultdict(list)
    frozen_tensors = frozen_elements = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            grouped[_group(name)].append((name, parameter))
        else:
            frozen_tensors += 1
            frozen_elements += parameter.numel()
    all_digest = hashlib.sha256()
    result = {}
    ordered = sorted((name, p) for pairs in grouped.values() for name, p in pairs)
    # Copy each task parameter only once; feed group and combined hash together.
    digests = {group: hashlib.sha256() for group in GROUPS}
    for name, parameter in ordered:
        header = json.dumps({"name": name, "dtype": str(parameter.dtype), "shape": list(parameter.shape)}, sort_keys=True, separators=(",", ":")).encode()
        cpu = parameter.detach().to(device="cpu").contiguous()
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        for digest in (all_digest, digests[_group(name)]):
            digest.update(len(header).to_bytes(8, "big")); digest.update(header)
            digest.update(len(raw).to_bytes(8, "big")); digest.update(raw)
    for group in GROUPS:
        pairs = grouped[group]
        result[group] = {"sha256": digests[group].hexdigest(), "tensor_count": len(pairs),
                         "numel": sum(p.numel() for _, p in pairs), "names": sorted(name for name, _ in pairs),
                         "dtypes": sorted({str(p.dtype) for _, p in pairs})}
    return {"schema": "vsr_trainable_state_sha256_v1", "groups": result,
            "all_trainable": {"sha256": all_digest.hexdigest(), "tensor_count": len(ordered),
                              "numel": sum(p.numel() for _, p in ordered)},
            "frozen_tensor_count": frozen_tensors, "frozen_numel": frozen_elements,
            "frozen_tensors_copied": False, "includes_names_shapes_dtypes_raw_bytes": True,
            "bitwise_training_reproducibility_verified": False}
