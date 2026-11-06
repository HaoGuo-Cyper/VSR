"""Wan, LoRA and LR residual conditioning, model construction and reproducibility.

No pretrained task adapters are accepted here. The caller loads official Wan
weights, owns VAE/text encoding and autocast, and performs one forward/backward
at a time. These block hooks retain the LRV graph through non-reentrant gradient
checkpoint recomputation; call ``clear_features`` after backward to release it.
"""

from __future__ import annotations

import functools
from collections import defaultdict

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


LORA_TARGETS = r"blocks\.\d+\.attn[12]\.(to_q|to_k|to_v|to_out\.0)"


class ConditionalWan(nn.Module):
    """Add a zero-initialized LR branch, fixed attention LoRA and optional MLP.

    ``forward(noisy, timestep, text, lrv_latent, enable_lrv=True)`` returns
    ``(flow, projected_tokens)``. Tokens are ``[B,T*H*W,teacher_dim]`` in Wan's
    time/height/width flattening order; ``token_grid`` records that grid.
    ``projection=False`` gives the Base architecture and returns ``None``.

    Layer indices are zero based. The default alignment output is after block
    9 (the tenth of thirty blocks), which has received all three LR injections.
    """

    def __init__(
        self,
        transformer,
        *,
        rank: int = 8,
        alpha: int = 8,
        injection_layers=(0, 5, 9),
        alignment_layer: int = 9,
        stem_channels: int = 128,
        projection: bool = True,
        projector_hidden: int = 1024,
        teacher_dim: int = 768,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        from peft import LoraConfig

        if getattr(transformer, "peft_config", {}):
            raise ValueError("Expected a freshly loaded official transformer without adapters")
        if rank <= 0 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        layers = tuple(int(i) for i in injection_layers)
        n_blocks = len(transformer.blocks)
        if not layers or len(set(layers)) != len(layers) or any(i < 0 or i >= n_blocks for i in layers):
            raise ValueError("Invalid or duplicate LRV injection layer")
        if not 0 <= alignment_layer < n_blocks or alignment_layer < min(layers):
            raise ValueError("Alignment must follow an LRV injection")
        if transformer.config.in_channels != transformer.config.out_channels:
            raise ValueError("Expected the unmodified T2V latent input and output channels")
        self.transformer = transformer
        transformer.requires_grad_(False)
        transformer.add_adapter(
            LoraConfig(
                r=rank, lora_alpha=alpha, target_modules=LORA_TARGETS,
                lora_dropout=0.0, bias="none", init_lora_weights="gaussian",
            ),
            adapter_name="vsr_fresh",
        )
        for name, parameter in transformer.named_parameters():
            trainable = ".lora_A." in name or ".lora_B." in name
            parameter.requires_grad_(trainable)
            if trainable:
                parameter.data = parameter.data.float()

        dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
        patch = tuple(transformer.config.patch_size)
        self.lrv_stem = nn.Sequential(
            nn.Conv3d(transformer.config.in_channels, stem_channels, kernel_size=patch, stride=patch),
            nn.SiLU(),
        )
        self.lrv_residuals = nn.ModuleDict({str(i): nn.Linear(stem_channels, dim) for i in layers})
        for residual in self.lrv_residuals.values():
            nn.init.zeros_(residual.weight)
            nn.init.zeros_(residual.bias)
        self.projector = (
            nn.Sequential(nn.Linear(dim, projector_hidden), nn.SiLU(), nn.Linear(projector_hidden, teacher_dim))
            if projection else None
        )
        self.injection_layers = layers
        self.alignment_layer = alignment_layer
        self.token_grid = None
        self._condition_tokens = None
        self._captured = None
        self._capture_enabled = False
        self._hooks = []
        for index in layers:
            self._hooks.append(transformer.blocks[index].register_forward_pre_hook(self._make_injector(index)))
        if projection:
            self._hooks.append(transformer.blocks[alignment_layer].register_forward_hook(self._capture))
        if gradient_checkpointing:
            transformer.enable_gradient_checkpointing(
                gradient_checkpointing_func=functools.partial(checkpoint, use_reentrant=False)
            )
        else:
            transformer.disable_gradient_checkpointing()
        transformer.set_attention_backend("native")

    def _make_injector(self, index):
        def inject(_module, args):
            if self._condition_tokens is None:
                return None
            hidden_states = args[0]
            if hidden_states.shape[:2] != self._condition_tokens.shape[:2]:
                raise ValueError("LRV condition and Wan hidden tokens have different grids")
            residual = self.lrv_residuals[str(index)](self._condition_tokens)
            return (hidden_states + residual.to(hidden_states.dtype), *args[1:])
        return inject

    def _capture(self, _module, _args, output):
        # Backward checkpoint replay must not replace the forward feature graph.
        if self._capture_enabled:
            self._captured = output

    def clear_features(self):
        """Release forward state only after its backward pass has completed."""
        self._captured = None
        self._condition_tokens = None

    def forward(self, noisy, timestep, text, lrv_latent, enable_lrv=True):
        self.clear_features()
        if noisy.ndim != 5 or noisy.shape != lrv_latent.shape:
            raise ValueError("Noisy HR and encoded upsampled LRV must share [B,C,T,H,W]")
        patch = tuple(self.transformer.config.patch_size)
        if any(size % stride for size, stride in zip(noisy.shape[2:], patch)):
            raise ValueError("Latent dimensions must be divisible by the DiT patch size")
        self.token_grid = tuple(size // stride for size, stride in zip(noisy.shape[2:], patch))
        if enable_lrv:
            self._condition_tokens = self.lrv_stem(lrv_latent).flatten(2).transpose(1, 2)
        self._capture_enabled = self.projector is not None
        try:
            flow = self.transformer(
                hidden_states=noisy, timestep=timestep, encoder_hidden_states=text, return_dict=False
            )[0]
        finally:
            self._capture_enabled = False
        tokens = None
        if self.projector is not None:
            if self._captured is None:
                raise RuntimeError("Alignment layer did not execute during forward")
            tokens = self.projector(self._captured)
            self._captured = None
        return flow, tokens


def gradient_norms(model: nn.Module):
    """Return finite L2 gradient norms and counts for the trainable groups."""
    squared = defaultdict(float)
    counts = defaultdict(int)
    finite = defaultdict(lambda: True)
    for name, parameter in model.named_parameters():
        if name.startswith("lrv_stem."):
            group = "lrv_stem"
        elif name.startswith("lrv_residuals."):
            group = "lrv_residuals"
        elif name.startswith("projector."):
            group = "projector"
        elif name.startswith("reference_adapter."):
            group = "hri"
        elif ".lora_" in name:
            group = "lora"
        else:
            group = "frozen"
        if parameter.grad is not None:
            value = parameter.grad.detach().float()
            counts[group] += parameter.numel()
            finite[group] = finite[group] and bool(torch.isfinite(value).all())
            squared[group] += float(value.square().sum())
    return {
        group: {"l2": squared[group] ** 0.5, "elements_with_grad": counts[group], "finite": finite[group]}
        for group in ("lrv_stem", "lrv_residuals", "lora", "projector", "hri", "frozen")
    }


# Reproducibility belongs to the model construction boundary.
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


METHOD_ARMS = ('flow_only', 'C1', 'A2')
NOREF_BASE_ARCHITECTURE = 'noref_base_v1'


def validate_restoration_architecture(config, variant):
    """Validate the NoRef mainline; the training-only projector lives in losses."""
    if not isinstance(config, dict) or variant not in ('Base', *METHOD_ARMS):
        raise ValueError('Mainline supports only NoRef Base/flow_only/C1/A2; historical variants remain archived')
    if config.get('reference', {}).get('enabled', False) is not False or config.get('conditioning', {}).get('hri_enabled', False) is not False:
        raise ValueError('Mainline forbids reference.enabled and conditioning.hri_enabled')
    marker = config.get('restoration_architecture')
    if marker is None:
        if variant != 'Base':
            raise ValueError('Method variants require explicit restoration_architecture=noref_base_v1')
        return False
    if marker != NOREF_BASE_ARCHITECTURE:
        raise ValueError('Unknown restoration_architecture marker')
    variants = config.get('training', {}).get('variants')
    if (not isinstance(variants, list) or not variants or len(variants) != len(set(variants))
            or any(name not in METHOD_ARMS for name in variants) or variant not in variants):
        raise ValueError('noref_base_v1 requires distinct flow_only/C1/A2 variants')
    return False


def seed_all(seed, device='cpu'):
    """Initialize a construction seed without touching CUDA on CPU paths."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.device(device).type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def make_model(config, variant, device='cpu'):
    """Build a fresh official Wan and unchanged NoRef task parameter names.

    Loading pretrained weights happens only when explicitly called. No text
    encoder, VAE, optimizer or task checkpoint is loaded by this constructor.
    """
    projection = validate_restoration_architecture(config, variant)
    from diffusers import WanTransformer3DModel
    training_seed = config['training'].get('seed', config['seed'])
    seed_all(training_seed, device)
    native, info = WanTransformer3DModel.from_pretrained(
        config['model']['local_dir'], subfolder='transformer',
        dtype=torch.bfloat16 if torch.device(device).type == 'cuda' else torch.float32,
        local_files_only=True, use_safetensors=True, output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'Official initialization loading mismatch: {info}')
    original_count = sum(p.numel() for p in native.parameters())
    for name, parameter in native.named_parameters():
        if any(key in name for key in (native._keep_in_fp32_modules or [])) and parameter.dtype != torch.float32:
            raise ValueError(f'Native FP32 exception changed: {name}')
    conditioning = config['conditioning']
    options = {key: conditioning[key] for key in ('rank', 'alpha', 'injection_layers', 'alignment_layer', 'stem_channels', 'projector_hidden', 'teacher_dim')}
    options.update(projection=projection, gradient_checkpointing=config['training']['gradient_checkpointing'])
    model = ConditionalWan(native, **options).to(device)
    trainables = {name: parameter.numel() for name, parameter in model.named_parameters() if parameter.requires_grad}
    return model, {'official_revision': config['model']['revision'], 'loading_info': info,
        'native_parameter_count': original_count, 'trainable_parameters': trainables,
        'trainable_count': sum(trainables.values()), 'seed': training_seed,
        'data_seed': config['seed'], 'training_seed': training_seed,
        'reference_enabled': False, 'trainable_initialization_hashes': tensor_state_hashes(model),
        'fresh_official_initialization': True, 'legacy_state_loaded': False, 'optimizer_fresh': True}
