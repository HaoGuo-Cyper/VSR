"""Auditable LRV residual conditioning for the official Wan T2V transformer.

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
