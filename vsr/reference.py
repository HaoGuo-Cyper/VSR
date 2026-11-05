"""Optional single-frame HRI cross-attention for the later reference stage.

The caller supplies only the user's encoded reference frame and its known
original-video frame index. This module neither loads HR target videos nor
chooses references. Reference dropout (default probability 0.5 in training) is
the caller's responsibility, expressed by the explicit boolean valid vector.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _batch_vector(value, batch, device, name):
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim == 0:
        tensor = tensor.expand(batch)
    if tensor.shape != (batch,):
        raise ValueError(f"{name} must be scalar or have one value per video")
    if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
        raise ValueError(f"{name} must contain integer frame values")
    return tensor.long()


def reference_position(ref_index, frame_count, dim):
    """Sinusoidal code of known index / max(T-1,1); no learned index table."""
    if dim <= 0:
        raise ValueError("Position dimension must be positive")
    half = (dim + 1) // 2
    frequencies = torch.exp(
        torch.arange(half, device=ref_index.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    relative = ref_index.float() / (frame_count.float() - 1).clamp_min(1)
    phases = 2 * math.pi * relative[:, None] * frequencies[None, :]
    return torch.stack((phases.sin(), phases.cos()), dim=-1).flatten(1)[:, :dim]


class HRIAdapter(nn.Module):
    """Produce a gated reference-attention residual for B,N,dim hidden tokens.

    Reference latent shape is B,latent_channels,1,Hlatent,Wlatent, allowing
    exactly one frame per video. Hlatent/Wlatent must be divisible by two.
    The core model decides where to add the returned residual (the proposed
    first integration point is block 9, shared by all four reference variants).

    Only valid rows are computed. Missing references yield exact zero residuals
    regardless of any placeholder values. If every reference is absent,
    ``ref_latent=None`` is accepted and no adapter parameter participates in
    autograd. New parameters remain FP32; the caller may use CUDA autocast.
    """

    def __init__(self, dim=1536, latent_channels=16, heads=12, stem_dim=128):
        super().__init__()
        if any(not isinstance(value, int) or value <= 0 for value in (dim, latent_channels, heads, stem_dim)):
            raise ValueError("Adapter dimensions and head count must be positive integers")
        if dim % heads:
            raise ValueError("Hidden dimension must be divisible by attention heads")
        self.dim = dim
        self.latent_channels = latent_channels
        self.patch_size = (1, 2, 2)
        self.stem = nn.Sequential(
            nn.Conv3d(latent_channels, stem_dim, kernel_size=self.patch_size, stride=self.patch_size),
            nn.SiLU(),
        )
        self.reference_projection = nn.Linear(stem_dim, dim)
        self.query_norm = nn.LayerNorm(dim)
        self.reference_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.output_projection = nn.Linear(dim, dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        # Exactly one zero-initialized factor. The gate starts at sigmoid(0)=.5
        # so output projection gradients can open the reference branch at step1.
        self.gate_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, hidden_tokens, ref_latent, ref_index, frame_count, valid):
        if hidden_tokens.ndim != 3 or hidden_tokens.shape[-1] != self.dim or min(hidden_tokens.shape[:2]) <= 0:
            raise ValueError("Hidden tokens must be nonempty B,N,dim")
        batch = hidden_tokens.shape[0]
        valid = torch.as_tensor(valid, device=hidden_tokens.device)
        if valid.dtype != torch.bool or valid.shape != (batch,):
            raise ValueError("Reference validity must be boolean with shape B")
        active = valid.nonzero(as_tuple=False).flatten()
        residual = torch.zeros_like(hidden_tokens)
        if active.numel() == 0:
            return residual
        if not isinstance(ref_latent, torch.Tensor) or ref_latent.ndim != 5:
            raise ValueError("An active reference requires a B,C,1,H,W latent")
        if ref_latent.shape[:3] != (batch, self.latent_channels, 1):
            raise ValueError("Provide exactly one reference frame per video with the configured channels")
        if min(ref_latent.shape[-2:]) < 2 or any(size % 2 for size in ref_latent.shape[-2:]):
            raise ValueError("Reference latent spatial dimensions must be divisible by two")
        indices = _batch_vector(ref_index, batch, hidden_tokens.device, "ref_index").index_select(0, active)
        counts = _batch_vector(frame_count, batch, hidden_tokens.device, "frame_count").index_select(0, active)
        if bool((counts < 1).any()) or bool((indices < 0).any()) or bool((indices >= counts).any()):
            raise ValueError("Active reference index must lie in the original video's frame range")
        reference = ref_latent.to(hidden_tokens.device).index_select(0, active)
        if not bool(torch.isfinite(reference).all()):
            raise ValueError("Active reference latents must be finite")
        tokens = self.stem(reference).flatten(2).transpose(1, 2)
        tokens = self.reference_projection(tokens)
        position = reference_position(indices, counts, self.dim).to(tokens.dtype)
        keys_values = self.reference_norm((tokens + position[:, None, :]).float())
        queries = self.query_norm(hidden_tokens.index_select(0, active).float())
        attended, _ = self.attention(queries, keys_values, keys_values, need_weights=False)
        active_residual = self.output_projection(attended) * self.gate_logit.sigmoid()
        return residual.index_copy(0, active, active_residual.to(hidden_tokens.dtype))
