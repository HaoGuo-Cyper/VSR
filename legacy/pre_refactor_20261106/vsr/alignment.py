"""Auditable anchor correspondence and frozen DINOv2-B supervision.

The selected HR anchors are learning targets, not a claim that Wan latent
timesteps have disjoint receptive fields. Token order is always T, H, W.
"""

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def _field(config, name):
    if isinstance(config, Mapping):
        if name not in config:
            raise ValueError(f"model config is missing {name!r}")
        return config[name]
    if not hasattr(config, name):
        raise ValueError(f"model config is missing {name!r}")
    return getattr(config, name)


def latent_geometry(vae_config, dit_config, frames, height, width):
    """Read compression and patch geometry; reject unsupported padding.

    ``temperal_downsample`` is the spelling in the Wan VAE configuration.
    The first-frame causal convention requires T = 1 mod temporal_stride.
    Runtime latent and token shapes must additionally be checked by the runner.
    """
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
           for v in (frames, height, width)):
        raise ValueError("frames, height and width must be positive integers")
    temporal_flags = tuple(_field(vae_config, "temperal_downsample"))
    dim_mult = tuple(_field(vae_config, "dim_mult"))
    if not dim_mult or len(temporal_flags) != len(dim_mult) - 1:
        raise ValueError("VAE downsampling flags must match len(dim_mult)-1")
    if any(flag not in (True, False) for flag in temporal_flags):
        raise ValueError("VAE temporal downsampling flags must be boolean")
    temporal_stride = 2 ** sum(bool(flag) for flag in temporal_flags)
    spatial_stride = 2 ** (len(dim_mult) - 1)
    patch_size = tuple(_field(dit_config, "patch_size"))
    if len(patch_size) != 3 or any(not isinstance(p, int) or p <= 0 for p in patch_size):
        raise ValueError("DiT patch_size must contain three positive integers")
    pt, ph, pw = patch_size
    if pt != 1:
        raise ValueError("stage-1 anchor scheme requires temporal patch_size=1")
    if (frames - 1) % temporal_stride:
        raise ValueError("causal VAE requires frames = 1 mod temporal_stride")
    if height % (spatial_stride * ph) or width % (spatial_stride * pw):
        raise ValueError("height/width must be divisible by VAE stride * DiT patch")
    latent_grid = [(frames - 1) // temporal_stride + 1,
                   height // spatial_stride, width // spatial_stride]
    token_grid = [latent_grid[0] // pt, latent_grid[1] // ph, latent_grid[2] // pw]
    return {
        "temporal_stride": temporal_stride,
        "spatial_stride": spatial_stride,
        "patch_size": list(patch_size),
        "latent_grid": latent_grid,
        "token_grid": token_grid,
        "anchors": list(range(0, frames, temporal_stride)),
        "flatten_order": "T,H,W",
    }


def teacher_to_student(teacher_patch_tokens, teacher_hw, grid):
    """Map B*Ta,Ht*Wt,C patch tokens spatially, preserving anchor order.

    No temporal interpolation, CLS tokens, or center cropping is permitted.
    ``grid`` is the student token grid (Ta, Hs, Ws), not the latent pixel grid.
    """
    if teacher_patch_tokens.ndim != 3:
        raise ValueError("teacher patches must have shape B*Ta,N,C")
    if len(teacher_hw) != 2 or len(grid) != 3:
        raise ValueError("teacher_hw must be H,W and grid must be Ta,H,W")
    ht, wt = map(int, teacher_hw)
    ta, gh, gw = map(int, grid)
    if min(ht, wt, ta, gh, gw) <= 0:
        raise ValueError("all teacher and student grid dimensions must be positive")
    bt, n, channels = teacher_patch_tokens.shape
    if n != ht * wt or bt == 0 or bt % ta:
        raise ValueError("teacher shape does not match patch grid / anchor count")
    features = teacher_patch_tokens.transpose(1, 2).reshape(bt, channels, ht, wt)
    # fp32 interpolation also works on CPU when the generator uses mixed precision.
    features = F.interpolate(features.float(), size=(gh, gw), mode="bilinear", align_corners=False)
    features = features.permute(0, 2, 3, 1).reshape(bt // ta, ta * gh * gw, channels)
    return features


def point_alignment(student, teacher, mask):
    """Mean cosine distance over valid tokens, with a detached HR teacher."""
    if student.ndim != 3 or student.shape != teacher.shape:
        raise ValueError("student and teacher must have equal B,N,C shapes")
    if mask.shape != student.shape[:-1] or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean with shape B,N")
    if student.device != teacher.device or student.device != mask.device:
        raise ValueError("student, teacher and mask must be on the same device")
    if not bool(mask.any()):
        raise ValueError("point alignment needs at least one valid token")
    # Index before normalization: masked padding (including NaN padding) must
    # neither affect the mean nor contaminate gradients through multiplication.
    u = F.normalize(student[mask].float(), dim=-1)
    v = F.normalize(teacher.detach()[mask].float(), dim=-1)
    return (1 - (u * v).sum(dim=-1)).mean()


class DinoTeacher(nn.Module):
    """Local-only official DINOv2-B/14 patch features, frozen throughout.

    Source revision and checkpoint SHA256 are audited by the run manifest.
    HR input is RGB B,3,T,H,W. Full anchor images are resized without cropping
    to a 2x finer patch grid than the student, then mapped with bilinear spatial
    interpolation. Both the resize and grid multiplier are part of the protocol.
    """

    def __init__(self, repo_path, checkpoint_path, *, input_range="zero_one", patch_grid_multiplier=2):
        super().__init__()
        repo_path, checkpoint_path = Path(repo_path), Path(checkpoint_path)
        if not (repo_path / "hubconf.py").is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError("local DINOv2 source and checkpoint are required")
        if input_range not in {"zero_one", "minus_one_one"}:
            raise ValueError("input_range must be zero_one or minus_one_one")
        if not isinstance(patch_grid_multiplier, int) or patch_grid_multiplier < 1:
            raise ValueError("patch_grid_multiplier must be a positive integer")
        self.input_range = input_range
        self.patch_grid_multiplier = patch_grid_multiplier
        self.model = torch.hub.load(str(repo_path), "dinov2_vitb14", source="local", pretrained=False)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.requires_grad_(False)
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.eval()

    def train(self, mode=True):
        # An enclosing trainable model must never enable teacher stochasticity.
        return super().train(False)

    @torch.no_grad()
    def forward(self, hr, anchors, target_grid):
        if hr.ndim != 5 or hr.shape[1] != 3:
            raise ValueError("HR must have shape B,3,T,H,W")
        if len(target_grid) != 3:
            raise ValueError("target_grid must be Ta,H,W")
        ta, gh, gw = map(int, target_grid)
        anchors = tuple(anchors)
        if len(anchors) != ta or ta <= 0 or min(gh, gw) <= 0:
            raise ValueError("anchor count must equal positive student temporal grid")
        if any(not isinstance(i, int) or i < 0 or i >= hr.shape[2] for i in anchors):
            raise ValueError("anchor index outside HR video")
        if any(a >= b for a, b in zip(anchors, anchors[1:])):
            raise ValueError("anchors must preserve strict temporal order")
        if not torch.isfinite(hr).all():
            raise ValueError("HR input must be finite")
        lower = -1.0 if self.input_range == "minus_one_one" else 0.0
        if float(hr.min()) < lower - 1e-5 or float(hr.max()) > 1 + 1e-5:
            raise ValueError(f"HR outside declared {self.input_range} range")
        b, _, _, h, w = hr.shape
        frames = hr[:, :, anchors].permute(0, 2, 1, 3, 4).reshape(b * ta, 3, h, w).float()
        if self.input_range == "minus_one_one":
            frames = (frames + 1) / 2
        teacher_hw = (gh * self.patch_grid_multiplier, gw * self.patch_grid_multiplier)
        frames = F.interpolate(frames, size=(teacher_hw[0] * 14, teacher_hw[1] * 14),
                               mode="bilinear", align_corners=False, antialias=True)
        frames = (frames - self.rgb_mean.float()) / self.rgb_std.float()
        # Use layer-normalized patch tokens, never the CLS or register tokens.
        tokens = self.model.forward_features(frames)["x_norm_patchtokens"]
        if tokens.shape[-1] != 768:
            raise RuntimeError("expected official DINOv2-B feature dimension 768")
        return F.normalize(teacher_to_student(tokens, teacher_hw, target_grid), dim=-1)
