"""Flow-matching restoration and masked student/EMA feature losses."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

def _require(value, message):
    if not value:
        raise ValueError(message)


def _latent_mask(mask, batch, frames):
    _require(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool and tuple(mask.shape) == (batch, frames), "Latent mask must be bool [B,T]")
    _require(not bool(mask[:, 0].any()), "The first latent position must remain clean")


def temporal_token_mask(mask, grid):
    """Expand [B,T] to [B,T*H*W] in the core model's T,H,W flatten order."""
    _require(len(grid) == 3 and all(type(value) is int and value > 0 for value in grid), "Positive T,H,W token grid required")
    _require(isinstance(mask, torch.Tensor) and mask.ndim == 2, "Latent mask must have B,T axes")
    _latent_mask(mask, mask.shape[0], grid[0])
    return mask[:, :, None, None].expand(-1, -1, grid[1], grid[2]).reshape(mask.shape[0], math.prod(grid))


class MaskedTemporalAlignment(nn.Module):
    """Student MLP and parameter-free teacher LayerNorm, only masked t>.3 tokens.

    This module belongs to the student optimizer, outside the model passed to
    TrainableEMA. Inputs are raw block features [B,T*H*W,C]; normalized_timestep
    is in [0,1], never native Wan [0,1000] units. Loss arithmetic is FP32.
    """

    def __init__(self, student_dim, teacher_dim, hidden_dim=1024, threshold=0.3):
        super().__init__()
        _require(all(type(value) is int and value > 0 for value in (student_dim, teacher_dim, hidden_dim)), "Positive projector dimensions required")
        _require(type(threshold) in (int, float) and 0 <= threshold < 1, "Timestep threshold must lie in [0,1)")
        self.student_dim, self.teacher_dim, self.threshold = student_dim, teacher_dim, float(threshold)
        self.projector = nn.Sequential(nn.Linear(student_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, teacher_dim))
        self.teacher_norm = nn.LayerNorm(teacher_dim, elementwise_affine=False)

    def forward(self, student, teacher, latent_mask, grid, normalized_timestep, *, capture_diagnostics=False):
        _require(type(capture_diagnostics) is bool, "capture_diagnostics must be boolean")
        _require(isinstance(student, torch.Tensor) and isinstance(teacher, torch.Tensor) and student.ndim == teacher.ndim == 3
                 and student.is_floating_point() and teacher.is_floating_point(), "Block features must be floating [B,N,C]")
        _require(student.shape[:2] == teacher.shape[:2] and student.shape[-1] == self.student_dim
                 and teacher.shape[-1] == self.teacher_dim and student.device == teacher.device,
                 "Student/teacher feature batch/token/channel/device mismatch")
        tokens = temporal_token_mask(latent_mask, grid).to(student.device)
        _require(tuple(tokens.shape) == tuple(student.shape[:2]), "Feature token count differs from the T,H,W grid")
        timestep = torch.as_tensor(normalized_timestep, device=student.device, dtype=torch.float32)
        if timestep.ndim == 0:
            timestep = timestep.expand(student.shape[0])
        _require(tuple(timestep.shape) == (student.shape[0],) and bool(torch.isfinite(timestep).all())
                 and bool(((0 <= timestep) & (timestep <= 1)).all()), "Pass one normalized [0,1] timestep per example")
        tokens = tokens & (timestep > self.threshold)[:, None]
        selected = int(tokens.sum())
        if not selected:
            # An empty differentiable slice gives zero even if unused tokens
            # contain NaN; no projector or teacher graph is constructed.
            return student.reshape(-1)[:0].sum(), {"selected_tokens": 0, "cosine_mean": None, "window": "normalized_t > threshold"}
        _require(all(p.dtype == torch.float32 for p in self.projector.parameters()), "Keep the student projector in FP32")
        selected_student, selected_teacher = student[tokens].float(), teacher[tokens].detach().float()
        _require(bool(torch.isfinite(selected_student).all()) and bool(torch.isfinite(selected_teacher).all()), "Selected block features must be finite")
        # Index before the projector/normalization: unused NaNs cannot leak into
        # parameter gradients through zero-times-NaN matrix multiplication.
        with torch.autocast(student.device.type, enabled=False):
            projected = self.projector(selected_student)
            with torch.no_grad():
                target = self.teacher_norm(selected_teacher)
            cosines = (F.normalize(projected, dim=-1) * F.normalize(target, dim=-1)).sum(-1)
            loss = (1 - cosines).mean()
        _require(bool(torch.isfinite(loss)), "Nonfinite temporal alignment loss")
        diagnostic = {"selected_tokens": selected, "cosine_mean": cosines.detach().mean(),
                      "window": "normalized_t > threshold", "threshold": self.threshold}
        if capture_diagnostics:
            # Copies are part of the actual Step. CPU rank/statistic work is
            # deferred to an explicit drain; no graph or cached input aliases.
            diagnostic["feature_snapshot"] = {name:value.detach().to(device="cpu",dtype=torch.float32,copy=True)
                for name,value in (("student_raw",selected_student),("teacher_raw",selected_teacher),
                    ("student_projected",projected),("teacher_normalized",target))}
        return loss, diagnostic



def flow_matching_loss(velocity, target):
    """Original FP32 velocity MSE; no teacher or optimizer side effects."""
    _require(isinstance(velocity,torch.Tensor) and isinstance(target,torch.Tensor)
             and velocity.shape == target.shape and velocity.device == target.device,
             "Velocity/target shape and device must match")
    result = F.mse_loss(velocity.float(), target)
    _require(bool(torch.isfinite(result)), "Nonfinite flow-matching objective")
    return result
