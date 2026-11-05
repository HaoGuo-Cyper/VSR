"""CPU-testable self-distillation primitives; not integrated into training.

Pass the restoration model (LoRA/LR trainables) to TrainableEMA. Keep the
student-only projector outside that model. Run the temporary teacher swap
BEFORE building the student graph: copy_ correctly increments version counters,
so swapping parameters underneath an outstanding student graph is unsupported.
No model, optimizer, CUDA context, checkpoint or background worker is created.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


def _require(value, message):
    if not value:
        raise ValueError(message)


def _trainables(model):
    pairs = dict(model.named_parameters())
    result = {name: parameter for name, parameter in pairs.items() if parameter.requires_grad}
    _require(result, "Restoration model has no trainable parameters")
    return result


def _frozen_signature(model):
    return {name: (id(p), p.data_ptr(), p._version, tuple(p.shape), p.dtype, p.device)
            for name, p in model.named_parameters() if not p.requires_grad}


class TrainableEMA:
    """EMA of exactly the captured restoration trainables, initialized to Base.

    Floating dtypes/shapes/names are exact contracts. Frozen backbone parameters
    are never copied into the EMA or modified by it. update() is explicit; no
    optimizer operation occurs here. State export returns detached copies.
    """

    def __init__(self, model, decay=0.999):
        _require(type(decay) in (int, float) and math.isfinite(decay) and 0 <= decay < 1, "EMA decay must lie in [0,1)")
        self.decay, self.num_updates = float(decay), 0
        params = _trainables(model)
        self._spec = {name: (tuple(p.shape), p.dtype, p.device) for name, p in params.items()}
        _require(all(p.is_floating_point() and bool(torch.isfinite(p).all()) for p in params.values()), "EMA initialization requires finite floating trainables")
        self._shadow = {name: p.detach().clone() for name, p in params.items()}
        self._swapping = False

    @property
    def names(self):
        return tuple(self._spec)

    def _validate(self, model):
        params = _trainables(model)
        _require(set(params) == set(self._spec), "EMA trainable parameter names changed")
        for name, p in params.items():
            _require((tuple(p.shape), p.dtype, p.device) == self._spec[name], "EMA shape/dtype/device changed: " + name)
            _require(bool(torch.isfinite(p).all()), "Nonfinite student trainable: " + name)
            shadow = self._shadow[name]
            _require((tuple(shadow.shape), shadow.dtype, shadow.device) == self._spec[name]
                     and not shadow.requires_grad and bool(torch.isfinite(shadow).all()), "Invalid EMA shadow: " + name)
        return params

    @torch.no_grad()
    def update(self, model):
        _require(not self._swapping, "Cannot update EMA during a temporary teacher swap")
        params = self._validate(model)  # validate every entry before mutation
        frozen = _frozen_signature(model)
        for name, p in params.items():
            self._shadow[name].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        self.num_updates += 1
        _require(frozen == _frozen_signature(model), "Frozen parameters changed during EMA update")

    def state_dict(self):
        return {"schema": "vsr_trainable_only_ema_v1", "decay": self.decay,
                "num_updates": self.num_updates, "trainable_names": list(self.names),
                "shadow": {name: value.detach().clone() for name, value in self._shadow.items()}}

    def load_state_dict(self, state):
        _require(not self._swapping, "Cannot load EMA state during a teacher swap")
        _require(state.get("schema") == "vsr_trainable_only_ema_v1" and state.get("decay") == self.decay,
                 "EMA state schema/decay mismatch")
        _require(type(state.get("num_updates")) is int and state["num_updates"] >= 0, "Invalid EMA update count")
        _require(state.get("trainable_names") == list(self.names) and isinstance(state.get("shadow"), dict)
                 and set(state["shadow"]) == set(self._spec), "EMA state parameter names mismatch")
        staged = {}
        for name, value in state["shadow"].items():
            _require(isinstance(value, torch.Tensor) and (tuple(value.shape), value.dtype, value.device) == self._spec[name]
                     and not value.requires_grad and bool(torch.isfinite(value).all()), "EMA state shape/dtype/device/value mismatch: " + name)
            staged[name] = value.detach().clone()
        self._shadow, self.num_updates = staged, state["num_updates"]

    @contextmanager
    def teacher_parameters(self, model):
        """Temporarily use EMA under eval/no_grad; always restore student/modes.

        Use before the student forward. Existing student gradients are retained.
        Caller-side modifications to frozen parameter versions are detected.
        """
        _require(not self._swapping, "Nested teacher swaps are forbidden")
        params = self._validate(model)
        saved = {name: p.detach().clone() for name, p in params.items()}
        flags = [(module, module.training) for module in model.modules()]
        frozen = _frozen_signature(model)
        self._swapping = True
        try:
            with torch.no_grad():
                for name, p in params.items():
                    p.copy_(self._shadow[name])
                model.eval()
                yield
        finally:
            with torch.no_grad():
                for name, p in params.items():
                    p.copy_(saved[name])
            for module, training in flags:
                module.training = training
            self._swapping = False
            _require(frozen == _frozen_signature(model), "Frozen parameters changed during the teacher call")

    def teacher_forward(self, model, call, *args, **kwargs):
        """Call a supplied forward/hook capture and return detached owned values."""
        def own(value):
            if isinstance(value, torch.Tensor):
                return value.detach().clone()
            if isinstance(value, tuple):
                return tuple(own(item) for item in value)
            if isinstance(value, list):
                return [own(item) for item in value]
            if isinstance(value, dict):
                return {key: own(item) for key, item in value.items()}
            _require(value is None or type(value) in (bool, int, float, str), "Unsupported teacher result type")
            return value
        with self.teacher_parameters(model):
            return own(call(*args, **kwargs))


@dataclass(frozen=True)
class SpanMask:
    mask: torch.Tensor
    spans: tuple
    rho_requested: float
    span: int
    masked_positions_per_example: int
    realized_ratio_all_latents: float
    realized_ratio_eligible_latents: float
    rounding: str = "nearest whole-span count to rho*T, half-up; cap at floor((T-1)/span); latent0 excluded"


def sample_span_mask(batch_size, latent_frames=9, *, rho=0.25, span=2, generator):
    """CPU mask with uniformly sampled nonoverlapping spans; never selects 0.

    rho's denominator is all T latent positions, including the protected first
    position. T=9/rho=.25/span=2 therefore selects 2/9, not an exact .25. All
    feasible arrangements of the chosen number of equal-length spans are
    sampled through compressed sorted start positions, without greedy failure.
    """
    _require(type(batch_size) is int and batch_size > 0 and type(latent_frames) is int and latent_frames > 0, "Positive integer batch/time sizes required")
    _require(type(span) is int and span > 0, "Span must be a positive integer")
    _require(type(rho) in (int, float) and math.isfinite(rho) and 0 <= rho <= 1, "rho must lie in [0,1]")
    _require(isinstance(generator, torch.Generator) and generator.device.type == "cpu", "Use an explicit independent CPU mask generator")
    eligible = latent_frames - 1
    _require(rho == 0 or span <= eligible, "Span cannot fit the eligible latent positions")
    count = min(int(math.floor(float(rho) * latent_frames / span + .5)), eligible // span)
    mask = torch.zeros(batch_size, latent_frames, dtype=torch.bool)
    spans = []
    for example in range(batch_size):
        starts = []
        if count:
            slots = eligible - count * (span - 1)
            compressed = torch.randperm(slots, generator=generator)[:count].sort().values.tolist()
            starts = [1 + value + index * (span - 1) for index, value in enumerate(compressed)]
            for start in starts:
                mask[example, start:start + span] = True
        spans.append(tuple((start, start + span) for start in starts))
    masked = count * span
    return SpanMask(mask, tuple(spans), float(rho), span, masked, masked / latent_frames,
                    masked / eligible if eligible else 0.)


def _latent_mask(mask, batch, frames):
    _require(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool and tuple(mask.shape) == (batch, frames), "Latent mask must be bool [B,T]")
    _require(not bool(mask[:, 0].any()), "The first latent position must remain clean")


def corrupt_latent_condition(condition, mask, *, generator, mode="matched_noise"):
    """Return a new condition; the original cached tensor is never mutated.

    Matched-noise statistics are detached per-example/channel population mean
    and SD over the original T,H,W. Noise matches these moments in expectation,
    not exactly in a finite draw. A zero-variance channel stays constant. A zero
    mask returns an exact clone and does not advance the noise generator.
    """
    _require(isinstance(condition, torch.Tensor) and condition.ndim == 5 and condition.is_floating_point()
             and all(condition.shape) and not condition.requires_grad and bool(torch.isfinite(condition).all()), "Condition must be finite frozen floating [B,C,T,H,W]")
    batch, _, frames, _, _ = condition.shape
    _latent_mask(mask, batch, frames)
    _require(mode in {"matched_noise", "zero"}, "Unknown latent corruption mode")
    _require(isinstance(generator, torch.Generator) and generator.device.type == condition.device.type, "Noise generator must match condition device type")
    if not bool(mask.any()):
        return condition.clone()
    if mode == "zero":
        replacement = torch.zeros_like(condition)
    else:
        source = condition.detach().float()
        mean = source.mean(dim=(2, 3, 4), keepdim=True)
        std = source.var(dim=(2, 3, 4), correction=0, keepdim=True).sqrt()
        noise = torch.randn(condition.shape, generator=generator, device=condition.device, dtype=torch.float32)
        replacement = (noise * std + mean).to(condition.dtype)
    return torch.where(mask.to(condition.device)[:, None, :, None, None], replacement, condition)


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

    def forward(self, student, teacher, latent_mask, grid, normalized_timestep):
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
        return loss, {"selected_tokens": selected, "cosine_mean": cosines.detach().mean(),
                      "window": "normalized_t > threshold", "threshold": self.threshold}
