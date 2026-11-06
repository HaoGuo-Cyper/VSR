"""Trainable-only EMA teacher and safe intermediate-feature capture.

Teacher execution finishes before building the student graph. Importing this
module never loads a model, initializes CUDA or starts an experiment.
"""
from __future__ import annotations
from contextlib import contextmanager, nullcontext
import math
import torch

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


def _require_finite(entries):
    """Check every (failure message, tensor), with one verdict per device.

    Shape/name/dtype/device contracts belong to the caller. All finite-value
    reductions stay on their tensor's device until a single combined verdict
    is read. Only a failing device needs individual reads to locate the first
    offender. This changes synchronization, not the set of values checked.
    No tensor is mutated, so callers can validate before committing a change.
    """
    per_device = {}
    for message, value in entries:
        _require(isinstance(value, torch.Tensor), "Finite validation requires tensors")
        verdict = torch.isfinite(value.detach()).all()
        per_device.setdefault(value.device, []).append((message, verdict))
    for checks in per_device.values():
        if not bool(torch.stack([verdict for _, verdict in checks]).all()):
            for message, verdict in checks:
                _require(bool(verdict), message)


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
        _require(all(p.is_floating_point() for p in params.values()), "EMA initialization requires finite floating trainables")
        _require_finite(("EMA initialization requires finite floating trainables: " + name, p)
                        for name, p in params.items())
        self._shadow = {name: p.detach().clone() for name, p in params.items()}
        self._swapping = False

    @property
    def names(self):
        return tuple(self._spec)

    def _validate(self, model):
        params = _trainables(model)
        _require(set(params) == set(self._spec), "EMA trainable parameter names changed")
        _require(set(self._shadow) == set(self._spec), "EMA shadow parameter names changed")
        finite = []
        for name, p in params.items():
            _require((tuple(p.shape), p.dtype, p.device) == self._spec[name], "EMA shape/dtype/device changed: " + name)
            shadow = self._shadow[name]
            _require(isinstance(shadow, torch.Tensor) and (tuple(shadow.shape), shadow.dtype, shadow.device) == self._spec[name]
                     and not shadow.requires_grad, "Invalid EMA shadow: " + name)
            finite.extend((("Nonfinite student trainable: " + name, p),
                           ("Invalid EMA shadow: " + name, shadow)))
        _require_finite(finite)
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
        for name, value in state["shadow"].items():
            _require(isinstance(value, torch.Tensor) and (tuple(value.shape), value.dtype, value.device) == self._spec[name]
                     and not value.requires_grad, "EMA state shape/dtype/device/value mismatch: " + name)
        _require_finite(("EMA state shape/dtype/device/value mismatch: " + name, value)
                        for name, value in state["shadow"].items())
        staged = {name: value.detach().clone() for name, value in state["shadow"].items()}
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


def _autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def capture_block_output(model, block_index, noisy, timestep, text, low):
    """Observe one output, then remove the hook before checkpoint replay.

    Does not clear the model's LR feature graph. Student callers must retain it
    through backward and clear it afterward. A teacher caller may clear it as
    soon as its no-grad result has been obtained.
    """
    _require(type(block_index) is int and 0 <= block_index < len(model.transformer.blocks), "Invalid captured block")
    captured = []

    def hook(_module, _args, output):
        _require(isinstance(output, torch.Tensor) and output.ndim == 3, "Expected Wan block output [B,N,C]")
        captured.append(output)

    handle = model.transformer.blocks[block_index].register_forward_hook(hook)
    try:
        with _autocast(noisy.device):
            flow, projection = model(noisy, timestep, text, low)
        _require(projection is None and len(captured) == 1, "Base block must be observed exactly once on the forward pass")
        return flow, captured[0]
    finally:
        handle.remove()
