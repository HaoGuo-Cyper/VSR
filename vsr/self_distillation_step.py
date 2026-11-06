"""Future NoRef flow/C1/A2 continuation step; staged CPU integration only.

No command, training queue, model load or CUDA initialization runs on import.
The teacher sees the SAME noised HR and native timestep as the student; only
its LR condition is clean. EMA swapping finishes before the student graph is
built. Device-batched finite checks are CPU-tested, but do not establish the
G3 <=1.5 GPU cost requirement; actual GPU profiling and gates remain pending.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .self_distillation import (TrainableEMA, MaskedTemporalAlignment,
                               sample_span_mask, corrupt_latent_condition,
                               _require_finite)


ARMS = ("flow_only", "C1", "A2")


def _need(value, message):
    if not value:
        raise ValueError(message)


def _modes(module):
    return [(child, child.training) for child in module.modules()]


def _restore_modes(flags):
    for module, training in flags:
        module.training = training


def _frozen(model):
    return {name: (id(p), p.data_ptr(), p._version, tuple(p.shape), p.dtype, p.device)
            for name, p in model.named_parameters() if not p.requires_grad}


def _autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def capture_block_output(model, block_index, noisy, timestep, text, low):
    """Observe one output, then remove the hook before checkpoint replay.

    Does not clear the model's LR feature graph. Student callers must retain it
    through backward and clear it afterward. A teacher caller may clear it as
    soon as its no-grad result has been obtained.
    """
    _need(type(block_index) is int and 0 <= block_index < len(model.transformer.blocks), "Invalid captured block")
    captured = []

    def hook(_module, _args, output):
        _need(isinstance(output, torch.Tensor) and output.ndim == 3, "Expected Wan block output [B,N,C]")
        captured.append(output)

    handle = model.transformer.blocks[block_index].register_forward_hook(hook)
    try:
        with _autocast(noisy.device):
            flow, projection = model(noisy, timestep, text, low)
        _need(projection is None and len(captured) == 1, "Base block must be observed exactly once on the forward pass")
        return flow, captured[0]
    finally:
        handle.remove()


class MatchedStepRNG:
    """Independent explicit streams; auxiliary draws cannot perturb FM draws."""

    def __init__(self, seed, device):
        _need(type(seed) is int and seed >= 0, "Explicit nonnegative continuation seed required")
        self.seed, self.device = seed, torch.device(device)
        self.noise = torch.Generator(device=self.device).manual_seed(seed + 1)
        self.timestep = torch.Generator(device="cpu").manual_seed(seed + 2)
        self.mask = torch.Generator(device="cpu").manual_seed(seed + 3)
        self.corruption = torch.Generator(device=self.device).manual_seed(seed + 4)

    def flow_inputs(self, clean, scheduler):
        _need(clean.device == self.device, "FM generator device differs from the latent device")
        _need(len(scheduler.timesteps) == 1000 and len(scheduler.sigmas) >= 1000,
              "Use the original native UniPC 1000-index training grid")
        index = int(torch.randint(0, 1000, (), generator=self.timestep))
        sigma = scheduler.sigmas[index].to(device=clean.device, dtype=torch.float32)
        timestep = scheduler.timesteps[index].to(clean.device).expand(clean.shape[0])
        _need(sigma.ndim == 0 and bool(torch.isfinite(sigma)) and 0 <= float(sigma) <= 1,
              "Expected a finite scalar native scheduler sigma in [0,1]")
        noise = torch.randn(clean.shape, generator=self.noise, device=clean.device, dtype=torch.float32)
        return (1 - sigma) * clean + sigma * noise, noise - clean, timestep, sigma, index


def _restoration_contract(model, student_layer, teacher_layer):
    _need(getattr(model, "projector", "not Base") is None and not hasattr(model, "reference_adapter"),
          "Pass a NoRef Base model; keep the student-only projector outside the restoration model")
    blocks = model.transformer.blocks
    _need(type(student_layer) is int and type(teacher_layer) is int
          and max(model.injection_layers) <= student_layer < teacher_layer < len(blocks),
          "Student must follow all LR injections and teacher must be deeper")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            _need(name.startswith(("lrv_stem.", "lrv_residuals.")) or ".lora_" in name,
                  "Unexpected restoration trainable: " + name)
    _need(all(not isinstance(module, nn.Dropout) or module.p == 0 for module in model.modules()),
          "Matched flow/C1/A2 numerics require the Base model's zero-dropout contract")


class NoRefContinuationStep:
    """One fresh-AdamW continuation arm, without a dataset/queue/checkpoint writer.

    A2/rho=0 is the clean flow-only path. A2/gamma=0 with rho>0 is C1's
    augmentation-only path. Every arm uses matched data order and FM streams.
    Teacher output is detached, from block21 under EMA, before student block9.
    """

    def __init__(self, model, *, arm, training_seed, rho=.25, span=2, gamma=.8,
                 student_layer=9, teacher_layer=21, hidden_dim=1024, ema_decay=.999,
                 learning_rate=1e-4, weight_decay=.01, max_grad_norm=1.,
                 feature_diagnostics_first=True, feature_diagnostics_every=512):
        _need(arm in ARMS, "Unknown continuation arm")
        _need(type(gamma) in (int, float) and math.isfinite(gamma) and gamma >= 0, "gamma must be finite and nonnegative")
        _need(type(max_grad_norm) in (int, float) and math.isfinite(max_grad_norm) and max_grad_norm > 0, "Positive gradient norm limit required")
        _need(type(feature_diagnostics_first) is bool and type(feature_diagnostics_every) is int
              and feature_diagnostics_every >= 0,"Use boolean first-active and nonnegative integer feature diagnostic frequency")
        _restoration_contract(model, student_layer, teacher_layer)
        self.model, self.arm, self.training_seed = model, arm, training_seed
        self.rho, self.span, self.gamma = rho, span, float(gamma)
        self.student_layer, self.teacher_layer = student_layer, teacher_layer
        self.max_grad_norm, self.steps = float(max_grad_norm), 0
        self.feature_diagnostics_first = feature_diagnostics_first
        self.feature_diagnostics_every = feature_diagnostics_every
        self._first_feature_snapshot_collected = False
        self._pending_feature_diagnostic = None
        device = next(model.parameters()).device
        self.rng = MatchedStepRNG(training_seed, device)
        self.ema = TrainableEMA(model, decay=ema_decay) if arm == "A2" else None
        self.alignment = None
        if arm == "A2":
            dimension = model.transformer.config.num_attention_heads * model.transformer.config.attention_head_dim
            # Isolate CPU projector initialization; do not seed or advance any
            # default CUDA generator or the independently matched FM streams.
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(torch.Generator(device="cpu").manual_seed(training_seed + 5).get_state())
                self.alignment = MaskedTemporalAlignment(dimension, dimension, hidden_dim=hidden_dim).to(device)
        named = [("restoration." + name, p) for name, p in model.named_parameters() if p.requires_grad]
        if self.alignment is not None:
            named.extend(("student_projector." + name, p) for name, p in self.alignment.projector.named_parameters())
        parameters = [p for _, p in named]
        _need(len({id(p) for p in parameters}) == len(parameters), "Optimizer parameter aliases are forbidden")
        # Explicit foreach setting is common to every future comparison arm.
        self.optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay, foreach=False)
        _need(not self.optimizer.state, "Continuation optimizer must start fresh")
        self.parameters = parameters
        self._named_optimizer_parameters = tuple(named)
        self.base_load_audit = None

    def data_index(self, training_clip_count):
        _need(type(training_clip_count) is int and training_clip_count > 0, "Positive fixed training clip count required")
        return self.steps % training_clip_count

    def teacher_features(self, noisy, timestep, text, clean_low, *, layer=None):
        _need(self.ema is not None, "Only A2 owns an EMA teacher")
        target_layer = self.teacher_layer if layer is None else layer

        def call():
            try:
                return capture_block_output(self.model, target_layer, noisy, timestep, text, clean_low)[1]
            finally:
                self.model.clear_features()

        return self.ema.teacher_forward(self.model, call)

    def drain_feature_diagnostics(self):
        """CPU-only low-frequency stats; call outside any Step timing block."""
        if self._pending_feature_diagnostic is None:return None
        from .method_diagnostics import analyze_feature_snapshot
        result = analyze_feature_snapshot(self._pending_feature_diagnostic)
        self._pending_feature_diagnostic = None
        return result

    def step(self, clean, low, text, scheduler):
        _need(torch.is_grad_enabled(), "Continuation step requires enabled student autograd")
        _need(clean.ndim == 5 and clean.shape == low.shape and clean.shape[2] == 9
              and clean.dtype == low.dtype == torch.float32 and clean.device == low.device
              and not clean.requires_grad and not low.requires_grad
              and bool(torch.isfinite(clean).all()) and bool(torch.isfinite(low).all()),
              "Use frozen FP32 matched HR/LR latents with nine time positions")
        model_modes = _modes(self.model)
        projector_modes = _modes(self.alignment) if self.alignment is not None else []
        frozen = _frozen(self.model)
        self.optimizer.zero_grad(set_to_none=True)
        self.model.clear_features()
        try:
            self.model.train()
            if self.alignment is not None:
                self.alignment.train()
            noisy, target, timestep, sigma, index = self.rng.flow_inputs(clean, scheduler)
            plan = sample_span_mask(clean.shape[0], 9, rho=0. if self.arm == "flow_only" else self.rho,
                                    span=self.span, generator=self.rng.mask)
            student_low = (low.clone() if self.arm == "flow_only" else
                           corrupt_latent_condition(low, plan.mask, generator=self.rng.corruption))
            # Compare in sigma's tensor precision, matching the loss window.
            # Converting FP32 .3 to Python float would spuriously exceed .3.
            active = self.arm == "A2" and self.gamma > 0 and bool(plan.mask.any()) and bool(sigma > .3)
            collect_features = active and ((self.feature_diagnostics_first and not self._first_feature_snapshot_collected)
                or (self.feature_diagnostics_every > 0 and (self.steps+1) % self.feature_diagnostics_every == 0))
            _need(not collect_features or self._pending_feature_diagnostic is None,
                  "Drain the previous detached CPU feature snapshot before another scheduled snapshot")
            # An active teacher validates source AND shadows before its swap.
            # The disabled-auxiliary path must also reject a corrupt EMA before
            # a student optimizer mutation; the post-step update checks again.
            if self.ema is not None and not active:
                self.ema._validate(self.model)
            # SAME noisy HR/timestep/text; teacher's condition alone stays clean.
            teacher = self.teacher_features(noisy, timestep, text, low) if active else None
            if active:
                flow, student = capture_block_output(self.model, self.student_layer, noisy, timestep, text, student_low)
            else:
                with _autocast(clean.device):
                    flow, projection = self.model(noisy, timestep, text, student_low)
                _need(projection is None, "Base must not include the historical DINO projector")
                student = None
            fm = F.mse_loss(flow.float(), target)
            if active:
                _need(teacher is not None and not teacher.requires_grad, "Teacher feature graph must be detached")
                sd, diagnostic = self.alignment(student, teacher, plan.mask, tuple(self.model.token_grid), sigma,
                                                capture_diagnostics=collect_features)
            else:
                sd = flow.reshape(-1)[:0].sum()
                diagnostic = {"selected_tokens": 0, "cosine_mean": None}
            total = fm + self.gamma * sd
            _need(bool(torch.isfinite(total)), "Nonfinite continuation objective")
            total.backward()
            _need(any(p.grad is not None for p in self.model.parameters() if p.requires_grad), "No restoration gradient")
            _need(all(p.grad is None for p in self.model.parameters() if not p.requires_grad), "Frozen backbone received a gradient")
            _require_finite(("Nonfinite continuation gradients: " + name, p.grad)
                            for name, p in self._named_optimizer_parameters if p.grad is not None)
            norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update(self.model)
            _need(frozen == _frozen(self.model), "Frozen backbone parameters changed")
            self.steps += 1
            cosine_mean = float(diagnostic['cosine_mean']) if diagnostic['cosine_mean'] is not None else None
            if collect_features:
                self._pending_feature_diagnostic = {'step':self.steps,'student_layer':self.student_layer,
                    'teacher_layer':self.teacher_layer,'sigma':float(sigma),'selected_tokens':diagnostic['selected_tokens'],
                    'cosine_mean':cosine_mean,'schedule':{'first_active':self.feature_diagnostics_first,
                        'every_steps':self.feature_diagnostics_every},'features':diagnostic['feature_snapshot']}
                self._first_feature_snapshot_collected = True
            return {"step": self.steps, "arm": self.arm, "timestep_index": index, "native_timestep": float(timestep[0]),
                    "sigma": float(sigma), "fm": float(fm.detach()), "sd": float(sd.detach()),
                    "total": float(total.detach()), "grad_norm_before_clip": float(norm),
                    "teacher_forward": active, "teacher_student_same_noisy_HR_and_timestep": True,
                    "rho_requested": plan.rho_requested, "rho_realized_all_latents": plan.realized_ratio_all_latents,
                    "rho_realized_eligible_latents": plan.realized_ratio_eligible_latents,
                    "mask_spans": plan.spans, "mask_rounding": plan.rounding,
                    "selected_tokens": diagnostic["selected_tokens"],
                    "cosine_mean":cosine_mean,"cosine_layers":{"student":self.student_layer,"teacher":self.teacher_layer},
                    "feature_diagnostics_snapshot_collected":collect_features,
                    "student_layer": self.student_layer, "teacher_layer": self.teacher_layer,
                    "optimizer_state_fresh_at_initialization": True, "EMA_updates": self.ema.num_updates if self.ema else 0}
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.clear_features()
            _restore_modes(model_modes)
            _restore_modes(projector_modes)


def load_matched_base_continuation(checkpoint, *, expected_checkpoint_sha256, expected_training_seed,
                                   allowed_root, arm, device="cpu", **options):
    """Explicit future loader: task parameters restored, optimizer state ignored.

    The actual caller must supply the new-run allowed_root and declared matched
    Base SHA/seed. This function is not called by import, tests use a tiny mock.
    """
    from .checkpoints import load_training_checkpoint
    model, audit = load_training_checkpoint(checkpoint, device=device, allowed_root=Path(allowed_root), variant="Base")
    config = audit["config"]
    _need(audit["checkpoint_sha256"] == expected_checkpoint_sha256 and audit["step"] == 8192 and audit["variant"] == "Base",
          "Continuation must use the declared final matched Base checkpoint")
    _need(config["training"].get("seed", config["seed"]) == expected_training_seed
          and [config["data"][k] for k in ("num_frames", "temporal_stride", "height", "width", "scale")] == [33, 1, 128, 224, 4]
          and not config.get("reference", {}).get("enabled", False), "Base training seed/protocol/family mismatch")
    _need(all(audit[key] is False for key in ("optimizer_state_restored", "scheduler_state_restored", "rng_state_restored")),
          "Do not resume Base optimizer/scheduler/RNG state")
    options.setdefault("learning_rate", config["training"]["learning_rate"])
    options.setdefault("weight_decay", config["training"]["weight_decay"])
    options.setdefault("max_grad_norm", config["training"]["max_grad_norm"])
    result = NoRefContinuationStep(model, arm=arm, training_seed=expected_training_seed, **options)
    result.base_load_audit = audit
    return result
