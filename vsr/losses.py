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


def _hr_patch_size(patch_size):
    _require(isinstance(patch_size, (tuple, list)) and len(patch_size) == 3
             and all(type(value) is int and value > 0 for value in patch_size),
             "Latent patch size must contain positive integer pt,ph,pw")
    return tuple(patch_size)


def patchify_hr_latent(clean, patch_size=(1, 2, 2)):
    """Detach normalized HR latents into Wan-native target patches.

    Input is the frozen VAE's already normalized [B,C,T,H,W] clean latent.
    Tokens flatten T,H,W; each vector flattens pt,ph,pw,C (C varies fastest),
    exactly inverting Wan's native output unpatchify. No channel/token
    normalization, masking, timestep selection or random sampling is applied.
    """
    patch = _hr_patch_size(patch_size)
    _require(isinstance(clean, torch.Tensor) and clean.ndim == 5
             and clean.is_floating_point() and all(size > 0 for size in clean.shape),
             "Clean HR target must be nonempty floating [B,C,T,H,W]")
    _require(bool(torch.isfinite(clean).all()), "Clean HR target must be finite")
    batch, channels, frames, height, width = clean.shape
    pt, ph, pw = patch
    _require(all(size % stride == 0 for size, stride in zip((frames, height, width), patch)),
             "Clean HR latent dimensions must be divisible by the patch size")
    target = clean.detach().float()
    return (target.reshape(batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw)
            .permute(0, 2, 4, 6, 3, 5, 7, 1)
            .reshape(batch, (frames // pt) * (height // ph) * (width // pw), channels * pt * ph * pw))


class HRLatentRegression(nn.Module):
    """One FP32 student head regressing all fixed clean HR latent elements.

    The head is training-only and belongs outside the restoration model.
    Production Wan uses 1536 features -> 16*1*2*2=64 targets per token.
    The caller isolates head initialization RNG and applies the declared loss
    weight outside this module. SmoothL1 beta=1 and elementwise mean are fixed.
    """

    def __init__(self, student_dim=1536, latent_channels=16, patch_size=(1, 2, 2)):
        super().__init__()
        _require(type(student_dim) is int and student_dim > 0
                 and type(latent_channels) is int and latent_channels > 0,
                 "Positive integer student and latent channel dimensions required")
        self.student_dim, self.latent_channels = student_dim, latent_channels
        self.patch_size = _hr_patch_size(patch_size)
        self.target_dim = latent_channels * math.prod(self.patch_size)
        self.projector = nn.Linear(student_dim, self.target_dim, dtype=torch.float32)

    def forward(self, student, clean_hr, grid):
        _require(isinstance(grid, (tuple, list)) and len(grid) == 3
                 and all(type(value) is int and value > 0 for value in grid),
                 "Positive integer T,H,W token grid required")
        _require(isinstance(student, torch.Tensor) and student.ndim == 3
                 and student.is_floating_point() and all(size > 0 for size in student.shape),
                 "Student features must be nonempty floating [B,N,C]")
        target = patchify_hr_latent(clean_hr, self.patch_size)
        _require(clean_hr.shape[1] == self.latent_channels,
                 "Clean HR latent channels differ from the target head")
        expected_grid = tuple(size // stride for size, stride in zip(clean_hr.shape[2:], self.patch_size))
        _require(tuple(grid) == expected_grid, "HR target and student token grids differ")
        _require(tuple(student.shape) == (clean_hr.shape[0], math.prod(grid), self.student_dim)
                 and tuple(target.shape) == (student.shape[0], student.shape[1], self.target_dim),
                 "Student/HR target batch, token or feature dimensions differ")
        _require(student.device == target.device
                 and all(parameter.device == student.device for parameter in self.projector.parameters()),
                 "Student, HR target and projector devices must match")
        _require(all(parameter.dtype == torch.float32 for parameter in self.projector.parameters()),
                 "Keep the HR latent projector in FP32")
        _require(bool(torch.isfinite(student).all()), "Student features must be finite")
        with torch.autocast(student.device.type, enabled=False):
            prediction = self.projector(student.float())
            loss = F.smooth_l1_loss(prediction, target, beta=1., reduction="mean")
        _require(bool(torch.isfinite(loss)), "Nonfinite HR latent regression objective")
        return loss, {"selected_tokens": student.shape[0] * student.shape[1],
                      "target_elements": target.numel(), "token_grid": list(grid),
                      "patch_order": "pt,ph,pw,C", "token_order": "T,H,W",
                      "loss": "SmoothL1", "beta": 1., "reduction": "element_mean",
                      "target_requires_grad": False, "all_tokens_supervised": True,
                      "target": "frozen_clean_HR_normalized_VAE_latent"}


class HRDINOCosineRegression(nn.Module):
    """Single FP32 student projection to frozen temporally pooled HR DINO.

    Both patch representations are L2-normalized along their channel axis;
    loss is mean(1-cosine) over every batch/token. Target pooling happens in
    data preparation, before this L2 normalization. No extra LayerNorm, mask,
    timestep window, EMA or teacher execution belongs to this loss.
    """

    def __init__(self, student_dim=1536, teacher_dim=768):
        super().__init__()
        _require(type(student_dim) is int and student_dim > 0
                 and type(teacher_dim) is int and teacher_dim > 0,
                 "Positive integer student and DINO dimensions required")
        self.student_dim, self.teacher_dim = student_dim, teacher_dim
        self.target_dim = teacher_dim
        self.projector = nn.Linear(student_dim, teacher_dim, dtype=torch.float32)

    def forward(self, student, teacher, grid):
        _require(isinstance(grid, (tuple, list)) and len(grid) == 3
                 and all(type(value) is int and value > 0 for value in grid),
                 "Positive integer T,H,W token grid required")
        _require(isinstance(student, torch.Tensor) and isinstance(teacher, torch.Tensor)
                 and student.ndim == teacher.ndim == 3
                 and student.is_floating_point() and teacher.is_floating_point()
                 and all(size > 0 for size in student.shape) and all(size > 0 for size in teacher.shape),
                 "Student and DINO target must be nonempty floating [B,N,C]")
        _require(student.shape[:2] == teacher.shape[:2] and student.shape[1] == math.prod(grid)
                 and student.shape[-1] == self.student_dim and teacher.shape[-1] == self.teacher_dim,
                 "Student/DINO batch, token, grid or channel dimensions differ")
        _require(student.device == teacher.device
                 and all(parameter.device == student.device for parameter in self.projector.parameters()),
                 "Student, DINO target and projector devices must match")
        _require(all(parameter.dtype == torch.float32 for parameter in self.projector.parameters()),
                 "Keep the HR DINO projector in FP32")
        _require(bool(torch.isfinite(student).all()) and bool(torch.isfinite(teacher).all()),
                 "Student and DINO target must be finite")
        with torch.autocast(student.device.type, enabled=False):
            projected = F.normalize(self.projector(student.float()), dim=-1, eps=1e-12)
            target = F.normalize(teacher.detach().float(), dim=-1, eps=1e-12)
            loss = (1 - (projected * target).sum(dim=-1)).mean()
        _require(bool(torch.isfinite(loss)), "Nonfinite HR DINO cosine objective")
        return loss, {"selected_tokens": student.shape[0] * student.shape[1],
                      "target_elements": teacher.numel(), "token_grid": list(grid),
                      "token_order": "T,H,W", "loss": "one_minus_cosine",
                      "reduction": "token_mean", "normalization": "channel_L2", "eps": 1e-12,
                      "target_requires_grad": False, "all_tokens_supervised": True,
                      "target": "frozen_temporally_pooled_clean_HR_DINO_patch_tokens"}


def shared_gradient_norms(fm_loss, unit_aux_loss, restoration_parameters):
    """Measure unweighted loss gradients without an optimizer or .grad writes.

    Caller passes the same full restoration-trainable list for both losses,
    excluding the auxiliary head. Unused entries contribute zero. The first
    autograd.grad retains the graph; the second frees it, so this diagnostic
    forward cannot subsequently be used for a training backward. Norm sums
    use FP64 to avoid overflow of otherwise finite FP32 gradient elements.
    """
    parameters = tuple(restoration_parameters)
    _require(parameters and len({id(value) for value in parameters}) == len(parameters)
             and all(isinstance(value, torch.Tensor) and value.requires_grad and value.is_floating_point()
                     for value in parameters), "Distinct shared restoration trainables required")
    device = parameters[0].device
    _require(all(value.device == device for value in parameters), "Shared restoration parameters must share one device")
    for value in (fm_loss, unit_aux_loss):
        _require(isinstance(value, torch.Tensor) and value.ndim == 0 and value.is_floating_point()
                 and value.requires_grad and value.device == device and bool(torch.isfinite(value)),
                 "Finite scalar differentiable FM and unit-aux losses required")
    fm_gradients = torch.autograd.grad(fm_loss, parameters, allow_unused=True, retain_graph=True)
    aux_gradients = torch.autograd.grad(unit_aux_loss, parameters, allow_unused=True, retain_graph=False)

    def norm(values):
        squared = torch.zeros((), dtype=torch.float64, device=device)
        for value in values:
            if value is not None:
                _require(bool(torch.isfinite(value).all()), "Nonfinite shared calibration gradient")
                squared = squared + value.detach().double().square().sum()
        result = float(squared.sqrt())
        _require(math.isfinite(result) and result > 0, "Shared calibration gradient norm must be finite and positive")
        return result

    return {"fm_grad_norm": norm(fm_gradients), "aux_grad_norm": norm(aux_gradients),
            "shared_parameter_tensors": len(parameters), "shared_parameter_elements": sum(p.numel() for p in parameters),
            "norm": "global_L2_FP64_accumulation_before_clipping", "optimizer_updates": 0,
            "parameter_grad_fields_modified": False, "graph_freed_after_aux_gradient": True}


def calibrated_auxiliary_weights(observations):
    """Freeze two weights from three train-only unit-gradient observations.

    Observation sigma fields are the nominal requested Python floats .2/.5/.8
    in that exact order, rather than an FP32 tensor's printed approximation.
    No validation score, model update, clipping or fallback enters this rule.
    """
    _require(isinstance(observations, (tuple, list)) and len(observations) == 3
             and all(isinstance(row, dict) for row in observations),
             "Exactly three ordered calibration observations required")
    _require([row.get("sigma") for row in observations] == [.2, .5, .8],
             "Calibration nominal sigma order must be .2,.5,.8")
    ratios = []
    for row in observations:
        fm, aux = row.get("fm_grad_norm"), row.get("aux_grad_norm")
        _require(all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in (fm, aux)),
                 "Calibration FM and unit-aux norms must be finite and positive")
        ratio = fm / aux
        _require(math.isfinite(ratio) and ratio > 0, "Invalid FM/unit-aux gradient ratio")
        ratios.append(ratio)
    median = sorted(ratios)[1]
    candidates = []
    for strength in (.1, 1.):
        unrounded = strength * median
        weight = float(format(unrounded, ".2g"))
        _require(math.isfinite(weight) and weight > 0, "Rounded calibration weight must be finite and positive")
        candidates.append({"relative_strength": strength, "weight": weight, "unrounded_weight": unrounded,
                           "realized_median_relative_strength": weight / median})
    _require(candidates[0]["weight"] != candidates[1]["weight"], "Calibration must yield two distinct weights")
    return {"schema": "auxiliary_gradient_calibration_v1", "sigmas": [.2, .5, .8],
            "FM_to_unit_aux_ratios": ratios, "median_FM_to_unit_aux_ratio": median,
            "relative_strengths": [.1, 1.], "rounding": "two_significant_digits",
            "weights": [row["weight"] for row in candidates], "candidates": candidates,
            "scope": "all_shared_restoration_trainables_excluding_auxiliary_head",
            "validation_used": False, "optimizer_updates": 0}
