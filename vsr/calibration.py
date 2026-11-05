"""Separate-objective generator-gradient calibration with no model updates.

Auxiliary weights are suggestions for a subsequently frozen experiment plan,
not evidence of useful restoration. This diagnostic explicitly uses noised
clean HR latents. It is not a restoration inference path.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import hashlib
import math
import statistics

import torch
import torch.nn.functional as F

from .alignment import point_alignment
from .relation import relation_loss


GENERATOR_GROUPS = ("lrv_stem", "lrv_residuals", "lora")


def _gradients(model):
    squares = defaultdict(float)
    counts = defaultdict(int)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not parameter.requires_grad:
            raise RuntimeError(f"Calibration gradient leaked into frozen parameter: {name}")
        if name.startswith("lrv_stem."):
            group = "lrv_stem"
        elif name.startswith("lrv_residuals."):
            group = "lrv_residuals"
        elif ".lora_" in name:
            group = "lora"
        elif name.startswith("projector."):
            group = "projector"
        else:
            group = "other_trainable"
        grad = parameter.grad.detach()
        if not bool(torch.isfinite(grad).all()):
            raise RuntimeError(f"Nonfinite calibration gradient in {name}")
        squares[group] += float(grad.double().square().sum())
        counts[group] += parameter.numel()
    groups = {group: {"l2": math.sqrt(squares[group]), "elements_with_grad": counts[group]}
              for group in (*GENERATOR_GROUPS, "projector", "other_trainable")}
    return {"generator_l2": math.sqrt(sum(squares[group] for group in GENERATOR_GROUPS)),
            "projector_l2": groups["projector"]["l2"], "groups": groups,
            "frozen_elements_with_grad": 0, "finite": True}


def _suggest(fm_gradient, auxiliary_gradient, fraction, bounds):
    if fm_gradient <= 0:
        return {"raw": None, "clamped": None, "reason": "zero_fm_generator_gradient"}
    if auxiliary_gradient <= 0:
        return {"raw": None, "clamped": None, "reason": "zero_auxiliary_generator_gradient"}
    raw = fraction * fm_gradient / auxiliary_gradient
    if not math.isfinite(raw):
        return {"raw": None, "clamped": None, "reason": "nonfinite_gradient_ratio"}
    return {"raw": raw, "clamped": min(bounds[1], max(bounds[0], raw)), "reason": None}


def _summary(rows, key):
    valid = [row[key] for row in rows if row[key]["raw"] is not None]
    return {"valid_sigma_count": len(valid),
            "raw_median": statistics.median(x["raw"] for x in valid) if valid else None,
            "clamped_median": statistics.median(x["clamped"] for x in valid) if valid else None,
            "excluded_sigma_count": len(rows) - len(valid)}


def collect_auxiliary_gradient_scales(model, clean, low, text, teacher, grid, flow_pair=None, *,
                                      seed=20261203, sigmas=(.3, .7, .99), target_fraction=.1,
                                      weight_bounds=(.01, 10.), valid_mask=None,
                                      fb_alpha=.01, fb_beta=.5):
    """Recompute FM/A/R separately at the same fixed noise and sigma.

    No optimizer is accepted or constructed, and no warmup/update occurs here.
    A caller may run this after a separately recorded warmup if zero branches
    are still closed, while preserving equal initialization and budget rules.

    Generator norm combines only LRV stem, LR residual maps and Wan LoRA.
    Projection-head gradients are reported separately and excluded from the
    suggested weights: lambda_aux = target_fraction*g_FM/g_aux, median across
    sigmas, bounded to weight_bounds. Zero auxiliary gradients remain undefined
    rather than being hidden by a denominator epsilon.
    """
    sigmas = tuple(float(value) for value in sigmas)
    if not sigmas or len(set(sigmas)) != len(sigmas) or any(not math.isfinite(s) or s <= 0 or s > 1 for s in sigmas):
        raise ValueError("Calibration sigmas must be distinct finite values in (0,1]")
    if not math.isfinite(target_fraction) or target_fraction <= 0:
        raise ValueError("Target gradient fraction must be finite and positive")
    if len(weight_bounds) != 2 or not all(math.isfinite(x) and x > 0 for x in weight_bounds) or weight_bounds[0] > weight_bounds[1]:
        raise ValueError("Weight bounds must be finite positive ordered limits")
    if clean.ndim != 5 or clean.shape != low.shape or clean.shape[0] <= 0:
        raise ValueError("Clean and LR latents must have matching B,C,T,H,W shapes")
    if len(grid) != 3 or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in grid):
        raise ValueError("grid must contain positive T,H,W integers")
    if teacher.ndim != 3 or teacher.shape[:2] != (clean.shape[0], math.prod(grid)):
        raise ValueError("Teacher must have B,T*H*W,D shape matching grid")
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else clean.device
    clean, low, text, teacher = (value.detach().to(device) for value in (clean, low, text, teacher))
    clean = clean.float()
    if not all(bool(torch.isfinite(value).all()) for value in (clean, low, text, teacher)):
        raise ValueError("Calibration inputs must be finite")
    if valid_mask is None:
        mask = torch.ones(teacher.shape[:2], device=device, dtype=torch.bool)
    else:
        if valid_mask.dtype != torch.bool or valid_mask.shape != teacher.shape[:2]:
            raise ValueError("valid_mask must be boolean B,T*H*W")
        mask = valid_mask.detach().to(device)
        if not bool(mask.any()):
            raise ValueError("Calibration requires valid supervision tokens")
    forward = backward = None
    if flow_pair is not None:
        if isinstance(flow_pair, dict):
            forward, backward = flow_pair["forward"], flow_pair["backward"]
        else:
            if len(flow_pair) != 2:
                raise ValueError("flow_pair must contain forward and backward tensors")
            forward, backward = flow_pair
        forward, backward = forward.detach().to(device), backward.detach().to(device)
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    rows = []
    try:
        # This diagnostic intentionally clears stale gradients and leaves none.
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            for sigma in sigmas:
                key = f"auxiliary-gradient-calibration-v1:{int(seed)}:{sigma:.12g}".encode()
                noise_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)
                rng = torch.Generator(device=device).manual_seed(noise_seed)
                noise = torch.randn(clean.shape, generator=rng, device=device, dtype=torch.float32)
                noisy = (1 - sigma) * clean + sigma * noise
                timestep = torch.full((clean.shape[0],), sigma * 1000, device=device, dtype=torch.float32)
                row = {"sigma": sigma, "timestep": sigma * 1000, "noise_seed": noise_seed, "objectives": {}}
                for objective in ("FM", "A", "R"):
                    model.zero_grad(set_to_none=True)
                    model.clear_features()
                    try:
                        context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
                        with context:
                            velocity, student = model(noisy, timestep, text, low)
                        if tuple(model.token_grid) != tuple(grid):
                            raise ValueError("Runtime student token grid differs from calibration grid")
                        diagnostics = None
                        if objective == "FM":
                            loss = F.mse_loss(velocity.float(), noise - clean)
                        else:
                            if student is None:
                                raise ValueError("Auxiliary calibration requires an enabled student projection head")
                            if objective == "A":
                                loss = point_alignment(student, teacher, mask)
                            else:
                                loss, diagnostics = relation_loss(student, teacher, grid, forward, backward, mask,
                                                                    fb_alpha=fb_alpha, fb_beta=fb_beta)
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError(f"Nonfinite {objective} calibration loss")
                        loss.backward()
                        row["objectives"][objective] = {"loss": float(loss.detach()), **_gradients(model),
                                                       "relation_diagnostics": diagnostics}
                    finally:
                        model.clear_features()
                        model.zero_grad(set_to_none=True)
                fm = row["objectives"]["FM"]["generator_l2"]
                row["lambda_A"] = _suggest(fm, row["objectives"]["A"]["generator_l2"], target_fraction, weight_bounds)
                row["lambda_R"] = _suggest(fm, row["objectives"]["R"]["generator_l2"], target_fraction, weight_bounds)
                rows.append(row)
        alignment, relation = _summary(rows, "lambda_A"), _summary(rows, "lambda_R")
        return {"status": "completed", "kind": "auxiliary_generator_gradient_calibration", "rows": rows,
                "suggested_lambda_A": alignment["clamped_median"], "suggested_lambda_R": relation["clamped_median"],
                "lambda_A_summary": alignment, "lambda_R_summary": relation,
                "protocol": {"sigmas": list(sigmas), "seed": int(seed), "target_generator_gradient_fraction": target_fraction,
                             "weight_bounds": list(weight_bounds), "generator_groups": list(GENERATOR_GROUPS),
                             "projector_gradients": "reported separately; excluded from weight suggestions",
                             "noise": "independent local SHA256(seed,sigma) generators; same noised input for FM/A/R",
                             "objective_backwards": "fresh separate forward/backward for every objective",
                             "native_flow": "noisy=(1-sigma)*clean+sigma*noise; target=noise-clean; timestep=1000*sigma",
                             "relation_scope": "spatial_and_hr_anchor_flow_temporal" if forward is not None else "spatial_only_no_flow",
                             "suggestion_rule": "lambda=target_fraction*gFM/gAux; record raw and bounded values; median valid sigmas",
                             "zero_auxiliary_gradient": "undefined suggestion; no epsilon denominator",
                             "model_updates": 0, "optimizer_created": False, "gradients_cleared_afterwards": True,
                             "predeclared_first_align_pilot_lambda": .1, "weights_applied": False,
                             "restoration_inference": False, "restoration_quality_claim": False}}
    finally:
        model.clear_features()
        clear_reference = getattr(model, "clear_reference", None)
        if callable(clear_reference):
            clear_reference()
        model.zero_grad(set_to_none=True)
        for module, was_training in flags:
            module.training = was_training
