"""Spatial and motion-compensated teacher-relation supervision.

Flows must be estimated from the actual consecutive HR supervision anchors,
for example frame pairs (0,4) and (4,8). Adjacent raw-frame flows cannot be
substituted. This module never estimates motion or reads HR images itself.
All flow values are HR-pixel displacements, not normalized coordinates.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _centers(height: int, width: int, device, dtype):
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), -1)


def temporal_correspondence(forward_flow, backward_flow, token_hw,
                            fb_alpha: float = 0.01, fb_beta: float = 0.5):
    """Return a shared target sampling grid and HR forward/backward reliability.

    Inputs: N,2,Hhr,Whr forward/backward flow for the SAME HR anchor pair.
    Output grid: N,Htoken,Wtoken,2, normalized for align_corners=False.

    HR coordinates of token j are (j+0.5)*Whr/Wtoken-0.5. Therefore flow
    displacement in token units is dx*Wtoken/Whr (and similarly for height).
    Both HR image bounds and target token-center bounds are checked. The latter
    avoids interpreting a partially padded token interpolation as a valid edge.

    Forward/backward criterion is explicitly squared HR pixels:
      ||f + warp(b,f)||² <= alpha*(||f||²+||warp(b,f)||²) + beta.
    Reliable edges are weighted by exp(-error² / (threshold²+eps)).
    """
    if forward_flow.ndim != 4 or forward_flow.shape[1] != 2 or backward_flow.shape != forward_flow.shape:
        raise ValueError("Flow pair must have identical N,2,Hhr,Whr shapes")
    if len(token_hw) != 2 or min(token_hw) <= 0:
        raise ValueError("token_hw must contain positive Htoken,Wtoken")
    if fb_alpha < 0 or fb_beta < 0:
        raise ValueError("Forward/backward thresholds must be nonnegative")
    ht, wt = (int(x) for x in token_hw)
    n, _, hh, wh = forward_flow.shape
    if n == 0:
        raise ValueError("temporal_correspondence needs at least one anchor pair")
    # Float32 grids avoid half precision coordinate quantization. No gradients
    # may reach the HR-derived motion estimator through relation supervision.
    f_hr = forward_flow.detach().float()
    b_hr = backward_flow.detach().to(device=f_hr.device, dtype=torch.float32)
    finite_f = torch.isfinite(f_hr).all(1, keepdim=True)
    finite_b = torch.isfinite(b_hr).all(1, keepdim=True)
    f_hr = torch.nan_to_num(f_hr, nan=0.0, posinf=0.0, neginf=0.0)
    b_hr = torch.nan_to_num(b_hr, nan=0.0, posinf=0.0, neginf=0.0)
    f = F.interpolate(f_hr, size=(ht, wt), mode="bilinear", align_corners=False)
    f_valid = F.interpolate(finite_f.float(), size=(ht, wt), mode="bilinear", align_corners=False)[:, 0] >= (1 - 1e-6)
    token_flow = f * f.new_tensor([wt / wh, ht / hh]).view(1, 2, 1, 1)
    base = _centers(ht, wt, f.device, f.dtype).unsqueeze(0).expand(n, -1, -1, -1)
    displacement_normalized = f.permute(0, 2, 3, 1) * f.new_tensor([2.0 / wh, 2.0 / hh])
    grid = base + displacement_normalized
    backward_at_target = F.grid_sample(b_hr, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    backward_finite = F.grid_sample(finite_b.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0] >= (1 - 1e-6)
    error_sq = (f + backward_at_target).square().sum(1)
    threshold_sq = fb_alpha * (f.square().sum(1) + backward_at_target.square().sum(1)) + fb_beta
    # Convert normalized coordinates back to pixel/token centers for bounds.
    target_hr = (grid + 1) * grid.new_tensor([wh, hh]) / 2 - 0.5
    target_token = (grid + 1) * grid.new_tensor([wt, ht]) / 2 - 0.5
    tolerance = 1e-5
    hr_inside = ((target_hr[..., 0] >= -tolerance) & (target_hr[..., 0] <= wh - 1 + tolerance)
                 & (target_hr[..., 1] >= -tolerance) & (target_hr[..., 1] <= hh - 1 + tolerance))
    token_inside = ((target_token[..., 0] >= -tolerance) & (target_token[..., 0] <= wt - 1 + tolerance)
                    & (target_token[..., 1] >= -tolerance) & (target_token[..., 1] <= ht - 1 + tolerance))
    valid = f_valid & backward_finite & hr_inside & token_inside & (error_sq <= threshold_sq)
    weight = valid.float() * torch.exp(-error_sq / (threshold_sq + 1e-8))
    return {"grid": grid, "weight": weight, "valid": valid,
            "flow_tokens": token_flow, "destination_hr": target_hr,
            "fb_error_sq": error_sq, "fb_threshold_sq": threshold_sq}


def _feature_grid(value, grid, label):
    if value.ndim == 3:
        b, count, c = value.shape
        if count != grid[0] * grid[1] * grid[2]:
            raise ValueError(f"{label} flat token count differs from grid")
        value = value.reshape(b, *grid, c)
    if value.ndim != 5 or tuple(value.shape[1:4]) != tuple(grid):
        raise ValueError(f"{label} must have B,T,H,W,C matching grid")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} features must be finite floating point")
    return value


def relation_loss(student, teacher, grid, forward_flow=None, backward_flow=None,
                  valid_mask=None, *, fb_alpha: float = 0.01, fb_beta: float = 0.5):
    """Match teacher edge cosines; return (scalar loss, detached diagnostics).

    student/teacher: B,T,H,W,C or flattened B,(T*H*W),C, T/H/W order.
    grid: actual runtime student (T,H,W), not VAE latent dimensions.
    flows: B,T-1,2,Hhr,Whr for consecutive *supervision anchors*.
    valid_mask: B,T,H,W (or B,T*H*W), True only for valid student tokens.
    Spatial horizontal/vertical edges share one normalization. Temporal edges
    share another, and the two nonempty losses have equal weight. No edges
    return a differentiable zero. Teacher and HR flows are always detached.
    """
    if len(grid) != 3 or any(not isinstance(x, int) or x <= 0 for x in grid):
        raise ValueError("grid must be positive integer T,H,W")
    s = _feature_grid(student, grid, "student")
    v = _feature_grid(teacher.detach(), grid, "teacher").to(device=s.device)
    if s.shape != v.shape:
        raise ValueError("Student/teacher shapes must match")
    b, t, h, w, c = s.shape
    s, v = F.normalize(s.float(), dim=-1), F.normalize(v.float(), dim=-1)
    if valid_mask is None:
        mask = torch.ones((b, t, h, w), device=s.device, dtype=torch.bool)
    else:
        if valid_mask.shape == (b, t * h * w):
            valid_mask = valid_mask.reshape(b, t, h, w)
        if tuple(valid_mask.shape) != (b, t, h, w):
            raise ValueError("valid_mask must have B,T,H,W shape")
        mask = valid_mask.to(device=s.device, dtype=torch.bool)
    zero = s.sum() * 0.0
    spatial_sum, spatial_count = zero, 0
    # Compare relations, never force two feature vectors to become equal.
    for axis in (2, 3):
        src, dst = [slice(None)] * 5, [slice(None)] * 5
        src[axis], dst[axis] = slice(None, -1), slice(1, None)
        edge = mask[tuple(src[:4])] & mask[tuple(dst[:4])]
        cosine_s = (s[tuple(src)] * s[tuple(dst)]).sum(-1)
        cosine_v = (v[tuple(src)] * v[tuple(dst)]).sum(-1)
        spatial_sum = spatial_sum + ((cosine_s - cosine_v).square() * edge).sum()
        spatial_count += int(edge.sum().item())
    spatial = spatial_sum / spatial_count if spatial_count else zero
    temporal, temporal_count, weight_sum, fb_reliable_count = zero, 0, 0.0, 0
    if (forward_flow is None) != (backward_flow is None):
        raise ValueError("Forward and backward flow must be provided together")
    if forward_flow is not None:
        if forward_flow.ndim != 5 or tuple(forward_flow.shape[:3]) != (b, t - 1, 2) or backward_flow.shape != forward_flow.shape:
            raise ValueError("Flows must be B,T-1,2,Hhr,Whr for the actual HR anchors")
        if t > 1:
            hh, wh = forward_flow.shape[-2:]
            correspondence = temporal_correspondence(forward_flow.reshape(b * (t - 1), 2, hh, wh).to(s.device),
                                                      backward_flow.reshape(b * (t - 1), 2, hh, wh).to(s.device),
                                                      (h, w), fb_alpha, fb_beta)
            sampling_grid = correspondence["grid"]
            weight = correspondence["weight"]
            # SAME motion coordinates and interpolation for student and teacher.
            def sample_next(features):
                next_frames = features[:, 1:].reshape(b * (t - 1), h, w, c).permute(0, 3, 1, 2)
                sampled = F.grid_sample(next_frames, sampling_grid, mode="bilinear", padding_mode="zeros", align_corners=False)
                return F.normalize(sampled.permute(0, 2, 3, 1), dim=-1)
            student_target, teacher_target = sample_next(s), sample_next(v)
            student_source = s[:, :-1].reshape(b * (t - 1), h, w, c)
            teacher_source = v[:, :-1].reshape(b * (t - 1), h, w, c)
            next_mask = mask[:, 1:].reshape(b * (t - 1), 1, h, w).float()
            target_mask = F.grid_sample(next_mask, sampling_grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0] >= (1 - 1e-6)
            source_mask = mask[:, :-1].reshape(b * (t - 1), h, w)
            weight = weight * (source_mask & target_mask).float()
            diff = (student_source * student_target).sum(-1) - (teacher_source * teacher_target).sum(-1)
            total_weight = weight.sum()
            temporal_count = int((weight > 0).sum().item())
            weight_sum = float(total_weight.detach())
            fb_reliable_count = int(correspondence["valid"].sum().item())
            if temporal_count:
                temporal = (weight * diff.square()).sum() / total_weight.clamp_min(1e-8)
    nonempty = int(spatial_count > 0) + int(temporal_count > 0)
    loss = (spatial + temporal) / nonempty if nonempty else zero
    return loss, {"spatial_loss": float(spatial.detach()), "temporal_loss": float(temporal.detach()),
                  "spatial_edges": spatial_count, "temporal_edges": temporal_count,
                  "temporal_weight_sum": weight_sum, "fb_reliable_before_token_masks": fb_reliable_count,
                  "active_edge_types": nonempty, "fb_alpha": fb_alpha, "fb_beta_squared_hr_pixels": fb_beta,
                  "normalization": "each_edge_type_then_mean_of_nonempty_types",
                  "flow_contract": "HR pixel flow between actual consecutive supervision anchors; detached",
                  "grid_convention": "pixel_centers_align_corners_false"}
