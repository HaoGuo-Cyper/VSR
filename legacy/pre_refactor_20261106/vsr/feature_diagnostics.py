"""Detached feature-space diagnostics, not restoration-quality evidence.

Statistics use the same valid, normalized point features as alignment. A lower
cosine error, higher retrieval rate or particular effective rank does not by
itself establish useful video recovery or an information-equivalent space.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


NORMALIZATION_EPS = 1e-12


def _inputs(student, teacher, validmask):
    if student.ndim != 3 or student.shape != teacher.shape or min(student.shape) <= 0:
        raise ValueError("Student and teacher must share nonempty B,N,D shapes")
    if validmask.dtype != torch.bool or validmask.shape != student.shape[:2]:
        raise ValueError("validmask must be boolean with shape B,N")
    # CPU float64 avoids retaining training graphs or solver workspaces on the
    # GPU and makes constant-feature centering stable. Invalid values are masked
    # before normalization; invalid padding may therefore contain NaNs.
    student = student.detach().to(device="cpu", dtype=torch.float64)
    teacher = teacher.detach().to(device="cpu", dtype=torch.float64)
    validmask = validmask.detach().to(device="cpu")
    if not bool(validmask.any()):
        raise ValueError("Feature diagnostics require at least one valid token")
    if not bool(torch.isfinite(student[validmask]).all()) or not bool(torch.isfinite(teacher[validmask]).all()):
        raise ValueError("Valid feature vectors must be finite")
    return student, teacher, validmask


def _statistics(normalized):
    # Subtracting a common origin before computing the mean is algebraically
    # equivalent centering and gives exact zero for any repeated vector, even
    # when an irrational normalized component would accumulate mean roundoff.
    offset = normalized - normalized[:1]
    centered = offset - offset.mean(dim=0, keepdim=True)
    variance = float(centered.square().mean(dim=0).mean().clamp_min(0))
    singular_values = torch.linalg.svdvals(centered).clamp_min(0)
    total = float(singular_values.sum())
    maximum_rank = min(normalized.shape[0] - 1, normalized.shape[1])
    if total == 0 or maximum_rank <= 0:
        effective_rank = 0.0
    else:
        probabilities = singular_values / singular_values.sum()
        positive = probabilities[probabilities > 0]
        entropy = float(-(positive * positive.log()).sum())
        # Roundoff can place exp(H) infinitesimally beyond the algebraic bound.
        effective_rank = min(float(maximum_rank), max(0.0, math.exp(entropy)))
    return {"mean_channel_variance": variance, "effective_rank": effective_rank,
            "centered_singular_value_sum": total, "maximum_centered_rank": maximum_rank}


@torch.no_grad()
def feature_statistics(student, teacher, validmask):
    """Cosine error, population channel variance and centered effective rank.

    Select valid tokens, L2-normalize each with eps=1e-12, then center the M,D
    matrix across tokens. For singular values s, p=s/s.sum and effective rank
    is exp(-sum(p*log(p))). An exactly zero centered matrix has rank 0.
    Similarities are clipped to [-1,1]; negative similarities are retained.
    """
    student, teacher, mask = _inputs(student, teacher, validmask)
    selected_student, selected_teacher = student[mask], teacher[mask]
    student_zero = int((selected_student.norm(dim=-1) <= NORMALIZATION_EPS).sum())
    teacher_zero = int((selected_teacher.norm(dim=-1) <= NORMALIZATION_EPS).sum())
    u = F.normalize(selected_student, dim=-1, eps=NORMALIZATION_EPS)
    v = F.normalize(selected_teacher, dim=-1, eps=NORMALIZATION_EPS)
    cosine = (u * v).sum(dim=-1).clamp(-1, 1)
    student_stats, teacher_stats = _statistics(u), _statistics(v)
    return {"valid_token_count": int(mask.sum()), "feature_dimension": student.shape[-1],
            "valid_point_cosine_error": float((1 - cosine).mean().clamp(0, 2)),
            "student_mean_channel_variance": student_stats["mean_channel_variance"],
            "teacher_mean_channel_variance": teacher_stats["mean_channel_variance"],
            "student_effective_rank": student_stats["effective_rank"],
            "teacher_effective_rank": teacher_stats["effective_rank"],
            "student_centered_singular_value_sum": student_stats["centered_singular_value_sum"],
            "teacher_centered_singular_value_sum": teacher_stats["centered_singular_value_sum"],
            "maximum_centered_rank": student_stats["maximum_centered_rank"],
            "student_zero_norm_tokens": student_zero, "teacher_zero_norm_tokens": teacher_zero,
            "protocol": {"features": "valid token L2 normalization before all statistics", "normalization_eps": NORMALIZATION_EPS,
                         "cosine_clip": [-1, 1], "negative_cosine": "retained", "variance": "population variance per channel, then channel mean",
                         "centering": "valid token mean subtracted per channel", "svd_dtype": "float64", "svd_device": "CPU",
                         "effective_rank": "p=s/s.sum; exp(-sum(p*log(p))); zero centered matrix -> 0",
                         "numerical_rank_bound": "clipped to min(valid_token_count-1,feature_dimension)",
                         "autograd": "student and teacher detached", "restoration_quality_metric": False}}


@torch.no_grad()
def same_anchor_patch_identity_retrieval(student, teacher, validmask, token_grid):
    """Same-anchor spatial-patch top1 identity rate, without motion mapping.

    token_grid is (anchor_count,patch_height,patch_width), flattened T,H,W.
    Each student query searches valid teacher patches only in its own batch
    video and anchor. The correct answer is the same spatial patch index.
    Ties use torch.argmax's first valid patch; tied-query counts are reported.
    This checks the chosen spatial correspondence, not cross-frame tracking.
    """
    student, teacher, mask = _inputs(student, teacher, validmask)
    if len(token_grid) != 3 or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in token_grid):
        raise ValueError("token_grid must contain positive anchor,height,width integers")
    anchors, height, width = token_grid
    patches = height * width
    if anchors * patches != student.shape[1]:
        raise ValueError("token_grid does not match the feature token dimension")
    batch, _, channels = student.shape
    student = student.reshape(batch, anchors, patches, channels)
    teacher = teacher.reshape(batch, anchors, patches, channels)
    mask = mask.reshape(batch, anchors, patches)
    rows, hits, queries, empty = [], 0, 0, 0
    for b in range(batch):
        for anchor in range(anchors):
            valid = mask[b, anchor]
            count = int(valid.sum())
            if count == 0:
                empty += 1
                continue
            u = F.normalize(student[b, anchor, valid], dim=-1, eps=NORMALIZATION_EPS)
            v = F.normalize(teacher[b, anchor, valid], dim=-1, eps=NORMALIZATION_EPS)
            similarity = (u @ v.T).clamp(-1, 1)
            maxima = similarity.max(dim=-1).values
            selected = similarity.argmax(dim=-1)
            correct = selected == torch.arange(count)
            tied = (similarity == maxima[:, None]).sum(dim=-1) > 1
            anchor_hits = int(correct.sum())
            hits += anchor_hits
            queries += count
            rows.append({"batch_index": b, "anchor_index": anchor, "valid_patch_count": count,
                         "top1_identity_rate": anchor_hits / count,
                         "mean_corresponding_cosine": float(similarity.diag().mean()),
                         "tied_top1_query_count": int(tied.sum())})
    return {"same_anchor_spatial_patch_top1_identity_rate": hits / queries,
            "anchor_macro_top1_identity_rate": sum(row["top1_identity_rate"] for row in rows) / len(rows),
            "correct_query_count": hits, "valid_query_count": queries, "nonempty_anchor_count": len(rows),
            "empty_anchor_count": empty, "per_anchor": rows,
            "protocol": {"name": "same-anchor spatial-patch top1 identity retrieval", "flatten_order": "T,H,W",
                         "candidate_set": "valid spatial teacher patches in the same batch video and same anchor",
                         "target": "identical spatial patch index", "motion_compensated": False, "cross_frame": False,
                         "similarity": "L2-normalized cosine clipped[-1,1]; negatives retained",
                         "tie_policy": "first valid spatial patch via torch.argmax; exact-equality ties counted",
                         "autograd": "student and teacher detached", "restoration_quality_metric": False}}
