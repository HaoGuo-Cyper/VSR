"""Frame-preserving paired metric pools for supplied-reference evaluation.

Run the official perceptual evaluator on the complete original sequence once,
then filter its existing temporal edges. Never concatenate frames around an
excluded reference: that would invent new adjacent pairs and alter cutfr.
The caller chooses its spatial pixel contract (saved float RGB or q/255 PNG).
This helper does not quantize it; the TecoGAN evaluator retains its documented
own uint8/crop preprocessing for temporal metrics.
"""
from __future__ import annotations

import torch

from .evaluation import _validate_video, rgb_metrics
from .perceptual_metrics import TLP_KEY, TLP100_KEY, TOF_KEY


PAIRED_METRIC_KEYS = ("psnr_rgb_db", "ssim_rgb", "mse_rgb_01", "lpips_alex_v0_1", TLP_KEY, TLP100_KEY, TOF_KEY)
POOL_NAMES = ("all_frames", "matched_nonreference", "reference_frame_separate")


@torch.no_grad()
def paired_metric_pools(prediction, target, perceptual, *, reference_index=0):
    """Return all / identical-nonref / reference-only RGB+LPIPS+time pools.

    Inputs are full RGB [1,3,T,H,W] in [0,1]. ``perceptual`` supplies the
    project's pinned PerceptualMetrics.evaluate interface. An empty temporal
    pool returns None, never a fabricated zero. Frame and pair indices always
    refer to the complete original input sequence.
    """
    _validate_video(prediction)
    _validate_video(target)
    if prediction.shape != target.shape or prediction.shape[2] < 6:
        raise ValueError("Paired full-sequence temporal evaluation needs matching videos with >=6 frames")
    if (not isinstance(reference_index, int) or isinstance(reference_index, bool)
            or not 0 <= reference_index < prediction.shape[2]):
        raise ValueError("Reference index is outside the original evaluated sequence")
    full = perceptual.evaluate(prediction, target, include_temporal=True)
    lpips = full["lpips_per_frame"]
    if len(lpips) != prediction.shape[2]:
        raise ValueError("Official LPIPS must return every original frame")
    pairs = full["tecogan"]["per_pair"]
    frame_count = prediction.shape[2]
    pair_ids = []
    for pair in pairs:
        endpoints = pair["indices"]
        if (len(endpoints) != 2 or any(not isinstance(value, int) or isinstance(value, bool) for value in endpoints)
                or not 0 <= endpoints[0] < endpoints[1] < frame_count or endpoints[1] != endpoints[0]+1):
            raise ValueError("Perceptual evaluator must retain original adjacent temporal pairs")
        pair_ids.append(tuple(endpoints))
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("Duplicate temporal metric pairs")
    pools = {"all_frames": list(range(frame_count)),
             "matched_nonreference": [index for index in range(frame_count) if index != reference_index],
             "reference_frame_separate": [reference_index]}
    result = {"kind": "paired", "primary_pool": "matched_nonreference", "excluded_reference_index": reference_index,
              "perceptual_audit": full.get("audit"), "lpips_protocol": full.get("lpips_protocol"),
              "temporal_protocol": full["tecogan"].get("protocol"),
              "input_pixel_contract": "caller-provided full RGB; spatial scores not requantized by metric pool helper"}
    for name, indices in pools.items():
        rgb = rgb_metrics(prediction[:, :, indices], target[:, :, indices])
        for original, row in zip(indices, rgb["per_frame"]):
            row["original_frame_index"] = original
        selected = [pair for pair in pairs if all(endpoint in indices for endpoint in pair["indices"])]
        result[name] = {**rgb, "frame_indices": indices,
            "lpips_alex_v0_1": sum(float(lpips[index]) for index in indices)/len(indices),
            "lpips_per_frame": [float(lpips[index]) for index in indices],
            **{key: sum(float(pair[key]) for pair in selected)/len(selected) if selected else None
               for key in (TLP_KEY, TLP100_KEY, TOF_KEY)},
            "temporal_pair_indices": [pair["indices"] for pair in selected],
            "temporal_pair_count": len(selected), "per_temporal_pair": selected}
    return result


def paired_metrics(prediction, target, perceptual, *, excluded_reference=0):
    """Compatibility alias; prefer reference_index on paired_metric_pools."""
    return paired_metric_pools(prediction, target, perceptual, reference_index=excluded_reference)
