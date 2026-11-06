"""Low-frequency detached CPU statistics for the new self-distillation arms.

No model/optimizer/cache is accepted, and no random operation is performed.
Call after the measured Step has returned. The GPU->CPU snapshot copies happen
inside Step; float64 centering/SVD below is explicitly outside Step timing.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .feature_diagnostics import _statistics, NORMALIZATION_EPS


@torch.no_grad()
def analyze_feature_snapshot(snapshot):
    fields = ('student_raw','teacher_raw','student_projected','teacher_normalized')
    features = snapshot['features']
    if not isinstance(features,dict) or set(features) != set(fields):
        raise ValueError('Complete raw/projected student and raw/LayerNorm teacher snapshots required')
    result = {}
    counts = set()
    for name in fields:
        value = features[name]
        if (not isinstance(value,torch.Tensor) or value.device.type != 'cpu' or value.requires_grad
                or value.ndim != 2 or min(value.shape) <= 0 or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Feature diagnostics accept finite detached nonempty CPU [tokens,channels] only')
        counts.add(value.shape[0])
        value = value.to(dtype=torch.float64)
        offset = value - value[:1]
        centered = offset - offset.mean(dim=0,keepdim=True)
        channel_variance = centered.square().mean(dim=0).clamp_min(0)
        normalized = F.normalize(value,dim=-1,eps=NORMALIZATION_EPS)
        normalized_stats = _statistics(normalized)
        result[name] = {'tokens':value.shape[0],'channels':value.shape[1],
            'mean_population_channel_std':float(channel_variance.sqrt().mean()),
            'rms_population_channel_std':float(channel_variance.mean().sqrt()),
            'normalized_rms_channel_std':math.sqrt(normalized_stats['mean_channel_variance']),
            'normalized_centered_effective_rank':normalized_stats['effective_rank'],
            'maximum_centered_rank':normalized_stats['maximum_centered_rank'],
            'normalized_centered_singular_value_sum':normalized_stats['centered_singular_value_sum']}
    if len(counts) != 1 or snapshot.get('selected_tokens') != next(iter(counts)):
        raise ValueError('All feature snapshots must describe the same selected tokens')
    return {key:value for key,value in snapshot.items() if key != 'features'} | {
        'statistics':result,'scope':{'quality_metric':False,'selection':'corrupted tokens in sigma>0.3 only',
            'feature_state':'features used in this step before its optimizer update',
            'dtype':'CPU float64','SVD_frequency':'only explicitly drained first-active/every-N snapshots',
            'raw_std':'population variance across selected tokens per channel; report mean SD and RMS SD',
            'effective_rank':'L2 normalize tokens eps1e-12, center tokens, p=s/s.sum, exp(-sum(p*log(p))); zero ->0',
            'timing':'CPU statistics/SVD outside Step; owned snapshot copy is inside the complete Step',
            'model_optimizer_gradients_RNG_accessed':False}}
