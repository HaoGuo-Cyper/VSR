"""Explicit inference architecture for new NoRef continuation arms.

This stdlib-only marker never changes model/text-cache identity. Unmarked
historical Base/Align/Relation/Both and HRI retain their existing projection
behavior. The method MLP/EMA are training-only and outside the restoration DiT.
"""
from __future__ import annotations

METHOD_ARMS = ('flow_only', 'C1', 'A2')
NOREF_BASE_ARCHITECTURE = 'noref_base_v1'


def validate_restoration_architecture(config,variant):
    """Return whether the restoration wrapper needs its legacy DINO projector."""
    marker = config.get('restoration_architecture')
    if marker is None:
        if variant in METHOD_ARMS:
            raise ValueError('Method variants require explicit restoration_architecture=noref_base_v1')
        return variant != 'Base'
    if marker != NOREF_BASE_ARCHITECTURE:
        raise ValueError('Unknown restoration_architecture marker')
    variants = config.get('training',{}).get('variants')
    if (not isinstance(variants,list) or not variants or any(name not in METHOD_ARMS for name in variants)
            or len(variants) != len(set(variants)) or variant not in variants):
        raise ValueError('noref_base_v1 only supports the explicitly named flow_only/C1/A2 variants')
    if (config.get('reference',{}).get('enabled',False) is not False
            or config.get('conditioning',{}).get('hri_enabled',False) is not False):
        raise ValueError('noref_base_v1 forbids reference.enabled and conditioning.hri_enabled')
    return False
