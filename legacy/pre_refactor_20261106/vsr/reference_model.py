"""Optional-HRI wrapper preserving the stage-1 ConditionalWan data path.

All reference variants use the same HRI module and initialization seed. This
module provides integration mechanics only; its tests establish no reference
restoration or propagation benefit.
"""
from __future__ import annotations

import torch

from .model import ConditionalWan
from .reference import HRIAdapter


class ReferenceConditionalWan(ConditionalWan):
    """Add reference attention before one Wan block, after its LRV injection.

    The reference index is in the current input clip's coordinates. An absent
    reference defaults to a disabled branch. A supplied reference defaults to
    valid for every video unless an explicit boolean ``ref_valid`` is given.

    Call ``clear_features()`` and ``clear_reference()`` after backward. During
    non-reentrant checkpoint backward, the reference context must still refer
    to that forward. As with ConditionalWan, interleaving several forwards
    before their backwards is unsupported. Starting the next forward replaces
    the old reference context, including when the next input has no reference.
    """

    def __init__(self, transformer, *, reference_layer=9, reference_heads=None,
                 reference_stem_dim=128, reference_seed=20261202, **conditioning_options):
        super().__init__(transformer, **conditioning_options)
        if not isinstance(reference_layer, int) or not 0 <= reference_layer < len(transformer.blocks):
            raise ValueError("Reference injection layer must be a valid zero-based Wan block index")
        dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
        heads = transformer.config.num_attention_heads if reference_heads is None else reference_heads
        # Every factory is explicitly CPU: fork_rng need only preserve the CPU
        # default generator. Seeding it directly does not alter CUDA RNG states.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(reference_seed))
            with torch.device("cpu"):
                self.reference_adapter = HRIAdapter(
                    dim=dim, latent_channels=transformer.config.in_channels,
                    heads=heads, stem_dim=reference_stem_dim,
                )
        self.reference_layer = reference_layer
        self.reference_seed = int(reference_seed)
        self._reference_context = None
        # Registration order is deliberate: super already registered the LRV
        # pre-hook on this block. Reference queries see the LR-conditioned h.
        self._hooks.append(transformer.blocks[reference_layer].register_forward_pre_hook(self._inject_reference))

    def _inject_reference(self, _module, args):
        if self._reference_context is None:
            return None
        reference, indices, frame_count, valid = self._reference_context
        hidden = args[0]
        residual = self.reference_adapter(hidden, reference, indices, frame_count, valid)
        return (hidden + residual, *args[1:])

    def clear_reference(self):
        """Release supplied reference inputs after their backward has finished."""
        self._reference_context = None

    def forward(self, noisy, timestep, text, lrv_latent, enable_lrv=True, *,
                ref_latent=None, ref_index=None, frame_count=None, ref_valid=None):
        # The base forward invokes self.clear_features(). It deliberately only
        # clears the LRV graph/capture; it must not erase this reference state.
        self.clear_reference()
        if ref_latent is None and ref_valid is None:
            return super().forward(noisy, timestep, text, lrv_latent, enable_lrv=enable_lrv)
        batch = noisy.shape[0]
        valid = (torch.ones(batch, dtype=torch.bool, device=noisy.device) if ref_valid is None
                 else torch.as_tensor(ref_valid, device=noisy.device))
        if valid.dtype != torch.bool or valid.shape != (batch,):
            raise ValueError("ref_valid must be a boolean B vector")
        if bool(valid.any()) and (ref_latent is None or ref_index is None or frame_count is None):
            raise ValueError("Active HRI requires reference latent, known clip frame index and original frame count")
        self._reference_context = (ref_latent, ref_index, frame_count, valid)
        return super().forward(noisy, timestep, text, lrv_latent, enable_lrv=enable_lrv)
