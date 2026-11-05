"""Instance-only, inference-only Wan T1 future-cache retention policy.

``policy = install(vae)`` installs immediately on the same real VAE. Use
``with install(vae) as policy: ...`` or call ``policy.remove()`` explicitly.
Install before AuditedNativeRestore adds its temporary encode observer; remove
after that observer has restored this wrapper. No production source is patched.
This is a scoped implementation, not evidence of numerical/capacity parity.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
from types import MethodType

import torch


_ATTRIBUTE = "_vsr_t1_cache_policy"
_MISSING = object()


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _cache_metadata(vae):
    """Metadata only; never retain a tensor or copy its contents."""
    result = {}
    for name in ("_enc_feat_map", "_feat_map"):
        cache = getattr(vae, name, None)
        if cache is None:
            result[name] = {"exists": False, "slots": 0, "tensor_entries": 0, "tensor_bytes": 0}
            continue
        tensors = [value for value in cache if isinstance(value, torch.Tensor)]
        result[name] = {"exists": True, "slots": len(cache), "tensor_entries": len(tensors),
                       "tensor_bytes": sum(value.numel() * value.element_size() for value in tensors),
                       "all_None": all(value is None for value in cache)}
    return result


def _identity_digest(vae):
    # No parameter/buffer values, tensor references, RNG calls or device moves.
    rows = [(kind, name, id(value), str(value.device), str(value.dtype), list(value.shape),
             bool(value.requires_grad))
            for kind, values in (("parameter", vae.named_parameters()), ("buffer", vae.named_buffers()))
            for name, value in values]
    flags = [(name, id(module), bool(module.training)) for name, module in vae.named_modules()]
    return hashlib.sha256(json.dumps([rows, flags], separators=(",", ":")).encode()).hexdigest()


class T1CachePolicy:
    """Own one encode wrapper and per-call temporary encoder hooks."""
    def __init__(self, vae, *, max_records=256):
        _require(isinstance(max_records, int) and not isinstance(max_records, bool) and max_records > 0,
                 "max_records must be a positive integer")
        _require(getattr(vae, _ATTRIBUTE, None) is None, "A T1 policy is already attached")
        _require(callable(getattr(vae, "encode", None)) and hasattr(vae, "encoder")
                 and callable(getattr(vae, "clear_cache", None)), "Expected a real Wan VAE encode/encoder/clear_cache")
        self.vae = vae
        self.max_records = max_records
        self._ledger = []
        self.dropped_records = 0
        self._active = False
        self._removed = False
        self._prior_instance_encode = vae.__dict__.get("encode", _MISSING)
        self._original_encode = vae.encode

        @functools.wraps(self._original_encode)
        def encode_wrapper(instance, *args, **kwargs):
            _require(instance is self.vae and not self._removed, "Detached T1 policy was called")
            return self._encode(*args, **kwargs)

        self._wrapper = MethodType(encode_wrapper, vae)
        vae.encode = self._wrapper
        setattr(vae, _ATTRIBUTE, self)

    @property
    def ledger(self):
        """Independent JSON-compatible observations; contains no tensors."""
        return copy.deepcopy(self._ledger)

    def _record(self, row):
        if len(self._ledger) < self.max_records:
            self._ledger.append(row)
        else:
            self.dropped_records += 1

    def _encode(self, *args, **kwargs):
        x = args[0] if args else kwargs.get("x")
        shape = list(x.shape) if isinstance(x, torch.Tensor) else None
        # All non-T1 calls delegate with the exact original positional/keyword
        # objects. No encoder hook, cache manipulation or numerical guard runs.
        if shape is None or len(shape) != 5 or shape[2] != 1:
            self._record({"route": "original_delegate", "input_shape": shape, "encoder_hooks_installed": False})
            return self._original_encode(*args, **kwargs)

        _require(not self._active, "Concurrent/reentrant T1 encoding is outside this policy")
        _require(shape[0] == 1 and shape[1] == 3, "T1 cache policy requires B1/C3 input")
        _require(not torch.is_grad_enabled(), "T1 cache policy is inference-only: use torch.no_grad()")
        _require(x.dtype == torch.float32, "This declared T1 policy requires original FP32 VAE input")
        _require(getattr(self.vae.config, "patch_size", None) is None
                 and self.vae.use_tiling is False and self.vae.use_slicing is False,
                 "T1 policy requires patch_size=None and original untiled/unsliced VAE path")
        _require(getattr(self.vae.config, "is_residual", False) is False,
                 "Only the reviewed Wan2.1 non-residual encoder is in scope")
        row = {"route": "T1_no_future_cache_retention", "input_shape": shape,
               "input_dtype": str(x.dtype), "input_device": str(x.device),
               "original_encode_called": True, "encoder_pre_calls": 0, "encoder_post_calls": 0,
               "feat_idx_argument_identity_preserved": True, "cleanup_errors": []}
        handles = []
        original_error = None
        before = _identity_digest(self.vae)
        self._active = True

        def before_encoder(module, positional, keywords):
            row["encoder_pre_calls"] += 1
            _require(row["encoder_pre_calls"] == 1, "Expected exactly one encoder call for T1")
            _require(len(positional) == 1 and list(positional[0].shape) == shape,
                     "Original encoder did not receive the exact declared T1 shape")
            cache = keywords.get("feat_cache", _MISSING)
            _require(isinstance(cache, list) and cache is getattr(self.vae, "_enc_feat_map", None)
                     and len(cache) > 0 and all(value is None for value in cache),
                     "Original _encode must establish its fresh all-None encoder cache")
            _require("feat_idx" in keywords and keywords["feat_idx"] is self.vae._enc_conv_idx
                     and keywords["feat_idx"] == [0], "Original fresh feat_idx is required")
            row["original_fresh_cache_slots"] = len(cache)
            row["original_cache_all_None"] = True
            row["encoder_exact_T1_shape"] = list(positional[0].shape)
            adjusted = dict(keywords)
            adjusted["feat_cache"] = None  # The sole numerical-call input change.
            _require(adjusted["feat_idx"] is keywords["feat_idx"], "feat_idx identity changed")
            return positional, adjusted

        def after_encoder(module, positional, keywords, output):
            # always_call=True also observes partial failures. Never raise from
            # this audit hook and hide an original convolution/attention error.
            row["encoder_post_calls"] += 1
            try:
                row["encoder_end_cache"] = _cache_metadata(self.vae)
            except BaseException as error:
                row["cleanup_errors"].append("encoder post observation: " + type(error).__name__ + ": " + str(error))

        try:
            handles.append(self.vae.encoder.register_forward_pre_hook(before_encoder, with_kwargs=True))
            handles.append(self.vae.encoder.register_forward_hook(after_encoder, with_kwargs=True, always_call=True))
            result = self._original_encode(*args, **kwargs)
            _require(row["encoder_pre_calls"] == row["encoder_post_calls"] == 1,
                     "Original T1 encode did not call encoder exactly once")
            _require(all(value["tensor_bytes"] == 0 for value in row["encoder_end_cache"].values()),
                     "T1 encoder retained a temporal feature tensor")
            row["status"] = "completed"
            return result
        except BaseException as error:
            original_error = error
            row["status"] = "failed"
            row["original_exception"] = type(error).__name__ + ": " + str(error)
            raise
        finally:
            for handle in reversed(handles):
                try:
                    handle.remove()
                except BaseException as error:
                    row["cleanup_errors"].append("hook removal: " + type(error).__name__ + ": " + str(error))
            if original_error is not None:
                try:
                    self.vae.clear_cache()
                except BaseException as error:
                    row["cleanup_errors"].append("failure cache cleanup: " + type(error).__name__ + ": " + str(error))
            try:
                row["parameter_buffer_names_devices_dtypes_identities_and_flags_unchanged"] = before == _identity_digest(self.vae)
                row["cache_after_call"] = _cache_metadata(self.vae)
            except BaseException as error:
                row["cleanup_errors"].append("final observation: " + type(error).__name__ + ": " + str(error))
            self._active = False
            self._record(row)
            if original_error is None:
                _require(not row["cleanup_errors"] and row.get("parameter_buffer_names_devices_dtypes_identities_and_flags_unchanged") is True,
                         "T1 policy cleanup/identity audit failed; see ledger")

    def remove(self):
        """Restore the prior instance encode attribute; remove in reverse stack order."""
        if self._removed:
            return
        _require(not self._active, "Cannot remove policy during an active encode")
        _require(self.vae.__dict__.get("encode", _MISSING) is self._wrapper,
                 "Remove outer encode observer first; refusing to overwrite another wrapper")
        if self._prior_instance_encode is _MISSING:
            del self.vae.encode
        else:
            self.vae.encode = self._prior_instance_encode
        if getattr(self.vae, _ATTRIBUTE, None) is self:
            delattr(self.vae, _ATTRIBUTE)
        self._removed = True

    def __enter__(self):
        _require(not self._removed, "Cannot re-enter a removed T1 policy")
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.remove()
        except BaseException as error:
            if value is None:
                raise
            value.add_note("T1 policy remove failed without replacing original exception: " + type(error).__name__ + ": " + str(error))
        return False


def install(vae, *, max_records=256):
    """Attach to, and return a policy for, this same real VAE instance."""
    return T1CachePolicy(vae, max_records=max_records)
