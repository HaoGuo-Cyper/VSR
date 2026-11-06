"""Frozen Wan VAE operations shared by data, inference and training.

Model imports are lazy. Import and CLI help never allocate a GPU or download.
Encoding keeps posterior mode and FP32; latent normalization is the pinned
Wan channelwise (z-mean)/std transform.
"""
from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from contextlib import nullcontext
import math

import torch

def _field(config, name):
    if isinstance(config, Mapping):
        if name not in config:
            raise ValueError(f"model config is missing {name!r}")
        return config[name]
    if not hasattr(config, name):
        raise ValueError(f"model config is missing {name!r}")
    return getattr(config, name)


def latent_geometry(vae_config, dit_config, frames, height, width):
    """Read compression and patch geometry; reject unsupported padding.

    ``temperal_downsample`` is the spelling in the Wan VAE configuration.
    The first-frame causal convention requires T = 1 mod temporal_stride.
    Runtime latent and token shapes must additionally be checked by the runner.
    """
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
           for v in (frames, height, width)):
        raise ValueError("frames, height and width must be positive integers")
    temporal_flags = tuple(_field(vae_config, "temperal_downsample"))
    dim_mult = tuple(_field(vae_config, "dim_mult"))
    if not dim_mult or len(temporal_flags) != len(dim_mult) - 1:
        raise ValueError("VAE downsampling flags must match len(dim_mult)-1")
    if any(flag not in (True, False) for flag in temporal_flags):
        raise ValueError("VAE temporal downsampling flags must be boolean")
    temporal_stride = 2 ** sum(bool(flag) for flag in temporal_flags)
    spatial_stride = 2 ** (len(dim_mult) - 1)
    patch_size = tuple(_field(dit_config, "patch_size"))
    if len(patch_size) != 3 or any(not isinstance(p, int) or p <= 0 for p in patch_size):
        raise ValueError("DiT patch_size must contain three positive integers")
    pt, ph, pw = patch_size
    if pt != 1:
        raise ValueError("stage-1 anchor scheme requires temporal patch_size=1")
    if (frames - 1) % temporal_stride:
        raise ValueError("causal VAE requires frames = 1 mod temporal_stride")
    if height % (spatial_stride * ph) or width % (spatial_stride * pw):
        raise ValueError("height/width must be divisible by VAE stride * DiT patch")
    latent_grid = [(frames - 1) // temporal_stride + 1,
                   height // spatial_stride, width // spatial_stride]
    token_grid = [latent_grid[0] // pt, latent_grid[1] // ph, latent_grid[2] // pw]
    return {
        "temporal_stride": temporal_stride,
        "spatial_stride": spatial_stride,
        "patch_size": list(patch_size),
        "latent_grid": latent_grid,
        "token_grid": token_grid,
        "anchors": list(range(0, frames, temporal_stride)),
        "flatten_order": "T,H,W",
    }



def module_device(module, fallback="cpu"):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def normalize_latent(z, vae, inverse=False):
    if not isinstance(z, torch.Tensor) or z.ndim != 5 or not z.is_floating_point() or not bool(torch.isfinite(z).all()):
        raise ValueError("Expected finite floating latent [B,C,T,H,W]")
    mean = z.new_tensor(_field(vae.config,"latents_mean")).view(1,-1,1,1,1)
    std = z.new_tensor(_field(vae.config,"latents_std")).view(1,-1,1,1,1)
    if mean.shape[1] != z.shape[1] or std.shape[1] != z.shape[1] or not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()) or not bool((std > 0).all()):
        raise ValueError("VAE normalization channels/values are invalid")
    result = z * std + mean if inverse else (z - mean) / std
    if not bool(torch.isfinite(result).all()):
        raise ValueError("VAE normalization overflow")
    return result


def inverse_normalize(z, vae):
    return normalize_latent(z, vae, inverse=True)


def load_frozen_vae(config_or_model_dir, device="cpu"):
    if isinstance(config_or_model_dir, Mapping):
        model_dir = config_or_model_dir.get("model",config_or_model_dir).get("local_dir")
    else:
        model_dir = config_or_model_dir
    if model_dir is None or not (Path(model_dir)/"vae/config.json").is_file():
        raise FileNotFoundError("Explicit local Wan VAE directory/config required")
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(str(model_dir), subfolder="vae", dtype=torch.float32, local_files_only=True, use_safetensors=True)
    return vae.requires_grad_(False).eval().to(device=device, dtype=torch.float32)


def _rgb(value):
    if not isinstance(value, torch.Tensor) or value.ndim != 5 or value.shape[0] != 1 or value.shape[1] != 3 or not all(value.shape) or not value.is_floating_point() or not bool(torch.isfinite(value).all()) or float(value.min()) < 0 or float(value.max()) > 1:
        raise ValueError("Expected finite RGB [1,3,T,H,W] in [0,1]")


def _frozen_fp32(vae):
    if any(p.requires_grad or p.dtype != torch.float32 for p in vae.parameters()):
        raise ValueError("Expected a frozen FP32 VAE")


@torch.no_grad()
def encode_video(vae, video):
    _rgb(video); _frozen_fp32(vae)
    device = module_device(vae)
    try:
        with torch.autocast(device.type, enabled=False):
            raw = vae.encode(video.to(device=device,dtype=torch.float32)*2-1).latent_dist.mode()
            result = normalize_latent(raw,vae).detach().cpu()
    finally:
        if hasattr(vae,"clear_cache"): vae.clear_cache()
    return result


@torch.no_grad()
def decode_latent(vae, normalized, device=None):
    _frozen_fp32(vae)
    target = module_device(vae) if device is None else torch.device(device)
    if target != module_device(vae):
        raise ValueError("Decode device differs from the loaded VAE")
    try:
        with torch.autocast(target.type, enabled=False):
            raw = inverse_normalize(normalized.to(device=target,dtype=torch.float32),vae)
            result = (vae.decode(raw).sample.float()/2+.5).clamp(0,1).detach().cpu()
            _rgb(result)
    finally:
        if hasattr(vae,"clear_cache"): vae.clear_cache()
    return result


@torch.no_grad()
def encode_pair(vae, clip, geometry=None, *, return_roundtrip=False):
    _rgb(clip["hr"]); _rgb(clip["lr_up"]); _frozen_fp32(vae)
    if clip["hr"].shape != clip["lr_up"].shape:
        raise ValueError("Paired HR and upsampledLR shapes differ")
    device = module_device(vae)
    # Preserve original order: HR encode, LR encode, HR decode. Cache boundaries
    # remain the VAE's own encode/decode call boundaries.
    try:
        with torch.autocast(device.type, enabled=False):
            clean_raw = vae.encode(clip["hr"].to(device=device,dtype=torch.float32)*2-1).latent_dist.mode()
            low_raw = vae.encode(clip["lr_up"].to(device=device,dtype=torch.float32)*2-1).latent_dist.mode()
            if geometry is not None:
                expected=(1,16,*geometry["latent_grid"])
                if tuple(clean_raw.shape)!=expected or tuple(low_raw.shape)!=expected:
                    raise ValueError("VAE latent geometry differs")
            pair={"clean":normalize_latent(clean_raw,vae).detach().cpu(),"low":normalize_latent(low_raw,vae).detach().cpu()}
            if not return_roundtrip:
                return pair
            restored=(vae.decode(clean_raw).sample.float()/2+.5).clamp(0,1).detach().cpu()
            if restored.shape!=clip["hr"].shape or not bool(torch.isfinite(restored).all()):
                raise ValueError("VAE roundtrip shape/finite-value failure")
            mse=float(torch.nn.functional.mse_loss(restored,clip["hr"].float().cpu()))
            return pair,restored,{"mse_rgb_01":mse,"psnr_rgb_db":-10*math.log10(max(mse,1e-12)),"definition":"allRGB, no shave"}
    finally:
        if hasattr(vae,"clear_cache"): vae.clear_cache()



# Instance-local inference cache policy: original delegate preserved for T>1.
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


install_cache_policy = install

def validate_cache_ledger(rows, enabled, tile_count):
    _require(isinstance(rows,list) and len(rows) == tile_count+int(enabled),'Missing or extra per-restore policy records')
    for index,row in enumerate(rows):
        if enabled and index == 0:
            _require(row['route'] == 'T1_no_future_cache_retention' and row['status'] == 'completed'
                 and row['encoder_pre_calls'] == row['encoder_post_calls'] == 1
                 and row['feat_idx_argument_identity_preserved'] is True and not row['cleanup_errors']
                 and row['parameter_buffer_names_devices_dtypes_identities_and_flags_unchanged'] is True
                 and all(item['tensor_bytes'] == 0 for item in row['encoder_end_cache'].values()),'T1 route/cleanup audit failed')
        else:
            _require(row['route'] == 'original_delegate' and row['encoder_hooks_installed'] is False,
                 'Every multi-frame tile must delegate the unchanged original encode')


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description="Frozen VAE API; GPU use requires an explicit caller and protocol")
    parser.parse_args(argv)
    parser.print_help()


if __name__=="__main__":main()
