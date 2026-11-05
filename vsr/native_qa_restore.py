"""Observed structural audits around the unchanged native restoration operator.

The hook delegates every tile to ``inference.sample_tile_latent``. This is an
infrastructure check, never a quality score or model-selection rule. Pass the
predeclared plans to :class:`AuditedNativeRestore` as an evaluator factory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from . import inference


class NativeQAAuditError(RuntimeError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def _tensor_sha(value):
    value = value.detach().cpu().contiguous()
    header = json.dumps([str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\n" + value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _network_state(network):
    digest = hashlib.sha256()
    items = [("parameter:" + n, p) for n, p in network.named_parameters()]
    items += [("buffer:" + n, p) for n, p in network.named_buffers()]
    for name, tensor in sorted(items):
        digest.update(name.encode() + b"\0" + _tensor_sha(tensor).encode())
    return digest.hexdigest()


def _rng_snapshot():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().clone(),
            "cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None}


def _rng_equal(a, b):
    x, y = a["numpy"], b["numpy"]
    cuda_same = (a["cuda"] is None and b["cuda"] is None) or (
        a["cuda"] is not None and b["cuda"] is not None and len(a["cuda"]) == len(b["cuda"])
        and all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"])))
    return (a["python"] == b["python"] and x[0] == y[0] and np.array_equal(x[1], y[1])
            and x[2:] == y[2:] and torch.equal(a["torch"], b["torch"]) and cuda_same)


def _restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if value["cuda"] is not None:
        torch.cuda.set_rng_state_all(value["cuda"])


def _contexts_empty(model):
    # token_grid is persistent geometry, not a held forward graph.
    names = ("_captured", "_condition_tokens", "_reference_context", "reference_context")
    return (all(getattr(model, name, None) is None for name in names)
            and not bool(getattr(model, "_capture_enabled", False)))


def _grads(networks):
    return [(label + ":" + name, p, p.requires_grad,
             None if p.grad is None else (id(p.grad), _tensor_sha(p.grad)))
            for label, network in networks for name, p in network.named_parameters()]


def _grads_equal(before, networks):
    after = _grads(networks)
    return (len(before) == len(after) and all(
        a[0] == b[0] and a[1] is b[1] and a[2:] == b[2:] for a, b in zip(before, after)))


def _plan(value):
    return json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value


class AuditedNativeRestore:
    """Callable ``frozen_evaluation`` restore factory, with no GPU orchestration.

    Plans are matched by exact input LR shape. The enclosing service must
    already enforce GPU exclusion and freeze source/checkpoint/code hashes.
    ``last_audit`` retains partial observations when the real operator fails.
    """
    def __init__(self, plans):
        if isinstance(plans, (dict, str, Path)):
            plans = [plans]
        self.plans = [json.loads(json.dumps(_plan(p))) for p in plans]
        if not self.plans:
            raise ValueError("At least one predeclared native QA plan is required")
        self.last_audit = None

    @torch.no_grad()
    def __call__(self, model, vae, lrv, text, config, *, steps=20, seed=42,
                 tile_hw=(256, 448), overlap=64, reference_image=None,
                 reference_index=None, **unexpected):
        started = time.perf_counter()
        self.last_audit = {"schema": "vsr_observed_native_operator_qa_v1",
                           "status": "failed", "phase": "precondition_validation"}
        if unexpected:
            raise ValueError(f"Native QA factory refuses extra restore options: {sorted(unexpected)}")
        matched = [p for p in self.plans if p["input_preparation"]["LR_shape"] == list(lrv.shape)]
        if len(matched) != 1:
            raise ValueError("Input shape must match exactly one predeclared native QA plan")
        plan = matched[0]
        spec = plan["restoration"]
        for actual, expected, label in [(steps, spec["steps"], "steps"), (seed, spec["sampling_seed"], "seed"),
                (list(tile_hw), spec["HR_tile_hw"], "tile_hw"), (overlap, spec["overlap"], "overlap")]:
            if actual != expected:
                raise ValueError(f"Native QA {label} differs from predeclaration")
        enabled = reference_image is not None
        if enabled and reference_index != 0:
            raise ValueError("Predeclared native QA supplies original frame0 only")
        if not _contexts_empty(model):
            raise ValueError("Clear outstanding model features/reference context before native QA")
        networks = [("model", model), ("vae", vae)]
        flags = [(module, module.training) for _, net in networks for module in net.modules()]
        before_grads = _grads(networks)
        before_state = {name: _network_state(net) for name, net in networks}
        rng_before = _rng_snapshot()
        expected_noise = torch.randn(tuple(spec["global_CPU_Gaussian_noise_shape"]),
            generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32)
        tiles = spec["tile_hr_tlhw"]
        latent_tiles = spec["tile_latent_tlhw"]
        if len(tiles) != spec["tile_count"] or len(latent_tiles) != len(tiles):
            raise ValueError("Predeclared tile list/count mismatch")
        encode_rows, sampler_rows = [], []
        audit = {"schema": "vsr_observed_native_operator_qa_v1", "status": "running",
            "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "input_LR_sha256": _tensor_sha(lrv), "text_sha256": _tensor_sha(text),
            "provided_HRI_sha256": _tensor_sha(reference_image) if enabled else None,
            "expected_global_noise_sha256": _tensor_sha(expected_noise),
            "parameters_buffers_sha256_before": before_state,
            "sampler_delegation": "observed hook calls unchanged vsr.inference.sample_tile_latent; no synthetic sampler",
            "sample_tile_latent_calls": sampler_rows, "VAE_encode_calls": encode_rows,
            "quality_score_acceptance_or_selection": False, "optimizer_updates": 0,
            "cost_scope": "audited restore wall includes tensor hashing/state/RNG observation, not an uninstrumented inference benchmark"}
        self.last_audit = audit
        original_encode = vae.encode
        sentinel = object()
        instance_encode = vae.__dict__.get("encode", sentinel)
        reference_object = None
        actual_global = None
        global_storage_pointer = None

        def observed_encode(pixels, *args, **kwargs):
            i = len(encode_rows)
            hri_call = enabled and i == 0
            expected_shape = ([1, 3, 1, *spec["padded_HR_T_HW"][1:]] if hri_call else
                [1, 3, spec["padded_HR_T_HW"][0], tiles[i-int(enabled)][2], tiles[i-int(enabled)][3]]
                if 0 <= i-int(enabled) < len(tiles) else None)
            row = {"ordinal": i, "role": "whole_HRI_T1" if hri_call else "LR_tile",
                   "shape": list(pixels.shape), "pixels_sha256": _tensor_sha(pixels)}
            encode_rows.append(row)
            if expected_shape is None or list(pixels.shape) != expected_shape:
                raise RuntimeError("Observed VAE encode order/shape differs from predeclared T1/tiles")
            return original_encode(pixels, *args, **kwargs)

        def observed_sampler(*args, **kwargs):
            nonlocal reference_object, actual_global, global_storage_pointer
            i = len(sampler_rows)
            if i >= len(tiles):
                raise RuntimeError("Observed more sampler calls than declared tiles")
            noise = kwargs["initial_noise"]
            top, left, height, width = latent_tiles[i]
            expected_crop = expected_noise[:, :, :, top:top+height, left:left+width]
            if noise.device.type != "cpu" or noise.dtype != torch.float32 or not torch.equal(noise, expected_crop):
                raise RuntimeError("Observed tile noise differs from the declared global seeded CPU field")
            root = noise
            while isinstance(getattr(root, "_base", None), torch.Tensor):
                root = root._base
            if tuple(root.shape) != tuple(expected_noise.shape) or not torch.equal(root, expected_noise):
                raise RuntimeError("Observed noise is not a view of the complete declared Gaussian field")
            pointer = root.untyped_storage().data_ptr()
            if global_storage_pointer is None:
                global_storage_pointer, actual_global = pointer, _tensor_sha(root)
            elif pointer != global_storage_pointer:
                raise RuntimeError("Different full Gaussian fields were allocated across tiles")
            ref = kwargs.get("ref_latent")
            if kwargs.get("ref_valid") is not enabled or kwargs.get("ref_index") != (reference_index if enabled else None):
                raise RuntimeError("Observed sampler reference validity/index differs from supplied mode")
            if kwargs.get("frame_count") != lrv.shape[2]:
                raise RuntimeError("Observed sampler changed original frame count")
            if enabled:
                if ref is None or tuple(ref.shape) != (1, expected_noise.shape[1], 1, *expected_noise.shape[-2:]):
                    raise RuntimeError("Observed HRI latent is not the complete single-frame spatial field")
                if reference_object is None:
                    reference_object = ref
                elif ref is not reference_object:
                    raise RuntimeError("Different whole HRI latent objects were supplied across tiles")
            elif ref is not None:
                raise RuntimeError("HRI-off sampler received reference pixels/latent")
            sampler_rows.append({"ordinal": i, "hr_tlhw": tiles[i], "latent_tlhw": latent_tiles[i],
                "noise_sha256": _tensor_sha(noise), "reference_valid": enabled,
                "reference_index": reference_index if enabled else None,
                "reference_latent_sha256": _tensor_sha(ref) if enabled else None})
            return inference.sample_tile_latent(*args, **kwargs)

        error = None
        result = None
        vae.encode = observed_encode
        try:
            result = inference.restore_video(model, vae, lrv, text, config, steps=steps, seed=seed,
                tile_hw=tile_hw, overlap=overlap, reference_image=reference_image,
                reference_index=reference_index, sampler_fn=observed_sampler)
        except BaseException as caught:
            error = caught
        finally:
            if instance_encode is sentinel:
                del vae.encode
            else:
                vae.encode = instance_encode
            checks = {"module_training_flags_restored": all(m.training == v for m, v in flags),
                      "contexts_cleared": _contexts_empty(model)}
            audit_errors = []
            try:
                checks["parameter_gradients_and_requires_grad_unchanged"] = _grads_equal(before_grads, networks)
                audit["parameters_buffers_sha256_after"] = {n: _network_state(v) for n, v in networks}
                checks["all_parameters_buffers_unchanged"] = before_state == audit["parameters_buffers_sha256_after"]
                checks["global_RNG_not_consumed"] = _rng_equal(rng_before, _rng_snapshot())
            except Exception as audit_error:
                audit_errors.append(f"{type(audit_error).__name__}: {audit_error}")
            # Preserve failure observations before defensive cleanup; restoring
            # flags/RNG cannot turn a failed observation into a passed audit.
            for module, training in flags:
                module.training = training
            try:
                inference._clear(model)
                checks["defensive_context_cleanup"] = _contexts_empty(model)
            except Exception as cleanup_error:
                checks["defensive_context_cleanup"] = False
                audit_errors.append(f"context cleanup: {type(cleanup_error).__name__}: {cleanup_error}")
            try:
                _restore_rng(rng_before)
            except Exception as cleanup_error:
                audit_errors.append(f"RNG cleanup: {type(cleanup_error).__name__}: {cleanup_error}")
            audit.update(checks=checks, audit_errors=audit_errors, observed_global_noise_sha256=actual_global,
                         full_noise_sha256=actual_global, actual_sampler_calls=len(sampler_rows),
                         actual_VAE_encode_calls=len(encode_rows), wall_seconds=time.perf_counter()-started)
        if error is not None:
            audit.update(status="failed", original_error=f"{type(error).__name__}: {error}")
            raise NativeQAAuditError(f"Native operator QA failed: {type(error).__name__}: {error}", audit) from error
        audit.update(status="failed", phase="structural_validation")
        restored, metadata = result
        checks = audit["checks"]
        checks.update({
            "VAE_encode_count_and_order": len(encode_rows) == len(tiles)+int(enabled),
            "sampler_count": len(sampler_rows) == len(tiles),
            "observed_global_noise_matches_expected": actual_global == audit["expected_global_noise_sha256"],
            "metadata_geometry_matches_plan": metadata["padded_shape"][2:] == spec["padded_HR_T_HW"]
                and [x["hr_tlhw"] for x in metadata["tiles"]] == tiles
                and [x["latent_tlhw"] for x in metadata["tiles"]] == latent_tiles
                and metadata["sampling"]["global_noise_shape"] == spec["global_CPU_Gaussian_noise_shape"],
            "output_shape_matches_plan": list(restored.shape) == plan["input_preparation"]["HR_shape"],
            "full_coverage": metadata["blend"]["all_pixels_covered"] and metadata["blend"]["minimum_weight_sum"] > 0,
            "output_finite_range": bool(torch.isfinite(restored).all()) and float(restored.min()) >= 0 and float(restored.max()) <= 1,
            "metadata_reference_mode": metadata["reference"]["enabled"] is enabled
                and metadata["reference"]["encode_count"] == int(enabled),
        })
        audit["output_RGB_sha256"] = _tensor_sha(restored)
        if audit["audit_errors"] or not all(checks.values()):
            audit["status"] = "failed"
            raise NativeQAAuditError(f"Native operator structural/state audit failed: {checks}", audit)
        audit["status"] = "passed"
        metadata["native_operator_audit"] = audit
        return restored, metadata


def make_audited_restore(plans):
    return AuditedNativeRestore(plans)
