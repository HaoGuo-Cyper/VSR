#!/usr/bin/env python3
"""Plan/measure synthetic Base forward-backward resources with zero updates.

prepare is standard-library only. cpu-selftest uses a new tiny CPU module.
execute is the only mode allowed to initialize CUDA or load the Base model.
No optimizer is constructed, no step is taken and no checkpoint is saved.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

sys.dont_write_bytecode = True
from reproduce_external_row import (OLD, CHECKPOINT, CHECKPOINT_SHA, LABEL, ENVIRONMENT,
                                    binding, digest, need, read, safe_run, save, sha)

PACKAGES = ("torch", "numpy", "diffusers", "transformers", "peft", "accelerate", "safetensors")


def collect_protocol(root):
    root = Path(root).resolve()
    cfg_path = CHECKPOINT.parent.parent / "config.json"
    cfg = read(cfg_path)
    need(cfg["seed"] == 20261102 and cfg["reproducibility"]["mode"] == "strict"
         and cfg["training"]["mixed_precision"] == "bf16"
         and cfg["training"]["gradient_checkpointing"] is True
         and not cfg.get("reference", {}).get("enabled", False), "Expected strict NoRef Base configuration")
    preflight = root / "scripts/frozen_external/outputs/VSR_final_evaluation_text_cache_preflight.json"
    caption = read(preflight)["entries"][LABEL]
    need(caption["checkpoint_path"] == str(CHECKPOINT) and caption["checkpoint_sha256"] == CHECKPOINT_SHA
         and caption["config_path"] == str(cfg_path) and sha(cfg_path) == caption["config_sha256"],
         "Cached caption preflight is not bound to the Base checkpoint/config")
    manifest_path = OLD / "docs/official-model-manifest.json"
    manifest = read(manifest_path)
    need(manifest["sha"] == cfg["model"]["revision"], "Official model revision changed")
    model_root = Path(cfg["model"]["local_dir"])
    transformer_cfg = read(model_root / "transformer/config.json")
    vae_cfg = read(model_root / "vae/config.json")
    need(transformer_cfg["in_channels"] == transformer_cfg["out_channels"] == 16
         and transformer_cfg["patch_size"] == [1, 2, 2]
         and vae_cfg["temperal_downsample"] == [False, True, True], "Unexpected Wan latent geometry")
    official = []
    for item in manifest["siblings"]:
        name = item["rfilename"]
        if name.startswith("transformer/"):
            path = model_root / name
            if "lfs" in item:
                official.append(binding(path, item["lfs"]["sha256"]))
            else:
                import hashlib
                raw = path.read_bytes()
                need(hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest() == item["blobId"],
                     "Official transformer JSON/blob changed")
                official.append(binding(path))
    need((root / ".deps").resolve() == (OLD / ".deps").resolve(), "Pinned local dependencies required")
    return {"schema": "vsr_synthetic_no_update_resource_profile_v1", "project_root": str(root),
        "script": binding(__file__), "helper": binding(Path(__file__).with_name("reproduce_external_row.py")),
        "checkpoint": binding(CHECKPOINT, CHECKPOINT_SHA), "checkpoint_config": binding(cfg_path), "config": cfg,
        "official_transformer_files": official, "official_manifest": binding(manifest_path),
        "vae_geometry_json_only": binding(model_root / "vae/config.json"),
        "production_core": {str(p): sha(p) for p in sorted((root / "vsr").glob("*.py"))},
        "caption": {"text": "A high quality video.", "preflight": binding(preflight),
            "cache": binding(caption["cache_path"], caption["cache_file_sha256"]),
            "embedding_sha256": caption["embedding_sha256"], "shape": caption["shape"], "dtype": caption["dtype"],
            "T5_fallback": False},
        "runtime": {"executable": str(Path(sys.executable).resolve()), "python": sys.version,
                    "packages": {name: importlib.metadata.version(name) for name in PACKAGES}, "environment": ENVIRONMENT},
        "shapes": [{"nominal_pixel_frames": frames, "latent_shape": [1, 16, (frames-1)//4+1, 16, 28],
                    "nominal_HR_hw": [128, 224], "nominal_LR_hw": [32, 56]}
                   for frames in (9, 17, 33)],
        "input": {"distribution": "independent standard normal normalized HR and upsampled-LR latents, plus Gaussian flow noise",
                  "seed": 20261105, "input_dtype": "torch.float32", "sigma": 0.5, "timestep": 500.0,
                  "noisy": "(1-sigma)*clean + sigma*noise", "target": "noise-clean",
                  "frame_spacing": "not represented by synthetic random latents; frame-count geometry only"},
        "conditions": ["Base_forward_backward", "one_no_grad_teacher_then_Base_forward_backward"],
        "schedule": {"warmup_cycles_per_case": 1, "timed_cycles_per_case": 3,
                     "total_shapes": 3, "total_conditions": 2, "total_forward_backward_cycles": 24},
        "teacher": {"weights": "same Base parameters; no copy, EMA, projector or layer selection",
                    "no_grad": True, "detached_output_held_until_student_backward_completes": True,
                    "output_used_in_loss": False, "interpretation": "one extra forward resource estimate only"},
        "timing": "CUDA-synchronized forward/flow-MSE/backward/zero_grad/feature-clear cycle; input generation and parameter audits excluded; allocated peak includes resident Base and inputs",
        "invariants": {"optimizer_constructed": False, "optimizer_steps": 0, "parameter_updates": 0,
                       "checkpoint_saved": False, "VAE_loaded": False, "T5_loaded": False,
                       "parameter_identity_and_versions_checked": True, "trainable_SHA_before_after": True},
        "stop_policy": "Any OOM records not_fit and stops all remaining cases; any other failure stops; no retry, batch/dtype/shape/checkpoint change or fallback",
        "limits": ["Synthetic inputs are not VAE reconstruction or fit-quality measurements.",
                   "No optimizer state, gradient clipping, data transfer, VAE/T5, EMA parameter storage or alignment loss is included.",
                   "Seconds are per no-update forward/backward cycle, not seconds per optimizer update.",
                   "8192-cycle hours are an incomplete lower cost estimate only, not formal RE cost; T1.3 remains partial."]}


def parameter_identity(model):
    return {name: {"object_id": id(p), "storage_pointer": p.data_ptr(), "version": p._version,
                   "shape": list(p.shape), "dtype": str(p.dtype), "device": str(p.device),
                   "requires_grad": p.requires_grad} for name, p in model.named_parameters()}


def features_empty(model):
    return all(getattr(model, name, None) is None for name in ("_condition_tokens", "_captured"))


def cycle(torch, model, inputs, *, teacher, device, audit_gradients=False):
    """Run no optimizer operation. Teacher output lives through student backward."""
    gpu = device == "cuda"
    model.zero_grad(set_to_none=True)
    model.clear_features()
    need(all(p.grad is None for p in model.parameters()) and features_empty(model), "Dirty pre-cycle state")
    if gpu:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    teacher_output = flow = loss = tokens = None
    try:
        if teacher:
            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                teacher_output, tokens = model(inputs["noisy"], inputs["time"], inputs["text"], inputs["low"])
            need(tokens is None and not teacher_output.requires_grad, "Base teacher must be detached with no projector")
            teacher_output = teacher_output.detach()
            model.clear_features()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            flow, tokens = model(inputs["noisy"], inputs["time"], inputs["text"], inputs["low"])
        need(tokens is None, "Base profile must not compute an alignment projector")
        loss = torch.nn.functional.mse_loss(flow.float(), inputs["noise"]-inputs["clean"])
        loss.backward()
        need(any(p.grad is not None for p in model.parameters() if p.requires_grad), "Backward did not produce trainable gradients")
        need(all(p.grad is None for p in model.parameters() if not p.requires_grad), "Frozen backbone received gradients")
        teacher_bytes = teacher_output.numel()*teacher_output.element_size() if teacher_output is not None else 0
        teacher_held_through_backward = teacher_output is not None if teacher else None
        # Full gradient-value audit is warmup-only, excluded from timed cycles.
        if audit_gradients:
            need(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),
                 "Nonfinite warmup gradients")
        model.zero_grad(set_to_none=True)
        model.clear_features()
        del flow, teacher_output, tokens
        if gpu:
            torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        value = float(loss.detach().cpu())
        need(math.isfinite(value) and all(p.grad is None for p in model.parameters()) and features_empty(model),
             "Nonfinite loss or cycle cleanup failure")
        return {"seconds_per_forward_backward_zero_cycle": seconds, "flow_mse": value,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated() if gpu else None,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved() if gpu else None,
                "teacher_output_bytes_held": teacher_bytes,
                "teacher_output_held_through_student_backward": teacher_held_through_backward,
                "grads_cleared": True, "features_cleared": True, "optimizer_steps": 0}
    finally:
        model.zero_grad(set_to_none=True)
        model.clear_features()


def make_inputs(torch, shape, text, *, device):
    # CPU generation makes inputs identical across conditions and independent of
    # CUDA model initialization RNG. These are normalized latent-space fixtures.
    generator = torch.Generator(device="cpu").manual_seed(20261105)
    clean, low, noise = [torch.randn(shape, generator=generator, dtype=torch.float32).to(device) for _ in range(3)]
    return {"noisy": .5*clean+.5*noise, "low": low, "clean": clean, "noise": noise,
            "time": torch.tensor([500.0], device=device), "text": text}


def cpu_selftest():
    import torch
    from torch.utils.checkpoint import checkpoint
    need(not torch.cuda.is_initialized(), "CPU fixture must not initialize CUDA")
    torch.set_num_threads(2)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 4)
            self._condition_tokens = self._captured = None
        def clear_features(self):
            self._condition_tokens = self._captured = None
        def forward(self, noisy, timestep, text, low):
            self._condition_tokens = low
            result = checkpoint(self.layer, (noisy+low).transpose(1, -1), use_reentrant=False).transpose(1, -1)
            return result, None
    torch.manual_seed(7)
    model = Tiny().train()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    identity = parameter_identity(model)
    inputs = make_inputs(torch, [1, 4, 3, 2, 2], torch.zeros(1, 2, 4), device="cpu")
    rows = [cycle(torch, model, inputs, teacher=value, device="cpu", audit_gradients=True) for value in (False, True)]
    need(identity == parameter_identity(model) and all(torch.equal(before[name], p) for name, p in model.named_parameters()),
         "CPU fixture changed parameters")
    need(rows[1]["teacher_output_held_through_student_backward"] is True
         and rows[1]["teacher_output_bytes_held"] > 0 and not torch.cuda.is_initialized(), "Teacher lifecycle or CPU-only check failed")
    print(json.dumps({"status": "passed", "tiny_CPU_only": True, "parameter_bytes_identity_versions_unchanged": True,
                      "gradients_and_features_cleared": True, "teacher_held_through_backward": True,
                      "optimizer_steps": 0, "CUDA_initialized": False}))


def execute(args):
    run = safe_run(args.run_dir, args.project_root)
    protocol = read(run / "protocol.json")
    need(args.expected_protocol_sha and sha(run / "protocol.json") == args.expected_protocol_sha,
         "Exact prepared raw-file protocol SHA required")
    state = read(run / "status.json")
    need(state["status"] == "planned" and state["protocol_sha256"] == args.expected_protocol_sha,
         "Only a never-started planned profile can execute; no retry/resume")
    save(run / "status.json", {"status": "running", "protocol_sha256": args.expected_protocol_sha, "parameter_updates": 0})
    model = torch = None
    before_identity = before_hash = None
    results = []
    current_case = "pre_GPU_bindings"
    started = time.perf_counter()
    try:
        need(protocol == collect_protocol(args.project_root), "Prepared code/assets/runtime changed; make a fresh plan")
        need(all(os.environ.get(k) == v for k, v in ENVIRONMENT.items()), "Exact strict environment required")
        sys.path.insert(0, protocol["project_root"])
        import torch as torch_module
        torch = torch_module
        need(not torch.cuda.is_initialized(), "Configure strict runtime before CUDA initialization")
        torch.set_num_threads(2)
        from vsr.reproducibility import configure_reproducibility, tensor_state_hashes
        from vsr.checkpoints import load_training_checkpoint
        cfg = protocol["config"]
        runtime = configure_reproducibility(cfg)
        need(runtime["flags"]["deterministic_algorithms"] and not runtime["flags"]["deterministic_warn_only"]
             and runtime["flags"]["sdp_math"] and not runtime["flags"]["sdp_flash"]
             and not runtime["flags"]["sdp_memory_efficient"] and not runtime["flags"]["sdp_cudnn"],
             "Strict math backend was not established")
        save(run / "runtime.json", runtime)
        current_case = "cached_text_only"
        cache = torch.load(protocol["caption"]["cache"]["path"], map_location="cpu", weights_only=True)
        text = cache.get(protocol["caption"]["text"]) if isinstance(cache, dict) else None
        need(isinstance(text, torch.Tensor) and list(text.shape) == protocol["caption"]["shape"]
             and str(text.dtype) == protocol["caption"]["dtype"] and bool(torch.isfinite(text).all()), "Invalid cached text; T5 fallback forbidden")
        import hashlib
        header = json.dumps([str(text.dtype), list(text.shape)], separators=(",", ":")).encode()+b"\n"
        need(hashlib.sha256(header+text.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() == protocol["caption"]["embedding_sha256"],
             "Cached embedding tensor digest differs")
        text = text.detach().to("cuda")
        del cache
        current_case = "Base_checkpoint_load"
        model, load_audit = load_training_checkpoint(CHECKPOINT, allowed_root=OLD / "runs", device="cuda", variant="Base")
        need(load_audit["checkpoint_sha256"] == CHECKPOINT_SHA and load_audit["step"] == 8192
             and load_audit["config"] == cfg and model.projector is None, "Unexpected Base checkpoint")
        need(any(getattr(module, "gradient_checkpointing", False) for module in model.transformer.modules()),
             "Gradient checkpointing must remain enabled")
        model.train()
        before_identity = parameter_identity(model)
        before_hash = tensor_state_hashes(model)["all_trainable"]["sha256"]
        save(run / "loaded_Base.json", {"audit": load_audit, "trainable_sha256_before": before_hash,
                                       "all_parameter_identity_version_snapshot_sha256": digest(before_identity)})
        for geometry in protocol["shapes"]:
            inputs = make_inputs(torch, geometry["latent_shape"], text, device="cuda")
            for condition in protocol["conditions"]:
                current_case = {"geometry": geometry, "condition": condition, "phase": "warmup"}
                teacher = condition.startswith("one_no_grad_teacher")
                warmup = cycle(torch, model, inputs, teacher=teacher, device="cuda", audit_gradients=True)
                rows = []
                for index in range(3):
                    current_case = {"geometry": geometry, "condition": condition, "phase": "timed", "cycle": index+1}
                    rows.append(cycle(torch, model, inputs, teacher=teacher, device="cuda"))
                seconds = [row["seconds_per_forward_backward_zero_cycle"] for row in rows]
                result = {"geometry": geometry, "condition": condition, "warmup_excluded": warmup, "timed_cycles": rows,
                          "mean_seconds_per_forward_backward_zero_cycle": statistics.mean(seconds),
                          "sample_std_seconds": statistics.stdev(seconds),
                          "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in rows),
                          "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in rows),
                          "8192_cycles_hours_lower_cost_estimate_excluding_optimizer": statistics.mean(seconds)*8192/3600,
                          "estimate_is_complete_optimizer_update_or_formal_RE_cost": False}
                need(before_identity == parameter_identity(model), "Parameter identity/version changed during profile")
                results.append(result)
                save(run / "profile_progress.json", {"results": results, "parameter_updates": 0})
            del inputs
            gc.collect()
            torch.cuda.empty_cache()
        after_hash = tensor_state_hashes(model)["all_trainable"]["sha256"]
        need(before_hash == after_hash and before_identity == parameter_identity(model)
             and all(p.grad is None for p in model.parameters()) and features_empty(model), "Profile changed weights or left gradients/features")
        summary = {"status": "passed_synthetic_partial_T1.3", "protocol_sha256": args.expected_protocol_sha,
                   "parameter_updates": 0, "optimizer_steps": 0, "trainable_sha256_before": before_hash,
                   "trainable_sha256_after": after_hash, "all_parameter_identities_versions_unchanged": True,
                   "gradients_and_features_cleared": True, "results": results,
                   "limitations": protocol["limits"], "wall_seconds": time.perf_counter()-started}
        save(run / "summary.json", summary)
        save(run / "status.json", {"status": "passed_synthetic_partial_T1.3", "summary_sha256": sha(run / "summary.json"),
                                   "protocol_sha256": args.expected_protocol_sha, "parameter_updates": 0})
        print(json.dumps({"status": summary["status"], "run": str(run), "parameter_updates": 0}))
    except BaseException as error:
        oom = torch is not None and isinstance(error, torch.cuda.OutOfMemoryError)
        unchanged = before_identity == parameter_identity(model) if before_identity is not None and model is not None else None
        failure = {"status": "not_fit" if oom else "failed", "case": current_case,
                   "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
                   "completed_cases": results, "parameter_updates": 0, "optimizer_steps": 0, "automatic_retry": False,
                   "parameter_identities_versions_unchanged": unchanged,
                   "post_failure_trainable_hash_recomputed": False, "remaining_cases_not_run": True}
        save(run / "failure.json", failure)
        save(run / "status.json", {"status": failure["status"], "failure_sha256": sha(run / "failure.json"),
                                   "protocol_sha256": args.expected_protocol_sha, "parameter_updates": 0})
        raise
    finally:
        if model is not None:
            model.zero_grad(set_to_none=True)
            model.clear_features()
        model = None
        gc.collect()
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute", "cpu-selftest"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--expected-protocol-sha")
    args = parser.parse_args()
    if args.mode == "cpu-selftest":
        need(args.run_dir is None and args.expected_protocol_sha is None, "CPU fixture does not use run directories")
        cpu_selftest()
    elif args.mode == "prepare":
        need(args.run_dir is not None and args.expected_protocol_sha is None, "Fresh run path required; prepare determines its protocol SHA")
        run = safe_run(args.run_dir, args.project_root)
        need(not run.exists() or not any(run.iterdir()), "Use a fresh empty profile run")
        protocol = collect_protocol(args.project_root)
        need("torch" not in sys.modules, "prepare must not import torch")
        run.mkdir(parents=True, exist_ok=True)
        save(run / "protocol.json", protocol)
        value = sha(run / "protocol.json")
        save(run / "status.json", {"status": "planned", "protocol_sha256": value, "parameter_updates": 0,
                                  "created_utc": datetime.now(timezone.utc).isoformat(), "torch_imported": False, "CUDA_initialized": False})
        print(json.dumps({"status": "planned", "run": str(run), "protocol_sha256": value, "parameter_updates": 0}))
    else:
        need(args.run_dir is not None, "Prepared run path required")
        execute(args)


if __name__ == "__main__":
    main()
