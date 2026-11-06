"""Isolated real-model optional-HRI smoke; run only after the GPU is free.

Uses one frozen V2 training clip from this VSR project, official fresh weights,
and a separately T=1 encoded HR reference. No task checkpoint is loaded and no
active-training/latest status is modified. This tests graphs, not recovery.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / ".deps"))

import torch
import torch.nn.functional as F

from vsr.alignment import point_alignment
from vsr.audit import dump, environment, file_hash, verify_official
from vsr.cache import _metadata, _temporal_contract
from vsr.checkpoints import _check_config
from vsr.data import DATA_VERSION
from vsr.model import gradient_norms
from vsr.preflight import normalize_latent
from vsr.reproducibility import configure_reproducibility, reproducibility_flags, tensor_state_hashes
from vsr.train import compute_relation_objective, make_model


def _inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Smoke cache must stay within this VSR run/root")
    return path


def _frozen_load(path, *, mmap=False):
    before = path.stat()
    value = torch.load(path, map_location="cpu", weights_only=True, **({"mmap": True} if mmap else {}))
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"Cache artifact changed during reading; wait for preparation to finish: {path}")
    return value


def load_v2_training_clip(cache_run, clip_index=0, *, allowed_root=None):
    """Load one stable train clip; large auxiliary lists are memory mapped."""
    allowed_root = ROOT / "runs" if allowed_root is None else Path(allowed_root)
    source = _inside(cache_run, allowed_root)
    if not isinstance(clip_index, int) or isinstance(clip_index, bool) or clip_index < 0:
        raise ValueError("clip_index must be nonnegative")
    required = ["config.json", "selection.json", "correspondence.json", "fresh_text_embeddings.pt",
                "fresh_teacher_features.pt", "fresh_anchor_flows.pt", f"clip_{clip_index}/metadata.json",
                f"clip_{clip_index}/paired_rgb.pt", f"clip_{clip_index}/fresh_latents.pt"]
    paths = {name: _inside(source / name, source) for name in required}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Wait for this V2 cache's frozen preparation artifacts: {missing}")
    config = json.loads(paths["config.json"].read_text())
    _check_config(config)
    selection = json.loads(paths["selection.json"].read_text())
    training_ids = selection["training_ids"]
    if clip_index >= len(training_ids) or selection.get("test_accessed") is not False:
        raise ValueError("Smoke must use one recorded training clip and no test pixels")
    metadata, binding = _metadata(source / f"clip_{clip_index}", source)
    if metadata.get("data_version") != DATA_VERSION or DATA_VERSION != "hq-vsr-paired-v2":
        raise ValueError("This prospective HRI smoke requires processing V2")
    if metadata.get("record_id") != training_ids[clip_index] or metadata.get("split") != "train":
        raise ValueError("Cached clip does not match the sealed training selection")
    if binding.get("manifest_sha256") != selection["manifest_sha256"]:
        raise ValueError("Training clip manifest binding mismatch")
    if binding.get("official_revision", binding.get("model_revision")) != config["model"]["revision"]:
        raise ValueError("Training clip official normalization/model revision differs from source config")
    if metadata.get("seed") != config["seed"]:
        raise ValueError("Training clip data seed differs from source config")
    if config.get("_sha256") is not None and binding.get("config_sha256") != config["_sha256"]:
        raise ValueError("Training clip originating config binding differs from source config")
    _temporal_contract(metadata, expected_stride=config["data"].get("temporal_stride", 1),
                       expected_frames=config["data"]["num_frames"])
    rgb = _frozen_load(paths[f"clip_{clip_index}/paired_rgb.pt"])
    expected_hr = (1, 3, config["data"]["num_frames"], config["data"]["height"], config["data"]["width"])
    for key, shape in (("hr", expected_hr), ("lr_up", expected_hr),
                       ("lr", (*expected_hr[:3], expected_hr[3]//4, expected_hr[4]//4))):
        value = rgb[key]
        if tuple(value.shape) != shape or value.requires_grad or not bool(torch.isfinite(value).all()) or float(value.min()) < 0 or float(value.max()) > 1:
            raise ValueError(f"Invalid frozen V2 {key}")
    pair = _frozen_load(paths[f"clip_{clip_index}/fresh_latents.pt"])
    if pair.get("binding") != binding:
        raise ValueError("Per-clip normalized latent binding differs from RGB binding")
    geometry = json.loads(paths["correspondence.json"].read_text())
    clean, low = (pair[key].detach().clone() for key in ("clean", "low"))
    if (clean.shape != low.shape or clean.shape[0] != 1 or list(clean.shape[2:]) != geometry["latent_grid"]
            or not bool(torch.isfinite(clean).all()) or not bool(torch.isfinite(low).all())):
        raise ValueError("Frozen normalized latent geometry/values are invalid")
    text_map = _frozen_load(paths["fresh_text_embeddings.pt"])
    text = text_map[metadata["caption"]].detach().clone()
    teacher_list = _frozen_load(paths["fresh_teacher_features.pt"], mmap=True)
    teacher = teacher_list[clip_index].detach().clone()
    del teacher_list
    flow_list = _frozen_load(paths["fresh_anchor_flows.pt"], mmap=True)
    flows = {key: flow_list[clip_index][key].detach().clone() for key in ("forward", "backward")}
    del flow_list
    if teacher.shape[:2] != (1, math.prod(geometry["token_grid"])) or not bool(torch.isfinite(teacher).all()):
        raise ValueError("Teacher tokens differ from the selected clip grid")
    expected_flow = (1, len(geometry["anchors"])-1, 2, expected_hr[-2], expected_hr[-1])
    if any(tuple(value.shape) != expected_flow or not bool(torch.isfinite(value).all()) for value in flows.values()):
        raise ValueError("HR-anchor flow cache geometry/values differ from the clip")
    audit = {"source_run": str(source), "clip_index": clip_index, "record_id": metadata["record_id"],
             "manifest_sha256": selection["manifest_sha256"], "source_binding": binding,
             "source_status": json.loads((source/"status.json").read_text()).get("status") if (source/"status.json").is_file() else "preparing_or_running",
             "artifact_sha256": {name: file_hash(path) for name, path in paths.items()},
             "task_checkpoint_loaded": False, "auxiliary_loading": "CPU mmap; clone only selected training sample",
             "test_accessed": False}
    clip = {**rgb, "metadata": metadata, "binding": binding, "fps": metadata["output_fps"]}
    return config, {"clip": clip, "clean": clean, "low": low, "text": text,
                    "teacher": teacher, "flows": flows, "geometry": geometry, "source_audit": audit}


def _clear(model):
    model.clear_features(); model.clear_reference()


def _context(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def run_graph_smoke(model, vae, inputs, config, output_dir, *, updates=3, reference_index=0, seed=20261204):
    """Reusable CPU-testable graph engine; real CLI injects official models."""
    if updates not in (2, 3) or not torch.are_deterministic_algorithms_enabled():
        raise ValueError("Smoke requires strict deterministic algorithms and 2 or 3 updates")
    if config.get("reproducibility", {}).get("mode") != "strict":
        raise ValueError("Smoke config must explicitly select strict CPU relation")
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    clip, grid = inputs["clip"], inputs["geometry"]["token_grid"]
    frame_count = clip["hr"].shape[2]
    if not 0 <= reference_index < frame_count or any(p.requires_grad for p in vae.parameters()):
        raise ValueError("Reference index must be known/in-range and VAE must be frozen")
    clean, low, text, teacher = (inputs[key].detach().to(device) for key in ("clean", "low", "text", "teacher"))
    mask = torch.ones(teacher.shape[:2], dtype=torch.bool, device=device)
    with torch.no_grad():
        vae.eval()
        frame = clip["hr"][:, :, reference_index:reference_index+1]
        raw = vae.encode((frame*2-1).to(device=next(vae.parameters()).device, dtype=torch.float32)).latent_dist.mode()
        reference = normalize_latent(raw, vae).detach().to(device)
    if reference.shape != (clean.shape[0], clean.shape[1], 1, clean.shape[3], clean.shape[4]):
        raise ValueError("Independent T=1 reference encoding differs from runtime target latent grid")
    torch.save({"reference_rgb": frame, "reference_latent": reference.cpu(), "clip_index": reference_index,
                "source_frame_index": clip["metadata"]["source_indices"][reference_index],
                "single_frame_encode": True, "source_binding": clip["binding"]}, output_dir/"single_frame_reference.pt")
    on = {"ref_latent": reference, "ref_index": reference_index, "frame_count": frame_count,
          "ref_valid": torch.ones(1, dtype=torch.bool, device=device)}
    off = {"ref_latent": None, "ref_index": None, "frame_count": frame_count,
           "ref_valid": torch.zeros(1, dtype=torch.bool, device=device)}
    phase, events = {"name": "initial"}, []
    def observe(_module, args):
        events.append({"phase": phase["name"], "grad_enabled": torch.is_grad_enabled(),
                       "reference_context_alive": model._reference_context is not None,
                       "reference_valid": bool(args[-1].any())})
    observer = model.reference_adapter.register_forward_pre_hook(observe)
    previous_checkpointing = model.transformer.gradient_checkpointing
    flags = [(module, module.training) for module in model.modules()]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    initial_hash = tensor_state_hashes(model)
    rows = []
    def conditions(step):
        sigma = (.3, .7, .99)[step % 3]
        rng = torch.Generator(device=device).manual_seed(seed + step)
        noise = torch.randn(clean.shape, generator=rng, device=device)
        return (1-sigma)*clean+sigma*noise, torch.tensor([1000*sigma], device=device), noise
    def objective(step, reference_kwargs):
        noisy, timestep, noise = conditions(step)
        with _context(device):
            velocity, student = model(noisy, timestep, text, low, **reference_kwargs)
        fm = F.mse_loss(velocity.float(), noise-clean)
        alignment = point_alignment(student, teacher, mask)
        relation, diagnostics = compute_relation_objective(student, teacher, grid, inputs["flows"]["forward"],
            inputs["flows"]["backward"], mask, config)
        if diagnostics["loss_device"] != "cpu" or not diagnostics["strict_cpu_backward"]:
            raise RuntimeError("Smoke relation did not use the strict CPU backward path")
        loss = fm + config["training"]["lambda_align"]*alignment + config["training"].get("lambda_relation", 0)*relation
        return loss, {"fm": float(fm.detach()), "alignment": float(alignment.detach()),
                      "relation": float(relation.detach()), "relation_diagnostics": diagnostics}
    def controls(label):
        phase["name"] = label
        model.eval()
        noisy, timestep, _ = conditions(0)
        with torch.no_grad(), _context(device):
            enabled, _ = model(noisy, timestep, text, low, **on); _clear(model)
            disabled, _ = model(noisy, timestep, text, low, **off); _clear(model)
            default, _ = model(noisy, timestep, text, low); _clear(model)
            ignored = {**off, "ref_latent": torch.full_like(reference, float("nan"))}
            ignored_output, _ = model(noisy, timestep, text, low, **ignored); _clear(model)
        exact_off = torch.equal(disabled, default) and torch.equal(disabled, ignored_output)
        if not exact_off:
            raise RuntimeError("Missing HRI depends on a placeholder or stale reference context")
        return {"missing_default_nan_paths_exact": exact_off,
                "flow_mae_on_minus_off": float((enabled-disabled).abs().mean()),
                "on_off_exact": torch.equal(enabled, disabled)}
    try:
        model.transformer.gradient_checkpointing = True
        initial_controls = controls("initial_controls")
        if not initial_controls["on_off_exact"]:
            raise RuntimeError("Zero-initialized HRI branch changed initial output")
        for step in range(updates):
            model.train(); model.zero_grad(set_to_none=True)
            phase["name"] = f"update_{step+1}"
            before = len(events)
            loss, values = objective(step, on)
            loss.backward()
            norms = gradient_norms(model)
            if norms["frozen"]["elements_with_grad"] or not all(row["finite"] for row in norms.values()):
                raise RuntimeError("Frozen/nonfinite gradient in HRI smoke")
            hri_stem = model.reference_adapter.stem[0].weight.grad
            stem_norm = float(hri_stem.float().norm()) if hri_stem is not None else 0.
            calls = events[before:]
            if len(calls) < 2 or not all(event["reference_context_alive"] for event in calls):
                raise RuntimeError("Non-reentrant checkpoint replay did not retain HRI context")
            torch.nn.utils.clip_grad_norm_(params, config["training"]["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step(); _clear(model)
            rows.append({"step": step+1, **values, "gradient_groups": norms,
                         "hri_stem_gradient_l2": stem_norm, "reference_adapter_calls_forward_and_replay": len(calls)})
        for group in ("lrv_stem", "lrv_residuals", "lora", "hri"):
            if rows[-1]["gradient_groups"][group]["l2"] <= 0:
                raise RuntimeError(f"HRI smoke did not reach {group} after branch opening")
        if rows[-1]["hri_stem_gradient_l2"] <= 0:
            raise RuntimeError("HRI stem remained closed after updates")
        final_controls = controls("final_controls")
        if final_controls["flow_mae_on_minus_off"] <= 0:
            raise RuntimeError("HRI condition did not affect the real model after updates")
        gradients = []
        for checkpointing in (True, False):
            phase["name"] = f"gradient_compare_checkpoint_{checkpointing}"
            model.train(); model.zero_grad(set_to_none=True)
            model.transformer.gradient_checkpointing = checkpointing
            loss, _ = objective(1, on); loss.backward()
            gradients.append({name: None if p.grad is None else p.grad.detach().cpu().clone()
                              for name, p in model.named_parameters() if p.requires_grad})
            _clear(model)
        mismatches, max_difference = [], 0.
        for name, first in gradients[0].items():
            second = gradients[1][name]
            same = (first is None and second is None) or (first is not None and second is not None and torch.equal(first, second))
            if not same:
                mismatches.append(name)
                if first is not None and second is not None:
                    max_difference = max(max_difference, float((first-second).abs().max()))
        model.transformer.gradient_checkpointing = True
        phase["name"] = "missing_backward"
        model.train(); model.zero_grad(set_to_none=True)
        loss, _ = objective(1, off); loss.backward()
        missing_no_gradient = all(p.grad is None for p in model.reference_adapter.parameters())
        if not missing_no_gradient:
            raise RuntimeError("Explicitly absent HRI acquired adapter gradients")
        if any(p.grad is not None for p in vae.parameters()):
            raise RuntimeError("Frozen single-frame VAE acquired gradients")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        summary = {"status": "passed" if not mismatches else "failed", "kind": "isolated_reference_graph_smoke",
            "updates": updates, "initial_controls": initial_controls, "final_controls": final_controls,
            "rows": rows, "checkpoint_gradients_exact": not mismatches, "checkpoint_gradient_mismatches": mismatches,
            "checkpoint_max_abs_gradient_difference": max_difference, "missing_hri_parameter_gradients_none": missing_no_gradient,
            "frozen_vae_gradients_none": True, "strict_flags": reproducibility_flags(),
            "diagnostic_wall_seconds": time.perf_counter()-started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "reference_latent_shape": list(reference.shape), "reference_single_frame_encode_count": 1,
            "reference_context_events": events, "initial_trainable_hashes": initial_hash,
            "updated_trainable_hashes": tensor_state_hashes(model),
            "source_audit": inputs["source_audit"], "reference_index": reference_index,
            "reference_source_frame_index": clip["metadata"]["source_indices"][reference_index],
            "task_checkpoint_loaded": False, "formal_initialization": "discard diagnostic model; reference comparisons reload official initialization",
            "quality_hypotheses_verified": False, "restoration_sampling": False,
            "scope": "one noised-HR training diagnostic; no recovery or propagation quality conclusion"}
        dump(output_dir/"summary.json", summary)
        if mismatches:
            raise RuntimeError("Strict checkpoint/direct backward gradients differ; see summary.json")
        return summary
    finally:
        observer.remove(); model.zero_grad(set_to_none=True); _clear(model)
        model.transformer.gradient_checkpointing = previous_checkpointing
        for module, flag in flags:
            module.training = flag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-run", required=True)
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--updates", type=int, choices=(2, 3), default=3)
    parser.add_argument("--seed", type=int, default=20261204)
    args = parser.parse_args()
    active_path = ROOT/"runs/active_training.json"
    if active_path.is_file() and json.loads(active_path.read_text()).get("status") == "running":
        raise RuntimeError("Formal no-reference training is active; run this isolated GPU smoke only after it finishes")
    config, inputs = load_v2_training_clip(args.cache_run, args.clip_index)
    config = copy.deepcopy(config); config.pop("_sha256", None)
    config.update(stage="reference_graph_smoke_only", experiment_name="reference_graph_smoke")
    config["reference"] = {"enabled": True, "layer": 9, "stem_dim": 128, "initialization_seed": args.seed+100}
    config["training"].update(seed=args.seed, variants=["Both"], updates=args.updates, gradient_checkpointing=True, resume=False)
    config["formal_training"] = {"enabled": False, "reason": "isolated graph diagnostic"}
    config["reproducibility"] = {"mode": "strict", "relation_loss_device": "cpu"}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = ROOT/"runs"/f"reference_graph_smoke_{stamp}"
    run.mkdir(exist_ok=False)
    dump(run/"config.json", config); dump(run/"source_cache.json", inputs["source_audit"])
    try:
        dump(run/"reproducibility.json", configure_reproducibility(config))
        env = environment(ROOT); dump(run/"environment.json", env)
        if not env["cuda_available"]:
            raise RuntimeError("This real reference smoke requires a free visible CUDA GPU")
        dump(run/"official_artifacts.json", verify_official(ROOT, config))
        model, initialization = make_model(config, "Both")
        dump(run/"initialization.json", initialization)
        from diffusers import AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(config["model"]["local_dir"], subfolder="vae", dtype=torch.float32,
            local_files_only=True, use_safetensors=True).requires_grad_(False).eval().to("cuda")
        summary = run_graph_smoke(model, vae, inputs, config, run, updates=args.updates,
                                  reference_index=args.reference_index, seed=args.seed)
        dump(run/"status.json", {"status": summary["status"], "run": str(run), "diagnostic_only": True,
             "task_checkpoint_loaded": False, "quality_hypotheses_verified": False})
        print(json.dumps({"status": summary["status"], "run": str(run), "checkpoint_gradients_exact": summary["checkpoint_gradients_exact"]}), flush=True)
    except Exception as error:
        failure = {"status": "failed", "error": str(error), "traceback": traceback.format_exc()}
        dump(run/"failure.json", failure); dump(run/"status.json", failure)
        raise


if __name__ == "__main__":
    main()
