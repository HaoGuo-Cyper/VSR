"""Fresh official initialization for all optional-HRI ablation variants.

Caption, LRV, FM convention, LoRA range and update budget follow train.py.
Each variant additionally trains the identical reference branch with one known
HR reference frame per clip and independent, matched reference dropout RNG.
This module never resumes a no-reference or legacy fine-tuned checkpoint.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".deps"))

import torch
import torch.nn.functional as F

from .alignment import point_alignment
from .audit import dump, environment, file_hash, verify_official
from .model import gradient_norms
from .preflight import autocast, normalize_latent, release
from .reference_evaluation import evaluate_reference_restoration
from .relation import relation_loss
from .train import make_model, prepare,compute_relation_objective
from .reproducibility import configure_reproducibility


def _timing_sync(device):
    """Synchronize a known CUDA phase boundary; CPU timing never initializes it."""
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class _PhaseTimings:
    """Wall observations only; never draws RNG or changes model/backend flags."""

    def __init__(self, path, *, started=None):
        self.path = Path(path)
        self.started = time.perf_counter() if started is None else started
        self.phases = {}
        self.status = "running"
        self.error = None
        self.write()

    def write(self):
        value = {"status": self.status, "phases": self.phases,
                 "entire_experiment_wall_seconds": time.perf_counter() - self.started,
                 "whole_wall_scope": "main entry through latest observation, including config/metadata, preparation, loads, updates, evaluation and releases; excludes interpreter/import startup and final timing-file serialization",
                 "GPU_hours_definition": "wall_seconds/3600 is allocated-device occupancy while this exclusive process owns the GPU; it is not measured CUDA-kernel active time",
                 "update_GPU_hours": "per-variant training_gpu_hours only covers synchronized optimizer-update blocks; excludes preparation, loading and evaluation",
                 "inference_timing": "internal evaluation sampling/decode consumes cached latents; not end-to-end RGB-to-RGB inference",
                 "failure_scope": "handled Python exceptions; abrupt OS termination must additionally use service status/journal",
                 "timing_observations_change_RNG_or_optimization": False}
        if self.error is not None:
            value["error"] = self.error
        dump(self.path, value)

    @contextmanager
    def phase(self, name, *, device=None):
        row = {"status": "running", "boundary_device": str(device) if device is not None else "cpu/no_cuda_sync"}
        self.phases[name] = row
        started = time.perf_counter()
        try:
            _timing_sync(device)
            started = time.perf_counter()
            yield row
            _timing_sync(device)
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            # Do not mask the primary failure if a failed CUDA context also
            # refuses the final timing-only synchronization.
            try:
                _timing_sync(device)
            except Exception as sync_error:
                row["final_sync_error"] = f"{type(sync_error).__name__}: {sync_error}"
            raise
        else:
            row["status"] = "completed"
        finally:
            row["wall_seconds"] = time.perf_counter() - started
            self.write()

    def finish(self, status, *, error=None):
        self.status, self.error = status, error
        self.write()


@torch.no_grad()
def encode_reference_frames(vae, clips, config, output_dir):
    """Encode every original HR frame independently as a T=1 VAE video.

    Returns one mapping {original_clip_frame_index: CPU normalized latent} per
    clip. No compressed multi-frame HR latent is accepted or sliced. This is
    training/reference-benchmark preparation, never implicit inference access
    to extra HR frames. Every cache is new and bound to its raw-data contract.
    """
    if any(parameter.requires_grad for parameter in vae.parameters()):
        raise ValueError("Reference cache VAE must be frozen before encoding")
    parameter = next(vae.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    dtype = parameter.dtype if parameter is not None else torch.float32
    flags = [(module, module.training) for module in vae.modules()]
    vae.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {"vsr/reference_training.py": file_hash(Path(__file__)),
                     "vsr/preflight.py": file_hash(Path(__file__).with_name("preflight.py"))}
    base = Path(config["model"]["local_dir"])
    vae_config = base / "vae/config.json"
    vae_config_hash = file_hash(vae_config) if vae_config.is_file() else None
    reference_latents, seals = [], []
    encoding_started = time.perf_counter()
    encoded_frames = 0
    encoding_status, encoding_error = "running", None
    peak_started = False
    try:
        _timing_sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            peak_started = True
        encoding_started = time.perf_counter()
        for i in range(len(clips)):
            clip = clips[i]
            hr = clip["hr"]
            if hr.ndim != 5 or hr.shape[:2] != (1, 3) or hr.shape[2] < 1:
                raise ValueError("Reference preparation requires RGB [1,3,T,H,W]")
            if not bool(torch.isfinite(hr).all()) or float(hr.min()) < 0 or float(hr.max()) > 1:
                raise ValueError("Reference preparation requires finite RGB[0,1]")
            if hr.shape[2] != config["data"]["num_frames"]:
                raise ValueError("Reference cache frame count differs from the frozen clip protocol")
            frames = {}
            for index in range(hr.shape[2]):
                # The temporal dimension is explicitly length one on EVERY VAE
                # encode. Encoding a whole video then slicing is forbidden.
                supplied_frame = hr[:, :, index:index+1].to(device=device, dtype=dtype)
                raw = vae.encode(supplied_frame * 2 - 1).latent_dist.mode()
                if raw.ndim != 5 or raw.shape[0] != 1 or raw.shape[2] != 1:
                    raise ValueError("Single-frame VAE encoding must preserve one latent time position")
                normalized = normalize_latent(raw, vae).detach().cpu().clone()
                if not bool(torch.isfinite(normalized).all()):
                    raise ValueError("Nonfinite encoded HRI")
                frames[index] = normalized
                encoded_frames += 1
            binding = {"source_binding": clip.get("binding", {}), "clip_metadata": clip["metadata"],
                       "official_revision": config["model"]["revision"], "vae_config_sha256": vae_config_hash,
                       "encoder_source_sha256": source_hashes, "config_sha256": config.get("_sha256"),
                       "frame_indices": list(range(hr.shape[2])), "encoding": "one separate T=1 VAE encode per original frame",
                       "posterior": "mode", "normalization": "official (latent-mean)/std", "vae_frozen": True,
                       "role": "training HRI or explicitly supplied reference benchmark only"}
            directory = output_dir / f"clip_{i}"
            directory.mkdir(parents=True, exist_ok=True)
            tensor_path = directory / "reference_latents.pt"
            torch.save({"reference_latents": frames, "binding": binding}, tensor_path)
            artifact_hash = file_hash(tensor_path)
            dump(directory / "reference_binding.json", {"binding": binding, "reference_latents_sha256": artifact_hash,
                 "shapes": {str(index): list(value.shape) for index, value in frames.items()}})
            seals.append({"clip_index": i, "record_id": clip["metadata"]["record_id"], "sha256": artifact_hash,
                          "frame_count": len(frames)})
            reference_latents.append(frames)
            if (i + 1) % 25 == 0 or i + 1 == len(clips):
                print(f"Encoded single-frame references {i+1}/{len(clips)} clips, {encoded_frames} frames, "
                      f"wall_seconds={time.perf_counter()-encoding_started:.2f}", flush=True)
        if any(parameter.grad is not None for parameter in vae.parameters()):
            raise ValueError("Frozen reference VAE acquired parameter gradients")
        dump(output_dir / "reference_cache_seal.json", {"new_cache": True, "legacy_state_loaded": False,
             "clip_count": len(clips), "per_clip": seals, "encoder_source_sha256": source_hashes,
             "single_frame_only": True, "video_latent_slicing": False})
        encoding_status = "completed"
        return reference_latents
    except Exception as error:
        encoding_status, encoding_error = "failed", f"{type(error).__name__}: {error}"
        raise
    finally:
        sync_error = None
        try:
            _timing_sync(device)
        except Exception as error:
            sync_error = error
            encoding_status = "failed"
        try:
            elapsed = time.perf_counter() - encoding_started
            peak, peak_error = None, None
            if peak_started and sync_error is None:
                try:
                    peak = torch.cuda.max_memory_allocated(device)
                except Exception as error:
                    peak_error = f"{type(error).__name__}: {error}"
            timing = {"status": encoding_status, "wall_seconds": elapsed,
                "allocated_GPU_hours": elapsed / 3600 if device.type == "cuda" else None,
                "completed_clips": len(reference_latents), "encoded_frames": encoded_frames,
                "requested_clips": len(clips), "requested_frames": len(clips)*config["data"]["num_frames"],
                "device": str(device), "peak_cuda_allocated_bytes": peak,
                "scope": "frozen VAE already loaded; separate T=1 encodes, normalization, CPU transfers, tensor/binding writes and cache seal; excludes VAE load and release",
                "peak_scope": "PyTorch allocated peak reset once at encoding-phase entry, including the resident VAE; not total driver VRAM",
                "error": encoding_error,
                "peak_observation_error": peak_error,
                "final_sync_error": f"{type(sync_error).__name__}: {sync_error}" if sync_error is not None else None}
            dump(output_dir / "reference_encoding_timings.json", timing)
        finally:
            for module, was_training in flags:
                module.training = was_training
        if sync_error is not None and encoding_error is None:
            raise sync_error


def draw_reference_condition(frame_latents, frame_count, reference_rng, *, drop_probability=.5, device="cpu"):
    """Draw one known reference index and one validity flag from a CPU RNG.

    Draw both on every update, even when reference is missing. Every ablation
    therefore consumes the same reference sequence without touching FM RNGs.
    A missing reference never reads frame_latents or transfers a black image.
    """
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if not math.isfinite(drop_probability) or not 0 <= drop_probability <= 1:
        raise ValueError("drop_probability must lie in [0,1]")
    if reference_rng.device.type != "cpu":
        raise ValueError("Reference sampling uses an independent CPU generator")
    sampled_index = int(torch.randint(frame_count, (), generator=reference_rng))
    provided = bool(torch.rand((), generator=reference_rng) >= drop_probability)
    valid = torch.tensor([provided], device=device, dtype=torch.bool)
    if provided:
        latent = frame_latents[sampled_index]
        if not isinstance(latent, torch.Tensor) or latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[2] != 1:
            raise ValueError("Selected HRI must be a single-frame B=1 latent")
        reference = latent.detach().to(device)
        index = torch.tensor([sampled_index], device=device, dtype=torch.long)
    else:
        reference, index = None, None
    kwargs = {"ref_latent": reference, "ref_index": index, "frame_count": frame_count, "ref_valid": valid}
    return kwargs, {"reference_sampled_index": sampled_index, "reference_valid": provided,
                    "supplied_reference_index": sampled_index if provided else None,
                    "drop_probability": drop_probability}


def save_reference_checkpoint(path, model, optimizer, step, config, initialization, noise_rng, index_rng, reference_rng):
    """Save all fresh trainable state, including HRI; no loading/resume path."""
    state = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    torch.save({"format": "vsr_training_v1", "reference_trained": True, "step": step,
                "trainable_state": state, "optimizer": optimizer.state_dict(),
                "noise_rng": noise_rng.get_state(), "index_rng": index_rng.get_state(),
                "reference_rng": reference_rng.get_state(), "config": config, "initialization": initialization}, path)


def train_reference_variant(variant, config, run, clips, latents, embeddings, features, flows, geometry, ntrain, reference_latents):
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
    if variant not in ("Base", "Align", "Relation", "Both"):
        raise ValueError("Unknown reference ablation variant")
    options, reference_options = config["training"], config["reference"]
    if not reference_options.get("enabled"):
        raise ValueError("Reference training requires reference.enabled=true")
    if not 0 < ntrain <= len(clips):
        raise ValueError("A nonempty training selection is required")
    target = run / variant
    target.mkdir(exist_ok=False)
    print(f"{variant}+HRI: independent official initialization", flush=True)
    _timing_sync("cuda")
    model_load_started = time.perf_counter()
    model, initialization = make_model(config, variant)
    _timing_sync("cuda")
    model_load_seconds = time.perf_counter()-model_load_started
    if not hasattr(model, "reference_adapter") or not callable(getattr(model, "clear_reference", None)):
        raise ValueError("Model factory did not create ReferenceConditionalWan")
    initialization["reference_trained"] = True
    initialization["reference_configuration"] = reference_options
    initialization["load_initialize_seconds"] = model_load_seconds
    dump(target / "initialization.json", initialization)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=options["learning_rate"], weight_decay=options["weight_decay"])
    scheduler = UniPCMultistepScheduler.from_pretrained(config["model"]["local_dir"], subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(1000, device="cuda")
    training_seed=options.get("seed",config["seed"])
    noise_rng = torch.Generator(device="cuda").manual_seed(training_seed + 1)
    index_rng = torch.Generator(device="cpu").manual_seed(training_seed + 2)
    reference_rng = torch.Generator(device="cpu").manual_seed(training_seed + reference_options.get("seed_offset", 3))
    _timing_sync("cuda")
    vae_load_started = time.perf_counter()
    vae = AutoencoderKLWan.from_pretrained(config["model"]["local_dir"], subfolder="vae", dtype=torch.float32,
          local_files_only=True, use_safetensors=True).requires_grad_(False).eval().to("cuda")
    _timing_sync("cuda")
    evaluation_vae_load_seconds = time.perf_counter()-vae_load_started
    dump(target / "initialization_timings.json", {"model_load_initialize_seconds": model_load_seconds,
         "evaluation_vae_load_seconds": evaluation_vae_load_seconds,
         "scope": "independent official model/adapter construction and frozen evaluation-VAE load; excludes optimizer/scheduler/text setup"})
    texts = {key: value.to("cuda") for key, value in embeddings.items()}
    metadata = [clips[i]["metadata"] for i in range(len(clips))]
    mask = torch.ones((1, math.prod(geometry["token_grid"])), device="cuda", dtype=torch.bool)
    history, train_seconds, provided_count = [], 0., 0
    index_counts = {str(i): 0 for i in range(config["data"]["num_frames"])}
    hri_gradient_records = []
    torch.cuda.reset_peak_memory_stats()
    phase_start = time.perf_counter()
    for step in range(options["updates"] + 1):
        if step in options["evaluate_at"]:
            started = time.perf_counter()
            final = step == options["updates"]
            evaluation = config["evaluation"]
            selected = (list(range(ntrain)) if config["data"]["selection"] == "debug"
                        else list(range(ntrain, len(clips))))
            if not selected:
                selected = list(range(min(ntrain, 2)))
            summary = evaluate_reference_restoration(model, vae, latents, clips, embeddings, reference_latents, config,
                target / f"step_{step:06d}",
                reference_indices=tuple(evaluation.get("reference_indices", (0, 4)) if final
                                        else evaluation.get("monitor_reference_indices", (0,))),
                seeds=tuple(evaluation["seeds"] if final else evaluation["monitor_seeds"]),
                steps=evaluation["sampling_steps"], sample_indices=selected)
            history.append({"step": step, "metrics": summary, "seconds": time.perf_counter() - started})
            dump(target / "evaluation_history.json", history)
            print(f"{variant}+HRI: matched on/off evaluation step {step} complete", flush=True)
        if step == options["updates"]:
            break
        started = time.perf_counter()
        model.train()
        i = step % ntrain
        clean, low = (latents[i][key].to("cuda") for key in ("clean", "low"))
        text = texts[metadata[i]["caption"]]
        timestep_index = int(torch.randint(0, 1000, (), generator=index_rng))
        sigma = scheduler.sigmas[timestep_index].to("cuda")
        timestep = scheduler.timesteps[timestep_index].expand(1)
        noise = torch.randn(clean.shape, generator=noise_rng, device="cuda")
        noisy = (1 - sigma) * clean + sigma * noise
        reference_kwargs, reference_record = draw_reference_condition(
            reference_latents[i], config["data"]["num_frames"], reference_rng,
            drop_probability=reference_options.get("drop_probability", .5), device="cuda")
        index_counts[str(reference_record["reference_sampled_index"])] += 1
        provided_count += int(reference_record["reference_valid"])
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            velocity, student = model(noisy, timestep, text, low, **reference_kwargs)
        fm = F.mse_loss(velocity.float(), noise - clean)
        teacher = features[i].to("cuda") if features is not None and variant!="Base" else None
        alignment = point_alignment(student, teacher, mask) if variant in ("Align", "Both") else fm.new_zeros(())
        relation, relation_diagnostics = fm.new_zeros(()), None
        if variant in ("Relation", "Both"):
            relation, relation_diagnostics = compute_relation_objective(student, teacher, geometry["token_grid"],
                flows[i]["forward"], flows[i]["backward"], mask, config)
        loss = fm + options["lambda_align"] * alignment + options.get("lambda_relation", 0.) * relation
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite reference loss at {step+1}")
        loss.backward()
        norms = gradient_norms(model) if (step+1) % options["gradient_log_every"] == 0 or step < 3 else None
        if norms is not None:
            if not all(value["finite"] for value in norms.values()) or norms["frozen"]["elements_with_grad"]:
                raise RuntimeError(f"Invalid trainable/frozen reference gradients: {norms}")
            if "hri" not in norms:
                raise RuntimeError("Reference gradient audit requires the hri parameter group")
            hri_gradient_records.append({"step": step+1, "reference_valid": reference_record["reference_valid"],
                                         "hri": norms["hri"]})
        total_norm = torch.nn.utils.clip_grad_norm_(params, options["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        model.clear_features()
        model.clear_reference()
        torch.cuda.synchronize()
        update_seconds = time.perf_counter() - started
        train_seconds += update_seconds
        row = {"step": step+1, "clip_id": metadata[i]["record_id"], "timestep_index": timestep_index,
               "sigma": float(sigma), "timestep": float(timestep[0]), "fm": float(fm.detach()),
               "alignment": float(alignment.detach()), "relation": float(relation.detach()), "loss": float(loss.detach()),
               "relation_diagnostics": relation_diagnostics, "grad_norm_before_clip": float(total_norm),
               "gradient_groups": norms, "update_seconds": update_seconds, "optimizer_updates_seconds": train_seconds,
               **reference_record}
        with (target / "training.jsonl").open("a") as log:
            log.write(json.dumps(row, allow_nan=False) + "\n")
        if (step+1) % options["log_every"] == 0:
            print(f"{variant}+HRI {step+1}/{options['updates']} FM={row['fm']:.6f} A={row['alignment']:.6f} R={row['relation']:.6f}", flush=True)
            dump(run / "progress.json", {"variant": variant, "reference_trained": True,
                 "step": step+1, "updates": options["updates"], "loss": row["loss"]})
        if (step+1) in options["checkpoint_at"]:
            save_reference_checkpoint(target / f"checkpoint_{step+1:06d}.pt", model, optimizer, step+1,
                config, initialization, noise_rng, index_rng, reference_rng)
    if any(parameter.grad is not None for parameter in vae.parameters()):
        raise RuntimeError("Frozen evaluation VAE acquired gradients")
    if not history:
        raise ValueError("Reference budget must include at least one evaluation point")
    summary = {"status": "completed", "variant": variant, "reference_trained": True, "updates": options["updates"],
               "training_clip_count": ntrain, "validation_clip_count": len(clips)-ntrain, "test_accessed": False,
               "optimizer_updates_seconds": train_seconds, "training_gpu_hours": train_seconds/3600,
               "phase_wall_seconds": time.perf_counter()-phase_start,
               "phase_wall_scope": "after model/VAE/optimizer/scheduler/text setup through updates and matched cached-latent evaluations; excludes final release",
               "model_load_initialize_seconds": model_load_seconds,
               "evaluation_vae_load_seconds": evaluation_vae_load_seconds,
               "complete_variant_wall_location": "run phase_timings.json phases.variant_<name>, includes model/VAE loading, setup, updates, evaluation, cleanup and release",
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "evaluation": history[-1],
               "caption_policy": config["caption_policy"], "quality_hypotheses_verified": False,
               "reference_sampling": {"drop_probability": reference_options.get("drop_probability", .5),
                                      "provided_updates": provided_count, "missing_updates": options["updates"]-provided_count,
                                      "sampled_index_counts": index_counts, "independent_rng_seed": training_seed+reference_options.get("seed_offset", 3)},
               "note": "Budget completed; provided reference benefits and missing-reference stability require assessment"}
    dump(target / "hri_gradient_audit.json", {"records": hri_gradient_records,
         "any_provided_hri_gradient": any(row["reference_valid"] and row["hri"]["l2"] > 0 for row in hri_gradient_records),
         "zero_initialization_note": "HRI stem gradients can be zero before output residual maps open"})
    dump(target / "summary.json", summary)
    optimizer.zero_grad(set_to_none=True)
    model.clear_features()
    model.clear_reference()
    del model, optimizer, vae, params, texts
    release()
    return summary


def main():
    program_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    config["_sha256"] = file_hash(config_path)
    if not config.get("reference", {}).get("enabled"):
        raise ValueError("reference.enabled=true is required")
    if config["training"].get("batch_size", 1) != 1:
        raise ValueError("First reference runner supports one video per update")
    if config["training"]["updates"] not in config["training"]["evaluate_at"]:
        raise ValueError("Final budget must include matched on/off evaluation")
    name = config["experiment_name"]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", name):
        raise ValueError("Experiment name must be a simple directory name")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = ROOT / "runs" / f"{name}_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    dump(run / "config.json", config)
    dump(ROOT / "runs/active_reference_training.json", {"status": "running", "run": str(run)})
    print(f"New independent reference experiment: {run}", flush=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    timings = _PhaseTimings(run / "phase_timings.json", started=program_started)
    try:
        dump(run/"reproducibility.json",configure_reproducibility(config))
        with timings.phase("environment"):
            env = environment(ROOT)
            dump(run / "environment.json", env)
        if not env["cuda_available"]:
            raise RuntimeError("CUDA unavailable; host GPU execution required")
        with timings.phase("official_artifact_verification"):
            dump(run / "official_artifacts.json", verify_official(ROOT, config))
        with timings.phase("shared_pair_preparation", device="cuda"):
            data = prepare(config, run, env)
        from diffusers import AutoencoderKLWan
        with timings.phase("reference_vae_load", device="cuda"):
            encoder = AutoencoderKLWan.from_pretrained(config["model"]["local_dir"], subfolder="vae", dtype=torch.float32,
                local_files_only=True, use_safetensors=True).requires_grad_(False).eval().to("cuda")
        with timings.phase("reference_single_frame_encoding"):
            # The encoder function owns synchronization and peak measurement
            # for this phase; do not reset its counters in an outer timer.
            reference_latents = encode_reference_frames(encoder, data[0], config, run / "reference_cache")
        with timings.phase("reference_encoder_release", device="cuda"):
            del encoder
            release()
        with timings.phase("reference_supervision_seal"):
            from .reference_diagnostics import seal_reference_supervision
            seal_reference_supervision(run, config)
        summaries = {}
        for variant in config["training"]["variants"]:
            with timings.phase(f"variant_{variant}", device="cuda"):
                summaries[variant] = train_reference_variant(variant, config, run, *data, reference_latents)
        status = {"status": "completed", "stage": config["stage"], "run": str(run), "variants": list(summaries),
                  "reference_trained": True, "test_accessed": False, "quality_hypotheses_verified": False,
                  "note": "Completion means fixed budgets exhausted; matched HRI gains require assessment"}
        dump(run / "status.json", status)
        dump(ROOT / "runs/active_reference_training.json", status)
        print(json.dumps(status), flush=True)
        timings.finish("completed")
    except Exception as error:
        status = {"status": "failed", "run": str(run), "error": str(error), "traceback": traceback.format_exc()}
        dump(run / "status.json", status)
        dump(ROOT / "runs/active_reference_training.json", status)
        timings.finish("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
