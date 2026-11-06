"""Stage 1 real-model preflight. Every invocation creates a fresh run directory.

This is a bounded diagnostic, not a formal training/evaluation entry point.
Inference is implemented separately and receives only an LRV latent and text.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import torch
import torch.nn.functional as F

from .alignment import DinoTeacher, latent_geometry, point_alignment
from .audit import dump, environment, file_hash, verify_official
from .data import decode_clip, validate_manifest, write_video
from .model import ConditionalWan, gradient_norms


@contextmanager
def autocast():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        yield


def release():
    gc.collect()
    torch.cuda.empty_cache()


def psnr(a, b):
    mse = float(F.mse_loss(a.float(), b.float()))
    return {"mse_rgb_01": mse, "psnr_rgb_db": -10 * math.log10(max(mse, 1e-12)),
            "definition": "all RGB pixels, no border shave, range [0,1]; debug clip only"}


def normalize_latent(z, vae, inverse=False):
    mean = z.new_tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = z.new_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    return z * std + mean if inverse else (z - mean) / std


@torch.no_grad()
def encode_text(root, config, captions, run):
    from transformers import AutoTokenizer, UMT5EncoderModel
    base = config["model"]["local_dir"]
    tokenizer = AutoTokenizer.from_pretrained(base, subfolder="tokenizer", local_files_only=True)
    text_model = UMT5EncoderModel.from_pretrained(base, subfolder="text_encoder", dtype=torch.bfloat16,
                                                local_files_only=True, use_safetensors=True).requires_grad_(False).eval().to("cuda")
    prompts = list(dict.fromkeys(captions))
    inputs = tokenizer(prompts, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    with autocast():
        out = text_model(input_ids=inputs.input_ids.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).last_hidden_state
    # Same masked-position handling as official WanPipeline._get_t5_prompt_embeds.
    out = out.cpu()
    for i, length in enumerate(inputs.attention_mask.sum(1).tolist()):
        out[i, length:] = 0
    result = {caption: out[i:i+1].contiguous() for i, caption in enumerate(prompts)}
    dump(run / "text_audit.json", {"prompts": prompts, "status": "generic_debug_prompts_only", "text_max_length": 512,
                                   "encoder_frozen": True, "encoder_has_grad": any(p.grad is not None for p in text_model.parameters())})
    torch.save(result, run / "fresh_text_embeddings.pt")
    del text_model, out
    release()
    return result


@torch.no_grad()
def encode_videos(config, clips, run, geometry):
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(config["model"]["local_dir"], subfolder="vae", dtype=torch.float32,
                                          local_files_only=True, use_safetensors=True).requires_grad_(False).eval().to("cuda")
    encoded = []
    for i, clip in enumerate(clips):
        hr = clip["hr"].to("cuda")
        up = clip["lr_up"].to("cuda")
        # Mode avoids an extra VAE posterior sample and is fixed across all groups.
        clean_raw = vae.encode(hr * 2 - 1).latent_dist.mode()
        low_raw = vae.encode(up * 2 - 1).latent_dist.mode()
        restored = (vae.decode(clean_raw).sample.float() / 2 + .5).clamp(0, 1).cpu()
        if tuple(restored.shape) != tuple(clip["hr"].shape):
            raise ValueError("VAE roundtrip changed frame count or spatial geometry")
        if list(clean_raw.shape[2:]) != geometry["latent_grid"]:
            raise ValueError(f"Runtime VAE latent shape {clean_raw.shape} differs from derived geometry {geometry}")
        clean = normalize_latent(clean_raw, vae).cpu()
        low = normalize_latent(low_raw, vae).cpu()
        error = psnr(restored, clip["hr"])
        dump(run / f"clip_{i}/vae_roundtrip.json", {**error, "shape": list(restored.shape), "posterior": "mode",
                                                   "vae_frozen": True, "vae_has_grad": any(p.grad is not None for p in vae.parameters())})
        write_video(run / f"clip_{i}/vae_roundtrip.mp4", restored, clip["fps"])
        encoded.append({"clean": clean, "low": low})
        torch.save({"clean": clean, "low": low, "binding": clip["binding"]}, run / f"clip_{i}/fresh_latents.pt")
        del hr, up, clean_raw, low_raw, restored
    del vae
    release()
    return encoded


@torch.no_grad()
def sample_lrv(model, low_latent, text, scheduler_dir, steps, seed, enable_lrv=True):
    """Only conditioning inputs are the encoded user LRV and text. No HRV/teacher."""
    from diffusers import UniPCMultistepScheduler
    scheduler = UniPCMultistepScheduler.from_pretrained(scheduler_dir, subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(steps, device="cuda")
    noise_rng = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(low_latent.shape, generator=noise_rng, device="cuda", dtype=torch.float32)
    model.eval()
    for t in scheduler.timesteps:
        with autocast():
            velocity, _ = model(x, t.expand(x.shape[0]), text, low_latent, enable_lrv=enable_lrv)
        x = scheduler.step(velocity.float(), t, x, return_dict=False)[0]
    model.clear_features()
    return x.cpu()


def train_variant(name, config, clips, encoded, embeddings, teacher_features, geometry, run):
    from diffusers import WanTransformer3DModel, UniPCMultistepScheduler
    seed = config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    opts = config["training"]
    target = run / name
    target.mkdir()
    print(f"{name}: loading fresh official transformer", flush=True)
    native, info = WanTransformer3DModel.from_pretrained(config["model"]["local_dir"], subfolder="transformer",
                                                       dtype=torch.bfloat16, local_files_only=True,
                                                       use_safetensors=True, output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Native model did not load exactly: {info}")
    if len(native.blocks) != 30 or native.config.in_channels != 16:
        raise ValueError("Expected official Wan2.1 T2V 1.3B")
    native_count = sum(p.numel() for p in native.parameters())
    for parameter_name, parameter in native.named_parameters():
        if any(token in parameter_name for token in (native._keep_in_fp32_modules or [])) and parameter.dtype != torch.float32:
            raise ValueError(f"Native FP32 parameter exception lost: {parameter_name}")
    c = config["conditioning"]
    model = ConditionalWan(native, **{k:c[k] for k in ("rank", "alpha", "injection_layers", "alignment_layer", "stem_channels", "projector_hidden", "teacher_dim")},
                           projection=(name == "Align"), gradient_checkpointing=opts["gradient_checkpointing"]).to("cuda")
    # Moving the module must preserve native FP32 time/norm exceptions.
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    dump(target / "initialization.json", {"official_revision": config["model"]["revision"], "loading_info": info,
         "legacy_state_loaded": False, "optimizer_fresh": True, "seed": seed,
         "trainable_parameters": {n:p.numel() for n,p in trainable}, "trainable_count": sum(p.numel() for _,p in trainable),
         "native_count": native_count, "tensor_dtypes": sorted({str(p.dtype) for p in model.parameters()})})
    optimizer = torch.optim.AdamW([p for _,p in trainable], lr=opts["learning_rate"], weight_decay=opts["weight_decay"])
    scheduler = UniPCMultistepScheduler.from_pretrained(config["model"]["local_dir"], subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(1000, device="cuda")
    # Explicit independent RNG keeps Base/Align noise and time samples identical.
    noise_rng = torch.Generator(device="cuda").manual_seed(seed + 1)
    index_rng = torch.Generator(device="cpu").manual_seed(seed + 2)
    rows = []
    model.train()
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    mask = torch.ones((1, math.prod(geometry["token_grid"])), device="cuda", dtype=torch.bool)
    for step in range(opts["updates_per_variant"]):
        i = step % len(clips)
        clean, low = (encoded[i][k].to("cuda") for k in ("clean", "low"))
        text = embeddings[clips[i]["metadata"]["caption"]].to("cuda")
        index = int(torch.randint(0, 1000, (), generator=index_rng))
        sigma = scheduler.sigmas[index].to("cuda")
        t = scheduler.timesteps[index].expand(1)
        noise = torch.randn(clean.shape, generator=noise_rng, device="cuda")
        noisy = (1 - sigma) * clean + sigma * noise
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            velocity, student = model(noisy, t, text, low)
        if tuple(model.token_grid) != tuple(geometry["token_grid"]):
            raise ValueError("Runtime DiT token grid differs from correspondence")
        loss_fm = F.mse_loss(velocity.float(), noise - clean)
        loss_a = point_alignment(student, teacher_features[i].to("cuda"), mask) if student is not None else loss_fm.new_zeros(())
        loss = loss_fm + opts["lambda_align"] * loss_a
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        gradients = gradient_norms(model)
        if not all(g["finite"] for g in gradients.values()) or gradients["frozen"]["elements_with_grad"]:
            raise ValueError(f"Invalid gradient or frozen gradient leak: {gradients}")
        torch.nn.utils.clip_grad_norm_([p for _,p in trainable], opts["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        model.clear_features()
        row = {"step": step + 1, "clip_id": clips[i]["metadata"]["record_id"], "timestep_index": index,
               "timestep": float(t[0]), "sigma": float(sigma), "fm": float(loss_fm.detach()), "alignment": float(loss_a.detach()),
               "total": float(loss.detach()), "gradients_before_clip": gradients, "elapsed_seconds": time.perf_counter() - start}
        rows.append(row)
        with (target / "training.jsonl").open("a") as log:
            log.write(json.dumps(row, allow_nan=False) + "\n")
        print(f"{name} update {step+1}: FM={row['fm']:.6f} A={row['alignment']:.6f}", flush=True)
    torch.cuda.synchronize()
    optimizer_updates_seconds = time.perf_counter() - start
    # Recompute a fresh graph and use alignment alone to audit the generator path.
    alignment_gradients = None
    if name == "Align":
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            _, student = model(noisy, t, text, low)
        alignment_only = point_alignment(student, teacher_features[i].to("cuda"), mask)
        alignment_only.backward()
        alignment_gradients = gradient_norms(model)
        for group in ("lrv_stem", "lrv_residuals", "lora", "projector"):
            if not alignment_gradients[group]["finite"] or alignment_gradients[group]["l2"] <= 0:
                raise ValueError(f"Alignment does not reach {group}: {alignment_gradients}")
        if alignment_gradients["frozen"]["elements_with_grad"]:
            raise ValueError("Alignment reached frozen weights")
        model.clear_features()
        dump(target / "alignment_only_gradients.json", alignment_gradients)
    final_grads = rows[-1]["gradients_before_clip"]
    for group in ("lrv_stem", "lrv_residuals", "lora"):
        if final_grads[group]["l2"] <= 0:
            raise ValueError(f"Generator has no effective gradient after updates: {group}")
    optimizer.zero_grad(set_to_none=True)
    # This newly created adapter is saved for audit only. No resume path exists.
    state = {n:p.detach().cpu().clone() for n,p in trainable}
    torch.save({"format": "stage1_diagnostic_only", "variant": name, "step": len(rows), "trainable_state": state,
                "config": config, "manifest_sha256": clips[0]["binding"]["manifest_sha256"]}, target / "fresh_adapter.pt")
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated()
    # LRV sensitivity at fixed noisy latent/text; no quality claim from this value.
    model.eval()
    with torch.no_grad(), autocast():
        correct, _ = model(noisy, t, text, low)
        changed, _ = model(noisy, t, text, low.flip(-1))
        disabled, _ = model(noisy, t, text, low, enable_lrv=False)
    sensitivity = {"flow_mae_flipped_lrv": float((correct-changed).abs().mean()),
                   "flow_mae_disabled_lrv": float((correct-disabled).abs().mean()),
                   "interpretation": "condition path sensitivity only; does not prove correct recovery"}
    outputs = {}
    for seed_value in config["sampling"]["seeds"]:
        sample_start = time.perf_counter()
        generated = sample_lrv(model, encoded[0]["low"].to("cuda"), embeddings[clips[0]["metadata"]["caption"]].to("cuda"),
                               config["model"]["local_dir"], config["sampling"]["steps"], seed_value)
        outputs[str(seed_value)] = generated
        dump(target / f"sampling_{seed_value}.json", {"seed": seed_value, "steps": config["sampling"]["steps"],
             "cfg": 1.0, "seconds": time.perf_counter()-sample_start, "inference_inputs": ["LRV", "caption"], "hri": False})
    dump(target / "summary.json", {"status": "passed", "updates": len(rows), "optimizer_updates_seconds": optimizer_updates_seconds,
         "optimizer_updates_gpu_hours": optimizer_updates_seconds/3600, "diagnostic_phase_seconds": train_seconds,
         "timing_scope": "optimizer updates exclude model loading/text/VAE/teacher; diagnostic phase also includes gradient probe and checkpoint save",
         "peak_allocated_bytes": peak, "condition_sensitivity": sensitivity,
         "alignment_only_gradients": alignment_gradients, "quality_conclusion": "not_evaluated"})
    del model, native, optimizer, state, trainable, correct, changed, disabled
    release()
    return outputs


@torch.no_grad()
def decode_samples(config, clips, outputs, run):
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(config["model"]["local_dir"], subfolder="vae", dtype=torch.float32,
                                          local_files_only=True, use_safetensors=True).requires_grad_(False).eval().to("cuda")
    for variant, samples in outputs.items():
        for seed, z in samples.items():
            video = (vae.decode(normalize_latent(z.to("cuda"), vae, inverse=True)).sample.float()/2+.5).clamp(0,1).cpu()
            if video.shape != clips[0]["hr"].shape:
                raise ValueError("Sampling changed original frame count/spatial output geometry")
            write_video(run / variant / f"sample_{seed}.mp4", video, clips[0]["fps"])
            # Lossless tensor, so any future diagnostic isn't measured on mp4v.
            torch.save(video, run / variant / f"sample_{seed}_rgb.pt")
    del vae
    release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    config = json.loads((root / args.config).read_text())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = root / "runs" / f"stage1_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    dump(run / "config.json", config)
    dump(root / "runs/latest.json", {"run": str(run), "status": "running"})
    print(f"Fresh run: {run}", flush=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    try:
        env = environment(root)
        dump(run / "environment.json", env)
        if not env["cuda_available"] or not env["bf16"]:
            raise RuntimeError("Real stage1 needs CUDA BF16; sandbox execution may hide GPU; use host execution")
        dump(run / "official_artifacts.json", verify_official(root, config))
        manifest = json.loads((root / config["data"]["manifest"]).read_text())
        validate_manifest(manifest)
        dump(run / "manifest.json", manifest)
        base = Path(config["model"]["local_dir"])
        geometry = latent_geometry(json.loads((base/"vae/config.json").read_text()),
                                   json.loads((base/"transformer/config.json").read_text()),
                                   config["data"]["num_frames"], config["data"]["height"], config["data"]["width"])
        dump(run / "correspondence.json", geometry)
        clips = []
        for i, record in enumerate(manifest["debug_records"][:config["data"]["debug_clips"]]):
            clip = decode_clip(record, **{k: config["data"][k] for k in ("num_frames", "height", "width", "scale")}, seed=config["seed"])
            clip["binding"] = {"data_cache_key": clip["cache_key"], "manifest_sha256": manifest["manifest_sha256"],
                               "model_revision": config["model"]["revision"], "code_commit": env["git_commit"],
                               "code_sha256": env["code_sha256"], "config_sha256": file_hash(root / args.config)}
            dump(run / f"clip_{i}/metadata.json", {**clip["metadata"], "binding": clip["binding"]})
            for key in ("hr", "lr", "lr_up"):
                write_video(run / f"clip_{i}/{key}.mp4", clip[key], clip["fps"])
            clips.append(clip)
        if not clips:
            raise ValueError("No debug HQ-VSR train clips")
        text = encode_text(root, config, [x["metadata"]["caption"] for x in clips], run)
        latents = encode_videos(config, clips, run, geometry)
        teacher = DinoTeacher(root / config["teacher"]["source_dir"], root / config["teacher"]["weights"]).to("cuda")
        with torch.no_grad():
            features = [teacher(x["hr"].to("cuda"), geometry["anchors"], tuple(geometry["token_grid"])).cpu() for x in clips]
        dump(run / "teacher_audit.json", {"frozen": not any(p.requires_grad for p in teacher.parameters()),
             "has_grad": any(p.grad is not None for p in teacher.parameters()), "feature_shapes": [list(x.shape) for x in features],
             "resize_policy": "full crop -> 2x student patch grid *14 pixels; bilinear spatial token mapping; no teacher crop"})
        del teacher
        release()
        outputs = {name: train_variant(name, config, clips, latents, text, features, geometry, run) for name in config["training"]["variants"]}
        decode_samples(config, clips, outputs, run)
        status = {"status": "passed", "stage": "stage1_preflight", "run": str(run), "quality_hypotheses_verified": False,
                  "formal_training_started": False, "caption_semantics_verified": False,
                  "pending": ["video-specific captions", "Base recovery and LRV corruption controls", "validation-selected loss weights and fixed formal budget", "Relation/Both", "HRI", "external evaluation"]}
        dump(run / "status.json", status)
        dump(root / "runs/latest.json", status)
        print(json.dumps(status), flush=True)
    except Exception as error:
        status = {"status": "failed", "run": str(run), "error": str(error), "traceback": traceback.format_exc()}
        dump(run / "status.json", status)
        dump(root / "runs/latest.json", status)
        raise


if __name__ == "__main__":
    main()
