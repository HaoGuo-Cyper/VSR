"""Streaming, Base-only preparation for the accepted 33-frame protocol.

One RGB clip and one frozen FP32 VAE are live during preparation. Only the
small normalized latent pairs accumulate on CPU. Text comes from an explicitly
bound cache; no T5, DINO, RAFT, task model or optimizer is constructed here.
Historical data.py/preflight.py and the training numerical path stay unchanged.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch

from .alignment import latent_geometry
from .audit import dump, file_hash
from .cache import PairedClipSequence, validate_cache
from .data import CAPTION_PLACEHOLDER, decode_clip, validate_manifest, write_video
from .preflight import normalize_latent, psnr

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_RUNS = Path("/home/hugo/Projects/VSR/runs")
ENCODER_FILES = ("vsr/data.py", "vsr/preflight.py", "vsr/stage2_preparation.py")
EXPECTED_GRID = [9, 16, 28]
DIAGNOSTIC_STAGE = "stage2_full_optimizer_resource_diagnostic"


def _need(value, message):
    if not value:
        raise ValueError(message)


def _json(path):
    return json.loads(Path(path).read_text())


def _new_run(run):
    run = Path(run).resolve()
    allowed = (ROOT / "runs").resolve()
    _need(run != allowed and run.is_relative_to(allowed) and run.is_dir(), "Stage 2 output must be an existing new ROOT/runs child")
    _need(not any(run.glob("clip_*")), "Refusing to reuse partial clip preparation")
    for name in ("selection.json", "correspondence.json", "fresh_text_embeddings.pt", "fresh_training_cache.pt", "cache_reuse.json", "preparation_complete.json"):
        _need(not (run / name).exists(), "Refusing to overwrite existing preparation: " + name)
    return run


def _base_config(config):
    options, data = config["training"], config["data"]
    _need(options["variants"] == ["Base"] and options.get("preparation_variants", ["Base"]) == ["Base"], "Stage 2 preparation is Base-only; auxiliary preparation is forbidden")
    _need(not config.get("reference", {}).get("enabled", False) and not config["conditioning"].get("hri_enabled", False), "HRI is forbidden in this Base preparation")
    _need(options.get("lambda_align", 0) == 0 and options.get("lambda_relation", 0) == 0, "Base auxiliary weights must be zero")
    _need([data[k] for k in ("num_frames", "height", "width", "scale", "temporal_stride")] == [33, 128, 224, 4, 1], "Expected accepted 33-frame stride1 crop geometry")
    _need(data["selection"] == "train" and config["reproducibility"]["mode"] == "strict", "Expected strict fixed train/validation preparation")
    _need(config["model"]["repo_id"] == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
          and config["model"]["revision"] == "0fad780a534b6463e45facd96134c9f345acfa5b", "Pinned official Wan model required")
    contract = config.get("stage2_contract", {})
    _need(contract.get("T5_load_allowed", False) is False and contract.get("DINO_load_allowed", False) is False, "Base cannot permit T5/DINO construction")
    if config.get("stage") == DIAGNOSTIC_STAGE:
        _need(contract.get("expected_training_clips") == 2 and contract.get("expected_validation_clips") == 0
              and data.get("train_limit") == 2 and data.get("val_limit") == 0 and options["updates"] == 8,
              "Only the explicit two-train/no-val/eight-update diagnostic may omit validation")
    else:
        _need(config.get("stage") == "stage2_protocol33_noref_base" and options["updates"] == 8192
              and contract.get("expected_training_clips") == 1641 and contract.get("expected_validation_clips") == 32
              and data.get("train_limit") is None and data.get("val_limit") == 32,
              "Formal Stage 2 requires the fixed full train1641/val32 and 8192 updates")


def _encoder_hashes(env):
    hashes = {name: file_hash(ROOT / name) for name in ENCODER_FILES}
    _need(all(env.get("code_sha256", {}).get(name) == value for name, value in hashes.items()), "Preparation encoder source differs from the recorded environment")
    return hashes


def _text_cache(config):
    spec = config["data"]["text_cache"]
    source = Path(spec["source_run"]).resolve()
    path = Path(spec["path"]).resolve()
    allowed = [(ROOT / "runs").resolve(), ORIGINAL_RUNS.resolve()]
    _need(any(source != root and source.is_relative_to(root) for root in allowed), "Text cache source must be an own new or historical VSR run")
    _need(path == source / "fresh_text_embeddings.pt", "Text cache must be the source run's fresh_text_embeddings.pt")
    _need(file_hash(path) == spec["sha256"], "Text cache file SHA256 mismatch; no T5 fallback")
    _need(_json(source / "config.json")["model"] == config["model"], "Text cache official-model contract mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _need(isinstance(payload, dict) and set(payload) == {CAPTION_PLACEHOLDER}, "Only the fixed generic cached caption is supported")
    value = payload[CAPTION_PLACEHOLDER]
    _need(isinstance(value, torch.Tensor) and list(value.shape) == [1, 512, 4096]
          and value.dtype == torch.bfloat16 and value.device.type == "cpu" and not value.requires_grad
          and bool(torch.isfinite(value).all()), "Cached text must be finite frozen CPU BF16 [1,512,4096]")
    header = json.dumps([str(value.dtype), list(value.shape)], separators=(",", ":")).encode() + b"\n"
    tensor_sha = hashlib.sha256(header + value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    _need(spec.get("embedding_sha256", tensor_sha) == tensor_sha, "Cached text tensor SHA256 mismatch")
    _need(file_hash(path) == spec["sha256"], "Text cache changed while being read")
    audit = {"caption": CAPTION_PLACEHOLDER, "source": "explicit_verified_frozen_text_cache", "source_run": str(source),
             "path": str(path), "sha256": spec["sha256"], "embedding_sha256": tensor_sha,
             "shape": list(value.shape), "dtype": str(value.dtype), "T5_loaded": False, "fallback": False}
    return payload, path, audit


def _save_text(run, source, audit):
    destination = run / "fresh_text_embeddings.pt"
    _need(not destination.exists(), "Refusing to overwrite cached text")
    shutil.copyfile(source, destination)
    _need(file_hash(destination) == audit["sha256"], "Copied text cache digest mismatch")
    dump(run / "text_audit.json", audit)


def _geometry(config):
    base = Path(config["model"]["local_dir"])
    data = config["data"]
    value = latent_geometry(_json(base / "vae/config.json"), _json(base / "transformer/config.json"),
                            data["num_frames"], data["height"], data["width"])
    _need(value["latent_grid"] == EXPECTED_GRID and value["token_grid"] == [9, 8, 14], "Unexpected accepted latent/token grid")
    return value


def _load_vae(config):
    from diffusers import AutoencoderKLWan
    return AutoencoderKLWan.from_pretrained(config["model"]["local_dir"], subfolder="vae", dtype=torch.float32,
        local_files_only=True, use_safetensors=True).requires_grad_(False).eval().to("cuda")


@torch.no_grad()
def _encode_pair(vae, clip, geometry):
    device = next(vae.parameters()).device
    _need(all(p.dtype == torch.float32 and not p.requires_grad for p in vae.parameters()), "Expected one frozen FP32 VAE")
    with torch.autocast(device.type, enabled=False):
        hr, up = (clip[key].to(device=device, dtype=torch.float32) for key in ("hr", "lr_up"))
        clean_raw = vae.encode(hr * 2 - 1).latent_dist.mode()
        low_raw = vae.encode(up * 2 - 1).latent_dist.mode()
        expected = (1, 16, *geometry["latent_grid"])
        _need(all(tuple(z.shape) == expected and z.dtype == torch.float32 and bool(torch.isfinite(z).all())
                  for z in (clean_raw, low_raw)), "VAE returned invalid raw latent geometry/dtype/values")
        restored = (vae.decode(clean_raw).sample.float() / 2 + .5).clamp(0, 1).cpu()
        _need(restored.shape == clip["hr"].shape and bool(torch.isfinite(restored).all()), "VAE roundtrip geometry/nonfinite failure")
        pair = {"clean": normalize_latent(clean_raw, vae).detach().cpu(), "low": normalize_latent(low_raw, vae).detach().cpu()}
        _need(all(bool(torch.isfinite(z).all()) for z in pair.values()), "Normalized VAE latent is nonfinite")
        error = psnr(restored, clip["hr"])
    return pair, restored, error


def _verify_seal(source, required):
    seal_path = source / "cache_asset_seal.json"
    _need(seal_path.is_file(), "A completed and sealed Stage 2 cache is required")
    seal = _json(seal_path)
    files = seal.get("artifact_sha256", {})
    _need(required.issubset(files), "Cache seal does not cover all required data artifacts")
    _need(seal.get("contains_task_initialization") is False, "Cache seal must exclude task initialization")
    for name, expected in files.items():
        path = (source / name).resolve()
        _need(path.is_relative_to(source) and path.is_file() and file_hash(path) == expected, "Frozen cache asset checksum/path mismatch: " + name)
    return file_hash(seal_path)


def _reuse(config, run, env, manifest, training, validation, embeddings, geometry, encoder_hashes):
    source = Path(config["data"]["cache_from"]).resolve()
    allowed = (ROOT / "runs").resolve()
    _need(source != run and source != allowed and source.is_relative_to(allowed), "Reuse only a different run under the actual new ROOT/runs")
    _base_config(_json(source / "config.json"))
    _need(_json(source / "status.json")["status"] == "completed", "Cache producer must have completed; do not reuse an active/failed attempt")
    expected_selection = {"manifest_sha256": manifest["manifest_sha256"], "training_ids": [r["id"] for r in training],
                          "validation_ids": [r["id"] for r in validation], "test_accessed": False,
                          "caption_policy": config["caption_policy"]}
    _need(_json(source / "selection.json") == expected_selection, "Cache selection/data-window order differs")
    count = len(training) + len(validation)
    required = {"config.json", "selection.json", "correspondence.json", "fresh_training_cache.pt", "fresh_text_embeddings.pt"}
    required.update(f"clip_{i}/{name}" for i in range(count) for name in ("metadata.json", "paired_rgb.pt", "fresh_latents.pt"))
    seal_sha = _verify_seal(source, required)
    verified = validate_cache(source, config, manifest["manifest_sha256"], encoder_hashes,
                              allowed_root=allowed, expected_manifest=manifest)
    _need(_json(source / "correspondence.json") == geometry, "Cache latent geometry differs")
    cached_text = torch.load(source / "fresh_text_embeddings.pt", map_location="cpu", weights_only=True)
    _need(isinstance(cached_text, dict) and set(cached_text) == set(embeddings)
          and all(isinstance(cached_text[k], torch.Tensor) and cached_text[k].dtype == embeddings[k].dtype
                  and torch.equal(cached_text[k], embeddings[k]) for k in embeddings), "Cached text differs from the explicitly bound text source")
    frozen = torch.load(source / "fresh_training_cache.pt", map_location="cpu", weights_only=True)
    _need(isinstance(frozen, dict) and frozen.get("teacher_features") is None, "Base cache cannot carry DINO targets")
    latents = frozen.get("latents")
    _need(isinstance(latents, list) and len(latents) == count, "Cache latent count differs")
    first_binding = _json(source / "clip_0/metadata.json")["binding"]
    _need(frozen.get("binding") == first_binding, "Top-level latent binding differs")
    shape = (1, 16, *geometry["latent_grid"])
    for i, pair in enumerate(latents):
        _need(isinstance(pair, dict) and set(pair) == {"clean", "low"}, "Invalid Base latent pair")
        _need(all(isinstance(z, torch.Tensor) and tuple(z.shape) == shape and z.dtype == torch.float32
                  and z.device.type == "cpu" and not z.requires_grad and bool(torch.isfinite(z).all()) for z in pair.values()), "Invalid cached latent tensor")
        item = torch.load(source / f"clip_{i}/fresh_latents.pt", map_location="cpu", weights_only=True)
        _need(item.get("binding") == _json(source / f"clip_{i}/metadata.json")["binding"]
              and all(torch.equal(pair[k], item[k]) for k in ("clean", "low")), "Per-clip/top-level latent mismatch")
    clips = PairedClipSequence(source, max_cached=2, allowed_root=allowed)
    verified.update(source_asset_seal_sha256=seal_sha, source_encoder_hashes=encoder_hashes,
                    task_checkpoint_loaded=False, T5_loaded=False, DINO_loaded=False)
    dump(run / "cache_reuse.json", verified)
    return clips, latents, cached_text, None, None, geometry, len(training)


def prepare_stage2(config, run, env):
    """Return train_variant's preparation tuple, with bounded CPU RGB retention."""
    from .train import select_records
    _base_config(config)
    run = _new_run(run)
    _need(_json(run / "config.json") == config and _json(run / "environment.json") == env,
          "Runner must persist the exact config/environment before preparation")
    encoder_hashes = _encoder_hashes(env)
    manifest = _json(ROOT / config["data"]["manifest"])
    validate_manifest(manifest)
    training, validation = select_records(manifest, config)
    contract = config.get("stage2_contract", {})
    _need(training and (validation or config.get("stage") == DIAGNOSTIC_STAGE), "Separate nonempty training and validation selections are required outside the explicit diagnostic")
    _need(len(training) == contract.get("expected_training_clips", len(training))
          and len(validation) == contract.get("expected_validation_clips", len(validation)), "Fixed training/validation counts differ")
    _need(all(r.get("caption") == CAPTION_PLACEHOLDER for r in training + validation), "Only the fixed generic caption is permitted")
    embeddings, text_path, text_audit = _text_cache(config)
    geometry = _geometry(config)
    dump(run / "selection.json", {"manifest_sha256": manifest["manifest_sha256"], "training_ids": [r["id"] for r in training],
        "validation_ids": [r["id"] for r in validation], "test_accessed": False, "caption_policy": config["caption_policy"]})
    dump(run / "correspondence.json", geometry)
    _save_text(run, text_path, text_audit)
    if config["data"].get("cache_from"):
        return _reuse(config, run, env, manifest, training, validation, embeddings, geometry, encoder_hashes)
    records = training + validation
    latents, first_binding, vae = [], None, None
    started = time.perf_counter()
    try:
        vae = _load_vae(config)
        for i, record in enumerate(records):
            clip = decode_clip(record, num_frames=33, height=128, width=224, scale=4,
                               seed=config["seed"], temporal_stride=1)
            directory = run / f"clip_{i}"
            directory.mkdir()
            clip["binding"] = {"data_cache_key": clip["cache_key"], "manifest_sha256": manifest["manifest_sha256"],
                "official_revision": config["model"]["revision"], "code_commit": env["git_commit"],
                "code_sha256": env["code_sha256"], "config_sha256": config["_sha256"]}
            binding = clip["binding"]
            if first_binding is None:
                first_binding = binding
            torch.save({key: clip[key] for key in ("hr", "lr", "lr_up")}, directory / "paired_rgb.pt")
            dump(directory / "metadata.json", {**clip["metadata"], "binding": binding})
            pair, restored, error = _encode_pair(vae, clip, geometry)
            latents.append(pair)
            torch.save({**pair, "binding": binding}, directory / "fresh_latents.pt")
            dump(directory / "vae_roundtrip.json", {**error, "shape": list(restored.shape), "posterior": "mode",
                "vae_frozen": True, "vae_has_grad": False, "precision": "FP32", "streaming_RGB": True})
            if i < min(config["data"].get("save_input_videos", 2), 2):
                for key in ("hr", "lr", "lr_up"):
                    write_video(directory / f"{key}.mp4", clip[key], clip["fps"])
                write_video(directory / "vae_roundtrip.mp4", restored, clip["fps"])
            del clip, restored, pair
            if (i + 1) % 25 == 0 or i + 1 == len(records):
                dump(run / "preparation_progress.json", {"clips_completed": i + 1, "clips_total": len(records),
                    "elapsed_seconds": time.perf_counter() - started, "RGB_resident_clip_bound": 1, "T5_loaded": False, "DINO_loaded": False})
                print(f"Stage2 streaming VAE preparation {i+1}/{len(records)}", flush=True)
        torch.save({"latents": latents, "teacher_features": None, "binding": first_binding}, run / "fresh_training_cache.pt")
        dump(run / "preparation_complete.json", {"status": "completed", "clip_count": len(records),
            "training_clips": len(training), "validation_clips": len(validation), "encoder_source_hashes": encoder_hashes,
            "RGB_resident_clip_bound": 1, "returned_RGB_LRU_bound": 2, "T5_loaded": False, "DINO_loaded": False,
            "task_model_or_optimizer_loaded": False, "elapsed_seconds": time.perf_counter() - started})
    finally:
        if vae is not None:
            vae.clear_cache()
        vae = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    clips = PairedClipSequence(run, max_cached=2, allowed_root=ROOT / "runs")
    return clips, latents, embeddings, None, None, geometry, len(training)
