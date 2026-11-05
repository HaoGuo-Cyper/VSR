"""Seal complete frozen preparation after an interrupted training process.

This module only reads a fixed allowlist of data/text/VAE artifacts. It never
opens task checkpoints, optimizer state, or adapter weights, and never changes
an experiment's training status. The seal is compatible with train.prepare.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import torch

from .alignment import latent_geometry
from .audit import file_hash
from .cache import PairedClipSequence, PRODUCTION_RUNS_ROOT, _run_path, _within, validate_cache
from .data import validate_manifest

ENCODER_FILES = ("vsr/data.py", "vsr/preflight.py", "vsr/alignment.py", "vsr/reproducibility.py")
_SHA = re.compile(r"[0-9a-f]{64}")
_STRICT_FLAGS = {
    "cuda_initialized": False, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "deterministic_algorithms": True, "deterministic_warn_only": False,
    "cudnn_benchmark": False, "cudnn_deterministic": True,
    "cudnn_allow_tf32": False, "matmul_allow_tf32": False,
    "matmul_allow_fp16_reduced_precision_reduction": False,
    "matmul_allow_bf16_reduced_precision_reduction": False,
    "sdp_flash": False, "sdp_memory_efficient": False, "sdp_cudnn": False,
    "sdp_math": True, "sdp_math_allow_fp16_bf16_reduction": False,
}


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _stat(path):
    s = Path(path).stat()
    if not Path(path).is_file() or s.st_size < 1:
        raise ValueError(f"Missing/empty preparation file: {path}")
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _tensor(value, shape, label):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
            or not value.is_floating_point() or value.device.type != "cpu"
            or value.requires_grad or not bool(torch.isfinite(value).all())):
        raise ValueError(f"Invalid finite frozen CPU tensor {label}; expected {tuple(shape)}")


def _selected_ids(manifest, config):
    """Mirror the sealed, metric-independent selection without importing train."""
    data, seed = config["data"], config["seed"]
    if data.get("selection") == "debug":
        training = list(manifest["debug_records"])
    elif data.get("selection") == "train":
        training = [r for r in manifest["records"] if r["split"] == "train"]
        training.sort(key=lambda r: hashlib.sha256(f"train:{seed}:{r['id']}".encode()).hexdigest())
        if data.get("train_limit"):
            training = training[:data["train_limit"]]
    else:
        raise ValueError("Unsupported originating data selection")
    validation = [r for r in manifest["records"] if r["split"] == "val"]
    validation.sort(key=lambda r: hashlib.sha256(f"val:{seed}:{r['id']}".encode()).hexdigest())
    validation = validation[:data.get("val_limit", 0)]
    return [r["id"] for r in training], [r["id"] for r in validation]


def _official_configs(project, config, audit):
    model = config["model"]
    pinned = _json(_within(project / "docs/official-model-manifest.json", project))
    if (pinned.get("id") != model.get("repo_id") or pinned.get("sha") != model.get("revision")
            or not re.fullmatch(r"[0-9a-f]{40}", str(model.get("revision", "")))
            or audit.get("status") != "passed" or audit.get("repo_id") != model["repo_id"]
            or audit.get("revision") != model["revision"] or audit.get("official_weights_only") is not True
            or audit.get("legacy_state_loaded") is not False):
        raise ValueError("Pinned official model identity/initial artifact audit mismatch")
    siblings = {x["rfilename"]: x for x in pinned["siblings"]}
    audited = {str(Path(x["path"]).resolve()): x for x in audit["files"]}
    base = Path(model["local_dir"]).resolve()
    configs, inputs = {}, {}
    for name in ("vae/config.json", "transformer/config.json", "text_encoder/config.json"):
        path = _within(base / name, base)
        payload = path.read_bytes()
        item = siblings.get(name, {})
        actual = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        row = audited.get(str(path), {})
        if actual != item.get("blobId") or row.get("algorithm") != "git_blob_sha1" or row.get("digest") != actual:
            raise ValueError(f"Official configuration checksum mismatch: {name}")
        configs[name.split("/")[0]] = json.loads(payload)
        inputs[str(path)] = hashlib.sha256(payload).hexdigest()
    vae, dit = configs["vae"], configs["transformer"]
    channels = vae.get("z_dim")
    if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1 or dit.get("in_channels") != channels:
        raise ValueError("Official VAE/DiT latent channel mismatch")
    for key in ("latents_mean", "latents_std"):
        values = vae.get(key, [])
        if (len(values) != channels or any(not isinstance(v, (float, int)) or isinstance(v, bool)
                or not math.isfinite(v) or (key == "latents_std" and v <= 0) for v in values)):
            raise ValueError("Official latent normalization parameters are invalid")
    return configs, inputs


def seal_prepared_cache(experiment_dir, *, project_root=None, allowed_root=PRODUCTION_RUNS_ROOT,
                        quiet_seconds=1.0, progress=None):
    """Fully validate and create an immutable cache_asset_seal.json.

    ``allowed_root`` and ``project_root`` overrides support new temporary CPU
    fixtures. The CLI exposes neither override. Actual raw source bytes are
    rehashed, without decoding. All required prep assets are stat-snapshotted
    before and after validation to detect concurrent writes. No GPU is used.
    """
    if not isinstance(quiet_seconds, (int, float)) or not math.isfinite(quiet_seconds) or quiet_seconds < 0:
        raise ValueError("quiet_seconds must be finite and nonnegative")
    project = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
    run = _run_path(experiment_dir, allowed_root)
    if run == Path(allowed_root).resolve() or Path(allowed_root).resolve() != project / "runs":
        raise ValueError("Only a child of this own project's runs directory may be sealed")
    status_path = _within(run / "status.json", run)
    status = _json(status_path)
    if status.get("status") != "interrupted":
        raise ValueError("Origin training status must be interrupted; running/completed/failed are refused")
    if status.get("preparation_completed") is False or status.get("preparation_status") in {"failed", "running", "incomplete"}:
        raise ValueError("Origin explicitly records failed/incomplete preparation")
    active_path = project / "runs/active_training.json"
    if active_path.exists():
        active = _json(active_path)
        if active.get("status") == "running" and Path(active.get("run", "")).resolve() == run:
            raise ValueError("Origin is still recorded as active/running")
    config = _json(_within(run / "config.json", run))
    env = _json(_within(run / "environment.json", run))
    if config.get("data", {}).get("cache_from"):
        raise ValueError("Seal the original fresh preparation, not an indirect cache reuse run")
    if config.get("reproducibility", {}).get("mode") != "strict":
        raise ValueError("Interrupted formal preparation requires strict numerical contract")
    repro = _json(_within(run / "reproducibility.json", run))
    if repro.get("mode") != "strict" or any(repro.get("flags", {}).get(k) != v for k, v in _STRICT_FLAGS.items()):
        raise ValueError("Recorded strict encoder flags are incomplete or incompatible")
    hashes = {name: file_hash(_within(project / name, project)) for name in ENCODER_FILES}
    manifest_path = _within(project / config["data"]["manifest"], project)
    manifest = _json(manifest_path)
    validate_manifest(manifest)
    verified = validate_cache(run, config, manifest["manifest_sha256"], hashes,
                              allowed_root=allowed_root, expected_manifest=manifest)
    clips = PairedClipSequence(run, allowed_root=allowed_root, max_cached=1)
    fixed = ["config.json", "environment.json", "selection.json", "correspondence.json", "reproducibility.json",
             "official_artifacts.json", "text_audit.json", "fresh_training_cache.pt", "fresh_text_embeddings.pt"]
    names = fixed + [f"clip_{i}/{name}" for i in range(len(clips))
                     for name in ("metadata.json", "paired_rgb.pt", "fresh_latents.pt", "vae_roundtrip.json")]
    paths = {name: _within(run / name, run) for name in names}
    # Freeze inputs before any heavy tensor/raw-byte validation, including all
    # metadata read above; a final hash comparison catches their early change.
    initial_hashes = {name: file_hash(paths[name]) for name in fixed if not name.endswith(".pt")}
    stats = {name: _stat(path) for name, path in paths.items()}
    status_stat, status_sha = _stat(status_path), file_hash(status_path)
    input_paths = [manifest_path, _within(project / "docs/official-model-manifest.json", project)] + [
        _within(project / name, project) for name in ENCODER_FILES]
    input_stats = {str(p): _stat(p) for p in input_paths}
    input_hashes = {str(p): file_hash(p) for p in input_paths}
    if quiet_seconds:
        time.sleep(quiet_seconds)
    if any(_stat(paths[name]) != value for name, value in stats.items()):
        raise ValueError("Preparation files are still changing; refusing to seal")
    # Revalidate after the snapshot: early metadata reads cannot conceal an
    # intervening edit that happened before the stat baseline was established.
    verified = validate_cache(run, config, manifest["manifest_sha256"], hashes,
                              allowed_root=allowed_root, expected_manifest=manifest)
    selected = _json(paths["selection.json"])
    expected_train, expected_val = _selected_ids(manifest, config)
    if selected.get("training_ids") != expected_train or selected.get("validation_ids") != expected_val:
        raise ValueError("Prepared selection differs from the configured deterministic full selection")
    official, official_inputs = _official_configs(project, config, _json(paths["official_artifacts.json"]))
    input_hashes.update(official_inputs)
    input_stats.update({name: _stat(name) for name in official_inputs})
    data = config["data"]
    geometry = latent_geometry(official["vae"], official["transformer"], data["num_frames"], data["height"], data["width"])
    if _json(paths["correspondence.json"]) != geometry:
        raise ValueError("Prepared geometry differs from derived official configurations")
    cfg_hash = config.get("_sha256", "")
    if not _SHA.fullmatch(cfg_hash):
        raise ValueError("Origin configuration SHA256 is missing")
    origin_paths = [name for name, value in env.get("code_sha256", {}).items()
                    if name.startswith("configs/") and value == cfg_hash]
    origin_config = {k: v for k, v in config.items() if k != "_sha256"}
    if not any(file_hash(_within(project / name, project)) == cfg_hash
               and _json(project / name) == origin_config for name in origin_paths):
        raise ValueError("Origin configuration no longer matches its recorded source SHA256")
    for name in origin_paths:
        path = _within(project / name, project)
        input_hashes[str(path)] = cfg_hash
        input_stats[str(path)] = _stat(path)
    frozen = torch.load(paths["fresh_training_cache.pt"], map_location="cpu", weights_only=True)
    if not isinstance(frozen, dict) or set(frozen) - {"latents", "binding", "teacher_features"}:
        raise ValueError("Frozen preparation has unexpected state keys")
    latents = frozen.get("latents")
    if not isinstance(latents, list) or len(latents) != len(clips):
        raise ValueError("Frozen latent length differs from full RGB selection")
    if frozen.get("binding") != clips[0]["binding"]:
        raise ValueError("Top-level normalized latent binding mismatch")
    shape = (1, official["vae"]["z_dim"], *geometry["latent_grid"])
    records = {r["id"]: r for r in manifest["records"]}
    captions, raw_stats, raw_hashes = set(), {}, {}
    for i, pair in enumerate(latents):
        clip = clips[i]  # Existing lazy reader validates every RGB value/shape/range.
        metadata = clip["metadata"]
        record = records[metadata["record_id"]]
        if metadata.get("caption") != record.get("caption") or metadata.get("caption_status") != record.get("caption_status"):
            raise ValueError("Cached caption identity/status differs from sealed manifest")
        captions.add(metadata["caption"])
        raw_path = Path(record["path"]).resolve()
        if raw_path != Path(metadata.get("source_path", "")).resolve() or not raw_path.is_relative_to(Path(manifest["raw_root"]).resolve()):
            raise ValueError("Cached raw source path differs from sealed manifest/root")
        if str(raw_path) not in raw_hashes:
            raw_stats[str(raw_path)] = _stat(raw_path)
            raw_hashes[str(raw_path)] = file_hash(raw_path)
        if raw_hashes[str(raw_path)] != record["sha256"]:
            raise ValueError("Raw source bytes differ from sealed manifest SHA256")
        per_clip = torch.load(paths[f"clip_{i}/fresh_latents.pt"], map_location="cpu", weights_only=True)
        if not isinstance(pair, dict) or set(pair) != {"clean", "low"} or not isinstance(per_clip, dict) or set(per_clip) != {"clean", "low", "binding"}:
            raise ValueError("Normalized latent pair contains unexpected/missing keys")
        if per_clip["binding"] != clip["binding"]:
            raise ValueError("Per-clip normalized latent binding mismatch")
        for key in ("clean", "low"):
            _tensor(pair[key], shape, f"top latent {i}/{key}")
            _tensor(per_clip[key], shape, f"per-clip latent {i}/{key}")
            if pair[key].dtype != torch.float32:
                raise ValueError("Normalized latents must preserve the official FP32 VAE encoder dtype")
            if pair[key].dtype != per_clip[key].dtype or not torch.equal(pair[key], per_clip[key]):
                raise ValueError("Top/per-clip normalized latents are not torch.equal with identical dtype")
        rt = _json(paths[f"clip_{i}/vae_roundtrip.json"])
        if (rt.get("posterior") != "mode" or rt.get("vae_frozen") is not True or rt.get("vae_has_grad") is not False
                or rt.get("shape") != list(clip["hr"].shape)):
            raise ValueError("Frozen normalized VAE encoding/roundtrip evidence incomplete")
        if progress and ((i + 1) % 25 == 0 or i + 1 == len(clips)):
            progress(f"Verified RGB/raw/normalized latents {i + 1}/{len(clips)}")
    embeddings = torch.load(paths["fresh_text_embeddings.pt"], map_location="cpu", weights_only=True)
    text_audit = _json(paths["text_audit.json"])
    if (not isinstance(embeddings, dict) or set(embeddings) != captions
            or set(text_audit.get("prompts", [])) != captions or len(text_audit.get("prompts", [])) != len(captions)
            or text_audit.get("encoder_frozen") is not True or text_audit.get("encoder_has_grad") is not False
            or text_audit.get("text_max_length") != 512):
        raise ValueError("Frozen text embeddings must exactly cover all selected captions")
    text_dim = official["text_encoder"].get("d_model")
    if not isinstance(text_dim, int) or text_dim < 1:
        raise ValueError("Official text hidden size is missing")
    for caption, value in embeddings.items():
        _tensor(value, (1, 512, text_dim), f"text embedding {caption!r}")
    artifacts = {name: file_hash(path) for name, path in paths.items()}
    if any(artifacts[name] != sha for name, sha in initial_hashes.items()):
        raise ValueError("Preparation metadata changed during validation")
    if (any(_stat(paths[name]) != value for name, value in stats.items())
            or any(_stat(name) != value for name, value in input_stats.items())
            or any(file_hash(name) != sha for name, sha in input_hashes.items())
            or any(_stat(name) != value for name, value in raw_stats.items())
            or _stat(status_path) != status_stat or file_hash(status_path) != status_sha
            or _json(status_path) != status
            or _json(paths["config.json"]) != config or _json(paths["environment.json"]) != env
            or _json(paths["reproducibility.json"]) != repro or _json(manifest_path) != manifest
            or any(file_hash(project / name) != sha for name, sha in hashes.items())):
        raise ValueError("Preparation/input/status changed during validation; no seal written")
    if active_path.exists():
        active = _json(active_path)
        if active.get("status") == "running" and Path(active.get("run", "")).resolve() == run:
            raise ValueError("Origin became active/running during validation; no seal written")
    seal = {"schema_version": 1, "format": "vsr_prepared_cache_seal_v1", "source_run": str(run),
            "artifact_sha256": artifacts, "contains_task_initialization": False,
            "purpose": "only frozen RGB, normalized VAE targets and text encodings; auxiliary teachers regenerated",
            "origin_training_status": "interrupted", "origin_status_sha256": status_sha,
            "preparation_completed": True, "entire_experiment_completed": False,
            "task_checkpoint_read": False, "optimizer_state_read": False,
            "validation": {**verified, "all_rgb_payloads_checked": True, "normalized_latents_checked": True,
                           "top_perclip_latents_torch_equal": True, "text_captions_exact": True,
                           "raw_source_bytes_rehashed": True, "raw_source_count": len(raw_hashes),
                           "strict_encoder_flags_checked": True, "official_geometry": geometry},
            "verification_input_sha256": input_hashes,
            "raw_source_sha256": raw_hashes}
    target = _within(run / "cache_asset_seal.json", run)
    # Exclusive creation prevents overwriting a seal created concurrently.
    serialized = json.dumps(seal, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=run, prefix=".cache_asset_seal-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        if _stat(status_path) != status_stat or _json(status_path) != status:
            raise ValueError("Origin status changed immediately before sealing")
        if active_path.exists():
            active = _json(active_path)
            if active.get("status") == "running" and Path(active.get("run", "")).resolve() == run:
                raise ValueError("Origin became active/running immediately before sealing")
        try:
            os.link(temporary, target)
        except FileExistsError:
            if _json(target) != seal:
                raise ValueError("Existing frozen cache seal differs; refusing overwrite") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return seal
