"""Shared frozen-protocol paired/unpaired evaluation; no model selection.

This entry point performs no optimizer updates or hyperparameter search. It
measures declared native prefixes or frozen prepared crops. Metrics use the exact
uint8 PNG pixels saved here, never the optional lossy playback previews.
"""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
import gc
import hashlib
from importlib import metadata as package_metadata
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps"))

from PIL import Image
import torch
import torch.nn.functional as F

from .metric_pools import (PAIRED_METRIC_KEYS as PAIR_KEYS, POOL_NAMES as POOLS,
                          paired_metrics)
from .perceptual_metrics import ALEXNET_SHA256, LINEAR_SHA256, TECOGAN_REVISION


ALLOWED_DATASETS = ("REDS4", "YouHQ40", "VideoLQ")


@dataclass(frozen=True)
class BenchmarkPolicy:
    """Reader identity and reporting rules, sealed before any pixel load."""
    family: str = "external"
    allowed_datasets: tuple = ALLOWED_DATASETS
    run_prefix: str = "external"
    manifest_relative: str = "manifests/external.json"
    manifest_hash_field: str = "external_manifest_sha256"
    reader_options: dict | None = None
    freeze_reader_catalogue: bool = False
    preserve_reader_order: bool = False
    no_reference_primary_pool: str = "matched_nonreference"
    reference_primary_pool: str = "matched_nonreference"
    selection_policy: str = "explicit caller checkpoint list fixed before external pixels; selection must use internal validation only"
    summary_scope: str = "explicit fixed native-resolution prefixes; not whole source videos"
    primary_policy_description: str = "all original prefix frames except reference index0, same for every model/on/off"


def _hash_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _tensor_sha256(value):
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def stable_hq_input_contract(metadata, lr, hr, *, fps, frames):
    """Bind fixed input bytes/window, excluding runtime cost and tmp paths.

    Complete provenance is retained separately in result.source. The original
    pre-quantization LR matters to model conditioning even when two nearby
    float values happen to round to the same PNG pixel.
    """
    binding = metadata.get("input_binding")
    keys = ("manifest_sha256", "base_pair_cache_key", "record_id", "robustness_mode", "noise_seed",
            "robustness_protocol_sha256", "base_lr_rgb_sha256", "robust_lr_rgb_sha256")
    if not isinstance(binding, dict) or any(binding.get(key) is None for key in keys):
        raise ValueError("HQ input needs its complete stable cache/robustness binding")
    for key in ("base_lr_rgb_sha256", "robust_lr_rgb_sha256", "robustness_protocol_sha256"):
        if not isinstance(binding[key], str) or not re.fullmatch(r"[0-9a-f]{64}", binding[key]):
            raise ValueError("HQ pixel/protocol bindings require full SHA256 digests")
    raw_lr = lr.detach().float().cpu().contiguous()
    raw_hr = hr.detach().float().cpu().contiguous()
    if _tensor_sha256(raw_lr) != binding["robust_lr_rgb_sha256"]:
        raise ValueError("HQ robustness LR float bytes differ from input_binding")
    qlr, _ = quantize_rgb(raw_lr)
    qhr, _ = quantize_rgb(raw_hr)
    return {"input_binding": {key: binding[key] for key in keys},
        "source": {key: metadata.get(key) for key in ("dataset", "source_dataset", "record_id", "sequence_id", "source_id", "source_sha256", "split")},
        "window_geometry": {key: metadata.get(key) for key in ("source_frame_indices", "source_indices", "crop_tlhw", "resize_hw",
            "temporal_stride", "original_fps", "output_fps", "fps_source", "prepared_lr_hw", "prepared_hr_hw", "valid_region")},
        "fps": float(fps), "frames": frames, "lr_shape": list(raw_lr.shape), "hr_shape": list(raw_hr.shape),
        "actual_lr_rgb_f32_sha256": _tensor_sha256(raw_lr), "actual_hr_rgb_f32_sha256": _tensor_sha256(raw_hr),
        "measured_lr_uint8_sha256": _tensor_sha256(qlr), "measured_hr_uint8_sha256": _tensor_sha256(qhr)}


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"External evaluation path escapes its selected root: {path}")
    return path


def _integer(value, label, minimum=1):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be integer >= {minimum}")


def quantize_rgb(video):
    """Return uint8 RGB and q/255; both are CPU [1,3,T,H,W]."""
    if not isinstance(video, torch.Tensor) or video.ndim != 5 or tuple(video.shape[:2]) != (1, 3) or not all(video.shape):
        raise ValueError("Expected nonempty RGB [1,3,T,H,W]")
    if not video.is_floating_point() or not bool(torch.isfinite(video).all()):
        raise ValueError("Quantization requires finite floating RGB")
    value = video.detach().float().cpu()
    if float(value.min()) < -1e-6 or float(value.max()) > 1 + 1e-6:
        raise ValueError("Metric/source RGB must be in [0,1]")
    quantized = value.clamp(0, 1).mul(255).round().to(torch.uint8).contiguous()
    return quantized, quantized.float().div(255)


def _artifact(path, run):
    path = _inside(path, run)
    return {"path": path.relative_to(run).as_posix(), "bytes": path.stat().st_size, "sha256": _hash_file(path)}


def _save_pngs(quantized, directory, run, *, overwrite=False):
    if quantized.dtype != torch.uint8:
        raise ValueError("Save exactly the quantized uint8 metric pixels")
    directory = _inside(directory, run)
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for index, frame in enumerate(quantized[0].permute(1, 2, 3, 0)):
        path = directory / f"frame_{index:08d}.png"
        stream = io.BytesIO()
        Image.fromarray(frame.contiguous().numpy()).save(stream, format="PNG")
        payload = stream.getvalue()
        if path.exists() and not overwrite and _hash_file(path) != hashlib.sha256(payload).hexdigest():
            raise ValueError("Frozen source PNG pixels changed within this evaluation run")
        temporary = path.with_name(path.name + ".partial")
        temporary.write_bytes(payload)
        temporary.replace(path)
        artifacts.append(_artifact(path, run))
    return artifacts


def _verify_artifacts(artifacts, run):
    for item in artifacts:
        path = _inside(Path(run) / item["path"], run)
        if not path.is_file() or path.stat().st_size != item["bytes"] or _hash_file(path) != item["sha256"]:
            raise ValueError(f"Completed evaluation PNG/preview artifact changed: {path}")


def _average_scores(scores):
    if not scores or len({score["kind"] for score in scores}) != 1:
        raise ValueError("Cannot average empty/mixed metric kinds")
    if scores[0]["kind"] == "unpaired":
        value = sum(score["musiq"]["mean"] for score in scores)/len(scores)
        return {"kind": "unpaired", "musiq": {"mean": value}, "musiq_koniq10k_pyiqa": value}
    result = {"kind": "paired"}
    for pool in POOLS:
        result[pool] = {}
        for key in PAIR_KEYS:
            values = [score[pool][key] for score in scores if score[pool].get(key) is not None]
            result[pool][key] = sum(values)/len(values) if values else None
        result[pool]["valid_sample_counts"] = {key: sum(score[pool].get(key) is not None for score in scores) for key in PAIR_KEYS}
    return result


def _macro_dataset(rows):
    modes = sorted(rows[0]["modes"])
    return {"video_count": len(rows), "baseline": _average_scores([row["baseline"] for row in rows]),
        "modes": {mode: _average_scores([row["modes"][mode]["mean"] for row in rows]) for mode in modes},
        "aggregation": "all seeds arithmetic mean within each video, then equal video weights; no best seed"}


def _default_reader(root, **kwargs):
    from .external_data import ExternalReader
    return ExternalReader(root, **kwargs)


def _default_preview(path, **kwargs):
    from .infer_video import preview_training_config
    return preview_training_config(path, **kwargs)


def _default_model(path, **kwargs):
    from .checkpoints import load_training_checkpoint
    return load_training_checkpoint(path, **kwargs)


def _default_vae(config, **kwargs):
    from .infer_video import load_frozen_vae
    return load_frozen_vae(config, **kwargs)


def _default_caption(config, caption, checkpoint, root, *, device):
    """Reuse only new VSR-run text caches from the same pinned official base."""
    candidates = [Path(checkpoint).parent.parent / "fresh_text_embeddings.pt"]
    cached_run = config.get("data", {}).get("cache_from")
    if cached_run:
        candidates.append(Path(cached_run) / "fresh_text_embeddings.pt")
    for path in candidates:
        path = path.resolve()
        if not path.is_relative_to((Path(root)/"runs").resolve()) or not path.is_file():
            continue
        origin = path.parent / "config.json"
        if not origin.is_file() or json.loads(origin.read_text()).get("model") != config.get("model"):
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        value = payload.get(caption) if isinstance(payload, dict) else None
        if (isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[0] == 1
                and value.is_floating_point() and bool(torch.isfinite(value).all())):
            return value.detach().cpu(), {"caption": caption, "source": "new_VSR_frozen_text_cache",
                "path": str(path), "sha256": _hash_file(path), "model": config["model"]}
    from .reproducibility import configure_reproducibility
    from .infer_video import encode_caption
    configure_reproducibility(config)
    return encode_caption(config, caption, device=device)


def _default_perceptual(root, *, device):
    from .perceptual_metrics import PerceptualMetrics
    return PerceptualMetrics(Path(root)/"weights/alexnet-owt-7be5be79.pth",
                             Path(root)/"weights/lpips_v0.1_alex.pth", device=device)


def _default_musiq(root, *, device):
    from .no_reference import MUSIQEvaluator, KONIQ_CHECKPOINT
    return MUSIQEvaluator(Path(root)/"weights"/KONIQ_CHECKPOINT, device=device, project_root=root)


def _default_restore(*args, **kwargs):
    from .inference import restore_video
    return restore_video(*args, **kwargs)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _preview_video(video, fps, path, max_side, run):
    from .data import write_video
    height, width = video.shape[-2:]
    scale = min(1.0, max_side/max(height, width))
    new_h, new_w = max(2, int(height*scale)//2*2), max(2, int(width*scale)//2*2)
    frames = video[0].permute(1, 0, 2, 3)
    resized = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
    preview = resized.permute(1, 0, 2, 3).unsqueeze(0)
    write_video(path, preview, fps)
    return {"artifact": _artifact(path, run), "preview_hw": [new_h, new_w],
            "source_hw": [height, width], "downscaled_for_playback_only": (new_h, new_w) != (height, width),
            "playback_only": True, "requested_max_side": max_side, "used_for_metrics": False, "fps": fps}


def _resume_result(directory, run, contract):
    marker = directory/"complete.json"
    metrics = directory/"metrics.json"
    if not marker.exists():
        return None
    seal = json.loads(marker.read_text())
    if seal.get("contract_sha256") != contract or not metrics.is_file() or seal.get("metrics_sha256") != _hash_file(metrics):
        raise ValueError("Completed sequence metric seal / protocol differs; use a new evaluation run")
    row = json.loads(metrics.read_text())
    if row.get("contract_sha256") != contract or row.get("status") != "completed":
        raise ValueError("Completed sequence metric contract differs")
    _verify_artifacts(row["artifacts"], run)
    return row


@torch.no_grad()
def run_frozen_evaluation(project_root, checkpoints, *, datasets=ALLOWED_DATASETS, frames=17,
                            seeds=(42, 43), steps=20, tile_hw=(256, 448), overlap=64,
                            device="cuda", metric_device="cpu", run_dir=None, preview_fps=30.0,
                            caption="A high quality video.", preview_max_side=None,
                            selection_note=None, factories=None, policy=None, reader_factory=None):
    """Freeze protocol, then evaluate; factory overrides support CPU-only tests.

    Overrides are reader/preview/caption/model/vae/restore/perceptual/musiq.
    Model loaders must return (fresh-official-base-plus-task-state, audit).
    Passing an existing run_dir resumes only matching completed, sealed rows.
    """
    root = Path(project_root).resolve()
    policy = BenchmarkPolicy() if policy is None else policy
    if policy.family not in ("external", "hq"):
        raise ValueError("Unknown frozen evaluation family")
    if any(pool not in ("all_frames", "matched_nonreference")
           for pool in (policy.no_reference_primary_pool, policy.reference_primary_pool)):
        raise ValueError("Primary report must use all frames or matched non-reference frames")
    device, metric_device = torch.device(device), torch.device(metric_device)
    _integer(frames, "frames")
    _integer(steps, "steps")
    _integer(overlap, "overlap", 0)
    datasets, seeds = tuple(datasets), tuple(seeds)
    if not datasets or len(set(datasets)) != len(datasets) or set(datasets)-set(policy.allowed_datasets):
        raise ValueError("Choose distinct datasets from the frozen reader policy")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use every distinct predeclared seed")
    for seed in seeds:
        _integer(seed, "seed", 0)
    if any(name != "VideoLQ" for name in datasets) and frames < 6:
        raise ValueError("Paired TecoGAN protocol requires at least six original frames")
    if len(tile_hw) != 2 or any(not isinstance(value, int) or value <= 0 or value % 16 for value in tile_hw) or overlap % 16 or overlap >= min(tile_hw):
        raise ValueError("Native HR tile/overlap geometry must be 16-aligned")
    if preview_max_side is not None:
        _integer(preview_max_side, "preview_max_side", 2)
    if not isinstance(checkpoints, dict) or not checkpoints:
        raise ValueError("Provide explicit label=checkpoint mappings selected before external evaluation")
    defaults = {"reader": reader_factory or _default_reader, "preview": _default_preview, "caption": _default_caption,
        "model": _default_model, "vae": _default_vae, "restore": _default_restore,
        "perceptual": _default_perceptual, "musiq": _default_musiq}
    supplied = factories or {}
    if set(supplied)-set(defaults):
        raise ValueError("Unknown external evaluation factory override")
    defaults.update(supplied)
    f = defaults
    run_root = root/"runs"
    run = _inside(run_dir if run_dir is not None else run_root/(policy.run_prefix+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")), run_root)
    if run.exists() and any(run.iterdir()) and not (run/"protocol.json").is_file():
        raise ValueError("Refusing to write external evaluation into a nonempty non-external run")
    run.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    specs = {}
    for label, path in checkpoints.items():
        if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", label):
            raise ValueError("Checkpoint labels must be simple directory-safe identifiers")
        path = _inside(path, run_root)
        actual_hash = _hash_file(path)
        preview = f["preview"](path, allowed_root=run_root)
        if preview["checkpoint_sha256"] != actual_hash:
            raise ValueError("Checkpoint changed during protocol preview")
        specs[label] = {**preview, "path": str(path)}
    reader = f["reader"](root, **(policy.reader_options if policy.reader_options is not None else {"preview_fps": preview_fps}))
    perceptual = musiq = None
    protocol_accepted = False
    try:
        jobs = {name: list(reader.records[name]) if policy.preserve_reader_order else sorted(reader.records[name]) for name in datasets}
        if any(not ids for ids in jobs.values()) or any(Path(value).name != value or value in (".", "..") or "\\" in value for ids in jobs.values() for value in ids):
            raise ValueError("Frozen reader IDs must be nonempty, path-safe original identifiers")
        versions = {}
        for name in ("torch", "torchvision", "diffusers", "transformers", "numpy", "Pillow", "opencv-python-headless", "lpips"):
            try:
                versions[name] = package_metadata.version(name)
            except package_metadata.PackageNotFoundError:
                versions[name] = None
        protocol = {"schema_version": 1, "checkpoints": specs, "datasets": jobs, "frames": frames,
            "start": 0, "seeds": list(seeds), "steps": steps, "tile_hw": list(tile_hw), "overlap": overlap,
            "device": str(device), "metric_device": str(metric_device), "caption": caption,
            "red_preview_fps": preview_fps, "preview_max_side": preview_max_side,
            policy.manifest_hash_field: _hash_file(root/policy.manifest_relative),
            "checkpoint_selection": {"caller_note": selection_note, "runner_selects_checkpoint": False,
                "policy": policy.selection_policy},
            "quantization": "round(clamp(rgb,0,1)*255) to uint8; every metric uses saved PNG-equivalent q/255",
            "reference": "paired: reference-trained on/off with GT frame0 clone only; Base all+same nonref pool; VideoLQ HRIoff only",
            "primary_paired_pool": policy.primary_policy_description,
            "temporal": "official TecoGAN full sequence cutfr2/crop first, then exclude pairs touching reference; no frame joining",
            "metric_versions": {"lpips_package": "0.1.4", "lpips_version": "0.1", "lpips_alexnet_sha256": ALEXNET_SHA256,
                "lpips_linear_sha256": LINEAR_SHA256, "tecogan_revision": TECOGAN_REVISION,
                "MUSIQ": "local pinned KonIQ10k IQA-PyTorch reimplementation; provenance returned by evaluator"},
            "code_sha256": {path.name: _hash_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
            "runtime_versions": versions,
            "factory_overrides": sorted(supplied), "training_updates": 0, "external_tuning": False}
        frozen_catalogue = None
        if policy.freeze_reader_catalogue:
            catalogue = getattr(reader, "metadata", None)
            if not isinstance(catalogue, dict) or catalogue.get("training_allowed") is not False:
                raise ValueError("Internal reader requires a metadata-only frozen catalogue")
            if any(ids != catalogue.get("selected_record_ids") for ids in jobs.values()):
                raise ValueError("Reader IDs/order differ from the predeclared HQ selection")
            if catalogue.get("manifest_file_sha256") is not None and catalogue["manifest_file_sha256"] != protocol[policy.manifest_hash_field]:
                raise ValueError("HQ reader manifest differs from frozen file digest")
            frozen_catalogue = json.loads(json.dumps(catalogue, ensure_ascii=False, allow_nan=False))
            protocol.update(evaluation_family="HQ-VSR_internal_holdout", reader_configuration=frozen_catalogue,
                hq_configuration=policy.reader_options, manifest_snapshot="hq_manifest.snapshot.json",
                reader_configuration_snapshot="hq_reader_configuration.json",
                test_used_for_checkpoint_selection=False,
                quantization_comparison="saved PNG uint8 q/255; separate measurement protocol from online training float32 evaluation")
            protocol.pop("red_preview_fps", None)
            protocol.pop("external_tuning", None)
            protocol["reference"] = "reference-trained paired on/off with supplied GT frame0 only; no-reference-trained main all_frames plus matched_nonreference for cross-model comparisons"
        protocol = json.loads(json.dumps(protocol, ensure_ascii=False, allow_nan=False))
        contract = _digest(protocol)
        protocol_path = run/"protocol.json"
        if protocol_path.exists():
            if json.loads(protocol_path.read_text()) != protocol:
                raise ValueError("Existing external run protocol differs; never retune an existing external run")
        else:
            _json(protocol_path, protocol)
        if frozen_catalogue is not None:
            manifest_payload = (root/policy.manifest_relative).read_bytes()
            if hashlib.sha256(manifest_payload).hexdigest() != protocol[policy.manifest_hash_field]:
                raise ValueError("HQ manifest changed while freezing the evaluation catalogue")
            snapshots = {
                "hq_manifest.snapshot.json": manifest_payload,
                "hq_reader_configuration.json": (json.dumps(frozen_catalogue, indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode(),
            }
            for name, payload in snapshots.items():
                destination = run/name
                if destination.exists() and destination.read_bytes() != payload:
                    raise ValueError("Frozen HQ catalogue/configuration snapshot changed")
                if not destination.exists():
                    temporary = destination.with_name(destination.name+".partial")
                    temporary.write_bytes(payload)
                    temporary.replace(destination)
        protocol_accepted = True
        _json(run/"status.json", {"status": "running", "protocol_sha256": contract, "training_updates": 0})
        results, skipped = {}, []
        def score(prediction, truth=None, *, primary_pool=None):
            nonlocal perceptual, musiq
            if truth is not None:
                if perceptual is None:
                    perceptual = f["perceptual"](root, device=metric_device)
                value = paired_metrics(prediction, truth, perceptual, excluded_reference=0)
                value["primary_pool"] = primary_pool or policy.no_reference_primary_pool
                return value
            if musiq is None:
                musiq = f["musiq"](root, device=metric_device)
            return {"kind": "unpaired", "musiq": musiq.evaluate_video(prediction, batch_size=1)}
        for label, spec in specs.items():
            rows, pending = {name: [] for name in datasets}, []
            for dataset, ids in jobs.items():
                for sequence in ids:
                    directory = run/"results"/label/dataset/sequence
                    row_contract = _digest({"protocol": contract, "label": label, "dataset": dataset, "id": sequence})
                    existing = _resume_result(directory, run, row_contract)
                    if existing is not None:
                        rows[dataset].append(existing)
                        skipped.append(f"{label}/{dataset}/{sequence}")
                    else:
                        pending.append((dataset, sequence, directory, row_contract))
            model = vae = None
            if pending:
                caption_started = time.perf_counter()
                text, caption_audit = f["caption"](spec["config"], caption, Path(spec["path"]), root, device=device)
                caption_seconds = time.perf_counter()-caption_started
                if not isinstance(text, torch.Tensor) or text.ndim != 3 or text.shape[0] != 1 or not bool(torch.isfinite(text).all()):
                    raise ValueError("Native caption encoder/cache returned invalid embedding")
                _json(run/"checkpoints"/label/"caption.json", {"seconds": caption_seconds, "audit": caption_audit})
                try:
                    load_started = time.perf_counter()
                    model, checkpoint_audit = f["model"](spec["path"], device=device, allowed_root=run_root)
                    if checkpoint_audit["checkpoint_sha256"] != spec["checkpoint_sha256"] or checkpoint_audit["config"] != spec["config"]:
                        raise ValueError("Checkpoint/config changed between preview and fresh task restore")
                    vae = f["vae"](spec["config"], device=device)
                    model.eval()
                    vae.requires_grad_(False).eval()
                    _sync(device)
                    _json(run/"checkpoints"/label/"initialization.json", {"audit": checkpoint_audit,
                        "model_and_vae_load_seconds": time.perf_counter()-load_started, "optimizer_updates": 0})
                    reference_trained = bool(spec["config"].get("reference", {}).get("enabled", False))
                    for dataset, sequence, directory, row_contract in pending:
                        source = reader.load(dataset, sequence, num_frames=frames, start=0)
                        lr, source_meta = source["lr"], source["metadata"]
                        hr = source.get("hr")
                        paired = dataset != "VideoLQ"
                        if paired != (hr is not None):
                            raise ValueError("External paired/unpaired source contract changed")
                        if lr.shape[2] != frames or (hr is not None and hr.shape[2] != frames):
                            raise ValueError("External reader did not return the entire fixed prefix")
                        if hr is not None and tuple(hr.shape[-2:]) != tuple(size*4 for size in lr.shape[-2:]):
                            raise ValueError("External GT differs from native x4 target geometry")
                        if policy.family == "hq":
                            options = policy.reader_options
                            frame_indices = source_meta.get("source_frame_indices", [])
                            expected_mode = dataset.rsplit("-", 1)[1]
                            if (source_meta.get("dataset") != dataset or source_meta.get("source_dataset") != "HQ-VSR"
                                    or source_meta.get("record_id") != sequence or source_meta.get("split") != options["split"]
                                    or source_meta.get("training_allowed") is not False or source_meta.get("parameters_selected_on_test") is not False):
                                raise ValueError("Prepared HQ source identity/split/test-use contract differs")
                            if tuple(hr.shape[-2:]) != (options["height"], options["width"]):
                                raise ValueError("Prepared HQ crop differs from frozen spatial configuration")
                            if (len(frame_indices) != frames or any(right-left != options["temporal_stride"]
                                    for left, right in zip(frame_indices, frame_indices[1:]))):
                                raise ValueError("Prepared HQ window differs from frozen temporal stride")
                            binding = source_meta.get("input_binding", {})
                            if (binding.get("record_id") != sequence or binding.get("robustness_mode") != expected_mode
                                    or binding.get("noise_seed") != frozen_catalogue["noise_seed_by_record_id"][sequence]):
                                raise ValueError("HQ prepared robustness differs from pre-pixel mode/noise binding")
                        source_dir = run/"inputs"/dataset/sequence
                        qlr, measured_lr = quantize_rgb(lr)
                        artifacts = _save_pngs(qlr, source_dir/"lr", run)
                        qhr = measured_hr = None
                        if paired:
                            qhr, measured_hr = quantize_rgb(hr)
                            artifacts.extend(_save_pngs(qhr, source_dir/"hr", run))
                        up = F.interpolate(lr[0].permute(1, 0, 2, 3).float().cpu(),
                            size=(lr.shape[-2]*4, lr.shape[-1]*4), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
                        qbase, baseline = quantize_rgb(up.permute(1, 0, 2, 3).unsqueeze(0))
                        artifacts.extend(_save_pngs(qbase, source_dir/"bicubic", run))
                        source_contract = (stable_hq_input_contract(source_meta, lr, hr, fps=source["fps"], frames=frames)
                                           if policy.family == "hq" else source_meta)
                        baseline_contract = _digest({"protocol": contract, "source": source_contract})
                        baseline_file, baseline_seal = source_dir/"baseline_metrics.json", source_dir/"baseline.complete.json"
                        if baseline_seal.exists():
                            seal = json.loads(baseline_seal.read_text())
                            if seal.get("contract_sha256") != baseline_contract or seal.get("metrics_sha256") != _hash_file(baseline_file):
                                raise ValueError("Frozen baseline metric contract changed")
                            baseline_scores = json.loads(baseline_file.read_text())
                        else:
                            baseline_scores = score(baseline, measured_hr)
                            _json(baseline_file, baseline_scores)
                            _json(baseline_seal, {"contract_sha256": baseline_contract, "metrics_sha256": _hash_file(baseline_file)})
                        modes = ("hri_on", "hri_off") if reference_trained and paired else (("hri_off",) if reference_trained else ("no_reference",))
                        row = {"status": "completed", "contract_sha256": row_contract, "checkpoint_sha256": spec["checkpoint_sha256"],
                            "label": label, "dataset": dataset, "sequence_id": sequence, "source": source_meta,
                            "source_fps": source["fps"], ("prefix_frames" if policy.family == "external" else "evaluated_frames"): frames,
                            "training_setting": "reference_trained_with_dropout" if reference_trained else "no_reference_trained",
                            "provided_hri_rule": "supplied GT first frame clone only" if reference_trained and paired else "no HRI supplied",
                            "baseline": baseline_scores, "modes": {}, "artifacts": artifacts,
                            "metric_pixels": "saved uint8 PNG /255, native dimensions; never MP4 pixels"}
                        if policy.family == "hq":
                            row["source_id"] = source_meta.get("source_id")
                            row["stable_input_contract"] = source_contract
                            row["stable_input_contract_sha256"] = _digest(source_contract)
                        for mode in modes:
                            entries = []
                            for seed in seeds:
                                if device.type == "cuda":
                                    torch.cuda.reset_peak_memory_stats(device)
                                supplied_reference = hr[:, :, :1].clone() if mode == "hri_on" else None
                                _sync(device)
                                started = time.perf_counter()
                                video, restoration_meta = f["restore"](model, vae, lr, text, spec["config"],
                                    steps=steps, seed=seed, tile_hw=tile_hw, overlap=overlap,
                                    reference_image=supplied_reference, reference_index=0 if mode == "hri_on" else None)
                                _sync(device)
                                restore_seconds = time.perf_counter()-started
                                restore_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                                reported_noise_seed = restoration_meta.get("sampling", {}).get("seed")
                                if policy.family == "hq" and reported_noise_seed is not None and reported_noise_seed != seed:
                                    raise ValueError("HQ sampler reported a different noise seed than the frozen request")
                                expected_shape = (1, 3, frames, lr.shape[-2]*4, lr.shape[-1]*4)
                                if tuple(video.shape) != expected_shape:
                                    raise ValueError("Native external inference changed frame count/target geometry")
                                quantized, measured = quantize_rgb(video)
                                output_dir = directory/mode/f"seed_{seed}"
                                save_started = time.perf_counter()
                                png_artifacts = _save_pngs(quantized, output_dir/"png", run, overwrite=True)
                                png_seconds = time.perf_counter()-save_started
                                score_started = time.perf_counter()
                                metrics = score(measured, measured_hr, primary_pool=policy.reference_primary_pool if reference_trained else policy.no_reference_primary_pool)
                                _sync(metric_device)
                                metric_seconds = time.perf_counter()-score_started
                                preview = None
                                if preview_max_side is not None:
                                    preview = _preview_video(measured, source["fps"], output_dir/f"preview_max{preview_max_side}.mp4", preview_max_side, run)
                                    png_artifacts.append(preview["artifact"])
                                row["artifacts"].extend(png_artifacts)
                                entries.append({"seed": seed, "hri_supplied": mode == "hri_on", "reference_index": 0 if mode == "hri_on" else None,
                                    "metrics": metrics, "png_directory": str((output_dir/"png").relative_to(run)),
                                    "png_count": frames, "preview": preview, "restoration": restoration_meta,
                                    "seconds": {"restore": restore_seconds, "png_save": png_seconds, "metrics": metric_seconds},
                                    "restore_cuda_peak_allocated_bytes": restore_peak,
                                    "restore_peak_scope": "restore phase after peak reset; includes any already-resident model/VAE/metric weights",
                                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
                                if policy.family == "hq":
                                    entries[-1].update(noise_seed=reported_noise_seed, requested_noise_seed=seed,
                                        noise_seed_verified_by_sampler_metadata=reported_noise_seed is not None)
                                del video, measured, quantized, supplied_reference
                            row["modes"][mode] = {"seeds": entries, "mean": _average_scores([entry["metrics"] for entry in entries])}
                        _json(directory/"metrics.json", row)
                        _json(directory/"complete.json", {"contract_sha256": row_contract, "metrics_sha256": _hash_file(directory/"metrics.json")})
                        rows[dataset].append(row)
                        _json(run/"progress.json", {"label": label, "dataset": dataset, "sequence_id": sequence, "status": "completed"})
                        del source, lr, hr, qlr, measured_lr, qhr, measured_hr, up, qbase, baseline
                finally:
                    del model, vae, text
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            results[label] = {name: _macro_dataset(sorted(value, key=lambda row: row["sequence_id"])) for name, value in rows.items()}
        summary = {"status": "completed", "run": str(run), "protocol_sha256": contract,
            "results": results, "resumed_complete_sequences": skipped,
            "training_updates": 0, "external_tuning": False, "best_seed_selected": False,
            "quality_hypotheses_verified": False, "scope": policy.summary_scope,
            "VideoLQ_interpretation": "MUSIQ image quality complement only; playback blind assessment/content/time fidelity remain separate"}
        if policy.family == "hq":
            summary.pop("VideoLQ_interpretation", None)
            summary.pop("external_tuning", None)
            summary.update(evaluation_family="HQ-VSR_internal_holdout", test_used_for_checkpoint_selection=False,
                reader_configuration=frozen_catalogue,
                primary_report={label: {dataset: {mode: {
                    "pool": policy.no_reference_primary_pool if mode == "no_reference" else policy.reference_primary_pool,
                    "metrics": values[policy.no_reference_primary_pool if mode == "no_reference" else policy.reference_primary_pool]}
                    for mode, values in item["modes"].items()} for dataset, item in groups.items()}
                    for label, groups in results.items()})
        _json(run/"summary.json", summary)
        _json(run/"status.json", {key: summary[key] for key in ("status", "protocol_sha256", "training_updates", "external_tuning", "test_used_for_checkpoint_selection") if key in summary})
        return summary
    except Exception as error:
        if protocol_accepted:
            _json(run/"status.json", {"status": "failed", "error": str(error), "traceback": traceback.format_exc(), "training_updates": 0})
        raise
    finally:
        reader.close()

