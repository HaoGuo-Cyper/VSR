"""Prepare or execute a frozen, VAE-only reconstruction/condition-gap audit.

Preparation uses file/header metadata and hashes, without importing torch or
opening pixels. Execution is a fresh dated attempt; existing runs are never
resumed. The initial implementation covers 128x224 crops only. E*warp, a new
metric named tLPIPS, and full-resolution encode-tiling comparisons stay pending.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

CODE_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "vsr_vae_reconstruction_audit_v1"
SELECTION_PREFIX = "vae-audit-20261105:"
START_PREFIX = "vae-audit-common-window-20261105:"
ALGORITHM = "sha256_sorted_val_records_first_eight_distinct_source_ids"
CLIP_LENGTHS = (9, 17, 33)
STRIDES = (1, 4)
HEIGHT, WIDTH = 128, 224
MAX_SPAN = 129
LPIPS_HASHES = {
    "alexnet-owt-7be5be79.pth": "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
    "lpips_v0.1_alex.pth": "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
}
CORE_FILES = ("vsr/data.py", "vsr/evaluation.py", "vsr/perceptual_metrics.py", "vsr/reproducibility.py")
DOVE_COMMIT = "1a499a84bb6733f71ba5a9cf20394f2373f635ba"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def write_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_bytes(value)
    if not replace:
        with path.open("xb") as stream:
            stream.write(payload)
        return
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        stream.write(payload)
    temporary.replace(path)


def binding(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), f"Expected regular input file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def validate_binding(item):
    path = Path(item["path"])
    require(path.is_file() and path.stat().st_size == item["bytes"], f"Input size/path changed: {path}")
    require(sha(path) == item["sha256"], f"Input bytes changed: {path}")


def relative_input(root, relative):
    require(not Path(relative).is_absolute(), "Manifest pixel paths must be relative")
    path = (Path(root) / relative).resolve(strict=True)
    require(path.is_relative_to(Path(root).resolve()), "Manifest input escaped the read root")
    return path


def crop_geometry(source_h, source_w):
    require(source_h >= HEIGHT and source_w >= WIDTH, "HR upscaling is forbidden")
    ratio = max(HEIGHT / source_h, WIDTH / source_w)
    resized_h = max(HEIGHT, round(source_h * ratio))
    resized_w = max(WIDTH, round(source_w * ratio))
    return {"source_hw": [source_h, source_w], "resize_hw": [resized_h, resized_w],
            "crop_tlhw": [(resized_h - HEIGHT) // 2, (resized_w - WIDTH) // 2, HEIGHT, WIDTH],
            "interpolation": "OpenCV INTER_AREA; identical aspect-cover/center-crop arithmetic to vsr.data.decode_clip"}


def select_hq(manifest, inventory):
    require(inventory["catalog_record_count"] == len(manifest["records"]), "Inventory/catalog count differs")
    records = {row["id"]: row for row in manifest["records"]}
    candidates = []
    for row in inventory["records"]:
        header = row["header"]
        record = records.get(row["id"])
        require(record is not None, "Inventory contains an unknown HQ record")
        require(row["split"] == record["split"] and row["source_id"] == record["source_id"],
                "HQ inventory identity/split differs")
        if (record["split"] == "val" and header.get("opened") is True
                and header.get("frame_count", 0) >= MAX_SPAN
                and header.get("height", 0) >= HEIGHT and header.get("width", 0) >= WIDTH
                and math.isfinite(float(header.get("fps", 0))) and float(header["fps"]) > 0):
            candidates.append((row, record))
    candidates.sort(key=lambda pair: (hashlib.sha256((SELECTION_PREFIX + pair[0]["id"]).encode()).hexdigest(), pair[0]["id"]))
    selected, sources = [], set()
    for row, record in candidates:
        if row["source_id"] in sources:
            continue
        selected.append((row, record))
        sources.add(row["source_id"])
        if len(selected) == 8:
            break
    require(len(selected) == 8, "Need eight eligible, distinct HQ validation sources")
    return selected, len(candidates)


def case_record(dataset, rank, identity, frames, stride, start, source_frames, fps, geometry):
    span = 1 + (frames - 1) * stride
    require(start >= 0 and start + span <= source_frames, "Case source interval is invalid")
    indices = list(range(start, start + span, stride))
    require(len(indices) == frames, "Case frame count mismatch")
    return {"case_id": f"{dataset.replace('-', '')}_{rank:02d}_T{frames}_s{stride}",
            "dataset": dataset, "record_id": identity, "frames": frames, "stride": stride,
            "source_frame_count_header": source_frames, "source_start": start,
            "source_indices": indices, "source_span_frames": span,
            "source_fps": fps, "sampled_fps": fps / stride, "geometry": geometry,
            "expected_hr_shape": [1, 3, frames, HEIGHT, WIDTH],
            "expected_raw_latent_shape": [1, 16, (frames - 1) // 4 + 1, HEIGHT // 8, WIDTH // 8]}


def prepare(args):
    read_root = args.read_root.resolve(strict=True)
    manifest_path = read_root / "manifests/hq_vsr.json"
    inventory_path = read_root / "manifests/hq_temporal_stride4_inventory.json"
    external_path = read_root / "manifests/external.json"
    config_path = read_root / "configs/formal_hq_v2_noref.json"
    manifest, inventory, external, config = map(read_json, (manifest_path, inventory_path, external_path, config_path))
    selected, eligible_count = select_hq(manifest, inventory)
    model_root = Path(config["model"]["local_dir"]).resolve(strict=True)
    official_manifest_path = read_root / "docs/official-model-manifest.json"
    official_manifest = read_json(official_manifest_path)
    require(official_manifest["id"] == config["model"]["repo_id"] and official_manifest["sha"] == config["model"]["revision"], "Pinned official model identity differs")
    official_files = {row["rfilename"]: row for row in official_manifest["siblings"]}
    vae_config_raw = (model_root / "vae/config.json").read_bytes()
    require(hashlib.sha1(b"blob " + str(len(vae_config_raw)).encode() + b"\0" + vae_config_raw).hexdigest() == official_files["vae/config.json"]["blobId"], "VAE config differs from official pinned Git blob")
    vae_config = read_json(model_root / "vae/config.json")
    require(vae_config["z_dim"] == 16 and vae_config["temperal_downsample"] == [False, True, True], "Unexpected pinned VAE geometry")
    hq_inputs, reds_inputs, cases, unsupported = [], [], [], []
    for rank, (row, record) in enumerate(selected):
        source = binding(record["path"])
        require(source["sha256"] == record["sha256"] == row["raw_sha256"], "HQ source changed since original manifest/inventory")
        header = row["header"]
        start = int(hashlib.sha256((START_PREFIX + record["id"]).encode()).hexdigest()[:16], 16) % (header["frame_count"] - MAX_SPAN + 1)
        geometry = crop_geometry(header["height"], header["width"])
        hq_inputs.append({"record_id": record["id"], "source_id": record["source_id"], "split": "val",
                          "header": header, "file": source, "common_source_start": start, "maximum_source_span": MAX_SPAN})
        for frames in CLIP_LENGTHS:
            for stride in STRIDES:
                cases.append(case_record("HQ-VSR", rank, record["id"], frames, stride, start,
                                         header["frame_count"], float(header["fps"]), geometry))
    reds_catalogue = {row["id"]: row for row in external["datasets"]["REDS4"]["records"]}
    for rank, identity in enumerate(("000", "011")):
        row = reds_catalogue[identity]
        source_frames, role = row["frame_count"], row["gt"]
        require(len(role["frames"]) == source_frames, "REDS GT frame catalogue differs")
        geometry = crop_geometry(role["height"], role["width"])
        required_indices = set()
        for frames in CLIP_LENGTHS:
            for stride in STRIDES:
                span = 1 + (frames - 1) * stride
                if span > source_frames:
                    unsupported.append({"dataset": "REDS4", "record_id": identity, "frames": frames,
                                        "stride": stride, "required_source_frames": span,
                                        "available_source_frames": source_frames,
                                        "status": "unsupported_source_too_short", "padding_or_wrap_used": False})
                    continue
                case = case_record("REDS4", rank, identity, frames, stride, 0, source_frames, 30.0, geometry)
                case["fps_caveat"] = "30 is declared preview playback FPS; original PNG capture timing unknown"
                cases.append(case)
                required_indices.update(case["source_indices"])
        frame_files = []
        for index in sorted(required_indices):
            item = role["frames"][index]
            source = binding(relative_input(read_root, item["path"]))
            require(source["bytes"] == item["bytes"] and source["sha256"] == item["sha256"], "REDS PNG changed since manifest")
            frame_files.append({"ordinal": index, "source_name": Path(item["path"]).name, "file": source})
        reds_inputs.append({"record_id": identity, "role": "GT", "source_frame_count": source_frames,
                            "source_hw": [role["height"], role["width"]], "frames": frame_files})
    require(len(cases) == 58 and len(unsupported) == 2, "Expected 48 HQ + 10 REDS supported crop cases and two unsupported cells")
    root_protocol = read_json(read_root / "docs/final_evaluation_v1/protocol.json")
    vae_source = root_protocol["bindings"]["Wan_VAE_source"]
    actual_vae_source = binding(vae_source["path"])
    require(actual_vae_source["sha256"] == vae_source["sha256"], "Official VAE implementation changed")
    source_files = {relative: sha(CODE_ROOT / relative) for relative in CORE_FILES}
    source_files["scripts/audit_vae_reconstruction.py"] = sha(__file__)
    weights = [binding(path) for path in sorted((model_root / "vae").glob("*.safetensors"))]
    require(len(weights) == 1, "This frozen Wan VAE must have one safetensors file")
    require(weights[0]["sha256"] == official_files["vae/diffusion_pytorch_model.safetensors"]["lfs"]["sha256"], "VAE weights differ from official pinned LFS SHA256")
    metrics = []
    for name, digest in LPIPS_HASHES.items():
        item = binding(read_root / "weights" / name)
        require(item["sha256"] == digest, "Pinned LPIPS metric weights differ")
        metrics.append(item)
    scope_cases = cases if args.scope == "full" else cases[:1]
    protocol = {
        "schema": SCHEMA, "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_by": "static_file_and_header_inspection_no_torch_or_pixel_decode", "read_root": str(read_root),
        "run_root": str(args.run_root.resolve()), "scope": args.scope, "audit_kind": "crop_128x224",
        "scope_is_complete_T1_1": False,
        "selection": {"algorithm": ALGORITHM, "sort_prefix": SELECTION_PREFIX, "start_prefix": START_PREFIX,
                      "eligible_val_header_records": eligible_count, "selected_hq_record_count": 8,
                      "selected_hq_distinct_source_count": 8, "manifest_split_changed": False,
                      "common_start_maximum_span": MAX_SPAN, "reds_ids": ["000", "011"],
                      "supported_full_crop_cases": 58, "executed_case_count": len(scope_cases)},
        "cases": scope_cases, "full_crop_case_catalogue": cases, "unsupported_cells": unsupported,
        "inputs": {"hq": hq_inputs, "reds": reds_inputs},
        "bindings": {"official_model_manifest": binding(official_manifest_path), "hq_manifest": binding(manifest_path), "hq_inventory": binding(inventory_path),
                     "external_manifest": binding(external_path), "original_config": binding(config_path),
                     "vae_config": binding(model_root / "vae/config.json"), "vae_weights": weights,
                     "vae_source": actual_vae_source, "metric_weights": metrics, "source_files": source_files},
        "model": {"repo_id": config["model"]["repo_id"], "revision": config["model"]["revision"],
                  "local_dir": str(model_root), "component": "AutoencoderKLWan_only", "dtype": "float32",
                  "device": "cuda", "posterior": "mode_no_sampling", "eval": True, "grad_enabled": False,
                  "use_tiling": False, "use_slicing": False, "optimizer_updates": 0,
                  "DiT_loaded": False, "T5_loaded": False, "DINO_loaded": False,
                  "latents_mean": vae_config["latents_mean"], "latents_std": vae_config["latents_std"]},
        "preprocessing": {"crop_hw": [HEIGHT, WIDTH], "degradation": "vsr.data.degrade_x4",
                          "scale": 4, "kernel": "torch_bicubic", "antialias": True,
                          "align_corners": False, "clamp_01": True,
                          "degradation_source": "new synthetic LR from selected HR crop for BOTH HQ and REDS; supplied REDS LQ is not used",
                          "window": "one fixed maximum-span start per HQ source across all six conditions; REDS starts at zero",
                          "native_VAE_temporal_cache": "retained within encode/decode; do not encode frames independently"},
        "metrics": {"spatial_pixel_contract": "full float32 RGB reconstruction and HR in [0,1]; no q/255 for spatial metrics",
                    "rgb": "vsr.evaluation.rgb_metrics; per-frame RGB PSNR/SSIM, zero border shave",
                    "lpips": "lpips==0.1.4; AlexNet v0.1; CPU; all original frames",
                    "temporal": "existing TecoGAN tLP/tLP100/tOF Farneback; cutfr=2; official central crop; documented uint8 conversion",
                    "full_sequence_metrics": True, "error_by_original_sample_index": True,
                    "training_seed_statistics": "not applicable: frozen deterministic VAE, no trained arms and no sampler seeds",
                    "population_std": True, "normalization_arithmetic": "float32_then_float64_population_statistics",
                    "pending": ["E*warp", "a separately specified metric named tLPIPS", "full_resolution_tiled_vs_untiled_encoding"]},
        "notes": {"T1_1_remains_partial": "E*warp/tLPIPS definitions and full-resolution tiling comparison are pending; crop full scope is 58 cases, not complete Stage 1",
                  "DOVE_primary_source": f"https://github.com/zhengchen1999/DOVE/blob/{DOVE_COMMIT}/metrics/raft/ewarp.py",
                  "DOVE_adapter": f"https://github.com/zhengchen1999/DOVE/blob/{DOVE_COMMIT}/metrics/ewarp.py",
                  "DOVE_not_implemented": "raft_rgb255 uses original raft-things.pth, half-resolution inputs, RAFT iters20, forward/backward+motion-boundary masks and CUDA Resample2d; adapter minimum frame 256x256. Existing torchvision C_T_SKHT_V2 weights and 128x224 audit crops do not match. No downloads/installations performed.",
                  "full_resolution_plan": "Separate future scope: REDS000 T9 stride1 native720x1280, raw untiled encode versus native enable_tiling(256x448,strides192x384), both decoded with identical untiled decoder; log OOM and stop without fallback.",
                  "no_stage2_authorization": "Continuing Stages0/1 does not settle D1–D4."},
        "execution": {"fresh_empty_run_only": True, "resume": False, "overwrite": False,
                      "stop_on_any_case_failure_or_OOM": True, "automatic_precision_or_tiling_retry": False,
                      "per_case_result_hashes": True, "save_binary_latents": False,
                      "metric_device": "cpu", "reproducibility": config.get("reproducibility", {}),
                      "environment": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                                      "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                                      "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}}
    }
    write_json(args.protocol, protocol)
    print(json.dumps({"status": "prepared", "scope": args.scope, "protocol": str(args.protocol.resolve()),
                      "protocol_sha256": sha(args.protocol), "cases": len(scope_cases),
                      "supported_full_crop_cases": 58, "unsupported_cells": len(unsupported),
                      "GPU_used": False, "pixels_decoded": False}))


def validate_protocol(protocol):
    require(protocol["schema"] == SCHEMA and protocol["audit_kind"] == "crop_128x224", "Unsupported audit protocol")
    require(protocol["scope"] in ("pilot", "full"), "Invalid frozen scope")
    require(len(protocol["cases"]) == (1 if protocol["scope"] == "pilot" else 58), "Frozen case count differs")
    require(protocol["cases"] == (protocol["full_crop_case_catalogue"][:1] if protocol["scope"] == "pilot" else protocol["full_crop_case_catalogue"]), "Frozen execution cases differ from scope")
    require(protocol["model"]["dtype"] == "float32" and protocol["model"]["device"] == "cuda" and
            protocol["model"]["posterior"] == "mode_no_sampling" and not protocol["model"]["use_tiling"], "VAE-only numerical protocol differs")
    require(len({r["source_id"] for r in protocol["inputs"]["hq"]}) == 8, "HQ sources must be distinct")
    require(all(r["split"] == "val" for r in protocol["inputs"]["hq"]), "HQ inputs must retain val split")
    ids = []
    for case in protocol["full_crop_case_catalogue"]:
        require(case["frames"] in CLIP_LENGTHS and case["stride"] in STRIDES, "Unplanned condition")
        expected = list(range(case["source_start"], case["source_start"] + 1 + (case["frames"] - 1) * case["stride"], case["stride"]))
        require(case["source_indices"] == expected and expected[-1] < case["source_frame_count_header"], "Invalid source index selection")
        require(case["expected_hr_shape"] == [1, 3, case["frames"], HEIGHT, WIDTH], "Crop geometry differs")
        ids.append(case["case_id"])
    require(len(ids) == len(set(ids)) == 58, "Case IDs must be unique")
    for relative, digest in protocol["bindings"]["source_files"].items():
        require(sha(CODE_ROOT / relative) == digest, f"Audit implementation changed: {relative}")
    for key in ("official_model_manifest", "hq_manifest", "hq_inventory", "external_manifest", "original_config", "vae_config", "vae_source"):
        validate_binding(protocol["bindings"][key])
    for item in protocol["bindings"]["vae_weights"] + protocol["bindings"]["metric_weights"]:
        validate_binding(item)
    for row in protocol["inputs"]["hq"]:
        validate_binding(row["file"])
    for row in protocol["inputs"]["reds"]:
        for item in row["frames"]:
            validate_binding(item["file"])


def decode_case(case, protocol):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from vsr.data import degrade_x4

    indices = case["source_indices"]
    geometry = case["geometry"]
    source_h, source_w = geometry["source_hw"]
    resize_h, resize_w = geometry["resize_hw"]
    top, left, height, width = geometry["crop_tlhw"]
    frames = []
    if case["dataset"] == "HQ-VSR":
        source = next(row for row in protocol["inputs"]["hq"] if row["record_id"] == case["record_id"])
        cap = cv2.VideoCapture(source["file"]["path"])
        try:
            require(cap.isOpened(), "HQ source failed to open")
            require(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == source_h and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == source_w, "Decoded HQ source dimensions changed")
            chosen = set(indices)
            for ordinal in range(indices[-1] + 1):
                okay, frame = cap.read()
                require(okay, f"HQ decode failed at source index {ordinal}")
                if ordinal in chosen:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
    else:
        source = next(row for row in protocol["inputs"]["reds"] if row["record_id"] == case["record_id"])
        catalogue = {row["ordinal"]: row for row in source["frames"]}
        for ordinal in indices:
            with Image.open(catalogue[ordinal]["file"]["path"]) as image:
                require(image.format == "PNG" and image.height == source_h and image.width == source_w, "REDS source geometry/format changed")
                frames.append(np.array(image.convert("RGB"), dtype=np.uint8, copy=True))
    require(len(frames) == case["frames"], "Selected frame count differs")
    processed = [cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)[top:top + height, left:left + width] for frame in frames]
    hr = torch.from_numpy(np.stack(processed)).permute(3, 0, 1, 2).unsqueeze(0).contiguous().float().div_(255)
    require(list(hr.shape) == case["expected_hr_shape"], "Prepared HR geometry differs")
    lr, up = degrade_x4(hr, 4)
    return hr, up, {"source_indices": indices, "geometry": geometry, "lr_shape": list(lr.shape),
                    "hr_rgb_f32_sha256": hashlib.sha256(hr.numpy().tobytes()).hexdigest(),
                    "up_lr_rgb_f32_sha256": hashlib.sha256(up.numpy().tobytes()).hexdigest()}


def latent_statistics(raw_hr, raw_lr, mean, std):
    """CPU float64 population statistics, without retaining binary latents."""
    import torch
    h32, l32 = raw_hr.detach().cpu().float(), raw_lr.detach().cpu().float()
    h, l = h32.double(), l32.double()
    require(h.shape == l.shape and h.ndim == 5 and h.shape[0] == 1, "Latent pair geometry differs")
    mean = h32.new_tensor(mean).view(1, -1, 1, 1, 1)
    std = h32.new_tensor(std).view(1, -1, 1, 1, 1)
    require(bool((std > 0).all()), "Official normalization std must be positive")
    nh, nl = ((h32 - mean) / std).double(), ((l32 - mean) / std).double()

    def summary(a, b):
        return {"hr_mean": float(a.mean()), "hr_std_population": float(a.std(unbiased=False)),
                "up_lr_mean": float(b.mean()), "up_lr_std_population": float(b.std(unbiased=False)),
                "gap_mean_up_lr_minus_hr": float((b - a).mean()),
                "gap_rmse": float((b - a).square().mean().sqrt()), "gap_mae": float((b - a).abs().mean())}

    channels, times = [], []
    for channel in range(h.shape[1]):
        channels.append({"channel": channel, "raw": summary(h[:, channel], l[:, channel]),
                         "normalized": summary(nh[:, channel], nl[:, channel])})
        for latent_time in range(h.shape[2]):
            times.append({"channel": channel, "latent_time": latent_time,
                          "raw": summary(h[:, channel, latent_time], l[:, channel, latent_time]),
                          "normalized": summary(nh[:, channel, latent_time], nl[:, channel, latent_time])})
    return {"raw_latent_shape": list(h.shape), "population_std": True, "statistics_dtype": "float64_cpu", "normalization_dtype": "float32_cpu_same_as_training_formula",
            "normalization": "(raw - official_mean) / official_std", "per_channel": channels,
            "per_channel_latent_time": times,
            "per_latent_time": [{"latent_time": t, "raw": summary(h[:, :, t], l[:, :, t]),
                                 "normalized": summary(nh[:, :, t], nl[:, :, t])} for t in range(h.shape[2])]}


def metric_records(hr, reconstructed, perceptual):
    from vsr.evaluation import rgb_metrics
    rgb = rgb_metrics(reconstructed, hr)
    lp = perceptual.evaluate(reconstructed, hr, include_temporal=True)
    frames = []
    for index, (row, distance) in enumerate(zip(rgb["per_frame"], lp["lpips_per_frame"])):
        frames.append({"sampled_frame_index": index, **row, "lpips_alex_v0_1": distance})
    return {"rgb": rgb, "perceptual": lp, "per_frame": frames,
            "pixel_contract": "float32_RGB01_for_spatial; existing_TecoGAN_uint8_conversion_for_temporal",
            "E_star_warp": {"status": "pending_definition_and_matching_implementation"},
            "tLPIPS": {"status": "pending_definition_not_an_alias_for_TecoGAN_tLP"}}


def csv_bytes(rows):
    import csv
    import io
    if not rows:
        return b""
    keys = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def write_csv(path, rows, *, replace=False):
    payload = csv_bytes(rows)
    path = Path(path)
    if replace:
        temporary = path.with_name(path.name + ".partial")
        with temporary.open("xb") as stream:
            stream.write(payload)
        temporary.replace(path)
    else:
        with path.open("xb") as stream:
            stream.write(payload)


def export_tables(run, completed):
    summaries, frames, channels, positions, pairs = [], [], [], [], []
    for result in completed:
        c, m = result["case"], result["metrics"]
        identity = {key: c[key] for key in ("case_id", "dataset", "record_id", "frames", "stride", "source_start")}
        temporal = m["perceptual"]["tecogan"]
        summaries.append({**identity, **{key: m["rgb"][key] for key in ("psnr_rgb_db", "ssim_rgb", "mse_rgb_01")},
                          "lpips_alex_v0_1": m["perceptual"]["lpips_alex01_rgb_allframes"],
                          **{key: temporal[key] for key in ("tLP_alex_v0_1_TecoGAN_crop", "tLP100_alex_v0_1_TecoGAN_crop", "tOF_Farneback_TecoGAN_crop")},
                          "case_wall_seconds": result["resources"]["case_wall_seconds"],
                          "cuda_peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in result["resources"]["cuda_phases"].values())})
        for row in m["per_frame"]:
            frames.append({**identity, **row, "source_frame_index": c["source_indices"][row["sampled_frame_index"]]})
        for row in temporal["per_pair"]:
            a, b = row["indices"]
            pairs.append({**identity, "sampled_first": a, "sampled_second": b,
                          "source_first": c["source_indices"][a], "source_second": c["source_indices"][b],
                          **{key: value for key, value in row.items() if key != "indices"}})
        for group, target in (("per_channel", channels), ("per_channel_latent_time", positions)):
            for row in result["latent_statistics"][group]:
                flat = {**identity, "channel": row["channel"]}
                if "latent_time" in row:
                    flat["latent_time"] = row["latent_time"]
                for kind in ("raw", "normalized"):
                    flat.update({f"{kind}_{key}": value for key, value in row[kind].items()})
                target.append(flat)
    for name, rows in (("case_metrics.csv", summaries), ("per_frame.csv", frames), ("latent_channels.csv", channels),
                       ("latent_channel_time.csv", positions), ("temporal_pairs.csv", pairs)):
        write_csv(run / name, rows, replace=True)


def execute(args):
    raw = args.protocol.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    require(actual_sha == args.expected_protocol_sha, "Frozen protocol SHA differs; prepare a new protocol rather than editing it")
    protocol = json.loads(raw)
    validate_protocol(protocol)
    run = args.run_dir.resolve()
    require(run.parent == Path(protocol["run_root"]).resolve(), "Run must be a direct dated child of the frozen new run root")
    require(run != Path(protocol["read_root"]) / "runs" and not run.is_relative_to(Path(protocol["read_root"]) / "runs"), "New outputs must not be inside original runs")
    require(not run.exists() or (run.is_dir() and not any(run.iterdir())), "Fresh empty run required; no resume or overwrite")
    run.mkdir(parents=True, exist_ok=True)
    with (run / "protocol.json").open("xb") as stream:
        stream.write(raw)
    write_json(run / "unsupported_cells.json", protocol["unsupported_cells"])
    status = {"status": "starting", "protocol_sha256": actual_sha, "scope": protocol["scope"],
              "planned_cases": len(protocol["cases"]), "completed_cases": 0, "T1_1_complete": False,
              "training_updates": 0, "GPU_execution": True, "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(run / "status.json", status)
    completed, vae = [], None
    started = time.perf_counter()
    stage, active_case = "runtime_imports", None
    try:
        os.environ.update(protocol["execution"]["environment"])
        os.environ.update(HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
        sys.path.insert(0, str(Path(protocol["read_root"]) / ".deps"))
        sys.path.insert(0, str(CODE_ROOT))
        import torch
        import diffusers
        from diffusers import AutoencoderKLWan
        from vsr.perceptual_metrics import PerceptualMetrics
        from vsr.reproducibility import configure_reproducibility

        require(not torch.cuda.is_initialized(), "Use a fresh process before CUDA initialization")
        require(Path(inspect.getsourcefile(AutoencoderKLWan)).resolve() == Path(protocol["bindings"]["vae_source"]["path"]), "Imported VAE implementation path differs")
        reproducibility = configure_reproducibility({"reproducibility": protocol["execution"]["reproducibility"]})
        require(torch.cuda.is_available(), "Frozen FP32 CUDA audit requires CUDA; no CPU/dtype fallback")
        runtime = {"python": sys.version, "executable": sys.executable, "torch": str(torch.__version__),
                   "diffusers": diffusers.__version__, "torch_cuda": torch.version.cuda,
                   "packages": {}, "reproducibility": reproducibility,
                   "vae_source": str(inspect.getsourcefile(AutoencoderKLWan)),
                   "metric_device": "cpu", "device_name": torch.cuda.get_device_name(0)}
        for package in ("torchvision", "numpy", "Pillow", "lpips", "scikit-image", "opencv-python-headless"):
            try:
                runtime["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                runtime["packages"][package] = None
        write_json(run / "environment.json", runtime)
        stage = "load_vae_only"
        load_started = time.perf_counter()
        vae = AutoencoderKLWan.from_pretrained(protocol["model"]["local_dir"], subfolder="vae", dtype=torch.float32,
                                             local_files_only=True, use_safetensors=True).requires_grad_(False).eval().to("cuda")
        require(all(p.dtype == torch.float32 and p.device.type == "cuda" and not p.requires_grad for p in vae.parameters()), "VAE must remain FP32 CUDA/frozen")
        require(not vae.use_tiling and not vae.use_slicing, "Crop primary protocol is untiled/unsliced")
        require(list(vae.config.latents_mean) == protocol["model"]["latents_mean"] and list(vae.config.latents_std) == protocol["model"]["latents_std"], "Loaded VAE normalization constants differ")
        torch.cuda.synchronize()
        write_json(run / "vae_initialization.json", {"seconds": time.perf_counter() - load_started,
                   "weight_elements": sum(p.numel() for p in vae.parameters()), "dtype": "float32",
                   "device": "cuda", "eval": not vae.training, "DiT_loaded": False, "T5_loaded": False, "optimizer_updates": 0})
        metric_weights = {Path(row["path"]).name: row["path"] for row in protocol["bindings"]["metric_weights"]}
        stage = "load_cpu_lpips"
        perceptual = PerceptualMetrics(metric_weights["alexnet-owt-7be5be79.pth"], metric_weights["lpips_v0.1_alex.pth"], device="cpu", batch_size=8)
        for case in protocol["cases"]:
            active_case = case["case_id"]
            directory = run / "cases" / active_case
            directory.mkdir(parents=True, exist_ok=False)
            case_start = time.perf_counter()
            stage = "decode_inputs"
            write_json(directory / "status.json", {"status": "running", "stage": stage, "case_id": active_case, "protocol_sha256": actual_sha})
            status.update(status="running", active_case=active_case, stage=stage)
            write_json(run / "status.json", status, replace=True)
            hr, up, input_audit = decode_case(case, protocol)
            resources = {"cuda_phases": {}}

            def cuda_phase(name, function):
                nonlocal stage
                stage = name
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                initial_allocated = torch.cuda.memory_allocated()
                stamp = time.perf_counter()
                try:
                    return function()
                finally:
                    torch.cuda.synchronize()
                    resources["cuda_phases"][name] = {"seconds": time.perf_counter() - stamp,
                        "initial_allocated_bytes": initial_allocated, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "scope": "phase_peak_including_resident_VAE_and_case_tensors; CPU metrics excluded"}
                    write_json(directory / "resource_progress.json", {"stage": stage, **resources}, replace=True)
                    write_json(directory / "status.json", {"status": "running", "stage": stage, "case_id": active_case, "protocol_sha256": actual_sha}, replace=True)

            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                raw_hr = cuda_phase("encode_hr", lambda: vae.encode((hr * 2 - 1).to("cuda", dtype=torch.float32)).latent_dist.mode())
                require(list(raw_hr.shape) == case["expected_raw_latent_shape"] and bool(torch.isfinite(raw_hr).all()), "HR latent geometry/nonfinite failure")
                reconstructed = cuda_phase("decode_hr", lambda: (vae.decode(raw_hr).sample.float() / 2 + 0.5).clamp(0, 1).cpu())
                require(list(reconstructed.shape) == case["expected_hr_shape"] and bool(torch.isfinite(reconstructed).all()), "Reconstruction geometry/nonfinite failure")
                raw_lr = cuda_phase("encode_up_lr", lambda: vae.encode((up * 2 - 1).to("cuda", dtype=torch.float32)).latent_dist.mode())
                require(raw_lr.shape == raw_hr.shape and bool(torch.isfinite(raw_lr).all()), "LR latent geometry/nonfinite failure")
                stage = "latent_statistics_cpu"
                statistics = latent_statistics(raw_hr, raw_lr, protocol["model"]["latents_mean"], protocol["model"]["latents_std"])
                del raw_hr, raw_lr
                vae.clear_cache()
                torch.cuda.empty_cache()
            stage = "metrics_cpu"
            metric_stamp = time.perf_counter()
            metrics = metric_records(hr, reconstructed, perceptual)
            resources.update(metric_seconds=time.perf_counter() - metric_stamp, case_wall_seconds=time.perf_counter() - case_start)
            result = {"status": "passed", "protocol_sha256": actual_sha, "case": case, "inputs": input_audit,
                      "latent_statistics": statistics, "metrics": metrics, "resources": resources,
                      "posterior": "mode", "no_training_or_sampler_seed": True, "binary_latents_saved": False,
                      "completed_utc": datetime.now(timezone.utc).isoformat()}
            write_json(directory / "result.json", result)
            write_json(directory / "complete.json", {"status": "passed", "protocol_sha256": actual_sha,
                       "case_id": active_case, "result_sha256": sha(directory / "result.json")})
            write_json(directory / "status.json", {"status": "passed", "protocol_sha256": actual_sha, "case_id": active_case,
                       "result_sha256": sha(directory / "result.json")}, replace=True)
            completed.append(result)
            export_tables(run, completed)
            status.update(status="running", completed_cases=len(completed), last_completed_case=active_case,
                          wall_seconds=time.perf_counter() - started)
            write_json(run / "status.json", status, replace=True)
            print(json.dumps({"case": active_case, "status": "passed", "completed": len(completed),
                              "planned": len(protocol["cases"]), "wall_seconds": resources["case_wall_seconds"]}), flush=True)
            del hr, up, reconstructed
        summary = {"status": "passed", "protocol_sha256": actual_sha, "scope": protocol["scope"],
                   "completed_cases": len(completed), "expected_cases": len(protocol["cases"]),
                   "supported_full_crop_cases": 58, "unsupported_cells": protocol["unsupported_cells"],
                   "T1_1_complete": False, "T1_2_scope": "selected_crop_cases_only",
                   "pending": protocol["metrics"]["pending"], "training_updates": 0,
                   "wall_seconds": time.perf_counter() - started,
                   "case_result_sha256": {r["case"]["case_id"]: sha(run / "cases" / r["case"]["case_id"] / "result.json") for r in completed}}
        write_json(run / "summary.json", summary)
        status.update(status="passed", summary_sha256=sha(run / "summary.json"), completed_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=summary["wall_seconds"])
        write_json(run / "status.json", status, replace=True)
        print(json.dumps({"status": "passed", "scope": protocol["scope"], "run": str(run), "cases": len(completed), "T1_1_complete": False}), flush=True)
    except BaseException as error:
        failure = {"status": "failed", "protocol_sha256": actual_sha, "stage": stage, "case_id": active_case,
                   "exception_type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(),
                   "completed_cases": len(completed), "wall_seconds": time.perf_counter() - started,
                   "no_automatic_retry": True, "dtype_or_tiling_changed": False, "T1_1_complete": False}
        if "torch" in locals() and torch.cuda.is_initialized():
            failure["cuda_memory"] = {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
                                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        write_json(run / "failure.json", failure)
        if active_case is not None and (run / "cases" / active_case).is_dir():
            write_json(run / "cases" / active_case / "failure.json", failure)
            write_json(run / "cases" / active_case / "status.json", {"status": "failed", "stage": stage, "case_id": active_case,
                       "protocol_sha256": actual_sha, "failure_sha256": sha(run / "failure.json")}, replace=True)
        status.update(status="failed", stage=stage, active_case=active_case, failure_sha256=sha(run / "failure.json"), wall_seconds=failure["wall_seconds"])
        write_json(run / "status.json", status, replace=True)
        print(json.dumps({"status": "failed", "stage": stage, "case": active_case, "run": str(run)}), flush=True)
        raise
    finally:
        if vae is not None:
            try:
                vae.clear_cache()
            except BaseException:
                pass


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Freeze source selections and hashes without torch/pixel/GPU use")
    prep.add_argument("--read-root", type=Path, default=Path("/home/hugo/Projects/VSR"))
    prep.add_argument("--run-root", type=Path, default=Path("/home/hugo/Projects/VSR_/runs"))
    prep.add_argument("--scope", choices=("pilot", "full"), required=True, help="pilot freezes only first crop case; full freezes 58 cases")
    prep.add_argument("--protocol", type=Path, required=True, help="New protocol JSON file; existing file is never replaced")
    run = commands.add_parser("execute", help="Execute exact frozen VAE crop cases in a fresh CUDA process")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--expected-protocol-sha", required=True)
    run.add_argument("--run-dir", type=Path, required=True, help="Fresh empty direct child of the frozen run root")
    return root


def main():
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
