"""Fresh HQ-VSR manifests and deterministic paired-video preprocessing.

Only raw HQ-VSR videos are read. Caption placeholders are explicitly marked and
must not be confused with video-specific text annotations. Temporal sampling is
an explicit paired-input preparation step: stride s selects every s-th source
frame and gives BOTH paired videos source_fps/s, preserving their sampled times.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

DATA_VERSION = "hq-vsr-paired-v2"
# Source catalogues/splits are unchanged by the prospective preprocessing
# protocol. Keep their original version and exact serialized manifest digest.
MANIFEST_DATA_VERSION = "hq-vsr-paired-v1"
CAPTION_PLACEHOLDER = "A high quality video."
VIDEO_NAME = re.compile(r"^(?P<source_id>[A-Za-z0-9_-]{11})_(?P<segment>\d+)_(?P<start>\d+)to(?P<end>\d+)\.mp4$")
SPLITS = ("train", "val", "test")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _quotas(n: int, fractions: tuple[float, float, float]) -> list[int]:
    if len(fractions) != 3 or any(x <= 0 for x in fractions) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ValueError("fractions must be three positive numbers summing to 1")
    exact = [n * x for x in fractions]
    quotas = [math.floor(x) for x in exact]
    order = sorted(range(3), key=lambda i: (-(exact[i] - quotas[i]), i))
    for i in order[:n - sum(quotas)]:
        quotas[i] += 1
    if n >= 3:
        for i in range(3):
            if quotas[i] == 0:
                donor = max(range(3), key=lambda j: quotas[j])
                quotas[donor] -= 1
                quotas[i] += 1
    return quotas


def select_debug_records(manifest: dict, limit: int = 2) -> list[dict]:
    """Use train clips only, without selecting examples on any observed metric."""
    seed = manifest["seed"]
    records = [r for r in manifest["records"] if r["split"] == "train"]
    return sorted(records, key=lambda r: (_digest(["debug", seed, r["id"]]), r["id"]))[:max(0, min(limit, 2))]


def build_manifest(raw_root: str | Path, seed: int = 20261102,
                   fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> dict:
    root = Path(raw_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"HQ-VSR raw directory does not exist: {root}")
    records, ignored, rejected = [], [], []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            ignored.append(path.name)
            continue
        if path.suffix.lower() != ".mp4":
            ignored.append(path.name)
            continue
        match = VIDEO_NAME.fullmatch(path.name)
        if not match:
            rejected.append({"path": str(path), "reason": "ambiguous_source_id_filename"})
            continue
        records.append({"id": path.stem, "source_id": match["source_id"],
                        "path": str(path), "dataset": "HQ-VSR", "bytes": path.stat().st_size,
                        "sha256": sha256_file(path), "caption": CAPTION_PLACEHOLDER,
                        "caption_status": "generic_placeholder_not_video_annotation"})
    # Quarantine byte-identical files under conflicting source IDs to prevent a
    # second naming convention from defeating source-level isolation.
    digest_sources: dict[str, set] = {}
    for record in records:
        digest_sources.setdefault(record["sha256"], set()).add(record["source_id"])
    ambiguous = {h for h, sources in digest_sources.items() if len(sources) > 1}
    kept = []
    for record in records:
        if record["sha256"] in ambiguous:
            rejected.append({"path": record["path"], "reason": "identical_bytes_under_different_sources", "sha256": record["sha256"]})
        else:
            kept.append(record)
    records = kept
    if not records:
        raise ValueError("No unambiguous HQ-VSR videos remain")
    groups = sorted({r["source_id"] for r in records}, key=lambda s: (_digest(["split", seed, s]), s))
    quotas = _quotas(len(groups), fractions)
    assignment = {}
    offset = 0
    for split, count in zip(SPLITS, quotas):
        for source in groups[offset:offset + count]:
            assignment[source] = split
        offset += count
    for record in records:
        record["split"] = assignment[record["source_id"]]
    manifest = {"schema_version": 1, "data_version": MANIFEST_DATA_VERSION, "dataset": "HQ-VSR", "raw_root": str(root),
                "seed": int(seed), "fractions": dict(zip(SPLITS, fractions)),
                "group_rule": "11-character YouTube source ID; all segments from a source share one split",
                "allocation": "SHA256 ordered sources; largest remainder quotas; minimum one per split when >=3 sources",
                "caption_policy": "Generic placeholder; actual video-specific captions unavailable; text semantic use not validated",
                "records": records, "ignored_files": ignored, "rejected": rejected,
                "source_count": len(groups), "split_source_counts": dict(zip(SPLITS, quotas)),
                "split_clip_counts": {s: sum(r["split"] == s for r in records) for s in SPLITS}}
    manifest["debug_records"] = select_debug_records(manifest)
    manifest["debug_ids"] = [r["id"] for r in manifest["debug_records"]]
    manifest["manifest_sha256"] = _digest(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> None:
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _digest(payload):
        raise ValueError("Manifest digest mismatch")
    sources, identities, hashes = {}, set(), {}
    for record in manifest["records"]:
        if record["id"] in identities:
            raise ValueError("Repeated clip ID")
        identities.add(record["id"])
        if record["split"] not in SPLITS:
            raise ValueError("Unknown split")
        previous = sources.setdefault(record["source_id"], record["split"])
        if previous != record["split"]:
            raise ValueError("Source leakage between splits")
        previous_hash_split = hashes.setdefault(record["sha256"], record["split"])
        if previous_hash_split != record["split"]:
            raise ValueError("Byte-identical video leakage between splits")
    if manifest["debug_records"] != select_debug_records(manifest):
        raise ValueError("Debug selection does not match deterministic train-only selection")


def degrade_x4(hr, scale: int = 4):
    """Bicubic x4 with antialiasing; same deterministic kernel for every frame."""
    import torch.nn.functional as F
    if hr.ndim != 5 or hr.shape[0] != 1 or hr.shape[1] != 3:
        raise ValueError("Expected RGB [1,3,T,H,W]")
    if scale < 1 or hr.shape[-2] % scale or hr.shape[-1] % scale:
        raise ValueError("Height and width must be divisible by scale")
    frames = hr[0].permute(1, 0, 2, 3)
    lr_frames = F.interpolate(frames, size=(hr.shape[-2] // scale, hr.shape[-1] // scale), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    up_frames = F.interpolate(lr_frames, size=hr.shape[-2:], mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    return lr_frames.permute(1, 0, 2, 3).unsqueeze(0).contiguous(), up_frames.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def decode_clip(record: dict, num_frames: int = 9, height: int = 128,
                width: int = 224, scale: int = 4, seed: int = 20261102,
                temporal_stride: int = 1) -> dict:
    import cv2
    import numpy as np
    import torch
    if num_frames < 1 or height < 1 or width < 1:
        raise ValueError("num_frames/height/width must be positive")
    if not isinstance(temporal_stride, int) or isinstance(temporal_stride, bool) or temporal_stride < 1:
        raise ValueError("temporal_stride must be a positive integer")
    source_span = 1 + (num_frames - 1) * temporal_stride
    path = Path(record["path"])
    source_sha = sha256_file(path)
    if record.get("sha256") and source_sha != record["sha256"]:
        raise ValueError(f"Raw source changed since manifest creation: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot decode video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_w, source_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Missing valid original FPS: {path}")
        if source_h < height or source_w < width:
            raise ValueError(f"Undersized HR video {source_h}x{source_w}; requested {height}x{width}; HR upscaling is forbidden")
        if total < source_span:
            raise ValueError(f"Video has {total} source frames; {source_span} required for {num_frames} output frames at temporal_stride={temporal_stride}")
        clip_seed = int(_digest(["window", seed, record["id"]])[:16], 16)
        start = clip_seed % (total - source_span + 1)
        source_indices = list(range(start, start + source_span, temporal_stride))
        # Sequential decode avoids keyframe-seek ambiguity in source indices.
        raw_frames = []
        for index in range(start + source_span):
            okay, bgr = cap.read()
            if not okay:
                raise ValueError(f"Video decode failed at source frame {index}: {path}")
            if index >= start and (index - start) % temporal_stride == 0:
                raw_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    ratio = max(height / source_h, width / source_w)
    resize_h, resize_w = max(height, round(source_h * ratio)), max(width, round(source_w * ratio))
    top, left = (resize_h - height) // 2, (resize_w - width) // 2
    processed = [cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)[top:top + height, left:left + width] for frame in raw_frames]
    hr = torch.from_numpy(np.stack(processed)).permute(3, 0, 1, 2).unsqueeze(0).contiguous().float().div_(255)
    lr, lr_up = degrade_x4(hr, scale)
    sampled_fps = fps / temporal_stride
    fps_policy = "preserve_original_no_temporal_resampling" if temporal_stride == 1 else "preserve_sampled_input_fps"
    metadata = {"data_version": DATA_VERSION, "record_id": record["id"], "source_id": record["source_id"],
                "source_path": str(path.resolve()), "source_sha256": source_sha, "split": record["split"],
                "seed": seed, "window_seed": clip_seed, "source_indices": source_indices,
                "temporal_stride": temporal_stride, "original_fps": fps, "output_fps": sampled_fps, "fps_policy": fps_policy,
                "source_hw": [source_h, source_w], "resize_hw": [resize_h, resize_w],
                "crop_tlhw": [top, left, height, width],
                "degradation": {"scale": scale, "kernel": "torch_bicubic", "antialias": True, "align_corners": False, "clamp_01": True, "temporal_parameters": "shared_for_all_frames"},
                "caption": record.get("caption", CAPTION_PLACEHOLDER), "caption_status": record.get("caption_status", "generic_placeholder_not_video_annotation"),
                "valid_region": "entire_crop_no_padding"}
    cache_key = _digest(metadata)
    # Pixel mask; the runtime derives token mask using its actual VAE/DiT grid.
    return {"hr": hr, "lr": lr, "lr_up": lr_up, "fps": sampled_fps, "metadata": metadata,
            "cache_key": cache_key, "valid_mask": torch.ones((1, num_frames, height, width), dtype=torch.bool)}


def write_video(path: str | Path, tensor, fps: float) -> None:
    import cv2
    import torch
    if tensor.ndim != 5 or tensor.shape[:2] != (1, 3):
        raise ValueError("Expected float RGB [1,3,T,H,W]")
    if not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("FPS must be positive")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Video contains nonfinite values")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = tensor.shape[-2:]
    if height % 2 or width % 2:
        raise ValueError("mp4v output requires even dimensions to avoid silent truncation")
    frames = tensor.detach().cpu()[0].permute(1, 2, 3, 0).clamp(0, 1).mul(255).round().byte().numpy()
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise ValueError(f"Cannot open MP4 writer: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"No video output was written: {path}")
