"""Data preparation, fixed-window readers, cache validation and latent masking.

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
from dataclasses import dataclass
from collections import OrderedDict
from collections.abc import Sequence
import copy
import importlib.metadata
import operator
import os
import shutil
import subprocess
import tempfile

import torch

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


# Shared metadata I/O never loads a model or initializes CUDA.
def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def file_hash(path):
    return sha256_file(path)


def write_json_new(path, payload):
    """Atomically publish a complete JSON file; never replace an existing path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _need(value, message):
    if not value:
        raise ValueError(message)


def _mapped_path(path, path_map=None):
    """Apply only an explicit longest-prefix migration mapping."""
    original = Path(path)
    candidates = sorted((path_map or {}).items(), key=lambda item: len(str(item[0])), reverse=True)
    for before, after in candidates:
        try:
            relative = original.relative_to(Path(before))
        except ValueError:
            continue
        return (Path(after) / relative).resolve()
    return original.resolve()


def _inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    _need(path.is_relative_to(root), "Asset path escapes selected cache root")
    return path


class PreparedTrainingData(Sequence):
    """Read-only protocol33 training latents, with a bounded two-item LRU.

    The old producer seal is retained as provenance. Every artifact used here
    is hashed against it before loading; optional verify_all validates the
    complete asset set without deserializing unrelated payloads. No task
    checkpoint, teacher target, optimizer or raw video is loaded.
    """

    def __init__(self, cache_root, training_count=None, *, expected_seal_sha256=None, verify_all=False):
        self.root = Path(cache_root).resolve()
        _need(self.root.is_dir(), "Prepared cache root is missing")
        seal_path = _inside(self.root / "cache_asset_seal.json", self.root)
        self.seal = read_json(seal_path)
        self.seal_sha256 = file_hash(seal_path)
        if expected_seal_sha256 is not None:
            _need(self.seal_sha256 == expected_seal_sha256, "Cache seal SHA256 mismatch")
        _need(self.seal.get("contains_task_initialization") is False, "Data seal must exclude task initialization")
        self._hashes = self.seal.get("artifact_sha256")
        _need(isinstance(self._hashes, dict) and self._hashes, "Cache seal has no artifact hashes")
        for name, digest in self._hashes.items():
            _need(isinstance(name, str) and isinstance(digest, str) and len(digest) == 64, "Invalid sealed asset entry")
            _inside(self.root / name, self.root)
        if verify_all:
            for name in self._hashes:
                self._verified(name)
        self.config = self._json("config.json")
        self.selection = self._json("selection.json")
        self.geometry = self._json("correspondence.json")
        _need(self.selection.get("test_accessed") is False, "Training cache must exclude test data")
        self.ids = self.selection.get("training_ids")
        self.validation_ids = self.selection.get("validation_ids", [])
        _need(isinstance(self.ids, list) and self.ids and isinstance(self.validation_ids, list), "Invalid training/validation selection")
        all_ids = self.ids + self.validation_ids
        _need(all(isinstance(x, str) and x for x in all_ids) and len(set(all_ids)) == len(all_ids), "Duplicate or invalid cached clip IDs")
        if training_count is not None:
            _need(type(training_count) is int and training_count == len(self.ids), "Training count differs from sealed IDs")
        data = self.config.get("data", {})
        _need([data.get(k) for k in ("num_frames", "height", "width", "scale", "temporal_stride")] == [33,128,224,4,1], "Prepared cache must use protocol33 geometry")
        _need(self.geometry.get("latent_grid") == [9,16,28] and self.geometry.get("token_grid") == [9,8,14], "Prepared latent/token geometry differs")
        _need(self.config.get("model", {}).get("repo_id") == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" and self.config["model"].get("revision") == "0fad780a534b6463e45facd96134c9f345acfa5b", "Prepared official model differs")
        _need(self.config.get("reference", {}).get("enabled", False) is False, "HRI cache is outside this mainline")
        self.text = torch.load(self._verified("fresh_text_embeddings.pt"), map_location="cpu", weights_only=True)
        _need(isinstance(self.text, dict) and set(self.text) == {CAPTION_PLACEHOLDER}, "Only the fixed generic cached text is supported")
        embedding = self.text[CAPTION_PLACEHOLDER]
        _need(isinstance(embedding, torch.Tensor) and tuple(embedding.shape) == (1,512,4096) and embedding.dtype == torch.bfloat16 and embedding.device.type == "cpu" and not embedding.requires_grad and bool(torch.isfinite(embedding).all()), "Invalid cached text tensor")
        self._metadata, sources = [], {"train": set(), "val": set()}
        for index, record_id in enumerate(all_ids):
            raw = self._json(f"clip_{index}/metadata.json")
            binding = raw.get("binding")
            metadata = {k:v for k,v in raw.items() if k != "binding"}
            _need(isinstance(binding, dict) and binding.get("data_cache_key") == _digest(metadata), "Cached clip metadata binding mismatch")
            expected_split = "train" if index < len(self.ids) else "val"
            _need(metadata.get("record_id") == record_id and metadata.get("split") == expected_split, "Cached ID/split order differs")
            _need(binding.get("manifest_sha256") == self.selection.get("manifest_sha256"), "Cached manifest binding mismatch")
            _need(metadata.get("temporal_stride") == 1 and len(metadata.get("source_indices", [])) == 33 and metadata["source_indices"] == list(range(metadata["source_indices"][0],metadata["source_indices"][0]+33)), "Cached source window differs")
            _need(metadata.get("crop_tlhw", [])[-2:] == [128,224] and metadata.get("degradation", {}).get("scale") == 4, "Cached crop/degradation differs")
            fps = metadata.get("original_fps")
            _need(type(fps) in (int,float) and math.isfinite(fps) and fps > 0 and metadata.get("output_fps") == fps, "Cached stride1 FPS differs")
            _need(metadata.get("caption") == CAPTION_PLACEHOLDER, "Cached caption differs")
            sources[expected_split].add(metadata.get("source_id"))
            self._metadata.append(raw)
            _need(f"clip_{index}/fresh_latents.pt" in self._hashes, "Cache seal lacks per-clip latent")
        _need(None not in sources["train"] | sources["val"] and not sources["train"].intersection(sources["val"]), "Source leaks across train/validation")
        self._cache = OrderedDict()
        self.metadata = {"cache_root":str(self.root),"source_seal_sha256":self.seal_sha256,"training_count":len(self.ids),"validation_count":len(self.validation_ids),"task_checkpoint_loaded":False,"CPU_only":True}

    def _verified(self, name):
        _need(name in self._hashes, "Requested artifact is absent from the data seal: " + name)
        path = _inside(self.root / name, self.root)
        _need(path.is_file() and file_hash(path) == self._hashes[name], "Sealed cache artifact changed: " + name)
        return path

    def _json(self, name):
        return read_json(self._verified(name))

    def __len__(self):
        return len(self.ids)

    def record(self, index):
        index = operator.index(index)
        if index < 0: index += len(self)
        if not 0 <= index < len(self): raise IndexError(index)
        return copy.deepcopy(self._metadata[index])

    def text_for(self, index_or_caption=CAPTION_PLACEHOLDER):
        caption = self.record(index_or_caption)["caption"] if isinstance(index_or_caption, int) else index_or_caption
        _need(caption in self.text, "Caption is absent from the fixed text cache")
        return self.text[caption]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0: index += len(self)
        if not 0 <= index < len(self): raise IndexError(index)
        # Recheck the authoritative metadata bytes on every access, even on LRU hits.
        _need(self._json(f"clip_{index}/metadata.json") == self._metadata[index], "Cached metadata changed after construction")
        path = self._verified(f"clip_{index}/fresh_latents.pt")
        if index not in self._cache:
            item = torch.load(path, map_location="cpu", weights_only=True)
            _need(isinstance(item, dict) and item.get("binding") == self._metadata[index]["binding"], "Cached latent binding differs")
            pair = {k:item.get(k) for k in ("clean","low")}
            _need(all(isinstance(z, torch.Tensor) and tuple(z.shape) == (1,16,*self.geometry["latent_grid"]) and z.dtype == torch.float32 and z.device.type == "cpu" and not z.requires_grad and bool(torch.isfinite(z).all()) for z in pair.values()), "Invalid frozen latent pair")
            self._cache[index] = pair
        self._cache.move_to_end(index)
        while len(self._cache) > 2: self._cache.popitem(last=False)
        return {key:value.clone() for key,value in self._cache[index].items()}

    def close(self):
        self._cache.clear()


# Whole-latent masking: copied numerics, now owned by data preparation.
def _require(value, message):
    if not value:
        raise ValueError(message)


@dataclass(frozen=True)
class SpanMask:
    mask: torch.Tensor
    spans: tuple
    rho_requested: float
    span: int
    masked_positions_per_example: int
    realized_ratio_all_latents: float
    realized_ratio_eligible_latents: float
    rounding: str = "nearest whole-span count to rho*T, half-up; cap at floor((T-1)/span); latent0 excluded"


def sample_span_mask(batch_size, latent_frames=9, *, rho=0.25, span=2, generator):
    """CPU mask with uniformly sampled nonoverlapping spans; never selects 0.

    rho's denominator is all T latent positions, including the protected first
    position. T=9/rho=.25/span=2 therefore selects 2/9, not an exact .25. All
    feasible arrangements of the chosen number of equal-length spans are
    sampled through compressed sorted start positions, without greedy failure.
    """
    _require(type(batch_size) is int and batch_size > 0 and type(latent_frames) is int and latent_frames > 0, "Positive integer batch/time sizes required")
    _require(type(span) is int and span > 0, "Span must be a positive integer")
    _require(type(rho) in (int, float) and math.isfinite(rho) and 0 <= rho <= 1, "rho must lie in [0,1]")
    _require(isinstance(generator, torch.Generator) and generator.device.type == "cpu", "Use an explicit independent CPU mask generator")
    eligible = latent_frames - 1
    _require(rho == 0 or span <= eligible, "Span cannot fit the eligible latent positions")
    count = min(int(math.floor(float(rho) * latent_frames / span + .5)), eligible // span)
    mask = torch.zeros(batch_size, latent_frames, dtype=torch.bool)
    spans = []
    for example in range(batch_size):
        starts = []
        if count:
            slots = eligible - count * (span - 1)
            compressed = torch.randperm(slots, generator=generator)[:count].sort().values.tolist()
            starts = [1 + value + index * (span - 1) for index, value in enumerate(compressed)]
            for start in starts:
                mask[example, start:start + span] = True
        spans.append(tuple((start, start + span) for start in starts))
    masked = count * span
    return SpanMask(mask, tuple(spans), float(rho), span, masked, masked / latent_frames,
                    masked / eligible if eligible else 0.)


def _latent_mask(mask, batch, frames):
    _require(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool and tuple(mask.shape) == (batch, frames), "Latent mask must be bool [B,T]")
    _require(not bool(mask[:, 0].any()), "The first latent position must remain clean")


def corrupt_latent_condition(condition, mask, *, generator, mode="matched_noise"):
    """Return a new condition; the original cached tensor is never mutated.

    Matched-noise statistics are detached per-example/channel population mean
    and SD over the original T,H,W. Noise matches these moments in expectation,
    not exactly in a finite draw. A zero-variance channel stays constant. A zero
    mask returns an exact clone and does not advance the noise generator.
    """
    _require(isinstance(condition, torch.Tensor) and condition.ndim == 5 and condition.is_floating_point()
             and all(condition.shape) and not condition.requires_grad and bool(torch.isfinite(condition).all()), "Condition must be finite frozen floating [B,C,T,H,W]")
    batch, _, frames, _, _ = condition.shape
    _latent_mask(mask, batch, frames)
    _require(mode in {"matched_noise", "zero"}, "Unknown latent corruption mode")
    _require(isinstance(generator, torch.Generator) and generator.device.type == condition.device.type, "Noise generator must match condition device type")
    if not bool(mask.any()):
        return condition.clone()
    if mode == "zero":
        replacement = torch.zeros_like(condition)
    else:
        source = condition.detach().float()
        mean = source.mean(dim=(2, 3, 4), keepdim=True)
        std = source.var(dim=(2, 3, 4), correction=0, keepdim=True).sqrt()
        noise = torch.randn(condition.shape, generator=generator, device=condition.device, dtype=torch.float32)
        replacement = (noise * std + mean).to(condition.dtype)
    return torch.where(mask.to(condition.device)[:, None, :, None, None], replacement, condition)



# Fixed pixel-stress assets preserve the original val32 reader contract.
need = _need
fresh_json = write_json_new
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def local_seed(namespace, seed, record, frame=None):
    value = [namespace, seed, record] + ([] if frame is None else [frame])
    return int.from_bytes(bytes.fromhex(digest(value))[:8], "big") % 2**63


def mask_for(record, cfg):
    starts = cfg["mask"]["start_candidates"]
    # Use the full unsigned first8bytes for start selection, before RNG's
    # signed-range reduction (which would change modulo5 for some records).
    key = [cfg["mask"]["namespace"], cfg["mask"]["seed"], record]
    start = starts[int(digest(key)[:16], 16) % len(starts)]
    return [start <= i < start+8 for i in range(33)]


def codec_binding():
    path = shutil.which("ffmpeg")
    need(path is not None, "FFmpeg unavailable; codec condition cannot be silently replaced")
    encoders = subprocess.run([path, "-hide_banner", "-encoders"], check=True, capture_output=True, text=True).stdout
    need(" libx264 " in encoders, "libx264 unavailable; no codec fallback allowed")
    version = subprocess.run([path, "-version"], check=True, capture_output=True, text=True).stdout
    return {"path": str(Path(path).resolve()), "sha256": file_sha(path), "version": version}


def codec_frame(frame, fps, cfg, binding):
    import numpy as np
    import torch
    from fractions import Fraction
    h,w = frame.shape[-2:]
    need(h%2==0 and w%2==0, "Codec requires even native LR dimensions; never resize")
    raw = frame.permute(1,2,0).mul(255).round().byte().contiguous().numpy().tobytes()
    rate = Fraction(str(fps)).limit_denominator(1_000_000)
    command = [binding['path'],'-hide_banner','-loglevel','error','-nostdin','-f','rawvideo','-pixel_format','rgb24',
        '-video_size',f'{w}x{h}','-framerate',f'{rate.numerator}/{rate.denominator}','-i','pipe:0','-frames:v','1',
        '-an','-c:v','libx264','-preset',cfg['codec_allintra']['preset'],'-crf',str(cfg['codec_allintra']['crf']),'-pix_fmt','yuv420p','-threads','1','-g','1',
        '-keyint_min','1','-sc_threshold','0','-bf','0','-map_metadata','-1','-fflags','+bitexact','-flags:v','+bitexact','-f','h264','pipe:1']
    encoded = subprocess.run(command,input=raw,capture_output=True,check=True,timeout=60).stdout
    decode = [binding['path'],'-hide_banner','-loglevel','error','-nostdin','-f','h264','-i','pipe:0','-frames:v','1',
              '-an','-threads','1','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    payload = subprocess.run(decode,input=encoded,capture_output=True,check=True,timeout=60).stdout
    need(len(payload)==h*w*3 and len(encoded)>0, "Codec changed single-frame geometry/count")
    result = torch.from_numpy(np.frombuffer(payload,dtype=np.uint8).copy()).reshape(h,w,3).permute(2,0,1).float().div(255)
    return result, {"encode_args":command,"decode_args":decode,"encoded_sha256":hashlib.sha256(encoded).hexdigest(),
                    "encoded_bytes":len(encoded),"decoded_RGB24_sha256":hashlib.sha256(payload).hexdigest()}


def corrupt_cpu(clean, condition, item, cfg, codec):
    import torch
    import torch.nn.functional as F
    need(clean.device.type=='cpu' and tuple(clean.shape)==(1,3,33,32,56) and clean.dtype==torch.float32,
         "Only authoritative CPU float32 LR crop33x32x56 accepted")
    need(condition in cfg['conditions'] and bool(torch.isfinite(clean).all()) and float(clean.min())>=0 and float(clean.max())<=1,
         "Invalid condition/LR RGB")
    out = clean.clone()
    indices = item['masked_indices']
    details = []
    if condition=='blur':
        radius=cfg['blur']['kernel']//2
        x=torch.arange(-radius,radius+1,dtype=torch.float64)
        k=torch.exp(-x.square()/(2*cfg['blur']['sigma_LR_pixels']**2));k/=k.sum()
        kernel=torch.outer(k,k).float().view(1,1,2*radius+1,2*radius+1).expand(3,1,-1,-1).contiguous()
        selected=clean[0,:,indices].permute(1,0,2,3)
        out[0,:,indices]=F.conv2d(F.pad(selected,(radius,)*4,mode='reflect'),kernel,groups=3).permute(1,0,2,3).clamp(0,1)
    else:
        for frame in indices:
            if condition=='noise':
                g=torch.Generator(device='cpu').manual_seed(item['noise_seeds_by_masked_frame'][str(frame)])
                out[0,:,frame]=(clean[0,:,frame]+torch.randn(clean[0,:,frame].shape,generator=g)*cfg['noise']['sigma_RGB01']).clamp(0,1)
            elif condition=='drop_repeat':out[0,:,frame]=clean[0,:,indices[0]-1]
            elif condition=='drop_black':out[0,:,frame]=0
            else:
                out[0,:,frame],detail=codec_frame(clean[0,:,frame],item['cached_header']['fps'],cfg,codec)
                details.append({"frame":frame,**detail})
    untouched=[i for i in range(33) if not item['mask'][i]]
    need(torch.equal(out[:,:,untouched],clean[:,:,untouched]) and out.shape==clean.shape and bool(torch.isfinite(out).all()),
         "Corruption leaked into clean frames or changed geometry")
    actual=[i for i in range(33) if not torch.equal(out[:,:,i],clean[:,:,i])]
    need(set(actual).issubset(indices), "Actual pixel support escaped mask")
    return out,{"condition":condition,"intended_masked_indices":indices,"actual_changed_indices":actual,
                "unmasked_float32_byte_identity":True,"codec_frames":details,"fallback":False}



def validate_seal(seal, *, path_map=None, verify_runtime=True):
    """Validate the unchanged seal, allowing explicitly recorded path relocation.

    path_map changes where bytes are found, never their expected hashes.
    verify_runtime=False is intended for synthetic tests or a separately
    declared migrated runtime protocol, never silent source validation bypass.
    """
    _need(type(verify_runtime) is bool, "verify_runtime must be boolean")
    _need(digest({k:v for k,v in seal.items() if k != "seal_digest"}) == seal["seal_digest"], "Cohort seal digest mismatch")
    for path_key, hash_key in [("config_path","config_sha256"),("manifest","manifest_file_sha256")]:
        _need(file_hash(_mapped_path(seal[path_key],path_map)) == seal[hash_key], "Sealed protocol/catalogue changed")
    _need(all(file_hash(_mapped_path(path,path_map)) == value for path,value in seal["implementation_bindings"].items()), "Sealed implementation bytes changed")
    if verify_runtime:
        current = codec_binding()
        expected = copy.deepcopy(seal["codec"])
        expected["path"] = str(_mapped_path(expected["path"],path_map))
        _need(current == expected, "Sealed FFmpeg/runtime differs; declare and review a new runtime binding")
        _need(all(importlib.metadata.version(name) == version for name,version in seal["CPU_runtime_packages"].items()), "CPU decode/corruption runtime versions differ")
    return {"seal_digest":seal["seal_digest"],"source_hashes_verified":True,"runtime_verified":verify_runtime,"path_map":{str(k):str(v) for k,v in (path_map or {}).items()}}


class FramewiseReader:
    """CPU asset reader for a later, separately frozen inference worker.

    Construction reads only dataset/metadata JSON. load() verifies and reads
    NPY pixels and exposes the original frozen_evaluation stable-HQ contract.
    This class does not construct models, choose checkpoints or run inference.
    """
    def __init__(self, dataset_manifest, seal, *, path_map=None, verify_runtime=True):
        self.path_map = path_map or {}
        validate_seal(seal, path_map=self.path_map, verify_runtime=verify_runtime)
        self.seal=seal
        dataset_manifest = _mapped_path(dataset_manifest, self.path_map)
        self.root=Path(dataset_manifest).resolve().parent
        self.document=read(dataset_manifest)
        need(self.document['status']=='completed_CPU_assets_no_predictions'
             and self.document['cohort_seal_digest']==seal['seal_digest'] and self.document['config']==seal['config'],
             'CPU asset manifest is not bound to this frozen cohort')
        need(len(self.document['assets']) == 32 and len(seal['cohort']) == 32, 'Reader requires exactly32 asset/cohort rows')
        self.assets={r['record_id']:r for r in self.document['assets']}
        self.cohort={r['record']['id']:r for r in seal['cohort']}
        need(set(self.assets)==set(self.cohort) and len(self.assets)==32,'Reader must expose the complete val32')
        conditions=['clean',*seal['config']['conditions']]
        self._modes={f'Stage2-val-{c}':c for c in conditions}
        self.records={dataset:{record:copy.deepcopy(item['record']) for record,item in self.cohort.items()} for dataset in self._modes}
        self.metadata={'reader':'Stage2FramewiseReader','source_dataset':'HQ-VSR','split':'val',
            'selected_record_count':32,'selected_record_ids':list(self.cohort),'cohort_seal_digest':seal['seal_digest'],
            'data_seed':20261102,'frames':33,'temporal_stride':1,'target_hw':[128,224],'scale':4,
            'conditions':conditions,'source_group_count':32,'training_allowed':False,'test_pixels_allowed':False,
            'pixel_decode_in_constructor':False,'metrics_computed':False}

    def _tensor(self, spec, expected):
        import numpy as np
        import torch
        path=Path(spec['path'])
        if not path.is_absolute():
            need('project_root' in self.seal and Path(self.seal['project_root']).is_absolute(), 'Relative asset path requires the sealed absolute project_root')
            path=Path(self.seal['project_root'])/path
        path=_mapped_path(path, self.path_map)
        need(path.is_relative_to(self.root) and file_sha(path)==spec['sha256'],'Asset path or NPY file digest changed')
        array=np.load(path,allow_pickle=False)
        need(array.dtype==np.float32 and list(array.shape)==expected and np.isfinite(array).all()
             and float(array.min())>=0 and float(array.max())<=1,'Invalid authoritative RGB asset')
        need(hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()==spec['rgb_f32_sha256'],
             'Authoritative float32 pixel digest changed')
        return torch.from_numpy(array.copy())

    def load(self,dataset,record_id,num_frames=33,start=0):
        need(dataset in self._modes and record_id in self.cohort and num_frames==33 and start==0,
             'Only sealed val33 windows and declared conditions can be read')
        row,item=self.assets[record_id],self.cohort[record_id]
        condition=self._modes[dataset]
        spec=row['clean'] if condition=='clean' else row['conditions'][condition]['asset']
        lr,hr=self._tensor(spec,[1,3,33,32,56]),self._tensor(row['HR'],[1,3,33,128,224])
        metadata=copy.deepcopy(row['source_metadata'])
        need(metadata['split']=='val' and metadata['source_indices']==item['source_indices'],'Asset window or heldout split changed')
        corruption_digest=digest({'condition':condition,'config':self.seal['config'],'mask':item['mask'],
                                  'noise_seeds':item['noise_seeds_by_masked_frame']})
        metadata.update(dataset=dataset,source_dataset='HQ-VSR',sequence_id=record_id,
            manifest_sha256=self.seal['manifest_digest'],source_frame_indices=item['source_indices'],
            prepared_lr_hw=[32,56],prepared_hr_hw=[128,224],fps_source='unchanged source FPS; ordinal stride1',
            framewise_mask=item['mask'],stage2_cohort_seal_digest=self.seal['seal_digest'],
            input_binding={'manifest_sha256':self.seal['manifest_digest'],'base_pair_cache_key':row['base_pair_cache_key'],
                'record_id':record_id,'robustness_mode':'framewise_'+condition,
                'noise_seed':local_seed('stage2-record-noise-master-v1',20261105,record_id),
                'robustness_protocol_sha256':corruption_digest,'base_lr_rgb_sha256':row['clean']['rgb_f32_sha256'],
                'robust_lr_rgb_sha256':spec['rgb_f32_sha256']})
        return {'lr':lr,'hr':hr,'fps':metadata['output_fps'],'metadata':metadata}

    read=load
    def close(self):return None




def select_records(manifest, config):
    """Original result-independent training/validation selection and order."""
    validate_manifest(manifest)
    options=config["data"]
    if options.get("selection")=="debug":
        training=list(manifest["debug_records"])
    else:
        training=[r for r in manifest["records"] if r["split"]=="train"]
        training.sort(key=lambda r:hashlib.sha256(f"train:{config['seed']}:{r['id']}".encode()).hexdigest())
        if options.get("train_limit"):training=training[:options["train_limit"]]
    validation=[r for r in manifest["records"] if r["split"]=="val"]
    validation.sort(key=lambda r:hashlib.sha256(f"val:{config['seed']}:{r['id']}".encode()).hexdigest())
    validation=validation[:options.get("val_limit",0)]
    _need(training and all(r["split"]=="train" for r in training),"A nonempty training-only selection is required")
    _need(all(r.get("caption")==CAPTION_PLACEHOLDER for r in training+validation),"Only the fixed generic caption is supported")
    return training,validation


def _prepare_training_cache(config, output, *, project_root=None, vae=None, device="cpu", text_cache_path=None, text_cache_sha256=None):
    """Stream protocol33 RGB/VAE preparation to a fresh source-sealed cache.

    This is data preparation only: no T5, DINO, task model, optimizer or task
    checkpoint is loaded. A supplied VAE/decoder can support synthetic CPU
    fixtures. Real model construction is explicit and local-files-only.
    """
    from . import vae as vae_ops
    cfg=copy.deepcopy(config);options=cfg["data"]
    _need([options.get(k) for k in ("num_frames","height","width","scale","temporal_stride")]==[33,128,224,4,1],"Preparation requires protocol33 geometry")
    _need(cfg.get("reference",{}).get("enabled",False) is False and cfg.get("conditioning",{}).get("hri_enabled",False) is False,"Only NoRef mainline preparation is supported")
    model=cfg.get("model",{})
    _need(model.get("repo_id")=="Wan-AI/Wan2.1-T2V-1.3B-Diffusers" and model.get("revision")=="0fad780a534b6463e45facd96134c9f345acfa5b","Pinned official Wan model required")
    def locate(value):
        path=Path(value)
        if path.is_absolute():return path.resolve()
        _need(project_root is not None,"Relative asset paths require explicit project_root")
        return (Path(project_root)/path).resolve()
    manifest_path=locate(options["manifest"])
    manifest=read_json(manifest_path);training,validation=select_records(manifest,cfg)
    text_spec=options.get("text_cache",{})
    text_path=locate(text_cache_path or text_spec.get("path",""))
    expected=text_cache_sha256 or text_spec.get("sha256")
    _need(isinstance(expected,str) and len(expected)==64 and file_hash(text_path)==expected,"Explicit frozen text-cache SHA256 required; no T5 fallback")
    embeddings=torch.load(text_path,map_location="cpu",weights_only=True)
    _need(isinstance(embeddings,dict) and set(embeddings)=={CAPTION_PLACEHOLDER},"Fixed generic text cache required")
    text=embeddings[CAPTION_PLACEHOLDER]
    _need(isinstance(text,torch.Tensor) and tuple(text.shape)==(1,512,4096) and text.dtype==torch.bfloat16 and text.device.type=="cpu" and not text.requires_grad and bool(torch.isfinite(text).all()),"Invalid frozen text embedding")
    output=Path(output).resolve()
    _need(not output.exists(),"Preparation output must be a fresh absent directory")
    output.mkdir(parents=True)
    own_vae=vae is None
    latents=[];completed=0;cleared=False;primary_error=None
    geometry={"temporal_stride":4,"spatial_stride":8,"patch_size":[1,2,2],"latent_grid":[9,16,28],"token_grid":[9,8,14],"anchors":list(range(0,33,4)),"flatten_order":"T,H,W"}
    config_sha=hashlib.sha256((json.dumps(cfg,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()).hexdigest()
    sources={"vsr/data.py":file_hash(Path(__file__)),"vsr/vae.py":file_hash(Path(vae_ops.__file__))}
    try:
        write_json_new(output/"config.json",cfg)
        write_json_new(output/"selection.json",{"manifest_sha256":manifest["manifest_sha256"],"training_ids":[r["id"] for r in training],"validation_ids":[r["id"] for r in validation],"test_accessed":False,"caption_policy":cfg.get("caption_policy","fixed generic text")})
        write_json_new(output/"correspondence.json",geometry)
        with text_path.open("rb") as source, (output/"fresh_text_embeddings.pt").open("xb") as destination:
            shutil.copyfileobj(source,destination)
        _need(file_hash(output/"fresh_text_embeddings.pt")==expected,"Text cache changed during copy")
        write_json_new(output/"environment.json",{"source_code_sha256":sources,"device_requested":str(device),"T5_loaded":False,"DINO_loaded":False,"task_model_loaded":False,"manifest_path":str(manifest_path),"manifest_file_sha256":file_hash(manifest_path)})
        if own_vae:vae=vae_ops.load_frozen_vae(model["local_dir"],device=device)
        actual,target=vae_ops.module_device(vae),torch.device(device)
        _need(actual.type==target.type and (target.type!="cuda" or target.index is None or actual.index==target.index), "Injected VAE device differs from the declared preparation device")
        first_binding=None
        for index,record in enumerate(training+validation):
            clip=decode_clip(record,num_frames=33,height=128,width=224,scale=4,seed=cfg["seed"],temporal_stride=1)
            binding={"data_cache_key":clip["cache_key"],"manifest_sha256":manifest["manifest_sha256"],"official_revision":model["revision"],"code_sha256":sources,"config_sha256":config_sha}
            if first_binding is None:first_binding=binding
            directory=output/f"clip_{index}";directory.mkdir()
            write_json_new(directory/"metadata.json",{**clip["metadata"],"binding":binding})
            with (directory/"paired_rgb.pt").open("xb") as stream:torch.save({key:clip[key] for key in ("hr","lr","lr_up")},stream)
            pair,restored,error=vae_ops.encode_pair(vae,clip,geometry,return_roundtrip=True)
            latents.append(pair)
            with (directory/"fresh_latents.pt").open("xb") as stream:torch.save({**pair,"binding":binding},stream)
            write_json_new(directory/"vae_roundtrip.json",{**error,"shape":list(restored.shape),"posterior":"mode","vae_frozen":True,"vae_has_grad":False,"streaming_RGB":True})
            completed+=1
            del clip,restored,pair
        with (output/"fresh_training_cache.pt").open("xb") as stream:torch.save({"latents":latents,"teacher_features":None,"binding":first_binding},stream)
        if vae is not None and hasattr(vae,"clear_cache"):vae.clear_cache()
        cleared=True
        write_json_new(output/"preparation_complete.json",{"status":"completed","training_clips":len(training),"validation_clips":len(validation),"RGB_resident_clip_bound":1,"T5_loaded":False,"DINO_loaded":False,"task_model_or_optimizer_loaded":False})
        names=sorted(str(p.relative_to(output)) for p in output.rglob("*") if p.is_file())
        write_json_new(output/"cache_asset_seal.json",{"schema":"vsr_prepared_data_v2","source_run":str(output),"contains_task_initialization":False,"artifact_sha256":{name:file_hash(output/name) for name in names},"source_code_sha256":sources})
        return {"status":"completed_data_preparation","cache_root":str(output),"training_count":len(training),"validation_count":len(validation),"cache_seal_sha256":file_hash(output/"cache_asset_seal.json"),"optimizer_updates":0}
    except BaseException as error:
        primary_error=error
        if not (output/"failure.json").exists():write_json_new(output/"failure.json",{"status":"failed_no_retry","completed_clips":completed,"error":type(error).__name__+": "+str(error),"optimizer_updates":0})
        raise
    finally:
        if not cleared and vae is not None and hasattr(vae,"clear_cache"):
            try:vae.clear_cache()
            except BaseException as cleanup_error:
                if primary_error is None:raise
                primary_error.add_note("VAE cleanup also failed: "+str(cleanup_error))
        if own_vae:vae=None


def prepare_training_cache(config, output, *, project_root=None, vae=None, device="cpu", text_cache_path=None, text_cache_sha256=None, gpu_lock=None, gpu_admission=None):
    """Prepare data under the same exclusive GPU lease as train/infer/evaluate.

    CPU remains the default and never queries GPU inventory. Explicit CUDA
    preparation requires a declared lock/admission before any VAE load; the
    shared lease includes the complete inner preparation/cleanup lifecycle.
    """
    target=torch.device(device)
    _need(target.type in ("cpu","cuda"), "Data preparation supports only explicit CPU or CUDA devices")
    options={"project_root":project_root,"vae":vae,"device":str(target),"text_cache_path":text_cache_path,"text_cache_sha256":text_cache_sha256}
    if target.type == "cpu":
        return _prepare_training_cache(config,output,**options)
    _need(gpu_lock is not None and isinstance(gpu_admission,dict), "CUDA data preparation requires explicit gpu_lock and gpu_admission")
    # Lazy import prevents a data/checkpoint import cycle and does not run a
    # process query until the explicit CUDA context is actually entered.
    from .checkpoints import gpu_lease
    with gpu_lease(gpu_lock,target,gpu_admission):
        return _prepare_training_cache(config,output,**options)

def main(argv=None):
    """CPU-only metadata command; preparing model latents is an explicit API."""
    import argparse
    parser = argparse.ArgumentParser(description="VSR data manifests and sealed CPU cache checks")
    sub = parser.add_subparsers(dest="mode", required=True)
    catalog = sub.add_parser("manifest"); catalog.add_argument("--raw-root", type=Path, required=True); catalog.add_argument("--output", type=Path, required=True); catalog.add_argument("--seed", type=int, default=20261102)
    prep = sub.add_parser("prepare-cache"); prep.add_argument("--config", type=Path, required=True); prep.add_argument("--output", type=Path, required=True); prep.add_argument("--project-root", type=Path); prep.add_argument("--device", default="cpu"); prep.add_argument("--gpu-lock", type=Path); prep.add_argument("--gpu-admission", type=Path)
    cache = sub.add_parser("check-cache"); cache.add_argument("--cache-root", type=Path, required=True); cache.add_argument("--seal-sha256"); cache.add_argument("--verify-all", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "manifest":
        result = build_manifest(args.raw_root, seed=args.seed); write_json_new(args.output, result)
        print(json.dumps({"status":"manifest_written","output":str(args.output),"GPU_used":False}))
    elif args.mode == "prepare-cache":
        result = prepare_training_cache(read_json(args.config), args.output, project_root=args.project_root, device=args.device, gpu_lock=args.gpu_lock, gpu_admission=read_json(args.gpu_admission) if args.gpu_admission else None); print(json.dumps(result))
    else:
        dataset = PreparedTrainingData(args.cache_root, expected_seal_sha256=args.seal_sha256, verify_all=args.verify_all)
        print(json.dumps(dataset.metadata)); dataset.close()


if __name__ == "__main__":
    main()
