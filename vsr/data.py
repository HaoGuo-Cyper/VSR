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

    def hr_rgb_binding(self,index):
        """Bind only a training HR crop to the exact FM record/window/cache seal."""
        index=operator.index(index)
        if not 0<=index<len(self):raise IndexError(index)
        raw=self._json(f'clip_{index}/metadata.json')
        _need(raw==self._metadata[index] and raw['split']=='train'
              and raw['record_id']==self.ids[index],'Online HR differs from the FM training record')
        rgb_name=f'clip_{index}/paired_rgb.pt'
        rgb=self._verified(rgb_name)
        latent_name=f'clip_{index}/fresh_latents.pt'
        _need(latent_name in self._hashes,'Online HR has no sealed matching FM latent')
        return {'schema':'vsr_sealed_online_HR_v1','index':index,'record_id':raw['record_id'],
                'source_id':raw['source_id'],'split':'train','source_indices':copy.deepcopy(raw['source_indices']),
                'crop_tlhw':copy.deepcopy(raw['crop_tlhw']),'source_hw':copy.deepcopy(raw.get('source_hw')),
                'resize_hw':copy.deepcopy(raw.get('resize_hw')),'source_sha256':raw.get('source_sha256'),
                'temporal_stride':raw['temporal_stride'],'degradation':copy.deepcopy(raw['degradation']),
                'FM_binding':copy.deepcopy(raw['binding']),'cache_root':str(self.root),
                'cache_seal_sha256':self.seal_sha256,
                'metadata_sha256':self._hashes[f'clip_{index}/metadata.json'],
                'paired_RGB':{'path':str(rgb),'bytes':rgb.stat().st_size,'sha256':self._hashes[rgb_name]},
                'FM_latent_sha256':self._hashes[latent_name],
                'HR_shape':[1,3,33,128,224],'HR_dtype':'torch.float32','HR_range':[0,1],
                'augmentation':'same_sealed_FM_region_no_new_random_augmentation','teacher_features_cached':False}

    def hr_rgb(self,index):
        """Stream a sealed CPU [0,1] HR tensor; do not decode or cache features."""
        binding=self.hr_rgb_binding(index)
        payload=torch.load(binding['paired_RGB']['path'],map_location='cpu',weights_only=True)
        _need(isinstance(payload,dict) and set(payload)=={'hr','lr','lr_up'},
              'Sealed paired RGB payload must contain the producer HR/LR/LRup triple')
        hr=payload['hr']
        _need(isinstance(hr,torch.Tensor) and tuple(hr.shape)==(1,3,33,128,224)
              and hr.dtype==torch.float32 and hr.device.type=='cpu' and not hr.requires_grad
              and bool(torch.isfinite(hr).all()) and float(hr.min())>=0 and float(hr.max())<=1,
              'Invalid sealed online HR tensor shape/dtype/range')
        return hr.contiguous()

    def verify_hr_rgb_assets(self):
        """Complete training-only bytes/provenance audit, without teacher tensors."""
        records=[]
        for index in range(len(self)):
            binding=self.hr_rgb_binding(index)
            self._verified(f'clip_{index}/fresh_latents.pt')
            records.append(binding)
        return {'schema':'vsr_online_HR_asset_audit_v1','training_count':len(records),
                'cache_root':str(self.root),'cache_seal_sha256':self.seal_sha256,
                'records':records,'validation_or_test_RGB_loaded':False,
                'RGB_payload_deserialized':False,'teacher_forward_calls':0,'teacher_features_cached':False}

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

def _dinov2_source_header(cache_root):
    """Bind an existing paired-RGB cache, never a historical DINO feature cache."""
    root=Path(cache_root).resolve();seal=read_json(root/'cache_asset_seal.json')
    hashes=seal.get('artifact_sha256',{})
    def verified(name):
        path=_inside(root/name,root)
        _need(name in hashes and path.is_file() and file_hash(path)==hashes[name],'DINO source cache binding changed: '+name)
        return path
    config=read_json(verified('config.json'));selection=read_json(verified('selection.json'))
    geometry=read_json(verified('correspondence.json'))
    _need([config['data'].get(k) for k in ('num_frames','temporal_stride','height','width','scale')]==[33,1,128,224,4]
          and geometry['latent_grid']==[9,16,28] and geometry['token_grid']==[9,8,14], 'DINO requires the matched33stride1 crop cache')
    training,validation=selection['training_ids'],selection['validation_ids']
    _need(training and selection.get('test_accessed') is False
          and len(set(training+validation))==len(training+validation),'DINO source train/val selection differs or accesses test')
    return {'root':str(root),'seal_sha256':file_hash(root/'cache_asset_seal.json'),
            'config_sha256':hashes['config.json'],'selection_sha256':hashes['selection.json'],
            'geometry_sha256':hashes['correspondence.json'],
            'training_ids':training,'validation_ids':validation}


def _dinov2_source_record(source,index):
    root=Path(source['root']);seal=read_json(root/'cache_asset_seal.json')['artifact_sha256']
    ids=source['training_ids']+source['validation_ids'];name=f'clip_{index}/metadata.json'
    path=_inside(root/name,root)
    _need(name in seal and file_hash(path)==seal[name],'DINO source metadata changed')
    meta=read_json(path);payload={k:v for k,v in meta.items() if k!='binding'}
    _need(meta['record_id']==ids[index] and meta['split']==('train' if index<len(source['training_ids']) else 'val')
          and meta['binding']['data_cache_key']==_digest(payload),'DINO metadata ID/split/cache key mismatch')
    _need(meta['temporal_stride']==1 and len(meta['source_indices'])==33
          and meta['source_indices']==list(range(meta['source_indices'][0],meta['source_indices'][0]+33))
          and meta['crop_tlhw'][-2:]==[128,224],'DINO source window/crop mismatch; oldstride4 targets are forbidden')
    rgb_name=f'clip_{index}/paired_rgb.pt'
    _need(rgb_name in seal,'Original paired HR RGB is not sealed')
    return meta,{'metadata_sha256':seal[name],'paired_RGB_sha256':seal[rgb_name]}


def _dinov2_teacher_contract(binding):
    from .teacher import DINOV2_REVISION,DINOV2_WEIGHT_SHA256,dinov2_feature_spec
    _need(isinstance(binding,dict) and binding.get('architecture')=='dinov2_vitb14'
          and binding.get('source_revision')==DINOV2_REVISION
          and binding.get('weights',{}).get('sha256')==DINOV2_WEIGHT_SHA256
          and binding.get('feature_spec')==dinov2_feature_spec()
          and binding.get('precision')=='FP32_no_autocast','DINO feature cache teacher/preprocessing differs')


def _prepare_dinov2_cache(training_cache_root,output,teacher,device,execution_receipt=None):
    from . import teacher as teacher_ops
    import time
    started=time.perf_counter();feature_seconds=0.;feature_bytes=0;serialized_bytes=0
    source=_dinov2_source_header(training_cache_root);teacher_binding=teacher.binding
    _dinov2_teacher_contract(teacher_binding)
    header={'schema':'vsr_dinov2_cache_v1','source':source,'teacher':teacher_binding,
            'feature_spec':teacher_ops.dinov2_feature_spec(),
            'producer_code':{'data.py':file_hash(Path(__file__)),'teacher.py':file_hash(Path(teacher_ops.__file__))},
            'extraction_device':str(device),'target_normalized_before_loss':False}
    output=Path(output).resolve();_need(not output.exists(),'DINO cache output must be a new directory')
    _need(not output.is_relative_to(Path(source['root'])),'DINO output cannot overwrite paired source cache')
    output.mkdir(parents=True);records=[]
    if execution_receipt is not None:
        write_json_new(output/'extraction_receipt.json',execution_receipt)
        header['extraction_receipt_sha256']=file_hash(output/'extraction_receipt.json')
    try:
        teacher.prepare(device)
        for index in range(len(source['training_ids'])+len(source['validation_ids'])):
            meta,inputs=_dinov2_source_record(source,index)
            rgb_path=Path(source['root'])/f'clip_{index}/paired_rgb.pt'
            _need(file_hash(rgb_path)==inputs['paired_RGB_sha256'],'Original HR RGB cache changed')
            rgb=torch.load(rgb_path,map_location='cpu',weights_only=True)['hr']
            _need(tuple(rgb.shape)==(1,3,33,128,224) and rgb.dtype==torch.float32 and not rgb.requires_grad
                  and bool(torch.isfinite(rgb).all()) and float(rgb.min())>=0 and float(rgb.max())<=1,'Invalid original HR RGB')
            hr_sha=hashlib.sha256(rgb.contiguous().numpy().tobytes()).hexdigest()
            feature_started=time.perf_counter()
            features=teacher.encode_hr(rgb)
            feature_seconds+=time.perf_counter()-feature_started
            _need(isinstance(features,torch.Tensor) and tuple(features.shape)==(1,1008,768)
                  and features.dtype==torch.float32 and features.device.type=='cpu' and not features.requires_grad
                  and bool(torch.isfinite(features).all()),'DINO target shape/dtype/device differs')
            provenance={key:meta[key] for key in ('record_id','source_id','source_sha256','split','source_indices',
                         'crop_tlhw','source_hw','resize_hw','temporal_stride','original_fps','output_fps')}
            provenance.update(inputs,HR_f32_sha256=hr_sha,header_digest=_digest(header))
            directory=output/f'clip_{index}';directory.mkdir()
            metadata={'binding':provenance,'feature_shape':[1,1008,768],'dtype':'torch.float32'}
            write_json_new(directory/'metadata.json',metadata)
            with (directory/'features.pt').open('xb') as stream:
                torch.save({'features':features.detach().clone().contiguous(),'binding':provenance},stream)
            feature_bytes+=features.numel()*features.element_size()
            serialized_bytes+=(directory/'features.pt').stat().st_size
            records.append({'index':index,'record_id':meta['record_id'],
                'metadata':{'path':f'clip_{index}/metadata.json','sha256':file_hash(directory/'metadata.json')},
                'features':{'path':f'clip_{index}/features.pt','sha256':file_hash(directory/'features.pt')}})
            del rgb,features
        _need(teacher.binding==teacher_binding,'DINO teacher binding changed during extraction')
        manifest=header|{'status':'completed','records':records,'test_pixels_accessed':False}
        write_json_new(output/'manifest.json',manifest)
        write_json_new(output/'seal.json',{'schema':'vsr_dinov2_cache_seal_v1',
            'manifest_sha256':file_hash(output/'manifest.json'),'records':records})
        total_seconds=time.perf_counter()-started
        timings={'teacher_feature_extraction_seconds':feature_seconds,'total_cache_build_seconds':total_seconds,
                 'other_io_binding_seconds':total_seconds-feature_seconds,
                 'feature_scope':'encode_hr wall: CPU resize/normalize + H2D + FP32 DINO forward + blocking CPU copy + time grouping; not pure GPU compute',
                 'other_scope':'source/teacher binding, model construction/loading, input/output IO, hashing and publication through cache seal',
                 'total_scope':'cache builder entry through completed seal; excludes caller metadata construction, GPU admission and final cleanup',
                 'pure_GPU_encoder_seconds':None}
        memory=None
        if torch.device(device).type=='cuda':
            memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
                    'physical_device_bytes':torch.cuda.get_device_properties(device).total_memory}
        result={'status':'completed','cache_root':str(output),'seal_sha256':file_hash(output/'seal.json'),
                'training_count':len(source['training_ids']),'validation_count':len(source['validation_ids']),
                'teacher_frames':len(records)*33,'optimizer_updates':0,'test_pixels_accessed':False}
        receipt={'schema':'vsr_dinov2_cache_completion_v1',**result,'timings':timings,
                 'feature_tensor_bytes':feature_bytes,'serialized_feature_bytes':serialized_bytes,
                 'teacher_weight_bytes':teacher_binding['weights']['bytes'],'teacher_precision':'FP32',
                 'GPU_memory':memory,'new_VAE_feature_extraction_seconds_this_round':0.,
                 'VAE_cache_cost':'existing shared sunk preparation, not recharged per trajectory'}
        write_json_new(output/'completion_receipt.json',receipt)
        return result|{'completion_receipt':{'path':str(output/'completion_receipt.json'),
                       'sha256':file_hash(output/'completion_receipt.json')},'timings':timings,'GPU_memory':memory}
    except BaseException as error:
        if not (output/'failure.json').exists():
            write_json_new(output/'failure.json',{'status':'incomplete','completed_records':len(records),
                'error':str(error),'automatic_retry':False,'optimizer_updates':0})
        raise
    finally:
        import sys
        import gc
        primary=sys.exc_info()[1];first=primary
        operations=[teacher.close,gc.collect]
        if torch.device(device).type=='cuda' and torch.cuda.is_initialized():operations.append(torch.cuda.empty_cache)
        for operation in operations:
            try:operation()
            except BaseException as secondary:
                if first is None:first=secondary
                elif first is not secondary:first.add_note('DINO cleanup error: '+str(secondary))
        if primary is None and first is not None:raise first


def prepare_dinov2_cache(training_cache_root,output,*,teacher,device='cpu',gpu_lock=None,gpu_admission=None):
    """Explicit offline teacher extraction; CUDA load/cleanup share one lease."""
    device=torch.device(device)
    _need(device.type in ('cpu','cuda'),'DINO extraction requires an explicit CPU/CUDA device')
    if device.type=='cpu':return _prepare_dinov2_cache(training_cache_root,output,teacher,device)
    _need(gpu_lock is not None and isinstance(gpu_admission,dict),'DINO CUDA extraction requires shared GPU admission/lock')
    _need(not torch.cuda.is_initialized(),'Configure strict DINO extraction in a fresh process')
    _need(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','DINO extraction requires strict CUBLAS environment before CUDA')
    from .checkpoints import gpu_lease
    from .model import configure_reproducibility
    with gpu_lease(gpu_lock,str(device),gpu_admission) as admission:
        runtime=configure_reproducibility(read_json(Path(training_cache_root)/'config.json'))
        flags=runtime['flags'];enabled=('deterministic_algorithms','cudnn_deterministic','sdp_math')
        disabled=('deterministic_warn_only','cudnn_benchmark','cudnn_allow_tf32','matmul_allow_tf32',
                  'matmul_allow_fp16_reduced_precision_reduction','matmul_allow_bf16_reduced_precision_reduction',
                  'sdp_flash','sdp_memory_efficient','sdp_cudnn','sdp_math_allow_fp16_bf16_reduction')
        _need(runtime['mode']=='strict' and flags.get('cuda_initialized') is False
              and all(flags.get(k) is True for k in enabled) and all(flags.get(k) is False for k in disabled),
              'DINO extraction strict precision/math flags differ')
        torch.cuda.init()  # Strict flags above must precede the first CUDA context.
        torch.cuda.reset_peak_memory_stats(device)
        return _prepare_dinov2_cache(training_cache_root,output,teacher,device,
            {'runtime_before_teacher_load':runtime,'GPU_admission':admission,'strict_flags_verified_before_CUDA':True})


class PreparedDINOData:
    """Paired training-index reader for the new complete DINO cache only."""
    def __init__(self,cache_root,training_cache_root,*,expected_teacher_binding=None,expected_seal_sha256=None):
        from . import teacher as teacher_ops
        self.root=Path(cache_root).resolve();seal_path=self.root/'seal.json'
        self.seal_sha256=file_hash(seal_path)
        _need(expected_seal_sha256 is None or self.seal_sha256==expected_seal_sha256,'DINO cache seal changed')
        seal=read_json(seal_path);self.manifest=read_json(self.root/'manifest.json')
        _need(seal.get('schema')=='vsr_dinov2_cache_seal_v1' and seal['manifest_sha256']==file_hash(self.root/'manifest.json')
              and self.manifest.get('schema')=='vsr_dinov2_cache_v1' and self.manifest.get('status')=='completed'
              and self.manifest.get('test_pixels_accessed') is False,'DINO cache incomplete, old or changed')
        self.completion_receipt=read_json(self.root/'completion_receipt.json')
        _need(self.completion_receipt.get('schema')=='vsr_dinov2_cache_completion_v1'
              and self.completion_receipt.get('status')=='completed'
              and self.completion_receipt.get('seal_sha256')==self.seal_sha256
              and self.completion_receipt.get('optimizer_updates')==0
              and self.completion_receipt.get('test_pixels_accessed') is False,
              'DINO cache completion receipt is absent or changed')
        _dinov2_teacher_contract(self.manifest['teacher'])
        _need(expected_teacher_binding is None or self.manifest['teacher']==expected_teacher_binding,'Different DINO teacher requested')
        self.source=_dinov2_source_header(training_cache_root)
        _need(self.manifest['source']==self.source,'DINO cache does not match this paired data cache')
        _need(self.manifest['producer_code']=={'data.py':file_hash(Path(__file__)),
              'teacher.py':file_hash(Path(teacher_ops.__file__))},'DINO cache producer source changed')
        if 'extraction_receipt_sha256' in self.manifest:
            _need(file_hash(self.root/'extraction_receipt.json')==self.manifest['extraction_receipt_sha256'],
                  'DINO extraction environment receipt changed')
        self.ids=self.source['training_ids'];self.validation_ids=self.source['validation_ids']
        self.records=self.manifest['records'];self.header={k:v for k,v in self.manifest.items()
            if k not in ('status','records','test_pixels_accessed')}
        _need(self.manifest['feature_spec']==teacher_ops.dinov2_feature_spec()
              and self.manifest.get('target_normalized_before_loss') is False,'DINO time/spatial normalization differs')
        _need(seal['records']==self.records and len(self.records)==len(self.ids)+len(self.validation_ids),
              'DINO cache must retain every original train/val record')
        source_sets={'train':set(),'val':set()}
        for index,record_id in enumerate(self.ids+self.validation_ids):
            record=self.records[index]
            _need(record['index']==index and record['record_id']==record_id,'DINO cache order/ID differs')
            metadata=self._metadata(index)
            source_sets[metadata['binding']['split']].add(metadata['binding']['source_id'])
        _need(not source_sets['train'].intersection(source_sets['val']),'DINO source leaks across training/validation')
        _need(self.completion_receipt.get('teacher_frames')==len(self.records)*33
              and self.completion_receipt.get('training_count')==len(self.ids)
              and self.completion_receipt.get('validation_count')==len(self.validation_ids),
              'DINO cache receipt counts differ')
        self.metadata={'cache_root':str(self.root),'seal_sha256':self.seal_sha256,
                       'completion_receipt_sha256':file_hash(self.root/'completion_receipt.json'),
                       'training_count':len(self.ids),'validation_count':len(self.validation_ids),
                       'target_shape':[1,1008,768],'CPU_only':True}

    def _metadata(self,index):
        record=self.records[index];path=_inside(self.root/record['metadata']['path'],self.root)
        _need(file_hash(path)==record['metadata']['sha256'],'DINO metadata changed')
        metadata=read_json(path);original,inputs=_dinov2_source_record(self.source,index)
        expected={key:original[key] for key in ('record_id','source_id','source_sha256','split','source_indices',
                     'crop_tlhw','source_hw','resize_hw','temporal_stride','original_fps','output_fps')}
        binding=metadata['binding'];hr_sha=binding.get('HR_f32_sha256')
        _need(isinstance(hr_sha,str) and len(hr_sha)==64 and all(c in '0123456789abcdef' for c in hr_sha),'DINO cache lacks HR float32 digest')
        expected.update(inputs,HR_f32_sha256=hr_sha,header_digest=_digest(self.header))
        _need(binding==expected and metadata['feature_shape']==[1,1008,768]
              and metadata['dtype']=='torch.float32','DINO clip source/window/crop/teacher binding differs')
        return metadata

    def __len__(self):return len(self.ids)

    def record(self,index):
        _need(type(index) is int and 0<=index<len(self.records),'DINO record index outside cache')
        return copy.deepcopy(self._metadata(index))

    def __getitem__(self,index):
        index=operator.index(index)
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        return self.load(index)

    def load(self,index):
        """Explicit source index, including validation for offline calibration."""
        index=operator.index(index)
        if not 0<=index<len(self.records):raise IndexError(index)
        metadata=self._metadata(index);record=self.records[index]
        path=_inside(self.root/record['features']['path'],self.root)
        _need(file_hash(path)==record['features']['sha256'],'DINO feature artifact changed')
        payload=torch.load(path,map_location='cpu',weights_only=True);features=payload.get('features')
        _need(payload.get('binding')==metadata['binding'] and isinstance(features,torch.Tensor)
              and tuple(features.shape)==(1,1008,768) and features.dtype==torch.float32
              and features.device.type=='cpu' and not features.requires_grad and bool(torch.isfinite(features).all()),
              'DINO feature tensor/binding changed')
        return features.clone()

    def close(self):pass


class FixedCleanReader:
    """Read only completed clean33 assets; never decode raw test video here."""
    def __init__(self,dataset_manifest,*,expected_split,cohort_seal=None,path_map=None,expected_dataset_sha256=None):
        _need(expected_split in ('val','test'),'Fixed clean reader requires the actual val/test split')
        self.split=expected_split;self.mapping=path_map or {};self.path=Path(dataset_manifest).resolve()
        self.dataset_binding={'path':str(self.path),'bytes':self.path.stat().st_size,'sha256':file_hash(self.path)}
        _need(expected_dataset_sha256 is None or self.dataset_binding['sha256']==expected_dataset_sha256,'Clean dataset changed')
        self.dataset=read_json(self.path);self.seal_binding=None
        supplied=cohort_seal
        if supplied is None and self.dataset.get('schema')=='vsr_fixed_clean_dataset_v1':supplied=self.dataset['cohort_seal']['path']
        if isinstance(supplied,(str,Path)):
            seal_path=Path(supplied).resolve();self.seal=read_json(seal_path)
            self.seal_binding={'path':str(seal_path),'bytes':seal_path.stat().st_size,'sha256':file_hash(seal_path)}
        else:
            _need(isinstance(supplied,dict),'Explicit fixed clean cohort seal required')
            self.seal=copy.deepcopy(supplied)
        generic=self.dataset.get('schema')=='vsr_fixed_clean_dataset_v1'
        if expected_split=='val' and not generic:
            _need(self.dataset.get('status')=='completed_CPU_assets_no_predictions'
                  and self.dataset['config']==self.seal['config'],'Wrong existing val dataset status/config')
            validate_seal(self.seal,path_map=self.mapping)
            self.project_root=Path(self.seal['project_root'])
        else:
            _need(generic and self.dataset.get('status')=='completed_CPU_clean_assets'
                  and self.dataset.get('split')==expected_split,'Generic clean assets absent or wrong split')
            _need(self.seal.get('schema')=='vsr_fixed_clean_cohort_v1' and self.seal.get('split')==expected_split
                  and self.seal.get('seal_digest')==_digest({k:v for k,v in self.seal.items() if k!='seal_digest'}),
                  'Clean cohort metadata seal differs')
            if expected_split=='test':
                receipt=self.dataset.get('selection_receipt',{})
                _need(set(('path','bytes','sha256')).issubset(receipt)
                      and file_hash(receipt['path'])==receipt['sha256'] and Path(receipt['path']).stat().st_size==receipt['bytes']
                      and self.dataset.get('materialization',{}).get('selection_frozen_before_decode') is True,
                      'Test pixels need a fixed selection receipt before materialization')
            else:
                _need(self.dataset.get('materialization',{}).get('validation_assets_before_training') is True,
                      'Validation assets require an explicitly declared prospective CPU producer')
            declared=self.dataset['cohort_seal']
            _need(self.seal_binding is not None and declared==self.seal_binding,'Clean dataset/cohort file binding differs')
            self.project_root=self.path.parent
        _need(self.dataset.get('cohort_seal_digest')==self.seal['seal_digest'],'Clean dataset/cohort digest differs')
        self.cohort={x['record']['id']:x for x in self.seal['cohort']}
        self.assets={x['record_id']:x for x in self.dataset['assets']}
        _need(len(self.cohort)==len(self.seal['cohort'])==len(self.assets)==len(self.dataset['assets'])==32
              and set(self.assets)==set(self.cohort),'Clean reader requires all32 unique fixed clips')
        _need(all(len({str(self._asset_path(row[key])) for row in self.assets.values()})==32
                  for key in ('HR','clean')),'Clean records must have distinct per-record asset paths')
        for record_id,item in self.cohort.items():
            row=self.assets[record_id]
            _need(item['record']['split']==expected_split and row['source_id']==item['record']['source_id'],
                  'Clean source/split differs; never relabel test as val')
            indices=item['source_indices']
            _need(len(indices)==33 and indices==list(range(indices[0],indices[0]+33)),'Clean window differs')
            metadata=row.get('metadata',row.get('source_metadata',{}))
            _need(row.get('split',expected_split)==expected_split and metadata.get('split',expected_split)==expected_split
                  and metadata.get('source_id',row['source_id'])==row['source_id']
                  and metadata.get('record_id',record_id)==record_id
                  and metadata.get('source_indices',indices)==indices
                  and metadata.get('source_sha256',item['record'].get('sha256'))==item['record'].get('sha256'),
                  'Clean metadata source/window/split mismatch')
            for key in ('HR','clean'):
                spec=row[key];path=self._asset_path(spec)
                _need(path.is_file() and file_hash(path)==spec['sha256'],'Clean RGB asset changed')

    def _asset_path(self,spec):
        path=Path(spec['path'])
        if not path.is_absolute():path=self.project_root/path
        return _mapped_path(path,self.mapping)

    def describe(self):
        cohort=[];assets=[]
        for record_id,item in self.cohort.items():
            row=self.assets[record_id];metadata=row.get('metadata',row.get('source_metadata',{}))
            fps=metadata.get('output_fps',item.get('cached_header',{}).get('fps'))
            cohort.append({'record':{k:item['record'][k] for k in ('id','source_id','split')},
                           'source_indices':item['source_indices'],'cached_header':{'fps':fps}})
            assets.append({'record_id':record_id,'source_id':row['source_id'],
                **{key:{'path':str(self._asset_path(row[key])),'sha256':row[key]['sha256']} for key in ('HR','clean')}})
        return {'split':self.split,'source_split':self.split,'cohort':cohort,'assets':assets,
                'dataset_binding':copy.deepcopy(self.dataset_binding),'seal_binding':copy.deepcopy(self.seal_binding),
                'seal_digest':self.seal['seal_digest'],
                'data_independence':copy.deepcopy(self.seal.get('data_independence',
                    {'source_split':'manifest_source_ID_isolated','near_duplicate_status':'not_verified_by_this_reader'}))}

    def load(self,record_id,condition='clean',num_frames=33,start=0):
        import numpy as np
        _need(condition=='clean' and num_frames==33 and start==0,'Fixed clean reader forbids condition/window changes')
        _need(record_id in self.assets,'Clip outside frozen clean cohort')
        row,item=self.assets[record_id],self.cohort[record_id];result={}
        for key,name,shape in (('HR','hr',(1,3,33,128,224)),('clean','lr',(1,3,33,32,56))):
            spec=row[key];path=self._asset_path(spec)
            _need(file_hash(path)==spec['sha256'],'Clean RGB asset changed after reader construction')
            value=np.load(path,allow_pickle=False,mmap_mode='r')
            _need(value.dtype==np.float32 and value.shape==shape and np.isfinite(value).all()
                  and value.min()>=0 and value.max()<=1,'Invalid clean RGB asset')
            if 'rgb_f32_sha256' in spec:
                _need(hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()==spec['rgb_f32_sha256'],
                      'Clean float32 RGB digest differs')
            result[name]=torch.from_numpy(np.array(value,copy=True))
        metadata=copy.deepcopy(row.get('metadata',row.get('source_metadata',{})))
        fps=metadata.get('output_fps',item.get('cached_header',{}).get('fps'))
        _need(type(fps) in (int,float) and math.isfinite(fps) and fps>0,'Invalid clean source FPS')
        metadata.update(split=self.split,source_id=row['source_id'],record_id=record_id,sequence_id=record_id,
                        source_frame_indices=item['source_indices'],output_fps=fps)
        return result|{'fps':fps,'metadata':metadata}

    read=load
    def close(self):pass


def main(argv=None):
    """CPU-only metadata command; preparing model latents is an explicit API."""
    import argparse
    parser = argparse.ArgumentParser(description="VSR data manifests and sealed CPU cache checks")
    sub = parser.add_subparsers(dest="mode", required=True)
    catalog = sub.add_parser("manifest"); catalog.add_argument("--raw-root", type=Path, required=True); catalog.add_argument("--output", type=Path, required=True); catalog.add_argument("--seed", type=int, default=20261102)
    prep = sub.add_parser("prepare-cache"); prep.add_argument("--config", type=Path, required=True); prep.add_argument("--output", type=Path, required=True); prep.add_argument("--project-root", type=Path); prep.add_argument("--device", default="cpu"); prep.add_argument("--gpu-lock", type=Path); prep.add_argument("--gpu-admission", type=Path)
    cache = sub.add_parser("check-cache"); cache.add_argument("--cache-root", type=Path, required=True); cache.add_argument("--seal-sha256"); cache.add_argument("--verify-all", action="store_true")
    dino = sub.add_parser('prepare-dino-cache',help='Explicit fixed DINO teacher extraction; no training updates')
    dino.add_argument('--training-cache-root',type=Path,required=True);dino.add_argument('--output',type=Path,required=True)
    dino.add_argument('--teacher-source',type=Path,required=True);dino.add_argument('--teacher-weights',type=Path,required=True)
    dino.add_argument('--frame-batch-size',type=int,default=8);dino.add_argument('--device',default='cpu')
    dino.add_argument('--gpu-lock',type=Path);dino.add_argument('--gpu-admission',type=Path)
    dino.add_argument('--execute',action='store_true',help='Required to construct teacher and extract features')
    args = parser.parse_args(argv)
    if args.mode == "manifest":
        result = build_manifest(args.raw_root, seed=args.seed); write_json_new(args.output, result)
        print(json.dumps({"status":"manifest_written","output":str(args.output),"GPU_used":False}))
    elif args.mode == "prepare-cache":
        result = prepare_training_cache(read_json(args.config), args.output, project_root=args.project_root, device=args.device, gpu_lock=args.gpu_lock, gpu_admission=read_json(args.gpu_admission) if args.gpu_admission else None); print(json.dumps(result))
    elif args.mode == 'prepare-dino-cache':
        _need(args.execute,'DINO extraction requires explicit --execute')
        from .teacher import FrozenDINOv2
        teacher=FrozenDINOv2(args.teacher_source,args.teacher_weights,frame_batch_size=args.frame_batch_size)
        result=prepare_dinov2_cache(args.training_cache_root,args.output,teacher=teacher,device=args.device,
            gpu_lock=args.gpu_lock,gpu_admission=read_json(args.gpu_admission) if args.gpu_admission else None)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    else:
        dataset = PreparedTrainingData(args.cache_root, expected_seal_sha256=args.seal_sha256, verify_all=args.verify_all)
        print(json.dumps(dataset.metadata)); dataset.close()


if __name__ == "__main__":
    main()
