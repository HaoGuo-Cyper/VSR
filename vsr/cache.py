"""Bounded lazy RGB clips and verified reuse of this experiment's frozen caches.

No adapter, optimizer, scheduler or legacy experiment state is loaded here.
Production cache paths are confined to /home/hugo/Projects/VSR/runs. A temporary
root can be supplied explicitly by unit tests; all resolved child paths remain
confined to their selected run, including symlinks.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
import hashlib
import json
import math
import operator
from pathlib import Path
import re

import torch

from .data import DATA_VERSION, MANIFEST_DATA_VERSION, _digest, validate_manifest

PRODUCTION_RUNS_ROOT = Path("/home/hugo/Projects/VSR/runs")
LEGACY_PROJECT_ROOT = Path("/home/hugo/Projects/ReprVSR-pretrained")
ENCODER_SOURCE_FILES = ("vsr/data.py", "vsr/preflight.py")
_GEOMETRY_FIELDS = ("num_frames", "height", "width", "scale", "fps")
_CLIP_NAME = re.compile(r"clip_(\d+)$")


def _within(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Cache path escapes selected run/root: {path}")
    return path


def _run_path(experiment_dir, allowed_root):
    run = _within(experiment_dir, allowed_root)
    if run.is_relative_to(LEGACY_PROJECT_ROOT):
        raise ValueError("Legacy-project experiment caches are not allowed")
    if not run.is_dir():
        raise FileNotFoundError(f"Fresh experiment directory missing: {run}")
    return run


def _json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clip_directories(run):
    indexed = []
    for path in run.iterdir():
        match = _CLIP_NAME.fullmatch(path.name)
        if match and path.is_dir():
            _within(path, run)
            indexed.append((int(match[1]), path))
    indexed.sort(key=lambda x: x[0])
    if not indexed or [i for i, _ in indexed] != list(range(len(indexed))):
        raise ValueError("Cache clip directories must be nonempty contiguous clip_0,clip_1,...")
    for _, directory in indexed:
        for name in ("paired_rgb.pt", "metadata.json"):
            path = _within(directory / name, run)
            if not path.is_file():
                raise FileNotFoundError(f"Clip cache file missing: {path}")
    return [path for _, path in indexed]


def _metadata(directory, run):
    raw = _json(_within(directory / "metadata.json", run))
    if not isinstance(raw, dict) or not isinstance(raw.get("binding"), dict):
        raise ValueError("Each cached clip requires metadata and a binding")
    binding = raw["binding"]
    metadata = {key: value for key, value in raw.items() if key != "binding"}
    if binding.get("data_cache_key") != _digest(metadata):
        raise ValueError("Clip metadata digest does not match data_cache_key")
    return metadata, binding


def _temporal_contract(metadata, *, expected_stride=None, expected_frames=None):
    """Read historical v1 or modern v2 paired timing without resampling it."""
    version = metadata.get("data_version")
    if version not in {MANIFEST_DATA_VERSION, DATA_VERSION}:
        raise ValueError("Unsupported cached processing data version")
    stride = metadata.get("temporal_stride", 1)
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("Cached temporal_stride must be a positive integer")
    if version == MANIFEST_DATA_VERSION and stride != 1:
        raise ValueError("Historical v1 processing requires temporal_stride=1")
    if expected_stride is not None and stride != expected_stride:
        raise ValueError("Cached temporal_stride differs from configured stride")
    indices = metadata.get("source_indices", [])
    if (not isinstance(indices, list) or not indices
            or any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in indices)):
        raise ValueError("Cached source frame indices must be nonempty nonnegative integers")
    if expected_frames is not None and len(indices) != expected_frames:
        raise ValueError("Cached frame count differs from configured clip length")
    if indices != [indices[0] + i * stride for i in range(len(indices))]:
        raise ValueError("Cached source frame indices differ from temporal_stride protocol")
    original, output = metadata.get("original_fps"), metadata.get("output_fps")
    if (not isinstance(original, (int, float)) or isinstance(original, bool) or not math.isfinite(original) or original <= 0
            or not isinstance(output, (int, float)) or isinstance(output, bool) or not math.isfinite(output) or output <= 0
            or not math.isclose(output, original / stride, rel_tol=1e-7, abs_tol=1e-8)):
        raise ValueError("Cached original/output FPS mismatch for temporal stride")
    expected_policy = "preserve_original_no_temporal_resampling" if stride == 1 else "preserve_sampled_input_fps"
    if metadata.get("fps_policy") != expected_policy:
        raise ValueError("Cached FPS policy mismatch for temporal stride")
    return stride, float(output)


class PairedClipSequence(Sequence):
    """Lazy CPU clips with a hard limit of two cached RGB payloads.

    Returned dictionaries mirror decode_clip: hr/lr/lr_up, metadata, fps,
    binding, cache_key and an all-true pixel valid_mask. Treat returned tensors
    and metadata as read-only. Constructor discovers files without loading RGB.
    Call validate_cache before reusing a frozen latent/teacher cache.
    """

    def __init__(self, experiment_dir, *, max_cached=2, allowed_root=PRODUCTION_RUNS_ROOT):
        if not isinstance(max_cached, int) or isinstance(max_cached, bool) or not 1 <= max_cached <= 2:
            raise ValueError("LRU max_cached must be 1 or 2")
        self.run = _run_path(experiment_dir, allowed_root)
        self.directories = tuple(_clip_directories(self.run))
        self.max_cached = max_cached
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.directories)

    @property
    def cached_indices(self):
        return tuple(self._cache)

    def clear(self):
        self._cache.clear()

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self))) ]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        directory = self.directories[index]
        metadata, binding = _metadata(directory, self.run)
        _, fps = _temporal_contract(metadata)
        path = _within(directory / "paired_rgb.pt", self.run)
        if binding.get("paired_rgb_sha256") and _file_hash(path) != binding["paired_rgb_sha256"]:
            raise ValueError("Paired RGB file digest mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or any(key not in payload for key in ("hr", "lr", "lr_up")):
            raise ValueError("paired_rgb.pt must contain hr,lr,lr_up tensors")
        t = len(metadata.get("source_indices", []))
        crop = metadata.get("crop_tlhw", [])
        if len(crop) != 4 or t < 1:
            raise ValueError("Cached metadata lacks clip geometry")
        h, w = crop[-2:]
        scale = metadata["degradation"]["scale"]
        if (any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in (h, w, scale))
                or h % scale or w % scale):
            raise ValueError("Cached spatial geometry must use positive dimensions divisible by scale")
        shapes = {"hr": (1, 3, t, h, w), "lr": (1, 3, t, h // scale, w // scale), "lr_up": (1, 3, t, h, w)}
        for key, shape in shapes.items():
            value = payload[key]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not value.is_floating_point():
                raise ValueError(f"Invalid {key} tensor shape or dtype")
            if value.device.type != "cpu" or value.requires_grad or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{key} must be finite frozen CPU RGB")
            if float(value.min()) < 0 or float(value.max()) > 1:
                raise ValueError(f"{key} RGB lies outside [0,1]")
        clip = {key: payload[key] for key in ("hr", "lr", "lr_up")}
        clip.update(metadata=metadata, fps=fps, binding=binding, cache_key=binding["data_cache_key"],
                    valid_mask=torch.ones((1, t, h, w), dtype=torch.bool))
        self._cache[index] = clip
        self._cache.move_to_end(index)
        while len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)
        return clip


def validate_cache(experiment_dir, expected_config, expected_manifest_sha256,
                   expected_encoder_hashes, *, allowed_root=PRODUCTION_RUNS_ROOT,
                   expected_manifest=None):
    """Verify frozen-cache compatibility before a new official initialization.

    Only model/data/encoder contracts are compared: optimizer budgets, loss
    weights and variant names may differ. expected_encoder_hashes must include
    current vsr/data.py and vsr/preflight.py hashes. Per-clip metadata is checked
    without loading all RGB tensors. Optional sealed expected_manifest additionally
    verifies every source ID, raw-video SHA and split against its record.
    """
    run = _run_path(experiment_dir, allowed_root)
    config = _json(_within(run / "config.json", run))
    environment = _json(_within(run / "environment.json", run))
    selection = _json(_within(run / "selection.json", run))
    for key in ("repo_id", "revision"):
        if config.get("model", {}).get(key) != expected_config.get("model", {}).get(key):
            raise ValueError(f"Frozen cache official model {key} mismatch")
    if config.get("reproducibility",{}).get("mode","seeded")!=expected_config.get("reproducibility",{}).get("mode","seeded"):
        raise ValueError("Frozen cache numerical reproducibility/precision contract mismatch")
    if config.get("seed") != expected_config.get("seed"):
        raise ValueError("Frozen cache data/window seed mismatch")
    for key in _GEOMETRY_FIELDS:
        if config.get("data", {}).get(key) != expected_config.get("data", {}).get(key):
            raise ValueError(f"Frozen cache data geometry {key} mismatch")
    stride = config.get("data", {}).get("temporal_stride", 1)
    expected_stride = expected_config.get("data", {}).get("temporal_stride", 1)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in (stride, expected_stride)):
        raise ValueError("Configured temporal_stride must be a positive integer")
    if stride != expected_stride:
        raise ValueError("Frozen cache data geometry temporal_stride mismatch")
    if selection.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("Frozen cache manifest mismatch")
    if selection.get("test_accessed") is not False:
        raise ValueError("Test data must not be used in a training cache")
    if any(key not in expected_encoder_hashes for key in ENCODER_SOURCE_FILES):
        raise ValueError("Current encoder hashes must include data.py and preflight.py")
    recorded_hashes = environment.get("code_sha256", {})
    for key, value in expected_encoder_hashes.items():
        if not value or recorded_hashes.get(key) != value:
            raise ValueError(f"Frozen cache encoder source mismatch: {key}")
    records = None
    if expected_manifest is not None:
        validate_manifest(expected_manifest)
        if expected_manifest["manifest_sha256"] != expected_manifest_sha256:
            raise ValueError("Expected manifest object/digest mismatch")
        records = {r["id"]: r for r in expected_manifest["records"]}
    training = selection.get("training_ids", [])
    validation = selection.get("validation_ids", [])
    ordered_ids = training + validation
    if not training or len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("Cache selection must include distinct training/validation clip IDs")
    directories = _clip_directories(run)
    if len(directories) != len(ordered_ids):
        raise ValueError("Cache clip count differs from recorded selection")
    for index, directory in enumerate(directories):
        metadata, binding = _metadata(directory, run)
        if metadata.get("record_id") != ordered_ids[index]:
            raise ValueError("Cache clip order differs from selected record IDs")
        expected_split = "train" if index < len(training) else "val"
        if metadata.get("split") != expected_split:
            raise ValueError("Training/validation cache split mismatch")
        if metadata.get("data_version") != DATA_VERSION or metadata.get("seed") != config["seed"]:
            raise ValueError("Cached data version or seed mismatch")
        if binding.get("manifest_sha256") != expected_manifest_sha256:
            raise ValueError("Per-clip manifest binding mismatch")
        revision = binding.get("official_revision", binding.get("model_revision"))
        if revision != expected_config["model"]["revision"]:
            raise ValueError("Per-clip official revision binding mismatch")
        if config.get("_sha256") and binding.get("config_sha256") != config["_sha256"]:
            raise ValueError("Per-clip originating config binding mismatch")
        for key, value in expected_encoder_hashes.items():
            if binding.get("code_sha256", {}).get(key) != value:
                raise ValueError(f"Per-clip encoder binding mismatch: {key}")
        _temporal_contract(metadata, expected_stride=stride, expected_frames=config["data"]["num_frames"])
        crop = metadata.get("crop_tlhw", [])
        if len(crop) != 4 or crop[-2:] != [config["data"]["height"], config["data"]["width"]]:
            raise ValueError("Per-clip crop geometry mismatch")
        if metadata.get("valid_region") != "entire_crop_no_padding":
            raise ValueError("Lazy clip cache does not support padded crops")
        degradation = metadata.get("degradation", {})
        expected_degradation = {"scale": config["data"]["scale"], "kernel": "torch_bicubic", "antialias": True,
                                "align_corners": False, "clamp_01": True, "temporal_parameters": "shared_for_all_frames"}
        if degradation != expected_degradation:
            raise ValueError("Cached degradation contract mismatch")
        if records is not None:
            record = records.get(metadata["record_id"])
            if record is None or any(metadata.get(a) != record[b] for a, b in (("source_id", "source_id"), ("source_sha256", "sha256"), ("split", "split"))):
                raise ValueError("Cached source/split differs from sealed manifest")
    return {"status": "verified", "source_run": str(run), "clip_count": len(directories),
            "training_clip_count": len(training), "validation_clip_count": len(validation),
            "manifest_sha256": expected_manifest_sha256, "official_revision": expected_config["model"]["revision"],
            "encoder_source_hashes": dict(expected_encoder_hashes), "legacy_state_read": False,
            "temporal_stride": stride, "processing_data_version": DATA_VERSION,
            "tensor_loading": "lazy LRU<=2; weights_only=True CPU", "manifest_records_checked": records is not None}
