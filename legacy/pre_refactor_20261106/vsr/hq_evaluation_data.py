"""Frozen internal HQ-VSR paired-evaluation reader with explicit LR stress modes.

The test partition is this project's source-grouped holdout, not an official
HQ-VSR test benchmark. Construction reads the sealed catalogue only. Pixel
access is confined to an explicit load() call; this module selects no model,
checkpoint, loss weight or restoration result.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import torch

from .data import decode_clip, validate_manifest
from .robustness import MODES, apply_robustness, protocol_spec


EXPECTED_INTERNAL_TEST_CLIPS = 211


def _positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _video_noise_seed(seed, record_id):
    # Exclude candidate mode and model identity: noise/combined therefore use
    # the same random field for the same fixed video and clip geometry.
    value = json.dumps(["hq-internal-evaluation-noise-v1", seed, record_id], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**63)


class HQEvaluationReader:
    """Catalogue-only initialization; fixed paired pixels are read by load().

    split='test' always exposes all 211 project-held-out videos. There is no
    test-limit argument. val_limit affects only validation and defaults to the
    predeclared 32 videos in exactly train.select_records' SHA256 order.
    """

    def __init__(self, project_root, *, split="test", val_limit=32, height=128,
                 width=224, temporal_stride=4, seed=20261102,
                 modes=("identity", "blur", "noise", "compression", "combined")):
        if split not in ("test", "val"):
            raise ValueError("HQ evaluation reader accepts val/test only; train is forbidden")
        height, width = _positive_integer(height, "height"), _positive_integer(width, "width")
        temporal_stride = _positive_integer(temporal_stride, "temporal_stride")
        if height % 4 or width % 4:
            raise ValueError("HQ paired target geometry must be divisible by x4 scale")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
            raise ValueError("Evaluation seed must be an integer in [0,2**63)")
        if split == "val":
            val_limit = _positive_integer(val_limit, "val_limit")
        modes = tuple(modes)
        if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
            raise ValueError("Provide distinct predeclared robustness modes")
        self.root = Path(project_root).resolve()
        manifest_path = (self.root / "manifests/hq_vsr.json").resolve()
        if not manifest_path.is_relative_to(self.root):
            raise ValueError("HQ manifest must remain inside the selected independent project")
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        validate_manifest(manifest)
        if manifest.get("dataset") != "HQ-VSR":
            raise ValueError("Sealed catalogue must identify HQ-VSR")
        selected = [copy.deepcopy(row) for row in manifest["records"] if row["split"] == split]
        if split == "test":
            if len(selected) != EXPECTED_INTERNAL_TEST_CLIPS:
                raise ValueError("This frozen project protocol requires the complete 211-video internal test partition")
            # Stable ID ordering is for enumeration only, never score selection.
            selected.sort(key=lambda row: row["id"])
        else:
            selected.sort(key=lambda row: hashlib.sha256(f"val:{seed}:{row['id']}".encode()).hexdigest())
            if len(selected) < val_limit:
                raise ValueError("Validation catalogue is smaller than the predeclared val_limit")
            selected = selected[:val_limit]
        self.manifest = manifest
        self.manifest_sha256 = manifest["manifest_sha256"]
        self.manifest_file_sha256 = hashlib.sha256(raw).hexdigest()
        self.split, self.height, self.width, self.temporal_stride, self.seed = split, height, width, temporal_stride, seed
        self.modes = modes
        self._source_records = {row["id"]: row for row in selected}
        self.noise_seeds = {row["id"]: _video_noise_seed(seed, row["id"]) for row in selected}
        self._dataset_modes = {f"HQ-{split}-{mode}": mode for mode in modes}
        self.records = {dataset: {row["id"]: copy.deepcopy(row) for row in selected} for dataset in self._dataset_modes}
        specifications = {}
        for mode in modes:
            specification = protocol_spec(mode, seed=0)
            # Fixed operation settings can be frozen before pixel access. The
            # per-video seed table below supplies actual seeds, without a fake
            # common seed being mistaken for an applied input parameter.
            specification["noise"].pop("seed")
            specification["noise"]["seed_source"] = "noise_seed_by_record_id[record_id]"
            specifications[mode] = specification
        self.metadata = {
            "reader": "HQEvaluationReader", "source_dataset": "HQ-VSR", "split": split,
            "split_origin": "project-created source-grouped internal holdout; not an official HQ-VSR test split",
            "official_test_split": False, "manifest_path": str(manifest_path), "manifest_sha256": self.manifest_sha256,
            "manifest_file_sha256": self.manifest_file_sha256, "selected_record_count": len(selected),
            "selected_record_ids": [row["id"] for row in selected], "complete_test_partition": split == "test",
            "required_internal_test_count": EXPECTED_INTERNAL_TEST_CLIPS, "test_limit_supported": False,
            "validation_limit": val_limit if split == "val" else None,
            "validation_selection": "SHA256('val:seed:id') ascending, fixed prefix; identical to train.select_records",
            "data_seed": seed, "target_hw": [height, width], "temporal_stride": temporal_stride, "scale": 4,
            "robustness_modes": list(modes), "candidate_specifications": specifications,
            "noise_seed_by_record_id": dict(self.noise_seeds),
            "noise_seed_rule": "SHA256(['hq-internal-evaluation-noise-v1',seed,record_id]) first8bytes mod2**63; independent of candidate/model",
            "pixel_decode_in_constructor": False, "training_allowed": False, "checkpoint_selection_performed": False,
            "metrics_computed": False, "parameters_selected_on_test": False,
        }

    def load(self, dataset, record_id, num_frames=9, start=0):
        """Return paired sampled HR and fixed stress-processed LR.

        start=0 means use decode_clip's already fixed seeded window, rather
        than source frame zero. Other starts are rejected to prevent silently
        changing this predeclared comparison window.
        """
        if not isinstance(start, int) or isinstance(start, bool) or start != 0:
            raise ValueError("HQ evaluation start must be zero; decode_clip fixes the seeded source window")
        num_frames = _positive_integer(num_frames, "num_frames")
        if dataset not in self._dataset_modes:
            raise KeyError(f"Unknown internal evaluation dataset: {dataset}")
        if record_id not in self._source_records:
            raise KeyError(f"Video is outside this fixed {self.split} selection: {record_id}")
        record = copy.deepcopy(self._source_records[record_id])
        if record["split"] != self.split or record["split"] == "train":
            raise ValueError("Evaluation record cannot come from the training partition")
        clip = decode_clip(record, num_frames=num_frames, height=self.height, width=self.width,
                           scale=4, seed=self.seed, temporal_stride=self.temporal_stride)
        if clip["metadata"].get("record_id") != record_id or clip["metadata"].get("source_id") != record["source_id"]:
            raise ValueError("Decoded video identity differs from the sealed record")
        if clip["metadata"].get("source_sha256") != record["sha256"] or clip["metadata"].get("split") != self.split:
            raise ValueError("Decoded source hash or split differs from the sealed record")
        if not math.isfinite(float(clip["fps"])) or clip["fps"] <= 0:
            raise ValueError("Prepared paired input must have a valid sampled FPS")
        mode = self._dataset_modes[dataset]
        noise_seed = self.noise_seeds[record_id]
        lr, provenance = apply_robustness(clip["lr"], mode, fps=clip["fps"], seed=noise_seed)
        if lr.shape != clip["lr"].shape or not bool(torch.isfinite(lr).all()):
            raise ValueError("Robustness changed paired LR geometry or produced nonfinite pixels")
        metadata = {**copy.deepcopy(clip["metadata"]), "dataset": dataset, "source_dataset": "HQ-VSR",
                    "sequence_id": record_id, "manifest_sha256": self.manifest_sha256,
                    "manifest_file_sha256": self.manifest_file_sha256, "raw_catalogue_record": record,
                    "internal_split_origin": self.metadata["split_origin"], "official_test_split": False,
                    "source_frame_indices": list(clip["metadata"]["source_indices"]),
                    "requested_num_frames": num_frames, "start": 0,
                    "frame_selection": "decode_clip deterministic seeded window; same sampling/crop/scale/seed for every candidate and model",
                    "benchmark_prefix_explicit": False, "paired_scale": 4,
                    "prepared_lr_hw": list(lr.shape[-2:]), "prepared_hr_hw": list(clip["hr"].shape[-2:]),
                    "fps": float(clip["fps"]), "fps_source": "paired sampled input FPS = original source FPS / temporal_stride",
                    "base_pair_cache_key": clip["cache_key"], "robustness_mode": mode,
                    "robustness_noise_seed": noise_seed, "robustness": provenance,
                    "input_binding": {"manifest_sha256": self.manifest_sha256, "base_pair_cache_key": clip["cache_key"],
                                      "record_id": record_id, "robustness_mode": mode, "noise_seed": noise_seed,
                                      "robustness_protocol_sha256": provenance.get("protocol_sha256"),
                                      "base_lr_rgb_sha256": provenance.get("canonical_input_rgb_f32_sha256"),
                                      "robust_lr_rgb_sha256": provenance.get("output_rgb_f32_sha256")},
                    "training_allowed": False, "checkpoint_selection_performed": False,
                    "gpu_used": False, "metrics_computed": False, "parameters_selected_on_test": False}
        return {"lr": lr, "hr": clip["hr"], "fps": float(clip["fps"]), "metadata": metadata}

    read = load

    def close(self):
        """No persistent decoder, archive handle or GPU resource is held."""
        return None
