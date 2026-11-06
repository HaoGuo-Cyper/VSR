"""Prospective stride-v2 tests on actual synthetic video and cache fixtures.

These tests do not change the interpretation of already completed stride-1
pilots or constitute video-restoration evidence.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from vsr.cache import PairedClipSequence, validate_cache
from vsr.data import (DATA_VERSION, MANIFEST_DATA_VERSION, _digest,
                      build_manifest, decode_clip, degrade_x4, write_video)


HASHES = {"vsr/data.py": "a" * 64, "vsr/preflight.py": "b" * 64}


def json_save(path, value):
    path.write_text(json.dumps(value))


def video_fixture(root, frames=41):
    path = root / f"abc_defghij_0_0to{frames}.mp4"
    pixels = torch.zeros(1, 3, frames, 32, 48)
    for index in range(frames):
        pixels[:, :, index] = index / max(frames, 1) * .5
        pixels[:, 0, index, 8:16, index % 40:index % 40+8] = 1
    write_video(path, pixels, 24.)
    manifest = build_manifest(root, seed=42)
    return manifest, manifest["records"][0]


def cache_fixture(root, clip, manifest):
    run = root / "new_stride_run"
    run.mkdir()
    config = {"seed": 42, "model": {"repo_id": "official", "revision": "official-revision"}, "_sha256": "config-digest",
              "data": {"num_frames": 9, "height": 32, "width": 48, "scale": 4,
                       "fps": "preserve_sampled_input_fps", "temporal_stride": 4}}
    json_save(run / "config.json", config)
    json_save(run / "environment.json", {"code_sha256": HASHES})
    json_save(run / "selection.json", {"manifest_sha256": manifest["manifest_sha256"],
        "training_ids": [clip["metadata"]["record_id"]], "validation_ids": [], "test_accessed": False})
    directory = run / "clip_0"
    directory.mkdir()
    binding = {"data_cache_key": _digest(clip["metadata"]), "manifest_sha256": manifest["manifest_sha256"],
               "official_revision": "official-revision", "config_sha256": "config-digest", "code_sha256": HASHES}
    json_save(directory / "metadata.json", {**clip["metadata"], "binding": binding})
    torch.save({key: clip[key] for key in ("hr", "lr", "lr_up")}, directory / "paired_rgb.pt")
    return run, config


def update_metadata(path, changes, remove=()):
    value = json.loads(path.read_text())
    value.update(changes)
    for name in remove:
        value.pop(name, None)
    value["binding"]["data_cache_key"] = _digest({key: item for key, item in value.items() if key != "binding"})
    json_save(path, value)


class TemporalStrideTests(unittest.TestCase):
    def test_stride_four_selects_exact_source_times_and_shared_paired_degradation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, record = video_fixture(root)
            whole = decode_clip(record, num_frames=41, height=32, width=48, seed=42)
            sampled = decode_clip(record, num_frames=9, height=32, width=48, seed=42, temporal_stride=4)
            indices = sampled["metadata"]["source_indices"]
            self.assertEqual(indices, list(range(indices[0], indices[0]+33, 4)))
            torch.testing.assert_close(sampled["hr"], whole["hr"][:, :, indices], rtol=0, atol=0)
            lr, up = degrade_x4(sampled["hr"])
            torch.testing.assert_close(sampled["lr"], lr, rtol=0, atol=0)
            torch.testing.assert_close(sampled["lr_up"], up, rtol=0, atol=0)
            self.assertEqual(sampled["metadata"]["original_fps"], 24)
            self.assertEqual(sampled["fps"], 6)
            self.assertEqual(sampled["metadata"]["output_fps"], 6)
            self.assertEqual(sampled["metadata"]["fps_policy"], "preserve_sampled_input_fps")
            self.assertAlmostEqual((indices[-1]-indices[0])/24, (len(indices)-1)/sampled["fps"])
            self.assertEqual(sampled["hr"].shape[2], sampled["lr"].shape[2])
            self.assertEqual(manifest["data_version"], MANIFEST_DATA_VERSION)

    def test_stride_one_reproduces_old_window_pixels_fps_and_catalogue_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, record = video_fixture(root)
            default = decode_clip(record, num_frames=9, height=32, width=48, seed=42)
            explicit = decode_clip(record, num_frames=9, height=32, width=48, seed=42, temporal_stride=1)
            whole = decode_clip(record, num_frames=41, height=32, width=48, seed=42)
            old_seed = int(_digest(["window", 42, record["id"]])[:16], 16)
            old_start = old_seed % (41-9+1)
            self.assertEqual(default["metadata"]["source_indices"], list(range(old_start, old_start+9)))
            torch.testing.assert_close(default["hr"], whole["hr"][:, :, old_start:old_start+9], rtol=0, atol=0)
            torch.testing.assert_close(default["hr"], explicit["hr"], rtol=0, atol=0)
            self.assertEqual(default["cache_key"], explicit["cache_key"])
            self.assertEqual(default["fps"], 24)
            self.assertEqual(default["metadata"]["fps_policy"], "preserve_original_no_temporal_resampling")
            self.assertEqual(default["metadata"]["data_version"], "hq-vsr-paired-v2")
            with patch("vsr.data.DATA_VERSION", "unrelated-future-processing-version"):
                unchanged_catalogue = build_manifest(root, seed=42)
            self.assertEqual(manifest, unchanged_catalogue)
            self.assertEqual(manifest["data_version"], "hq-vsr-paired-v1")

    def test_short_source_is_rejected_without_padding_or_duplicate_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            _, record = video_fixture(Path(directory), frames=32)
            with self.assertRaisesRegex(ValueError, "33 required"):
                decode_clip(record, num_frames=9, height=32, width=48, temporal_stride=4)
            for invalid in (0, -1, 1.5, True):
                with self.assertRaises(ValueError):
                    decode_clip(record, num_frames=9, height=32, width=48, temporal_stride=invalid)

    def test_new_cache_requires_configured_stride_and_sampled_fps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            manifest, record = video_fixture(raw)
            clip = decode_clip(record, num_frames=9, height=32, width=48, seed=42, temporal_stride=4)
            run, config = cache_fixture(root, clip, manifest)
            report = validate_cache(run, config, manifest["manifest_sha256"], HASHES, allowed_root=root, expected_manifest=manifest)
            self.assertEqual(report["temporal_stride"], 4)
            lazy = PairedClipSequence(run, allowed_root=root)
            self.assertEqual(lazy[0]["fps"], 6)
            wrong = copy.deepcopy(config)
            wrong["data"]["temporal_stride"] = 1
            with self.assertRaisesRegex(ValueError, "temporal_stride mismatch"):
                validate_cache(run, wrong, manifest["manifest_sha256"], HASHES, allowed_root=root)
            metadata = run / "clip_0/metadata.json"
            update_metadata(metadata, {"output_fps": 24})
            with self.assertRaisesRegex(ValueError, "FPS mismatch"):
                validate_cache(run, config, manifest["manifest_sha256"], HASHES, allowed_root=root)
            with self.assertRaisesRegex(ValueError, "FPS mismatch"):
                PairedClipSequence(run, allowed_root=root)[0]
            update_metadata(metadata, {"output_fps": 6, "fps_policy": "preserve_original_no_temporal_resampling"})
            with self.assertRaisesRegex(ValueError, "FPS policy"):
                validate_cache(run, config, manifest["manifest_sha256"], HASHES, allowed_root=root)

    def test_historical_v1_remains_readable_but_is_forbidden_for_new_training_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            manifest, record = video_fixture(raw)
            clip = decode_clip(record, num_frames=9, height=32, width=48, seed=42)
            run, config = cache_fixture(root, clip, manifest)
            config["data"].pop("temporal_stride")
            config["data"]["fps"] = "preserve_original"
            json_save(run / "config.json", config)
            update_metadata(run / "clip_0/metadata.json", {"data_version": MANIFEST_DATA_VERSION}, remove=("temporal_stride",))
            lazy = PairedClipSequence(run, allowed_root=root)
            self.assertEqual(lazy[0]["fps"], 24)
            torch.testing.assert_close(lazy[0]["hr"], clip["hr"], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "data version"):
                validate_cache(run, config, manifest["manifest_sha256"], HASHES, allowed_root=root)
            update_metadata(run / "clip_0/metadata.json", {"data_version": DATA_VERSION, "temporal_stride": 4})
            with self.assertRaisesRegex(ValueError, "source frame indices"):
                PairedClipSequence(run, allowed_root=root)[0]


if __name__ == "__main__":
    unittest.main()
