"""Cache compatibility and bounded memory tested on new temporary fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from vsr.cache import PairedClipSequence, validate_cache
from vsr.data import DATA_VERSION, _digest

HASHES = {"vsr/data.py": "a" * 64, "vsr/preflight.py": "b" * 64}
MANIFEST = "c" * 64


def dump(path, value):
    path.write_text(json.dumps(value))


def fixture(root, count=3):
    run = root / "stage_fresh_fixture"
    run.mkdir()
    config = {"seed": 42, "model": {"repo_id": "official", "revision": "official-revision"},
              "data": {"num_frames": 9, "height": 16, "width": 24, "scale": 4, "fps": "preserve_original"}, "_sha256": "config-origin-hash"}
    dump(run / "config.json", config)
    dump(run / "environment.json", {"code_sha256": HASHES})
    dump(run / "selection.json", {"manifest_sha256": MANIFEST, "training_ids": [f"clip{i}" for i in range(count)], "validation_ids": [], "test_accessed": False})
    for i in range(count):
        directory = run / f"clip_{i}"
        directory.mkdir()
        metadata = {"data_version": DATA_VERSION, "record_id": f"clip{i}", "source_id": f"source{i}", "source_sha256": str(i) * 64,
                    "split": "train", "seed": 42, "source_indices": list(range(10, 19)), "crop_tlhw": [0, 0, 16, 24],
                    "original_fps": 23.976, "output_fps": 23.976, "fps_policy": "preserve_original_no_temporal_resampling",
                    "valid_region": "entire_crop_no_padding", "degradation": {"scale": 4, "kernel": "torch_bicubic", "antialias": True, "align_corners": False, "clamp_01": True, "temporal_parameters": "shared_for_all_frames"}}
        binding = {"data_cache_key": _digest(metadata), "manifest_sha256": MANIFEST, "official_revision": "official-revision", "code_sha256": HASHES, "config_sha256": "config-origin-hash"}
        dump(directory / "metadata.json", {**metadata, "binding": binding})
        torch.save({"hr": torch.full((1, 3, 9, 16, 24), i / count), "lr": torch.zeros(1, 3, 9, 4, 6), "lr_up": torch.zeros(1, 3, 9, 16, 24)}, directory / "paired_rgb.pt")
    return run, config


class CacheTests(unittest.TestCase):
    def test_lazy_weights_only_cpu_lru_and_interface(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, _ = fixture(root)
            real_load = torch.load
            with patch("vsr.cache.torch.load", wraps=real_load) as loader:
                clips = PairedClipSequence(run, allowed_root=root)
                loader.assert_not_called()
                self.assertEqual(len(clips), 3)
                first = clips[0]
                clips[1]
                self.assertEqual(clips.cached_indices, (0, 1))
                self.assertIs(clips[0], first)
                clips[2]
                self.assertEqual(clips.cached_indices, (0, 2))
                clips[1]
                self.assertEqual(clips.cached_indices, (2, 1))
                self.assertEqual(loader.call_count, 4)
                for call in loader.call_args_list:
                    self.assertEqual(call.kwargs, {"map_location": "cpu", "weights_only": True})
                self.assertEqual(first["fps"], 23.976)
                self.assertTrue(bool(first["valid_mask"].all()))
                self.assertEqual(first["metadata"]["record_id"], "clip0")
                self.assertNotIn("binding", first["metadata"])
                self.assertEqual(clips[-1]["metadata"]["record_id"], "clip2")
                clips.clear()
                self.assertEqual(clips.cached_indices, ())
                with self.assertRaises(IndexError):
                    clips[3]

    def test_validation_accepts_budget_change_but_rejects_encoder_geometry_or_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, config = fixture(root)
            expected = copy.deepcopy(config)
            expected["training"] = {"variants": ["Align"], "updates": 4096}
            report = validate_cache(run, expected, MANIFEST, HASHES, allowed_root=root)
            self.assertEqual(report["clip_count"], 3)
            self.assertFalse(report["legacy_state_read"])
            wrong = copy.deepcopy(expected)
            wrong["data"]["height"] = 32
            with self.assertRaisesRegex(ValueError, "geometry height"):
                validate_cache(run, wrong, MANIFEST, HASHES, allowed_root=root)
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                validate_cache(run, expected, "wrong", HASHES, allowed_root=root)
            with self.assertRaisesRegex(ValueError, "encoder source mismatch"):
                validate_cache(run, expected, MANIFEST, {**HASHES, "vsr/data.py": "wrong"}, allowed_root=root)
            wrong["model"]["revision"] = "different"
            with self.assertRaisesRegex(ValueError, "model revision"):
                validate_cache(run, wrong, MANIFEST, HASHES, allowed_root=root)

    def test_perclip_metadata_tampering_binding_and_split_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, config = fixture(root)
            path = run / "clip_0/metadata.json"
            raw = json.loads(path.read_text())
            raw["output_fps"] = 8
            dump(path, raw)
            with self.assertRaisesRegex(ValueError, "metadata digest"):
                PairedClipSequence(run, allowed_root=root)[0]
            metadata = {k: v for k, v in raw.items() if k != "binding"}
            raw["binding"]["data_cache_key"] = _digest(metadata)
            dump(path, raw)
            with self.assertRaisesRegex(ValueError, "FPS mismatch"):
                validate_cache(run, config, MANIFEST, HASHES, allowed_root=root)
            raw["output_fps"] = raw["original_fps"]
            raw["split"] = "test"
            raw["binding"]["data_cache_key"] = _digest({k: v for k, v in raw.items() if k != "binding"})
            dump(path, raw)
            with self.assertRaisesRegex(ValueError, "cache split mismatch"):
                validate_cache(run, config, MANIFEST, HASHES, allowed_root=root)

    def test_path_confinement_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, _ = fixture(root)
            with self.assertRaisesRegex(ValueError, "escapes"):
                PairedClipSequence(run)
            payload = run / "clip_0/paired_rgb.pt"
            payload.unlink()
            external = root / "external.pt"
            external.write_bytes(b"must not load")
            payload.symlink_to(external)
            with self.assertRaisesRegex(ValueError, "escapes"):
                PairedClipSequence(run, allowed_root=root)

    def test_rgb_shape_and_finiteness_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, _ = fixture(root)
            payload = run / "clip_0/paired_rgb.pt"
            data = torch.load(payload, weights_only=True)
            data["hr"][0, 0, 0, 0, 0] = float("nan")
            torch.save(data, payload)
            with self.assertRaisesRegex(ValueError, "finite"):
                PairedClipSequence(run, allowed_root=root)[0]
            with self.assertRaisesRegex(ValueError, "1 or 2"):
                PairedClipSequence(run, allowed_root=root, max_cached=3)


if __name__ == "__main__":
    unittest.main()
