"""CPU-only streaming/cache fixtures: no official weights or CUDA initialization."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import weakref

import torch

from vsr import stage2_preparation as preparation
from vsr.audit import file_hash
from vsr.data import DATA_VERSION, CAPTION_PLACEHOLDER, _digest, build_manifest


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(latents_mean=[0.] * 16, latents_std=[1.] * 16)
        self.encodes, self.decodes, self.cleared = 0, 0, False

    def encode(self, pixels):
        assert pixels.shape == (1, 3, 33, 128, 224) and pixels.dtype == torch.float32
        self.encodes += 1
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: torch.zeros(1, 16, 9, 16, 28)))

    def decode(self, latent):
        assert latent.shape == (1, 16, 9, 16, 28)
        self.decodes += 1
        return SimpleNamespace(sample=torch.zeros(1, 3, 33, 128, 224))

    def clear_cache(self):
        self.cleared = True


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "new_project"
        self.root.mkdir()
        original_root = Path(preparation.__file__).resolve().parents[1]
        for name in preparation.ENCODER_FILES:
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((original_root / name).read_bytes())
        self.env = {"git_commit": "fixture-only", "code_sha256":
                    {name: file_hash(self.root / name) for name in preparation.ENCODER_FILES}}
        raw = self.root / "raw"
        raw.mkdir()
        for index in range(5):
            (raw / f"source{index:05d}_0_0to99.mp4").write_bytes(f"fixture source {index}".encode())
        self.manifest = build_manifest(raw)
        dump(self.root / "manifests/hq_vsr.json", self.manifest)
        self.model = {"repo_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                      "revision": "0fad780a534b6463e45facd96134c9f345acfa5b", "local_dir": str(self.root / "official")}
        dump(self.root / "official/vae/config.json", {"temperal_downsample": [False, True, True], "dim_mult": [1, 2, 4, 4]})
        dump(self.root / "official/transformer/config.json", {"patch_size": [1, 2, 2]})
        text_source = self.root / "runs/text_source"
        text_source.mkdir(parents=True)
        dump(text_source / "config.json", {"model": self.model})
        text = text_source / "fresh_text_embeddings.pt"
        torch.save({CAPTION_PLACEHOLDER: torch.zeros(1, 512, 4096, dtype=torch.bfloat16)}, text)
        self.config = {"stage": preparation.DIAGNOSTIC_STAGE, "seed": 20261102, "model": self.model,
            "data": {"manifest": "manifests/hq_vsr.json", "num_frames": 33, "height": 128, "width": 224,
                "scale": 4, "temporal_stride": 1, "fps": "preserve_original_no_temporal_resampling",
                "selection": "train", "train_limit": 2, "val_limit": 0, "save_input_videos": 0,
                "text_cache": {"source_run": str(text_source), "path": str(text), "sha256": file_hash(text)}},
            "training": {"variants": ["Base"], "preparation_variants": ["Base"], "lambda_align": 0., "lambda_relation": 0., "updates": 8},
            "conditioning": {"hri_enabled": False}, "reproducibility": {"mode": "strict"},
            "caption_policy": "fixed generic fixture", "_sha256": "fixture config digest",
            "stage2_contract": {"T5_load_allowed": False, "DINO_load_allowed": False,
                                "expected_training_clips": 2, "expected_validation_clips": 0}}
        self.root_patch = patch.object(preparation, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.temp.cleanup)
        self.vae = FakeVAE().eval()
        self.weak_rgb = []
        self.decode_order = []

    def run_directory(self, name, config=None):
        run = self.root / "runs" / name
        run.mkdir()
        dump(run / "config.json", config or self.config)
        dump(run / "environment.json", self.env)
        return run

    def decode(self, record, **options):
        self.assertTrue(all(reference() is None for reference in self.weak_rgb), "Previous RGB clip retained before next decode")
        self.assertEqual(options, {"num_frames": 33, "height": 128, "width": 224, "scale": 4, "seed": 20261102, "temporal_stride": 1})
        hr, lr, up = torch.full((1, 3, 33, 128, 224), .5), torch.full((1, 3, 33, 32, 56), .5), torch.full((1, 3, 33, 128, 224), .5)
        self.weak_rgb.extend(weakref.ref(value) for value in (hr, lr, up))
        self.decode_order.append(record["id"])
        metadata = {"data_version": DATA_VERSION, "record_id": record["id"], "source_id": record["source_id"],
            "source_path": record["path"], "source_sha256": record["sha256"], "split": record["split"],
            "seed": 20261102, "source_indices": list(range(33)), "temporal_stride": 1,
            "original_fps": 24., "output_fps": 24., "fps_policy": "preserve_original_no_temporal_resampling",
            "crop_tlhw": [0, 0, 128, 224], "valid_region": "entire_crop_no_padding", "caption": CAPTION_PLACEHOLDER,
            "degradation": {"scale": 4, "kernel": "torch_bicubic", "antialias": True, "align_corners": False,
                            "clamp_01": True, "temporal_parameters": "shared_for_all_frames"}}
        return {"hr": hr, "lr": lr, "lr_up": up, "fps": 24., "metadata": metadata, "cache_key": _digest(metadata)}

    def prepare(self, run, config=None):
        with patch.object(preparation, "decode_clip", side_effect=self.decode), \
             patch.object(preparation, "_load_vae", return_value=self.vae) as load, \
             patch("vsr.preflight.encode_text", side_effect=AssertionError("T5 forbidden")), \
             patch("vsr.train.DinoTeacher", side_effect=AssertionError("DINO forbidden")), \
             patch("vsr.train.FlowTeacher", side_effect=AssertionError("RAFT forbidden")):
            result = preparation.prepare_stage2(config or self.config, run, self.env)
        return result, load

    def seal(self, source):
        names = ["config.json", "selection.json", "correspondence.json", "fresh_training_cache.pt", "fresh_text_embeddings.pt"]
        for directory in sorted(source.glob("clip_*")):
            names += [str((directory / name).relative_to(source)) for name in ("metadata.json", "paired_rgb.pt", "fresh_latents.pt")]
        dump(source / "cache_asset_seal.json", {"source_run": str(source), "contains_task_initialization": False,
             "artifact_sha256": {name: file_hash(source / name) for name in names}})
        dump(source / "status.json", {"status": "completed"})

    def test_streaming_keeps_no_previous_RGB_and_returns_bounded_lazy_sequence(self):
        run = self.run_directory("fresh")
        result, load = self.prepare(run)
        clips, latents, text, features, flows, geometry, ntrain = result
        self.assertEqual((len(clips), len(latents), ntrain), (2, 2, 2))
        self.assertEqual(geometry["latent_grid"], [9, 16, 28])
        self.assertEqual(clips.max_cached, 2)
        self.assertEqual(clips.cached_indices, ())
        self.assertEqual(self.vae.encodes, 4)
        self.assertEqual(self.vae.decodes, 2)
        self.assertTrue(self.vae.cleared)
        self.assertTrue(all(reference() is None for reference in self.weak_rgb))
        self.assertIsNone(features); self.assertIsNone(flows)
        load.assert_called_once()
        self.assertEqual(text[CAPTION_PLACEHOLDER].dtype, torch.bfloat16)
        self.assertFalse(torch.cuda.is_initialized())
        with self.assertRaisesRegex(ValueError, "partial clip"):
            preparation.prepare_stage2(self.config, run, self.env)

    def test_sealed_reuse_under_new_root_skips_decode_VAE_and_rechecks_tensors(self):
        source = self.run_directory("producer")
        self.prepare(source)
        self.seal(source)
        config = copy.deepcopy(self.config); config["data"]["cache_from"] = str(source)
        run = self.run_directory("consumer", config)
        result, load = self.prepare(run, config)
        load.assert_not_called()
        self.assertEqual(result[0].run, source)
        self.assertEqual(result[0].max_cached, 2)
        self.assertTrue((run / "cache_reuse.json").is_file())
        # A forged new seal cannot hide a top-level/per-clip latent mismatch.
        payload = torch.load(source / "clip_1/fresh_latents.pt", weights_only=True)
        payload["low"] = payload["low"] + 1
        torch.save(payload, source / "clip_1/fresh_latents.pt")
        self.seal(source)
        run2 = self.run_directory("consumer_mismatch", config)
        with self.assertRaisesRegex(ValueError, "Per-clip/top-level"):
            self.prepare(run2, config)

    def test_cache_fails_before_payload_load_on_tampered_sealed_artifact(self):
        source = self.run_directory("producer")
        self.prepare(source); self.seal(source)
        with (source / "fresh_training_cache.pt").open("ab") as stream:
            stream.write(b"modified")
        config = copy.deepcopy(self.config); config["data"]["cache_from"] = str(source)
        run = self.run_directory("bad_consumer", config)
        with self.assertRaisesRegex(ValueError, "checksum/path"):
            self.prepare(run, config)

    def test_caption_hash_and_dtype_cannot_fallback(self):
        config = copy.deepcopy(self.config)
        config["data"]["text_cache"]["sha256"] = "0" * 64
        run = self.run_directory("bad_text_hash", config)
        with self.assertRaisesRegex(ValueError, "Text cache file SHA256"):
            self.prepare(run, config)
        path = Path(self.config["data"]["text_cache"]["path"])
        torch.save({CAPTION_PLACEHOLDER: torch.zeros(1, 512, 4096)}, path)
        config = copy.deepcopy(self.config); config["data"]["text_cache"]["sha256"] = file_hash(path)
        run = self.run_directory("bad_text_dtype", config)
        with self.assertRaisesRegex(ValueError, "CPU BF16"):
            self.prepare(run, config)

    def test_only_explicit_diagnostic_allows_empty_val_or_short_update_budget(self):
        preparation._base_config(self.config)
        wrong = copy.deepcopy(self.config); wrong["stage"] = "stage2_protocol33_noref_base"
        with self.assertRaisesRegex(ValueError, "train1641/val32"):
            preparation._base_config(wrong)
        formal = copy.deepcopy(wrong)
        formal["training"]["updates"] = 8192; formal["data"].pop("train_limit"); formal["data"]["val_limit"] = 32
        formal["stage2_contract"].update(expected_training_clips=1641, expected_validation_clips=32)
        preparation._base_config(formal)
        for field, value in [("expected_training_clips", 2), ("expected_validation_clips", 0)]:
            bad = copy.deepcopy(formal); bad["stage2_contract"][field] = value
            with self.assertRaisesRegex(ValueError, "train1641/val32"):
                preparation._base_config(bad)
        wrong = copy.deepcopy(self.config); wrong["training"]["preparation_variants"] = ["Base", "Align"]
        with self.assertRaisesRegex(ValueError, "Base-only"):
            preparation._base_config(wrong)

    def test_runtime_memory_tuple_is_canonicalized_before_exact_environment_check(self):
        from vsr.stage2_base_queue import json_record
        original_hashes = copy.deepcopy(self.env["code_sha256"])
        # The actual CUDA audit returns this tuple; persisted JSON returns list.
        self.env["free_total_bytes"] = (15271919616, 16606101504)
        rejected = self.run_directory("unserialized_environment")
        with self.assertRaisesRegex(ValueError, "exact config/environment"):
            self.prepare(rejected)
        self.env = json_record(self.env)
        accepted = self.run_directory("canonical_environment")
        result, load = self.prepare(accepted)
        self.assertEqual(self.env["free_total_bytes"], [15271919616, 16606101504])
        self.assertEqual(self.env["code_sha256"], original_hashes)
        self.assertEqual(len(result[0]), 2)
        load.assert_called_once()
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
