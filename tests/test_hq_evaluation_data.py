"""Synthetic sealed catalogues and mocked pixels only; no real HQ test reads."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from vsr.data import _digest, select_debug_records
from vsr.hq_evaluation_data import HQEvaluationReader


def sealed_fixture(root, test_count=211, val_count=40):
    records = []
    for i, split in enumerate(["train"]*3 + ["val"]*val_count + ["test"]*test_count):
        source = f"{i:011d}"
        video_id = source + "_0_0to100"
        records.append({"id": video_id, "source_id": source, "split": split, "dataset": "HQ-VSR",
            "path": str(root/"nonexistent-synthetic-pixels"/(video_id+".mp4")), "bytes": 123,
            "sha256": hashlib.sha256(video_id.encode()).hexdigest(), "caption": "A high quality video."})
    manifest = {"schema_version": 1, "dataset": "HQ-VSR", "data_version": "hq-vsr-paired-v1", "seed": 20261102,
                "records": records, "split_origin": "synthetic unit test catalogue, no dataset pixels"}
    manifest["debug_records"] = select_debug_records(manifest)
    manifest["manifest_sha256"] = _digest(manifest)
    directory = root/"manifests"; directory.mkdir()
    (directory/"hq_vsr.json").write_text(json.dumps(manifest))
    return manifest


def fake_decode(record, *, num_frames, height, width, scale, seed, temporal_stride):
    hr = torch.full((1,3,num_frames,height,width), .5)
    lr = torch.full((1,3,num_frames,height//scale,width//scale), .5)
    metadata = {"record_id": record["id"], "source_id": record["source_id"], "source_sha256": record["sha256"],
        "source_path": record["path"], "split": record["split"], "source_indices": list(range(7,7+num_frames*temporal_stride,temporal_stride)),
        "temporal_stride": temporal_stride, "seed": seed, "window_seed": 77, "crop_tlhw": [8,12,height,width],
        "source_hw": [512,896], "resize_hw": [height+16,width+24], "original_fps": 30.,
        "output_fps": 30/temporal_stride, "fps_policy": "preserve_sampled_input_fps",
        "valid_region": "entire_crop_no_padding", "degradation": {"scale": scale, "kernel": "torch_bicubic", "antialias": True}}
    return {"hr": hr, "lr": lr, "fps": 30/temporal_stride, "metadata": metadata, "cache_key": _digest(metadata)}


def fake_robustness(video, mode, *, fps, seed):
    return video.clone(), {"protocol": {"mode": mode, "noise": {"seed": seed}}, "protocol_sha256": mode+"-spec",
                          "canonical_input_rgb_f32_sha256": "same-base-lr", "output_rgb_f32_sha256": mode+"-pixels",
                          "fps_float": fps, "parameters_selected_on_test": False}


class HQEvaluationReaderTests(unittest.TestCase):
    def test_constructor_exposes_all_211_test_videos_without_decoding_or_test_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); manifest = sealed_fixture(root)
            with patch("vsr.hq_evaluation_data.decode_clip", side_effect=AssertionError("Constructor may not read pixels")), \
                 patch("vsr.hq_evaluation_data.apply_robustness", side_effect=AssertionError("Constructor may not transform pixels")), \
                 patch("torch.cuda.init", side_effect=AssertionError("GPU forbidden")):
                reader = HQEvaluationReader(root, val_limit=1)
            expected = {row["id"] for row in manifest["records"] if row["split"] == "test"}
            self.assertEqual(set(reader.records), {"HQ-test-"+mode for mode in ("identity","blur","noise","compression","combined")})
            self.assertTrue(all(set(records) == expected and len(records) == 211 for records in reader.records.values()))
            self.assertFalse(reader.metadata["official_test_split"])
            self.assertTrue(reader.metadata["complete_test_partition"])
            self.assertFalse(reader.metadata["test_limit_supported"])
            self.assertEqual(set(reader.metadata["noise_seed_by_record_id"]), expected)
            self.assertNotIn("seed", reader.metadata["candidate_specifications"]["noise"]["noise"])
            self.assertIsNone(reader.close())
            with self.assertRaises(TypeError):
                HQEvaluationReader(root, test_limit=1)

    def test_validation_order_matches_training_sha_rule_and_train_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); manifest = sealed_fixture(root)
            reader = HQEvaluationReader(root, split="val")
            expected = [row["id"] for row in sorted((row for row in manifest["records"] if row["split"] == "val"),
                key=lambda row: hashlib.sha256(f"val:20261102:{row['id']}".encode()).hexdigest())[:32]]
            self.assertEqual(list(reader.records["HQ-val-identity"]), expected)
            self.assertEqual(reader.metadata["selected_record_count"], 32)
            with self.assertRaises(ValueError):
                HQEvaluationReader(root, split="train")
            with self.assertRaises(ValueError):
                HQEvaluationReader(root, modes=("noise", "noise"))

    def test_candidate_windows_source_degradation_and_noise_seeds_are_model_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); sealed_fixture(root)
            reader = HQEvaluationReader(root, height=32, width=48)
            video_id = next(iter(reader.records["HQ-test-identity"]))
            with patch("vsr.hq_evaluation_data.decode_clip", side_effect=fake_decode) as decoder, \
                 patch("vsr.hq_evaluation_data.apply_robustness", side_effect=fake_robustness) as robustness, \
                 patch("torch.cuda.init", side_effect=AssertionError("GPU forbidden")):
                results = [reader.load("HQ-test-"+mode, video_id) for mode in ("identity", "noise", "combined")]
                repeated = reader.load("HQ-test-noise", video_id)
                self.assertEqual(len({call.kwargs["seed"] for call in robustness.call_args_list}), 1)
                self.assertTrue(all(call.kwargs["fps"] == 7.5 for call in robustness.call_args_list))
                self.assertTrue(all(call.kwargs == {"num_frames": 9, "height": 32, "width": 48, "scale": 4,
                    "seed": 20261102, "temporal_stride": 4} for call in decoder.call_args_list))
            for result in results:
                metadata = result["metadata"]
                self.assertEqual(result["fps"], 7.5)
                self.assertEqual(metadata["source_frame_indices"], list(range(7,40,4)))
                self.assertEqual(metadata["crop_tlhw"], [8,12,32,48])
                self.assertEqual(metadata["valid_region"], "entire_crop_no_padding")
                self.assertEqual(metadata["degradation"]["kernel"], "torch_bicubic")
                self.assertFalse(metadata["metrics_computed"])
                self.assertEqual(metadata["source_sha256"], reader.records["HQ-test-identity"][video_id]["sha256"])
            self.assertEqual(results[1]["metadata"]["input_binding"], repeated["metadata"]["input_binding"])
            for other in results[1:]:
                torch.testing.assert_close(results[0]["hr"], other["hr"], rtol=0, atol=0)
                self.assertEqual(results[0]["metadata"]["base_pair_cache_key"], other["metadata"]["base_pair_cache_key"])

    def test_seal_count_window_and_split_guards_fire_before_fake_pixel_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); manifest = sealed_fixture(root)
            reader = HQEvaluationReader(root)
            video_id = next(iter(reader.records["HQ-test-identity"]))
            train_id = next(row["id"] for row in manifest["records"] if row["split"] == "train")
            with patch("vsr.hq_evaluation_data.decode_clip", side_effect=AssertionError("No pixel read allowed")):
                with self.assertRaises(ValueError):
                    reader.load("HQ-test-identity", video_id, start=1)
                with self.assertRaises(KeyError):
                    reader.load("HQ-test-identity", train_id)
            corrupted = copy.deepcopy(manifest); corrupted["records"][0]["split"] = "test"
            (root/"manifests/hq_vsr.json").write_text(json.dumps(corrupted))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                HQEvaluationReader(root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); sealed_fixture(root, test_count=210)
            with self.assertRaisesRegex(ValueError, "211-video"):
                HQEvaluationReader(root)


if __name__ == "__main__":
    unittest.main()
