"""CPU factory fixtures only: no real external pixels, weights or GPU jobs."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from vsr.evaluate_external import (build_parser, paired_metrics, quantize_rgb,
                                   run_external_evaluation)
from vsr.perceptual_metrics import LPIPS_KEY, TLP_KEY, TLP100_KEY, TOF_KEY


class FakeReader:
    def __init__(self):
        self.records = {"YouHQ40": {"000": {}, "001": {}}, "VideoLQ": {"000": {}}}
        self.loads, self.targets, self.closed = [], {}, False

    def load(self, dataset, sequence, num_frames, start):
        assert start == 0
        self.loads.append((dataset, sequence, num_frames, start))
        offset = int(sequence)*.03
        lr = torch.full((1, 3, num_frames, 2, 2), .3143 + offset)
        lr[:, :, 0] = .1173 + offset
        result = {"lr": lr, "fps": 8.0, "metadata": {"dataset": dataset, "sequence_id": sequence,
            "source_frame_indices": list(range(num_frames)), "benchmark_prefix_explicit": True,
            "source_frame_count": 33, "native_lr_hw": [2, 2], "fps_source": "synthetic_container"}}
        if dataset == "YouHQ40":
            target = torch.full((1, 3, num_frames, 8, 8), .4317 + offset)
            target[:, :, 0] = .1299 + offset
            self.targets[(dataset, sequence)] = target
            result["hr"] = target
        return result

    def close(self):
        self.closed = True


class FakePerceptual:
    def __init__(self):
        self.calls = []

    def evaluate(self, prediction, target, include_temporal=True):
        assert include_temporal is True
        for value in (prediction, target):
            assert torch.equal(value, value.mul(255).round().div(255)), "Metrics must use quantized PNG pixels"
        self.calls.append((prediction.clone(), target.clone()))
        errors = (prediction-target).abs().mean(dim=(0, 1, 3, 4)).tolist()
        indices = list(range(2, prediction.shape[2]-2))
        pairs = [{"indices": [a, b], TLP_KEY: float(a), TLP100_KEY: float(a*100), TOF_KEY: float(b)}
                 for a, b in zip(indices, indices[1:])]
        return {LPIPS_KEY: sum(errors)/len(errors), "lpips_per_frame": errors,
            "audit": {"synthetic_fixture": True}, "lpips_protocol": {"synthetic_fixture": True},
            "tecogan": {"per_pair": pairs, "protocol": {"cutfr": 2, "synthetic_fixture": True}}}


class FakeMUSIQ:
    def __init__(self):
        self.calls = []

    def evaluate_video(self, video, batch_size=1):
        assert batch_size == 1
        assert torch.equal(video, video.mul(255).round().div(255))
        self.calls.append(video.clone())
        scores = video.mean(dim=(0, 1, 3, 4)).mul(100).tolist()
        return {"mean": sum(scores)/len(scores), "per_frame_scores": scores, "synthetic_fixture": True}


class Factories:
    def __init__(self, root):
        self.root = root
        self.reader = FakeReader()
        self.perceptual, self.musiq = FakePerceptual(), FakeMUSIQ()
        self.events, self.restores, self.models = [], [], []
        self.configs = {}
        self.paths = {}
        for label, reference in [("Base", False), ("RefBoth", True)]:
            path = root/"runs"/"selected_by_internal_val"/label/"checkpoint.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("synthetic-checkpoint-"+label).encode())
            self.paths[label] = path
            self.configs[str(path)] = {"reference": {"enabled": reference}, "model": {"fixture": True}}
        manifest = root/"manifests/external.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"synthetic_fixture": True}))

    def preview(self, path, **kwargs):
        path = str(path)
        return {"path": path, "checkpoint_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "config": self.configs[path], "variant": Path(path).parent.name, "step": 2048}

    def caption(self, config, caption, checkpoint, root, **kwargs):
        self.events.append(("caption", str(checkpoint)))
        return torch.zeros(1, 2, 4), {"synthetic_fixture": True, "caption": caption}

    def model(self, path, **kwargs):
        self.events.append(("model", str(path)))
        model = nn.Linear(1, 1)
        self.models.append(model)
        return model, self.preview(path)

    def restore(self, model, vae, lrv, text, config, **kwargs):
        assert set(kwargs) == {"steps", "seed", "tile_hw", "overlap", "reference_image", "reference_index"}
        supplied = kwargs["reference_image"]
        self.restores.append({**kwargs, "reference": supplied, "lr": lrv.clone(), "text": text.clone(),
                              "reference_trained": config["reference"]["enabled"]})
        up = F.interpolate(lrv[0].permute(1, 0, 2, 3), scale_factor=4,
                           mode="bicubic", align_corners=False, antialias=True)
        video = up.permute(1, 0, 2, 3).unsqueeze(0)
        video = (video + .0137*(kwargs["seed"]-40) + (float(supplied.mean())*.1 if supplied is not None else 0)).clamp(0, 1)
        return video, {"synthetic_fixture": True, "output_shape": list(video.shape), "temporal_windowing": False}

    def mapping(self):
        return {"reader": lambda root, **kwargs: self.reader,
            "preview": self.preview, "caption": self.caption, "model": self.model,
            "vae": lambda config, **kwargs: nn.Linear(1, 1), "restore": self.restore,
            "perceptual": lambda root, **kwargs: self.perceptual,
            "musiq": lambda root, **kwargs: self.musiq}


class EvaluateExternalTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def execute(self, root, factories, **kwargs):
        return run_external_evaluation(root, factories.paths, datasets=("YouHQ40", "VideoLQ"), frames=9,
            seeds=(42, 43), steps=2, tile_hw=(16, 16), overlap=0,
            device="cpu", metric_device="cpu", run_dir=root/"runs/external_fixture", factories=factories.mapping(), **kwargs)

    def test_checkpoint_and_seed_aggregation_reference_condition_and_png_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            factories = Factories(root)
            summary = self.execute(root, factories)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["training_updates"], 0)
            self.assertFalse(summary["external_tuning"])
            self.assertFalse(summary["best_seed_selected"])
            self.assertEqual(len(factories.models), 2)
            self.assertIsNot(factories.models[0], factories.models[1])
            for position, event in enumerate(factories.events):
                if event[0] == "model":
                    self.assertEqual(factories.events[position-1], ("caption", event[1]))
            run = root/"runs/external_fixture"
            base_rows = []
            for label in ("Base", "RefBoth"):
                for sequence in ("000", "001"):
                    row = json.loads((run/f"results/{label}/YouHQ40/{sequence}/metrics.json").read_text())
                    expected_modes = {"no_reference"} if label == "Base" else {"hri_on", "hri_off"}
                    self.assertEqual(set(row["modes"]), expected_modes)
                    for mode, values in row["modes"].items():
                        self.assertEqual([entry["seed"] for entry in values["seeds"]], [42, 43])
                        for entry in values["seeds"]:
                            metrics = entry["metrics"]
                            self.assertEqual(metrics["all_frames"]["frame_indices"], list(range(9)))
                            self.assertEqual(metrics["matched_nonreference"]["frame_indices"], list(range(1, 9)))
                            self.assertEqual(metrics["reference_frame_separate"]["frame_indices"], [0])
                            self.assertEqual(entry["png_count"], 9)
                            self.assertIsNone(entry["preview"])
                            path = run/entry["png_directory"]/"frame_00000000.png"
                            with Image.open(path) as image:
                                pixels = np.array(image)
                            self.assertEqual(pixels.dtype, np.uint8)
                            self.assertEqual(pixels.shape, (8, 8, 3))
                        expected_mean = sum(entry["metrics"]["matched_nonreference"]["psnr_rgb_db"] for entry in values["seeds"])/2
                        self.assertAlmostEqual(values["mean"]["matched_nonreference"]["psnr_rgb_db"], expected_mean)
                    if label == "Base":
                        base_rows.append(row["modes"]["no_reference"]["mean"]["matched_nonreference"]["psnr_rgb_db"])
                real = json.loads((run/f"results/{label}/VideoLQ/000/metrics.json").read_text())
                self.assertEqual(set(real["modes"]), {"no_reference"} if label == "Base" else {"hri_off"})
                self.assertTrue(all(entry["metrics"]["kind"] == "unpaired" for mode in real["modes"].values() for entry in mode["seeds"]))
                self.assertTrue(all(not entry["hri_supplied"] for mode in real["modes"].values() for entry in mode["seeds"]))
                self.assertNotIn("hr", {Path(item["path"]).parent.name for item in real["artifacts"]})
            macro = summary["results"]["Base"]["YouHQ40"]["modes"]["no_reference"]["matched_nonreference"]["psnr_rgb_db"]
            self.assertAlmostEqual(macro, sum(base_rows)/2)
            for call in factories.restores:
                if call["reference_image"] is not None:
                    reference = call["reference_image"]
                    self.assertEqual(tuple(reference.shape), (1, 3, 1, 8, 8))
                    self.assertEqual(call["reference_index"], 0)
                    target = next(target for target in factories.reader.targets.values()
                                  if torch.equal(reference, target[:, :, :1]))
                    self.assertNotEqual(reference.data_ptr(), target.data_ptr())
                else:
                    self.assertIsNone(call["reference_index"])
            self.assertTrue(factories.reader.closed)
            self.assertFalse(list(run.rglob("*.pt")), "No huge float32 RGB outputs saved by default")

    def test_temporal_filter_keeps_original_pairs_not_joined_selected_frames(self):
        prediction = torch.zeros(1, 3, 9, 8, 8)
        target = torch.ones_like(prediction)
        evaluator = FakePerceptual()
        result = paired_metrics(prediction, target, evaluator, excluded_reference=3)
        self.assertEqual(len(evaluator.calls), 1)
        self.assertEqual(evaluator.calls[0][0].shape[2], 9)
        self.assertEqual(result["all_frames"]["temporal_pair_indices"], [[2, 3], [3, 4], [4, 5], [5, 6]])
        self.assertEqual(result["matched_nonreference"]["temporal_pair_indices"], [[4, 5], [5, 6]])
        self.assertNotIn([2, 4], result["matched_nonreference"]["temporal_pair_indices"])
        self.assertEqual(result["reference_frame_separate"]["temporal_pair_count"], 0)
        self.assertIsNone(result["reference_frame_separate"][TLP_KEY])

    def test_resume_seals_skip_inference_and_reject_artifact_or_protocol_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = Factories(root)
            summary = self.execute(root, first)
            resumed = Factories(root)
            again = self.execute(root, resumed)
            self.assertEqual(len(again["resumed_complete_sequences"]), 6)
            self.assertEqual(resumed.restores, [])
            self.assertEqual(resumed.models, [])
            self.assertEqual(resumed.reader.loads, [])
            self.assertEqual(again["results"], summary["results"])
            changed = Factories(root)
            with self.assertRaisesRegex(ValueError, "protocol differs"):
                run_external_evaluation(root, changed.paths, datasets=("YouHQ40", "VideoLQ"), frames=9,
                    seeds=(42,), steps=2, tile_hw=(16, 16), overlap=0, device="cpu", metric_device="cpu",
                    run_dir=root/"runs/external_fixture", factories=changed.mapping())
            self.assertEqual(json.loads((root/"runs/external_fixture/status.json").read_text())["status"], "completed")
            path = next((root/"runs/external_fixture/results/Base/YouHQ40/000").rglob("*.png"))
            path.write_bytes(b"tampered-output")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                self.execute(root, Factories(root))

    def test_quantizer_and_cli_defaults(self):
        video = torch.full((1, 3, 2, 8, 8), .5001)
        quantized, measured = quantize_rgb(video)
        self.assertEqual(quantized.dtype, torch.uint8)
        self.assertTrue(torch.equal(quantized, torch.full_like(quantized, 128)))
        self.assertTrue(torch.equal(measured, quantized.float()/255))
        args = build_parser().parse_args(["--checkpoint", "Base=/own/Base/checkpoint.pt", "--checkpoint", "Ref=/own/Ref/checkpoint.pt"])
        self.assertEqual(args.frames, 17)
        self.assertEqual(args.seeds, [42, 43])
        self.assertEqual(args.steps, 20)
        self.assertEqual(args.tile, "256x448")
        self.assertEqual(args.overlap, 64)


if __name__ == "__main__":
    unittest.main()
