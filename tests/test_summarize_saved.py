"""Synthetic on-disk artifacts verify pairing, never supply experiment scores."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from vsr.summarize_saved import (LPIPS, TLP, TOF, load_saved_result,
                                 summarize_results, text_report)
from vsr.audit import file_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_run(root, name, gain=0):
    run = root / name
    directory = run / "Base" / "step_000010"
    write(run / "config.json", {"seed": 5, "training": {"seed": 7}})
    rgb = {"status": "completed", "sampling": {"seeds": [42, 43], "steps": 20, "cfg": 1., "hri": False, "matched_noise_across_modes": True},
           "scope": "synthetic test fixture only", "metric_protocol": {"source": "lossless"}, "per_video": []}
    perceptual = {"format": "vsr_saved_perceptual_v2", "status": "completed",
                  "audit": {"source": "synthetic test fixture only"}, "per_video": []}
    for i, split in enumerate(("val", "val", "train")):
        record = f"record{i}"
        metric = lambda value: {"psnr_rgb_db": value, "ssim_rgb": value / 100}
        score = lambda value: {LPIPS: value, "tecogan": {TLP: value, TOF: value}}
        write(run / f"clip_{i}" / "metadata.json", {"record_id": record, "source_id": f"source{i}", "split": split,
            "binding": {"data_cache_key": f"crop{i}", "manifest_sha256": "manifest", "official_revision": "official"}})
        (run / f"clip_{i}" / "paired_rgb.pt").write_bytes(f"synthetic target {i}".encode())
        directory.mkdir(parents=True, exist_ok=True)
        samples = {}
        for seed in (42, 43):
            path = directory / f"synthetic_output_{i}_{seed}.bin"
            path.write_bytes(f"synthetic output {i} {seed} {gain}".encode())
            samples[seed] = path
        rgb["per_video"].append({"record_id": record, "source_id": f"source{i}", "split": split, "clip_index": i,
            "bicubic": metric(10), "modes": {"correct": {"seeds": [{"seed": s, "noise_seed": s, "rgb_path": str(samples[s]), "metrics": metric(20 + gain)} for s in (42, 43)]},
                                               "disabled": {"seeds": [{"seed": s, "noise_seed": s, "metrics": metric(5)} for s in (42, 43)]}}})
        perceptual["per_video"].append({"record_id": record, "source_id": f"source{i}", "split": split, "bicubic": score(.8),
            "source_rgb_sha256": file_hash(run / f"clip_{i}/paired_rgb.pt"),
            "source_metadata_sha256": file_hash(run / f"clip_{i}/metadata.json"),
            "modes": {"correct": {"seeds": [{"seed": s, "noise_seed": s, "rgb_path": str(samples[s]),
                         "rgb_sha256": file_hash(samples[s]), "metrics": score(.4 - gain / 100)} for s in (42, 43)]}}})
    write(directory / "metrics.json", rgb)
    perceptual["rgb_metrics_sha256"] = file_hash(directory / "metrics.json")
    write(directory / "perceptual_metrics_v2.json", perceptual)
    return directory


class SavedSummaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_discard_train_diagnostics_and_keep_complete_matched_seeds(self):
        base = load_saved_result(make_run(self.root, "base"))
        candidate = load_saved_result(make_run(self.root, "candidate", gain=1))
        report = summarize_results({"Base": base, "Candidate": candidate}, "Base", replicates=100)
        self.assertEqual(report["video_count"], 2)
        self.assertEqual(report["source_count"], 2)
        self.assertEqual(report["provenance"]["Base"]["training_seed"], 7)
        self.assertAlmostEqual(report["comparisons"]["Candidate"][LPIPS]["delta_video_macro"], -.01)
        self.assertEqual(report["condition_controls"]["Base"]["disabled"]["sample_count"], 4)
        self.assertNotIn("record2", report["comparisons"]["Candidate"][LPIPS]["video_deltas"])
        self.assertIn("uncertainty across training seeds", text_report(report))

    def test_refuse_different_crop_metric_protocol_or_bicubic(self):
        base = load_saved_result(make_run(self.root, "base"))
        candidate = load_saved_result(make_run(self.root, "candidate"))
        for key in ("bindings", "compatibility", "bicubic"):
            bad = copy.deepcopy(candidate)
            if key == "bindings":
                bad[key]["record0"]["data_cache_key"] = "different_crop"
            elif key == "compatibility":
                bad[key]["steps"] = 30
            else:
                bad[key][0]["metrics"][LPIPS] = .9
            with self.assertRaises(ValueError):
                summarize_results({"Base": base, "Candidate": bad}, "Base", replicates=100)

    def test_refuse_missing_perceptual_seed_and_reference_pool(self):
        directory = make_run(self.root, "incomplete")
        path = directory / "perceptual_metrics_v2.json"
        content = json.loads(path.read_text())
        content["per_video"][0]["modes"]["correct"]["seeds"].pop()
        write(path, content)
        with self.assertRaises(ValueError):
            load_saved_result(directory)

    def test_refuse_wrong_source_failed_score_changed_pixels_or_actual_noise(self):
        for kind in ("source", "failed", "pixels", "noise", "matched"):
            directory = make_run(self.root, kind)
            path = directory / "perceptual_metrics_v2.json"
            content = json.loads(path.read_text())
            if kind == "source":
                content["per_video"][0]["source_id"] = "wrong_upload"
            elif kind == "failed":
                content["status"] = "failed"
            elif kind == "pixels":
                Path(content["per_video"][0]["modes"]["correct"]["seeds"][0]["rgb_path"]).write_bytes(b"changed")
            else:
                rgb_path = directory / "metrics.json"
                rgb = json.loads(rgb_path.read_text())
                if kind == "noise":
                    rgb["per_video"][0]["modes"]["disabled"]["seeds"][0]["noise_seed"] += 100
                else:
                    rgb["sampling"]["matched_noise_across_modes"] = False
                write(rgb_path, rgb)
                content["rgb_metrics_sha256"] = file_hash(rgb_path)
            write(path, content)
            with self.assertRaises(ValueError):
                load_saved_result(directory)
        directory = make_run(self.root, "reference")
        path = directory / "metrics.json"
        content = json.loads(path.read_text())
        content["sampling"]["hri"] = True
        write(path, content)
        perceptual_path = directory / "perceptual_metrics_v2.json"
        perceptual = json.loads(perceptual_path.read_text())
        perceptual["rgb_metrics_sha256"] = file_hash(path)
        write(perceptual_path, perceptual)
        with self.assertRaises(ValueError):
            load_saved_result(directory)


if __name__ == "__main__":
    unittest.main()
