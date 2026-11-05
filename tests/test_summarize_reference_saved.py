"""Synthetic saved-record tests; no weights, actual pixels or GPU work."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from vsr.audit import file_hash
from vsr.metric_pools import PAIRED_METRIC_KEYS, POOL_NAMES
from vsr.summarize_reference_saved import (TEMPORAL, load_reference_result,
    summarize_reference_results, text_report)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def pools(value, reference, frames=9):
    frame_rows = [{"psnr_rgb_db": value + i * .01, "ssim_rgb": value / 100 + i * .001,
        "mse_rgb_01": .1 / (value + i), "original_frame_index": i} for i in range(frames)]
    indices = {"all_frames": list(range(frames)), "matched_nonreference": [i for i in range(frames) if i != reference],
        "reference_frame_separate": [reference]}
    result = {}
    for name in POOL_NAMES:
        selected = indices[name]
        pairs = [{"indices": [i, i + 1], **{key: value * .0001 + i * .00001 for key in TEMPORAL}}
            for i in range(2, frames - 3) if i in selected and i + 1 in selected]
        result[name] = {key: sum(frame_rows[i][key] for i in selected) / len(selected)
            for key in ("psnr_rgb_db", "ssim_rgb", "mse_rgb_01")}
        result[name].update(lpips_alex_v0_1=value / 100,
            lpips_per_frame=[value / 100 for i in selected],
            **{key: sum(pair[key] for pair in pairs) / len(pairs) if pairs else None for key in TEMPORAL},
            frame_count=len(selected), frame_indices=selected, per_frame=[frame_rows[i] for i in selected],
            temporal_pair_count=len(pairs), temporal_pair_indices=[pair["indices"] for pair in pairs],
            per_temporal_pair=pairs)
    return result


class Fixture:
    def __init__(self, root):
        self.root = root
        self.source = root / "runs/own_new_rgb_cache"
        self.run = root / "runs/own_fresh_HRI_run"
        self.ids = ["uploadA_clip1", "uploadA_clip2", "uploadB_clip1"]
        self.sources = ["uploadA", "uploadA", "uploadB"]
        self.selection = {"manifest_sha256": "m" * 64, "training_ids": [], "validation_ids": self.ids,
            "test_accessed": False}
        save(self.source / "selection.json", self.selection)
        save(self.run / "selection.json", self.selection)
        self.targets = {}
        for i, (record, source) in enumerate(zip(self.ids, self.sources)):
            metadata = {"record_id": record, "source_id": source, "split": "val", "output_fps": 8.,
                "binding": {"data_cache_key": record, "manifest_sha256": "m" * 64, "official_revision": "synthetic_pinned_official"}}
            path = self.source / f"clip_{i}/metadata.json"
            save(path, metadata)
            rgb = path.parent / "paired_rgb.pt"
            rgb.write_bytes(("synthetic-cache-bytes-" + record).encode())
            self.targets.update({str(path): file_hash(path), str(rgb): file_hash(rgb)})
        seal = self.source / "cache_asset_seal.json"
        save(seal, {"artifact_sha256": {str(Path(path).relative_to(self.source)): value for path, value in self.targets.items()}})
        save(self.run / "cache_reuse.json", {"source_run": str(self.source), "legacy_state_read": False,
            "source_asset_seal_sha256": file_hash(seal)})
        self.config = {"seed": 20261102, "data": {"num_frames": 9},
            "training": {"resume": False, "official_initialization_per_variant": True, "updates": 8192, "learning_rate": .0001},
            "conditioning": {"rank": 8},
            "model": {"revision": "synthetic_pinned_official"}, "reference": {"enabled": True, "drop_probability": .5},
            "reproducibility": {"mode": "strict"}}
        save(self.run / "config.json", self.config)

    def create(self, variant):
        directory = self.run / variant / "step_008192"
        groups = {name: {"sha256": "a" * 64, "numel": 1, "tensor_count": 1} for name in ("lrv", "lora", "hri")}
        groups["projector"] = {"sha256": "b" * 64, "numel": 0 if variant == "Base" else 1, "tensor_count": 0 if variant == "Base" else 1}
        save(directory.parent / "initialization.json", {"fresh_official_initialization": True,
            "legacy_state_loaded": False, "optimizer_fresh": True, "seed": 20261102, "data_seed": 20261102,
            "trainable_initialization_hashes": {"groups": groups}})
        sampling = {"seeds": [42, 43], "steps": 20, "cfg": 1., "matched_noise_across_modes": True}
        rgb = {"status": "completed", "scope": "synthetic fixtures only", "model_setting": "reference_trained_with_dropout",
            "sampling": sampling, "metric_protocol": {"pixel_source": "synthetic_float"}, "reference_cases": {}}
        score = {"status": "completed", "scope": rgb["scope"], "sampling": sampling, "audit": {"synthetic": True},
            "protocol": {"delta": "on-minus-off", "synthetic": True}, "scorer_sha256": "a" * 64,
            "sampled_artifact_sha256": {}, "target_artifact_sha256": self.targets, "reference_cases": {}}
        for ref in (0, 4):
            original, measured = [], []
            for i, (record, source) in enumerate(zip(self.ids, self.sources)):
                baseline = pools(24 + i, ref)
                a = {"clip_index": i, "record_id": record, "source_id": source, "split": "val", "reference_index": ref,
                    "frame_count": 9, "fps": 8., "evaluation_frame_indices": [j for j in range(9) if j != ref],
                    "bicubic": baseline["matched_nonreference"], "modes": {}}
                b = {key: a[key] for key in ("record_id", "source_id", "split", "reference_index", "frame_count", "fps")}
                b.update(bicubic=baseline, modes={})
                for mode in ("hri_on", "hri_off"):
                    ar, br = [], []
                    for seed in sampling["seeds"]:
                        value = 20 + i + seed * .0001
                        if variant == "Align":
                            value += .5
                        if mode == "hri_on":
                            value += (2 if ref == 0 else -1) + (1 if variant == "Align" else 0)
                        metrics = pools(value, ref)
                        path = directory / f"reference_{ref}/{record}/{mode}/seed_{seed}_rgb.pt"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(f"synthetic-{variant}-{record}-{mode}-{seed}-{ref}".encode())
                        score["sampled_artifact_sha256"][str(path)] = file_hash(path)
                        ar.append({"seed": seed, "noise_seed": seed, "hri_valid": mode == "hri_on",
                            "supplied_reference_index": ref if mode == "hri_on" else None,
                            "frame_count": 9, "fps": 8., "rgb_path": str(path),
                            "metrics": metrics["matched_nonreference"], "provided_reference_frame_metrics": metrics["reference_frame_separate"]})
                        br.append({"seed": seed, "metrics": metrics})
                    a["modes"][mode], b["modes"][mode] = {"seeds": ar}, {"seeds": br}
                original.append(a)
                measured.append(b)
            rgb["reference_cases"][str(ref)] = {"reference_index": ref, "per_video": original}
            score["reference_cases"][str(ref)] = {"reference_index": ref, "per_video": measured}
        save(directory / "metrics.json", rgb)
        score["rgb_metrics_sha256"] = file_hash(directory / "metrics.json")
        save(directory / "perceptual_reference_metrics.json", score)
        return directory


class ReferenceSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fixture = Fixture(self.root)
        self.base = self.fixture.create("Base")
        self.align = self.fixture.create("Align")

    def tearDown(self):
        self.temp.cleanup()

    def load(self, path):
        return load_reference_result(path, project_root=self.root)

    def mutate_rgb(self, change):
        path = self.base / "metrics.json"
        value = json.loads(path.read_text())
        change(value)
        save(path, value)
        score = self.base / "perceptual_reference_metrics.json"
        data = json.loads(score.read_text())
        data["rgb_metrics_sha256"] = file_hash(path)
        save(score, data)

    def test_case_separation_source_clusters_crossmethod_and_interaction(self):
        report = summarize_reference_results({"Base": self.load(self.base), "Align": self.load(self.align)}, replicates=100)
        self.assertEqual(set(report["reference_cases"]), {"0", "4"})
        for ref, expected in (("0", 2.), ("4", -1.)):
            case = report["reference_cases"][ref]
            within = case["within_method_on_minus_off"]["Base"]
            self.assertEqual((within["video_count"], within["source_count"], within["sampling_seeds"]), (3, 2, [42, 43]))
            self.assertAlmostEqual(within["comparisons"]["hri_on"]["psnr_rgb_db"]["delta_video_macro"], expected)
            cross = case["cross_method"]["hri_on"]["comparisons"]["Align"]["psnr_rgb_db"]
            self.assertAlmostEqual(cross["delta_video_macro"], 1.5)
            interaction = case["interaction_on_minus_off_vs_reference"]["comparisons"]["Align"]["psnr_rgb_db"]
            self.assertAlmostEqual(interaction["delta_video_macro"], 1.)
            self.assertEqual(set(case["unavailable_metrics"]["Base:reference_frame_separate"]), set(TEMPORAL))
        self.assertFalse(report["training_repeat_uncertainty_included"])
        self.assertIn("Reference index 4", text_report(report))

    def test_partial_matrix_only_base_supported_without_claiming_training_repeat(self):
        report = summarize_reference_results({"Base": self.load(self.base)}, replicates=100)
        self.assertEqual(report["reference_cases"]["0"]["interaction_on_minus_off_vs_reference"]["comparisons"], {})
        self.assertFalse(report["quality_hypotheses_verified"])

    def test_tampered_generated_or_target_bytes_and_extra_digest_are_rejected(self):
        score_path = self.base / "perceptual_reference_metrics.json"
        score = json.loads(score_path.read_text())
        sample = Path(next(iter(score["sampled_artifact_sha256"])))
        original = sample.read_bytes()
        sample.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.load(self.base)
        sample.write_bytes(original)
        extra = self.base / "extra.pt"
        extra.write_bytes(b"unbound")
        score["sampled_artifact_sha256"][str(extra)] = file_hash(extra)
        save(score_path, score)
        with self.assertRaisesRegex(ValueError, "complete original path sets"):
            self.load(self.base)

    def test_noise_missing_seed_and_actual_reference_validity_refused(self):
        for kind in ("noise", "missing_seed", "validity", "supplied_index"):
            with self.subTest(kind=kind):
                self.fixture.create("Base")
                def mutate(rgb):
                    samples = rgb["reference_cases"]["0"]["per_video"][0]["modes"]["hri_on"]["seeds"]
                    if kind == "noise": samples[0]["noise_seed"] += 1
                    elif kind == "missing_seed": samples.pop()
                    elif kind == "validity": samples[0]["hri_valid"] = 1
                    else: samples[0]["supplied_reference_index"] = False
                self.mutate_rgb(mutate)
                with self.assertRaises(ValueError): self.load(self.base)

    def test_target_binding_step_or_training_seed_cannot_differ_across_methods(self):
        base, align = self.load(self.base), self.load(self.align)
        for kind in ("binding", "training_seed", "step", "case", "conditioning", "optimizer", "common_init"):
            bad = copy.deepcopy(align)
            if kind == "binding": bad["bindings"][self.fixture.ids[0]]["target_rgb_sha256"] = "0" * 64
            elif kind == "case": del bad["cases"]["4"]
            elif kind == "conditioning": bad["compatibility"]["conditioning"]["rank"] = 16
            elif kind == "optimizer": bad["compatibility"]["numerical_training"]["learning_rate"] = .001
            elif kind == "common_init": bad["compatibility"]["common_trainable_initialization"]["hri"]["sha256"] = "c" * 64
            else: bad["compatibility"][kind] = "different"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                summarize_reference_results({"Base": base, "Align": bad}, replicates=100)

    def test_original_edges_and_empty_temporal_nulls_cannot_be_replaced(self):
        path = self.base / "perceptual_reference_metrics.json"
        original = json.loads(path.read_text())
        for kind in ("edges", "null"):
            bad = copy.deepcopy(original)
            metric = bad["reference_cases"]["4"]["per_video"][0]["modes"]["hri_on"]["seeds"][0]["metrics"]
            if kind == "edges": metric["matched_nonreference"]["temporal_pair_indices"] = [[2, 3], [3, 5], [5, 6]]
            else: metric["reference_frame_separate"][TEMPORAL[0]] = 0.
            save(path, bad)
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.load(self.base)

    def test_fresh_initialization_and_exact_rgb_file_sha_required(self):
        init = self.base.parent / "initialization.json"
        value = json.loads(init.read_text());value["optimizer_fresh"] = False;save(init, value)
        with self.assertRaisesRegex(ValueError, "official fresh"):
            self.load(self.base)
        value["optimizer_fresh"] = True;save(init, value)
        with (self.base / "metrics.json").open("a") as stream: stream.write(" ")
        with self.assertRaisesRegex(ValueError, "SHA-bound"):
            self.load(self.base)

    def test_missing_reused_asset_seal_is_not_silently_accepted(self):
        (self.fixture.source / "cache_asset_seal.json").unlink()
        with self.assertRaisesRegex(ValueError, "requires its existing prepared asset seal"):
            self.load(self.base)

    def test_lpips_and_provided_frame_scores_remain_bound_to_original_frames(self):
        path = self.base / "perceptual_reference_metrics.json"
        original = json.loads(path.read_text())
        bad = copy.deepcopy(original)
        bad["reference_cases"]["0"]["per_video"][0]["modes"]["hri_on"]["seeds"][0]["metrics"]["matched_nonreference"]["lpips_per_frame"][0] += .1
        save(path, bad)
        with self.assertRaisesRegex(ValueError, "LPIPS"):
            self.load(self.base)
        save(path, original)
        self.mutate_rgb(lambda rgb: rgb["reference_cases"]["0"]["per_video"][0]["modes"]["hri_on"]["seeds"][0]["provided_reference_frame_metrics"].update(psnr_rgb_db=-1))
        with self.assertRaisesRegex(ValueError, "original record"):
            self.load(self.base)


if __name__ == "__main__":
    unittest.main()
