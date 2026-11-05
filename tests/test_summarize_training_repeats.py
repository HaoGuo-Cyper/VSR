"""CPU synthetic records: paired training seeds, immutable data and no pseudo-replication."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from test_summarize_saved import make_run
from test_summarize_reference_saved import Fixture as ReferenceFixture
from vsr.audit import file_hash
from vsr.summarize_training_repeats import (
    aggregate_training_repeats, load_repeat_result, parse_pairs,
    summarize_repeat_results,
)


DATA_SEED = 20261102


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def config(training_seed, *, reference=False, fallback=False):
    training = {"variants": ["Base", "Align"], "updates": 8192,
        "resume": False, "official_initialization_per_variant": True,
        "learning_rate": .0001, "weight_decay": .01, "lambda_align": .01,
        "lambda_relation": 1., "batch_size": 1, "mixed_precision": "bf16"}
    if not fallback:
        training["seed"] = training_seed
    return {"seed": DATA_SEED, "training": training,
        "data": {"num_frames": 9, "height": 128, "width": 224,
            "scale": 4, "temporal_stride": 4, "fps": "preserve_sampled_input_fps"},
        "model": {"revision": "synthetic_pinned_official"},
        "conditioning": {"rank": 8, "alpha": 8, "alignment_layer": 9,
            "injection_layers": [0, 5, 9], "stem_channels": 128,
            "projector_hidden": 1024, "teacher_dim": 768, "hri_enabled": reference},
        "reference": {"enabled": reference, "drop_probability": .5},
        "reproducibility": {"mode": "strict", "relation_loss_device": "cpu"},
        "caption_policy": "synthetic fixed generic text fixture"}


def initialization(training_seed, variant, *, reference=False, fallback=False):
    groups = {name: {"sha256": sha(f"{training_seed}:{name}"), "numel": 1,
                      "tensor_count": 1} for name in ("lrv", "lora", "hri", "projector")}
    if not reference:
        groups["hri"] = {"sha256": sha(""), "numel": 0, "tensor_count": 0}
    if variant == "Base":
        groups["projector"] = {"sha256": sha(""), "numel": 0, "tensor_count": 0}
    value = {"fresh_official_initialization": True, "legacy_state_loaded": False,
        "optimizer_fresh": True, "seed": training_seed, "data_seed": DATA_SEED,
        "official_revision": "synthetic_pinned_official",
        "trainable_initialization_hashes": {"groups": groups}}
    if not fallback:
        value["training_seed"] = training_seed
    return value


def complete_metadata(metadata, *, fps):
    value = copy.deepcopy(metadata)
    record = value["record_id"]
    value.update(data_version="hq-vsr-paired-v2", source_sha256=sha("raw:" + record),
        seed=DATA_SEED, window_seed=int(sha(record)[:12], 16),
        source_indices=list(range(0, 33, 4)), temporal_stride=4,
        original_fps=fps * 4, output_fps=fps, fps_policy="preserve_sampled_input_fps",
        source_hw=[256, 448], resize_hw=[128, 224], crop_tlhw=[0, 0, 128, 224],
        degradation={"scale": 4, "kernel": "torch_bicubic", "antialias": True,
            "align_corners": False, "clamp_01": True,
            "temporal_parameters": "shared_for_all_frames"},
        caption="A high quality video.", caption_status="generic_placeholder_not_video_annotation",
        valid_region="entire_crop_no_padding")
    return value


def summary(variant, val_count):
    return {"status": "completed", "variant": variant, "updates": 8192,
        "training_clip_count": 1, "validation_clip_count": val_count, "test_accessed": False}


class NoReferenceFixture:
    """Decorate the existing score fixture with a realistic completed-run envelope."""
    def __init__(self, root):
        self.root = root
        self.source = root / "runs/shared_prepared_source"
        self.selection = {"manifest_sha256": "manifest", "training_ids": ["record2"],
            "validation_ids": ["record0", "record1"], "test_accessed": False}
        self.indices = {"record2": 0, "record0": 1, "record1": 2}
        for record, i in self.indices.items():
            number = int(record[-1])
            meta = {"record_id": record, "source_id": f"source{number}",
                "split": "train" if number == 2 else "val",
                "binding": {"data_cache_key": f"crop{number}",
                    "manifest_sha256": "manifest", "official_revision": "synthetic_pinned_official"}}
            save(self.source / f"clip_{i}/metadata.json", complete_metadata(meta, fps=7.5))
            (self.source / f"clip_{i}/paired_rgb.pt").write_bytes(f"synthetic target {number}".encode())
        save(self.source / "selection.json", self.selection)
        self.seal()

    def seal(self):
        files = list(self.source.glob("clip_*/*"))
        save(self.source / "cache_asset_seal.json", {"artifact_sha256": {
            str(f.relative_to(self.source)): file_hash(f) for f in files if f.is_file()}})

    def create(self, seed, *, fallback=False):
        result = {}
        for variant in ("Base", "Align"):
            original = make_run(self.root / "runs", f"noref_{seed}_{variant}",
                                gain=0 if variant == "Base" else 1)
            run = original.parent.parent
            directory = run / variant / "step_008192"
            directory.parent.mkdir(exist_ok=True)
            original.rename(directory)
            rgb = json.loads((directory / "metrics.json").read_text())
            score = json.loads((directory / "perceptual_metrics_v2.json").read_text())
            for row in rgb["per_video"]:
                row.update(clip_index=self.indices[row["record_id"]], fps=7.5, frame_count=9)
                for mode in row["modes"].values():
                    for sample in mode["seeds"]:
                        sample.update(fps=7.5, frame_count=9)
                        if "rgb_path" in sample:
                            sample["rgb_path"] = str(directory / Path(sample["rgb_path"]).name)
            for row in score["per_video"]:
                index = self.indices[row["record_id"]]
                row["source_metadata_sha256"] = file_hash(self.source / f"clip_{index}/metadata.json")
                row["source_rgb_sha256"] = file_hash(self.source / f"clip_{index}/paired_rgb.pt")
                for mode in row["modes"].values():
                    for sample in mode["seeds"]:
                        sample["rgb_path"] = str(directory / Path(sample["rgb_path"]).name)
                        sample["rgb_sha256"] = file_hash(Path(sample["rgb_path"]))
            save(directory / "metrics.json", rgb)
            score["rgb_metrics_sha256"] = file_hash(directory / "metrics.json")
            save(directory / "perceptual_metrics_v2.json", score)
            cfg = config(seed, fallback=fallback)
            cfg["data"]["cache_from"] = str(self.source)
            save(run / "config.json", cfg)
            save(run / "selection.json", self.selection)
            save(run / "cache_reuse.json", {"source_run": str(self.source), "legacy_state_read": False,
                "source_asset_seal_sha256": file_hash(self.source / "cache_asset_seal.json")})
            save(directory.parent / "initialization.json", initialization(seed, variant, fallback=fallback))
            save(directory.parent / "summary.json", summary(variant, 2))
            result[variant] = directory
        return result

    def refresh_source_bindings(self, pair):
        """Refresh all signatures so semantic guards, not stale hashes, decide."""
        self.seal()
        for directory in pair.values():
            run = directory.parent.parent
            reuse = json.loads((run / "cache_reuse.json").read_text())
            reuse["source_asset_seal_sha256"] = file_hash(self.source / "cache_asset_seal.json")
            save(run / "cache_reuse.json", reuse)
            score = json.loads((directory / "perceptual_metrics_v2.json").read_text())
            score["rgb_metrics_sha256"] = file_hash(directory / "metrics.json")
            for row in score["per_video"]:
                index = self.indices[row["record_id"]]
                row["source_metadata_sha256"] = file_hash(self.source / f"clip_{index}/metadata.json")
                row["source_rgb_sha256"] = file_hash(self.source / f"clip_{index}/paired_rgb.pt")
            save(directory / "perceptual_metrics_v2.json", score)


class ReferenceRepeatFixture:
    def __init__(self, root):
        self.fixture = ReferenceFixture(root)
        self.root = root
        for path in self.fixture.source.glob("clip_*/metadata.json"):
            save(path, complete_metadata(json.loads(path.read_text()), fps=8.))
        self.fixture.targets = {str(f): file_hash(f) for f in self.fixture.source.glob("clip_*/*") if f.is_file()}
        save(self.fixture.source / "cache_asset_seal.json", {"artifact_sha256": {
            str(Path(path).relative_to(self.fixture.source)): value for path, value in self.fixture.targets.items()}})

    def create(self, seed, *, fallback=False):
        f = self.fixture
        f.run = self.root / "runs" / f"reference_seed_{seed}"
        f.config = config(seed, reference=True, fallback=fallback)
        f.config["data"]["cache_from"] = str(f.source)
        save(f.run / "config.json", f.config)
        save(f.run / "selection.json", f.selection)
        save(f.run / "cache_reuse.json", {"source_run": str(f.source), "legacy_state_read": False,
            "source_asset_seal_sha256": file_hash(f.source / "cache_asset_seal.json")})
        result = {}
        for variant in ("Base", "Align"):
            directory = f.create(variant)
            save(directory.parent / "initialization.json", initialization(seed, variant, reference=True, fallback=fallback))
            save(directory.parent / "summary.json", summary(variant, 3))
            result[variant] = directory
        return result


def normalized(seed, variant, deltas=(0., 0., 0.)):
    observations = []
    for index, (record, source) in enumerate((("clipA1", "uploadA"), ("clipA2", "uploadA"), ("clipB1", "uploadB"))):
        for sample_seed in (42, 43):
            value = 10. + (deltas[index] if variant == "Align" else 0.)
            observations.append({"record_id": record, "source_id": source, "seed": sample_seed,
                "metrics": {"psnr_rgb_db": value, "lpips_alex01_rgb_allframes": .2}})
    return {"family": "noref", "variant": variant, "training_seed": seed, "data_seed": DATA_SEED,
        "protocol": {"sampling_seeds": [42, 43], "steps": 20, "updates": 8192},
        "data_contract": {"targets": {r: {"source_id": so, "rgb_sha256": sha(r), "indices": list(range(0, 33, 4))}
            for r, so in (("clipA1", "uploadA"), ("clipA2", "uploadA"), ("clipB1", "uploadB"))}},
        "common_initialization": {key: value for key, value in initialization(seed, "Base")["trainable_initialization_hashes"]["groups"].items() if key != "projector"},
        "projection_initialization": initialization(seed, variant)["trainable_initialization_hashes"]["groups"]["projector"],
        "contexts": {"all_frames": observations}, "context_metadata": {"all_frames": {"pool": "all_frames"}},
        "provenance": {"synthetic_fixture": True}, "fresh_verified": True}


class TrainingRepeatSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_equal_training_seed_weight_source_clusters_and_no_pseudoreplication(self):
        pairs = {20261103: {"Base": normalized(20261103, "Base"), "Align": normalized(20261103, "Align", (1., 1., 10.))},
                 20261104: {"Base": normalized(20261104, "Base"), "Align": normalized(20261104, "Align", (-1., -1., -4.))}}
        report = summarize_repeat_results(pairs, family="noref", replicates=100)
        self.assertEqual(report["training_seed_count"], 2)
        self.assertEqual(report["training_seeds"], [20261103, 20261104])
        self.assertEqual(report["sampling_seeds"], [42, 43])
        c = report["contexts"]["all_frames"]
        for per_seed in c["per_training_seed"].values():
            self.assertEqual((per_seed["video_count"], per_seed["source_count"]), (3, 2))
            self.assertEqual(per_seed["sampling_seeds"], [42, 43])
        delta = c["across_training_seeds"]["Align_minus_Base"]["psnr_rgb_db"]
        self.assertAlmostEqual(delta["mean"], 1.)  # mean(+4,-2), not mean source macros(+5.5,-2.5)
        self.assertEqual((delta["min"], delta["max"]), (-2., 4.))
        self.assertEqual((delta["positive_training_seeds"], delta["negative_training_seeds"], delta["zero_training_seeds"]), (1, 1, 0))
        self.assertEqual(delta["values_by_training_seed"], {"20261103": 4., "20261104": -2.})
        self.assertFalse(any(any(word in key.lower() for word in ("ci95", "variance", "pvalue")) for key in delta))
        source = c["per_source_per_training_seed"]["uploadA"]["20261103"]
        self.assertEqual(set(source["record_ids"]), {"clipA1", "clipA2"})
        self.assertAlmostEqual(source["Align_minus_Base"]["psnr_rgb_db"], 1.)

    def test_normalized_pair_labels_seeds_targets_protocol_and_common_init_fail_closed(self):
        original = {20261103: {"Base": normalized(20261103, "Base"), "Align": normalized(20261103, "Align", (1., 1., 1.))},
                    20261104: {"Base": normalized(20261104, "Base"), "Align": normalized(20261104, "Align", (1., 1., 1.))}}
        for kind in ("unpaired", "training_label", "data_seed", "target", "protocol", "common_init", "missing_sample", "source", "fresh"):
            pairs = copy.deepcopy(original)
            a = pairs[20261104]["Align"]
            if kind == "unpaired": del pairs[20261104]["Align"]
            elif kind == "training_label": a["training_seed"] = 20261105
            elif kind == "data_seed": a["data_seed"] += 1
            elif kind == "target": a["data_contract"]["targets"]["clipA1"]["rgb_sha256"] = "0" * 64
            elif kind == "protocol": a["protocol"]["steps"] = 21
            elif kind == "common_init": a["common_initialization"]["lrv"]["sha256"] = "0" * 64
            elif kind == "missing_sample": a["contexts"]["all_frames"].pop()
            elif kind == "source": a["contexts"]["all_frames"][0]["source_id"] = "different upload"
            else: a["fresh_verified"] = False
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                summarize_repeat_results(pairs, family="noref", replicates=100)

    def test_across_seed_targets_and_protocol_must_stay_fixed_but_initialization_can_vary(self):
        pairs = {seed: {name: normalized(seed, name) for name in ("Base", "Align")}
                 for seed in (20261103, 20261104)}
        self.assertNotEqual(pairs[20261103]["Base"]["common_initialization"], pairs[20261104]["Base"]["common_initialization"])
        summarize_repeat_results(pairs, family="noref", replicates=100)
        for field in ("data_contract", "protocol"):
            bad = copy.deepcopy(pairs)
            for row in bad[20261104].values(): row[field]["changed_across_training_seeds"] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize_repeat_results(bad, family="noref", replicates=100)

    def test_across_seed_observation_identities_and_metric_sets_cannot_drift(self):
        original = {seed: {name: normalized(seed, name) for name in ("Base", "Align")}
                    for seed in (20261103, 20261104)}
        for kind in ("whole_video_missing", "source_relabelled", "metric_added", "metric_missing"):
            bad = copy.deepcopy(original)
            for row in bad[20261104].values():
                observations = row["contexts"]["all_frames"]
                if kind == "whole_video_missing":
                    row["contexts"]["all_frames"] = [x for x in observations if x["record_id"] != "clipA1"]
                else:
                    for observation in observations:
                        if kind == "source_relabelled" and observation["record_id"] == "clipA1":
                            observation["source_id"] = "changed_source"
                        elif kind == "metric_added": observation["metrics"]["undeclared_extra_metric"] = 1.
                        elif kind == "metric_missing": observation["metrics"].pop("lpips_alex01_rgb_allframes")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                summarize_repeat_results(bad, family="noref", replicates=100)

    def test_completed_noref_disk_records_and_legacy_seed_fallback(self):
        fixture = NoReferenceFixture(self.root)
        first = fixture.create(DATA_SEED, fallback=True)
        second = fixture.create(20261103)
        report = aggregate_training_repeats({DATA_SEED: first, 20261103: second}, family="noref", project_root=self.root, replicates=100)
        self.assertEqual(report["training_seeds"], [DATA_SEED, 20261103])
        self.assertEqual(load_repeat_result(first["Base"], family="noref", project_root=self.root)["training_seed"], DATA_SEED)
        delta = report["contexts"]["all_frames"]["across_training_seeds"]["Align_minus_Base"]["psnr_rgb_db"]
        self.assertAlmostEqual(delta["mean"], 1.)

    def test_completed_HRI_disk_records_keep_reference_cases_and_modes_separate(self):
        fixture = ReferenceRepeatFixture(self.root)
        pairs = {seed: fixture.create(seed) for seed in (20261103, 20261104)}
        report = aggregate_training_repeats(pairs, family="reference", project_root=self.root, replicates=100)
        self.assertTrue({"ref0/hri_on", "ref0/hri_off", "ref4/hri_on", "ref4/hri_off"} <= set(report["contexts"]))
        for context in ("ref0/hri_on", "ref4/hri_on"):
            delta = report["contexts"][context]["across_training_seeds"]["Align_minus_Base"]["psnr_rgb_db"]
            self.assertAlmostEqual(delta["mean"], 1.5)

    def test_explicit_all_four_preparation_and_original_variants_fallback_are_equivalent(self):
        fixture = ReferenceRepeatFixture(self.root)
        original, repeat = fixture.create(DATA_SEED, fallback=True), fixture.create(20261103)
        all_four = ["Base", "Align", "Relation", "Both"]
        original_config_path = original["Base"].parent.parent / "config.json"
        original_config = json.loads(original_config_path.read_text())
        original_config["training"]["variants"] = all_four
        save(original_config_path, original_config)
        repeat_config_path = repeat["Base"].parent.parent / "config.json"
        repeat_config = json.loads(repeat_config_path.read_text())
        repeat_config["training"]["preparation_variants"] = all_four
        save(repeat_config_path, repeat_config)
        report = aggregate_training_repeats({DATA_SEED: original, 20261103: repeat},
            family="reference", project_root=self.root, replicates=100)
        self.assertEqual(report["training_seed_count"], 2)
        for seed in (DATA_SEED, 20261103):
            for variant in ("Base", "Align"):
                self.assertEqual(report["training_pair_provenance"][str(seed)][variant]["preparation_variants"], all_four)

    def test_optimizer_model_and_teacher_flow_changes_are_not_preparation_metadata(self):
        fixture = ReferenceRepeatFixture(self.root)
        pairs = {seed: fixture.create(seed) for seed in (20261103, 20261104)}
        path = pairs[20261104]["Base"].parent.parent / "config.json"
        original = json.loads(path.read_text())
        for kind in ("lambda", "model", "teacher", "flow"):
            changed = copy.deepcopy(original)
            if kind == "lambda": changed["training"]["lambda_align"] = .1
            elif kind == "model": changed["model"]["revision"] = "different_official_revision"
            elif kind == "teacher": changed["teacher"] = {"revision": "different_teacher_revision"}
            else: changed["flow"] = {"updates": 24}
            save(path, changed)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                aggregate_training_repeats(pairs, family="reference", project_root=self.root, replicates=100)
        save(path, original)

    def test_training_and_preparation_variant_lists_and_membership_are_validated(self):
        fixture = NoReferenceFixture(self.root)
        directory = fixture.create(20261103)["Align"]
        path = directory.parent.parent / "config.json"
        original = json.loads(path.read_text())
        invalid = (None, [], "Base", {"Base": True}, ["Base", "Base"],
                   ["Base", "Unknown"], ["Base", False], ["Base", 1])
        for field in ("variants", "preparation_variants"):
            for value in invalid:
                changed = copy.deepcopy(original)
                changed["training"][field] = value
                save(path, changed)
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "variants"):
                    load_repeat_result(directory, family="noref", project_root=self.root)
        changed = copy.deepcopy(original)
        changed["training"]["variants"] = ["Base"]  # this directory records Align
        changed["training"]["preparation_variants"] = ["Base", "Align"]
        save(path, changed)
        with self.assertRaisesRegex(ValueError, "Saved directory variant"):
            load_repeat_result(directory, family="noref", project_root=self.root)
        changed = copy.deepcopy(original)
        changed["training"]["preparation_variants"] = ["Base"]
        save(path, changed)
        with self.assertRaisesRegex(ValueError, "preparation_variants"):
            load_repeat_result(directory, family="noref", project_root=self.root)
        save(path, original)

    def test_official_initialization_revision_is_required_and_must_match_model(self):
        fixture = NoReferenceFixture(self.root)
        directory = fixture.create(20261103)["Align"]
        path = directory.parent / "initialization.json"
        original = json.loads(path.read_text())
        for kind in ("missing", "different"):
            changed = copy.deepcopy(original)
            if kind == "missing": changed.pop("official_revision")
            else: changed["official_revision"] = "different_official_revision"
            save(path, changed)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "Official initialization revision"):
                load_repeat_result(directory, family="noref", project_root=self.root)
        save(path, original)

    def test_manifest_and_official_binding_drift_rejected_with_refreshed_signatures(self):
        fixture = NoReferenceFixture(self.root)
        pair = fixture.create(20261103)
        path = fixture.source / "clip_1/metadata.json"
        original = json.loads(path.read_text())
        for field in ("manifest_sha256", "official_revision"):
            changed = copy.deepcopy(original)
            changed["binding"][field] = "different_binding_value"
            save(path, changed)
            fixture.refresh_source_bindings(pair)
            # Both model records and scorer digests now agree with changed
            # physical metadata; the declared manifest/model must still win.
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Prepared input manifest/official revision binding"):
                aggregate_training_repeats({20261103: pair}, family="noref", project_root=self.root, replicates=100)
        save(path, original)
        fixture.refresh_source_bindings(pair)

    def test_matching_changed_model_configs_cannot_conceal_wrong_init_or_input_binding(self):
        fixture = NoReferenceFixture(self.root)
        pair = fixture.create(20261103)
        for directory in pair.values():
            path = directory.parent.parent / "config.json"
            cfg = json.loads(path.read_text())
            cfg["model"]["revision"] = "different_official_revision"
            save(path, cfg)
        with self.assertRaisesRegex(ValueError, "Official initialization revision"):
            aggregate_training_repeats({20261103: pair}, family="noref", project_root=self.root, replicates=100)
        for directory in pair.values():
            path = directory.parent / "initialization.json"
            init = json.loads(path.read_text())
            init["official_revision"] = "different_official_revision"
            save(path, init)
        # Making both config/init headers match still leaves the sealed input
        # cache bound to the original official model and must be rejected.
        fixture.refresh_source_bindings(pair)
        with self.assertRaisesRegex(ValueError, "Prepared input manifest/official revision binding"):
            aggregate_training_repeats({20261103: pair}, family="noref", project_root=self.root, replicates=100)

    def test_original_completed_Base_and_fresh_recovery_Align_share_only_data(self):
        fixture = NoReferenceFixture(self.root)
        pair = fixture.create(20261103)
        original, recovery = pair["Base"].parent.parent, pair["Align"].parent.parent
        # The original process may have been interrupted in a different group;
        # its completed Base and independently sealed preparation remain valid.
        for source in fixture.source.glob("clip_*/*"):
            destination = original / source.relative_to(fixture.source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        shutil.copyfile(fixture.source / "cache_asset_seal.json", original / "cache_asset_seal.json")
        (original / "cache_reuse.json").unlink()
        cfg = json.loads((original / "config.json").read_text())
        cfg["data"].pop("cache_from")
        save(original / "config.json", cfg)
        save(original / "status.json", {"status": "interrupted", "other_variant": "unusable_partial"})
        save(recovery / "cache_reuse.json", {"source_run": str(original), "legacy_state_read": False,
            "source_asset_seal_sha256": file_hash(original / "cache_asset_seal.json")})
        cfg = json.loads((recovery / "config.json").read_text())
        cfg["data"]["cache_from"] = str(original)
        save(recovery / "config.json", cfg)
        report = aggregate_training_repeats({20261103: pair}, family="noref", project_root=self.root, replicates=100)
        self.assertEqual(report["training_seed_count"], 1)
        self.assertAlmostEqual(report["contexts"]["all_frames"]["across_training_seeds"]["Align_minus_Base"]["psnr_rgb_db"]["mean"], 1.)

    def test_semantically_invalid_FPS_rejected_even_with_current_artifact_hashes(self):
        fixture = NoReferenceFixture(self.root)
        pair = fixture.create(20261103)
        source = fixture.source / "clip_1/metadata.json"  # validation record0
        meta = json.loads(source.read_text())
        meta["output_fps"] = 6.  # original30 / stride4 must remain7.5
        save(source, meta)
        fixture.seal()
        for directory in pair.values():
            run = directory.parent.parent
            save(run / "cache_reuse.json", {"source_run": str(fixture.source), "legacy_state_read": False,
                "source_asset_seal_sha256": file_hash(fixture.source / "cache_asset_seal.json")})
            rgb = json.loads((directory / "metrics.json").read_text())
            for row in rgb["per_video"]:
                if row["record_id"] == "record0":
                    row["fps"] = 6.
                    for mode in row["modes"].values():
                        for sample in mode["seeds"]: sample["fps"] = 6.
            save(directory / "metrics.json", rgb)
            score = json.loads((directory / "perceptual_metrics_v2.json").read_text())
            score["rgb_metrics_sha256"] = file_hash(directory / "metrics.json")
            for row in score["per_video"]:
                if row["record_id"] == "record0": row["source_metadata_sha256"] = file_hash(source)
            save(directory / "perceptual_metrics_v2.json", score)
        with self.assertRaisesRegex(ValueError, "FPS"):
            load_repeat_result(pair["Base"], family="noref", project_root=self.root)

    def test_disk_envelope_rejects_warm_start_bad_seed_and_unfinished_budget(self):
        fixture = NoReferenceFixture(self.root)
        pair = fixture.create(20261103)
        directory = pair["Align"]
        run = directory.parent.parent
        cfg = json.loads((run / "config.json").read_text())
        init = json.loads((directory.parent / "initialization.json").read_text())
        finished = json.loads((directory.parent / "summary.json").read_text())
        for kind in ("resume", "official", "seed", "legacy", "optimizer", "init_seed", "updates", "status"):
            c, i, f = copy.deepcopy(cfg), copy.deepcopy(init), copy.deepcopy(finished)
            if kind == "resume": c["training"]["resume"] = True
            elif kind == "official": c["training"]["official_initialization_per_variant"] = False
            elif kind == "seed": c["training"]["seed"] += 1
            elif kind == "legacy": i["legacy_state_loaded"] = True
            elif kind == "optimizer": i["optimizer_fresh"] = False
            elif kind == "init_seed": i["training_seed"] += 1
            elif kind == "updates": f["updates"] = 4096
            else: f["status"] = "interrupted"
            save(run / "config.json", c)
            save(directory.parent / "initialization.json", i)
            save(directory.parent / "summary.json", f)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                load_repeat_result(directory, family="noref", project_root=self.root)
        save(run / "config.json", cfg)
        save(directory.parent / "initialization.json", init)
        save(directory.parent / "summary.json", finished)

    def test_no_ref_reused_source_requires_prepared_asset_seal(self):
        fixture = NoReferenceFixture(self.root)
        pair = fixture.create(20261103)
        (fixture.source / "cache_asset_seal.json").unlink()
        with self.assertRaises(ValueError):
            load_repeat_result(pair["Base"], family="noref", project_root=self.root)

    def test_duplicate_cli_training_seed_variant_and_unsupported_role_are_rejected(self):
        good = parse_pairs(["20261103:Base=/tmp/b", "20261103:Align=/tmp/a"])
        self.assertEqual(set(good), {20261103})
        self.assertEqual(set(good[20261103]), {"Base", "Align"})
        for entries in (["20261103:Base=/tmp/a", "20261103:Base=/tmp/b"],
                        ["not-a-seed:Base=/tmp/a"], ["20261103:Both=/tmp/a"]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                parse_pairs(entries)


if __name__ == "__main__":
    unittest.main()
