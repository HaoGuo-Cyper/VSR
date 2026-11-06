"""Synthetic CPU HQ-engine tests; no real HQ val/test pixels or GPU jobs."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

import torch

from test_evaluate_external import Factories
from vsr.evaluate_hq import build_parser, run_hq_evaluation
from vsr.frozen_evaluation import stable_hq_input_contract


class FakeHQReader:
    def __init__(self, root, *, split, val_limit, height, width, temporal_stride, seed, modes,
                 stop_before_pixels=False):
        self.root, self.split = Path(root), split
        self.options = {"split": split, "val_limit": val_limit, "height": height, "width": width,
                        "temporal_stride": temporal_stride, "seed": seed, "modes": list(modes)}
        count = 211 if split == "test" else 40
        ids = [f"synthetic_{index:03d}_16to143" for index in range(count)]
        ids = sorted(ids) if split == "test" else sorted(ids, key=lambda value: hashlib.sha256(f"val:{seed}:{value}".encode()).hexdigest())[:val_limit]
        self.records = {f"HQ-{split}-{mode}": {value: {"id": value, "split": split} for value in ids} for mode in modes}
        raw = (self.root/"manifests/hq_vsr.json").read_bytes()
        self.metadata = {"reader": "synthetic_HQ_fixture", "source_dataset": "HQ-VSR", "split": split,
            "training_allowed": False, "selected_record_ids": ids, "selected_record_count": len(ids),
            "manifest_file_sha256": hashlib.sha256(raw).hexdigest(), "manifest_sha256": "synthetic_canonical_seal",
            "complete_test_partition": split == "test", "validation_limit": val_limit if split == "val" else None,
            "target_hw": [height, width], "temporal_stride": temporal_stride, "data_seed": seed,
            "robustness_modes": list(modes), "noise_seed_by_record_id": {value: position+100 for position, value in enumerate(ids)},
            "pixel_decode_in_constructor": False, "checkpoint_selection_performed": False,
            "parameters_selected_on_test": False}
        self.loads, self.targets, self.closed = [], {}, False
        self.stop_before_pixels = stop_before_pixels

    def load(self, dataset, sequence, num_frames, start):
        # A load is forbidden until the complete source catalogue, reader
        # options, candidate modes and actual seed map have been sealed.
        run = self.root/"runs"/f"hq_{self.split}_fixture"
        assert (run/"protocol.json").is_file()
        assert (run/"hq_manifest.snapshot.json").read_bytes() == (self.root/"manifests/hq_vsr.json").read_bytes()
        frozen = json.loads((run/"hq_reader_configuration.json").read_text())
        assert frozen["selected_record_ids"] == self.metadata["selected_record_ids"]
        assert frozen["noise_seed_by_record_id"] == self.metadata["noise_seed_by_record_id"]
        self.loads.append((dataset, sequence, num_frames, start))
        if self.stop_before_pixels:
            raise RuntimeError("synthetic stop before any test pixels")
        assert start == 0 and sequence in self.records[dataset]
        mode = dataset.rsplit("-", 1)[1]
        value = .3143 + self.metadata["noise_seed_by_record_id"][sequence]*.0001
        lr = torch.full((1, 3, num_frames, self.options["height"]//4, self.options["width"]//4), value)
        base_lr = lr.clone()
        if mode != "identity":
            lr += .01
        hr = torch.full((1, 3, num_frames, self.options["height"], self.options["width"]), .4317)
        hr[:, :, 0] = .1299
        self.targets[(dataset, sequence)] = hr
        tensor_sha = lambda value: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
        binding = {"manifest_sha256": self.metadata["manifest_file_sha256"],
            "base_pair_cache_key": hashlib.sha256((sequence+":crop").encode()).hexdigest(), "record_id": sequence,
            "robustness_mode": mode, "noise_seed": self.metadata["noise_seed_by_record_id"][sequence],
            "robustness_protocol_sha256": hashlib.sha256((mode+":fixed_spec").encode()).hexdigest(),
            "base_lr_rgb_sha256": tensor_sha(base_lr), "robust_lr_rgb_sha256": tensor_sha(lr)}
        return {"lr": lr, "hr": hr, "fps": 7.5, "metadata": {"dataset": dataset, "source_dataset": "HQ-VSR",
            "record_id": sequence, "sequence_id": sequence, "source_id": sequence.split("_16to")[0], "split": self.split,
            "source_frame_indices": list(range(17, 17+num_frames*4, 4)), "crop_tlhw": [0, 0, self.options["height"], self.options["width"]],
            "robustness_mode": mode, "robustness_noise_seed": self.metadata["noise_seed_by_record_id"][sequence],
            "input_binding": binding, "robustness": {"elapsed_seconds": len(self.loads)*.0001,
                "compression": {"command": ["ffmpeg", "-i", f"/tmp/volatile_attempt_{len(self.loads)}/input.mp4"]}},
            "fps_source": "synthetic_original30/temporal_stride4", "benchmark_prefix_explicit": False,
            "training_allowed": False, "parameters_selected_on_test": False}}

    def close(self):
        self.closed = True


class HQFactories(Factories):
    def __init__(self, root, *, stop_before_pixels=False):
        super().__init__(root)
        # The HQ engine must not require or disguise its inputs as an external
        # REDS/YouHQ catalogue. Only this explicit HQ manifest is present.
        (root/"manifests/external.json").unlink()
        (root/"manifests/hq_vsr.json").write_text(json.dumps({"synthetic_HQ_catalogue": True, "split_contract": "val32/test211"}))
        self.reader = None
        self.stop_before_pixels = stop_before_pixels

    def create_reader(self, root, **options):
        self.reader = FakeHQReader(root, **options, stop_before_pixels=self.stop_before_pixels)
        return self.reader

    def mapping(self):
        result = super().mapping()
        result["reader"] = self.create_reader
        return result

    def restore(self, *args, **kwargs):
        video, metadata = super().restore(*args, **kwargs)
        metadata["sampling"] = {"seed": kwargs["seed"], "synthetic_fixture": True}
        return video, metadata


class EvaluateHQTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def execute(self, root, factories, split="val", modes=("identity",)):
        return run_hq_evaluation(root, factories.paths, split=split, modes=modes, frames=9,
            height=8, width=8, temporal_stride=4, seed=20261102, seeds=(42, 43), steps=2,
            tile_hw=(16, 16), overlap=0, device="cpu", metric_device="cpu",
            run_dir=root/"runs"/f"hq_{split}_fixture", factories=factories.mapping())

    def test_val32_freeze_quantization_primary_pools_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            f = HQFactories(root)
            summary = self.execute(root, f)
            self.assertEqual(summary["evaluation_family"], "HQ-VSR_internal_holdout")
            self.assertFalse(summary["test_used_for_checkpoint_selection"])
            self.assertNotIn("VideoLQ_interpretation", summary)
            self.assertNotIn("external_tuning", summary)
            run = root/"runs/hq_val_fixture"
            protocol = json.loads((run/"protocol.json").read_text())
            selected = f.reader.metadata["selected_record_ids"]
            self.assertEqual(len(selected), 32)
            self.assertEqual(protocol["datasets"], {"HQ-val-identity": selected})
            self.assertEqual(protocol["hq_configuration"]["val_limit"], 32)
            self.assertEqual(protocol["hq_configuration"]["temporal_stride"], 4)
            self.assertEqual(protocol["hq_configuration"]["seed"], 20261102)
            self.assertIn("online training float32", protocol["quantization_comparison"])
            self.assertNotIn("external_manifest_sha256", protocol)
            self.assertNotIn("red_preview_fps", protocol)
            self.assertEqual(protocol["hq_manifest_file_sha256"], hashlib.sha256((run/"hq_manifest.snapshot.json").read_bytes()).hexdigest())
            report = summary["primary_report"]
            self.assertEqual(report["Base"]["HQ-val-identity"]["no_reference"]["pool"], "all_frames")
            for mode in ("hri_on", "hri_off"):
                self.assertEqual(report["RefBoth"]["HQ-val-identity"][mode]["pool"], "matched_nonreference")
            for label, primary in (("Base", "all_frames"), ("RefBoth", "matched_nonreference")):
                row = json.loads((run/"results"/label/"HQ-val-identity"/selected[0]/"metrics.json").read_text())
                self.assertEqual(row["dataset"], "HQ-val-identity")
                self.assertEqual(row["source"]["source_dataset"], "HQ-VSR")
                self.assertNotIn("prefix_frames", row)
                self.assertEqual(row["evaluated_frames"], 9)
                self.assertIn("stable_input_contract_sha256", row)
                self.assertNotIn("robustness", row["stable_input_contract"])
                self.assertEqual(row["source"]["source_frame_indices"], list(range(17, 53, 4)))
                for mode in row["modes"].values():
                    self.assertEqual([entry["seed"] for entry in mode["seeds"]], [42, 43])
                    self.assertTrue(all(entry["metrics"]["primary_pool"] == primary for entry in mode["seeds"]))
                    self.assertTrue(all(entry["metrics"]["matched_nonreference"]["frame_indices"] == list(range(1, 9)) for entry in mode["seeds"]))
            base_row = json.loads((run/"results/Base/HQ-val-identity"/selected[0]/"metrics.json").read_text())
            ref_row = json.loads((run/"results/RefBoth/HQ-val-identity"/selected[0]/"metrics.json").read_text())
            self.assertNotEqual(base_row["source"]["robustness"], ref_row["source"]["robustness"])
            self.assertEqual(base_row["stable_input_contract_sha256"], ref_row["stable_input_contract_sha256"])
            self.assertFalse(list(run.rglob("YouHQ40")))
            self.assertTrue(f.reader.closed)
            resumed = HQFactories(root)
            again = self.execute(root, resumed)
            self.assertEqual(again["results"], summary["results"])
            self.assertEqual(len(again["resumed_complete_sequences"]), 64)
            self.assertEqual(resumed.reader.loads, [])
            self.assertEqual(resumed.models, [])
            # Simulate a process ending after one result JSON was written but
            # before its completion seal. Only the last reference-model case
            # reruns; its elapsed/tmp provenance now starts at1, not64.
            pending = run/"results/RefBoth/HQ-val-identity"/selected[-1]
            original_row = json.loads((pending/"metrics.json").read_text())
            (pending/"complete.json").unlink()
            partial = HQFactories(root)
            after_partial = self.execute(root, partial)
            new_row = json.loads((pending/"metrics.json").read_text())
            self.assertEqual(len(partial.reader.loads), 1)
            self.assertEqual(len(after_partial["resumed_complete_sequences"]), 63)
            self.assertNotEqual(original_row["source"]["robustness"], new_row["source"]["robustness"])
            self.assertEqual(original_row["stable_input_contract_sha256"], new_row["stable_input_contract_sha256"])
            self.assertEqual(after_partial["results"], summary["results"])

    def test_stable_contract_excludes_elapsed_and_tmp_but_retains_raw_pixel_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            f = HQFactories(root)
            # Exercise the reader only through the engine's pre-pixel freeze in
            # the normal integration test; this independent tiny contract uses
            # explicit tensors and complete stable binding fields.
            lr, hr = torch.full((1, 3, 9, 2, 2), .3143), torch.full((1, 3, 9, 8, 8), .4)
            pixel_sha = lambda value: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
            binding = {"manifest_sha256": "a"*64, "base_pair_cache_key": "b"*64, "record_id": "raw0",
                "robustness_mode": "compression", "noise_seed": 123, "robustness_protocol_sha256": "c"*64,
                "base_lr_rgb_sha256": pixel_sha(lr), "robust_lr_rgb_sha256": pixel_sha(lr)}
            metadata = {"input_binding": binding, "dataset": "HQ-test-compression", "record_id": "raw0",
                "source_frame_indices": list(range(0, 36, 4)), "crop_tlhw": [0, 0, 8, 8],
                "robustness": {"elapsed_seconds": .01, "command": ["/tmp/first.mp4"]}}
            first = stable_hq_input_contract(metadata, lr, hr, fps=7.5, frames=9)
            metadata["robustness"] = {"elapsed_seconds": 3.7, "command": ["/tmp/another.mp4"]}
            self.assertEqual(first, stable_hq_input_contract(metadata, lr, hr, fps=7.5, frames=9))
            changed_lr = lr+1e-6  # Still rounds to the same uint8 pixels.
            self.assertTrue(torch.equal((lr*255).round(), (changed_lr*255).round()))
            with self.assertRaisesRegex(ValueError, "float bytes differ"):
                stable_hq_input_contract(metadata, changed_lr, hr, fps=7.5, frames=9)
            metadata["input_binding"] = {**binding, "robust_lr_rgb_sha256": pixel_sha(changed_lr)}
            changed = stable_hq_input_contract(metadata, changed_lr, hr, fps=7.5, frames=9)
            self.assertNotEqual(first, changed)
            self.assertEqual(first["measured_lr_uint8_sha256"], changed["measured_lr_uint8_sha256"])

    def test_test_catalogue_has_all211_and_all_fixed_mode_labels_before_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            f = HQFactories(root, stop_before_pixels=True)
            with self.assertRaisesRegex(RuntimeError, "before any test pixels"):
                self.execute(root, f, split="test", modes=("identity", "blur", "noise", "compression", "combined"))
            run = root/"runs/hq_test_fixture"
            protocol = json.loads((run/"protocol.json").read_text())
            expected = {f"HQ-test-{mode}" for mode in ("identity", "blur", "noise", "compression", "combined")}
            self.assertEqual(set(protocol["datasets"]), expected)
            self.assertTrue(all(len(ids) == 211 for ids in protocol["datasets"].values()))
            self.assertEqual(len(f.reader.loads), 1)
            self.assertFalse(list(run.rglob("*.png")))
            self.assertFalse(protocol["test_used_for_checkpoint_selection"])
            self.assertTrue(protocol["reader_configuration"]["complete_test_partition"])

    def test_train_arbitrary_modes_and_test_limits_are_not_exposed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kwargs in ({"split": "train"}, {"modes": ("custom_best_test_sigma",)}, {"modes": ("noise", "noise")}):
                with self.assertRaises(ValueError):
                    run_hq_evaluation(root, {}, **kwargs)
        parser = build_parser()
        args = parser.parse_args(["--checkpoint", "Base=/own/Base/checkpoint.pt"])
        self.assertEqual((args.frames, args.height, args.width, args.temporal_stride, args.seed), (9, 128, 224, 4, 20261102))
        self.assertEqual(args.seeds, [42, 43])
        self.assertEqual((args.steps, args.tile, args.overlap), (20, "256x448", 64))
        self.assertFalse(any("--test-limit" in action.option_strings or "--val-limit" in action.option_strings for action in parser._actions))


if __name__ == "__main__":
    unittest.main()
