"""CPU wrapper integration: immutable pre-pixel contract and real MP4 checks."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from vsr.data import write_video
from vsr.frozen_evaluation import BenchmarkPolicy, _artifact, _json
from vsr.native_operator_qa import run_native_operator_qa


class SyntheticReader:
    def __init__(self, root, *, plans):
        self.records = {"HQ-val-nativecrop": {"fixture": {}}, "HQ-val-nativefull": {"fixture": {}}}
        self.metadata = {"training_allowed": False, "fixture": "metadata_only"}

    def policy(self, dataset):
        return BenchmarkPolicy(family="hq", allowed_datasets=(dataset,),
            reader_options={"height": 16, "width": 64}, run_prefix="synthetic_qa")

    def close(self):
        pass


class NativeOperatorQAIntegration(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / "runs/qa_fixture"
        self.paths = {}
        for name in ("NorefBase", "RefBase"):
            path = self.root / "runs" / name / "Base/checkpoint_008192.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(("synthetic-" + name).encode())
            self.paths[name] = path
        self.plans = [{"selection": {"split": "val"},
            "input_preparation": {"source_frame_indices": list(range(17)), "num_frames": 17},
            "restoration": {"steps": 20, "sampling_seed": 42, "HR_tile_hw": [256, 448], "overlap": 64},
            "checkpoint_policy": {"no_reference_SHA256": hashlib.sha256(self.paths["NorefBase"].read_bytes()).hexdigest()},
            "case": name} for name in ("crop", "whole")]
        self.events = []
        self.mismatch_noise = False
        self.restore_failure = None

    def tearDown(self):
        self.temporary.cleanup()

    def preview(self, path, **kwargs):
        return {"checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "variant": "Base", "step": 8192,
            "config": {"reference": {"enabled": "RefBase" in path.parts}}}

    def engine(self, root, checkpoints, **options):
        case = options["run_dir"]
        self.assertTrue((case.parent / "qa_protocol.json").is_file())
        frozen = json.loads((case.parent / "qa_protocol.json").read_text())
        self.assertEqual(set(frozen["checkpoints"]), set(checkpoints))
        self.assertFalse(frozen["test_or_external_data_used"])
        self.events.append(("synthetic_pixel_entry", options["datasets"][0]))
        if self.restore_failure is not None:
            raise self.restore_failure
        for label in checkpoints:
            modes = ("hri_on", "hri_off") if label == "RefBase" else ("no_reference",)
            entries = {}
            for mode in modes:
                video_path = case / f"{label}_{mode}.mp4"
                # This is a synthetic fixture, not a model output or dataset.
                write_video(video_path, torch.full((1, 3, 17, 16, 64), .25), 24.0)
                noise = "f" * 64 if self.mismatch_noise and mode == "hri_off" else "a" * 64
                entries[mode] = {"seeds": [{"seed": 42,
                    "restoration": {"native_operator_audit": {"status": "passed", "full_noise_sha256": noise}},
                    "preview": {"artifact": _artifact(video_path, case), "source_hw": [16, 64],
                        "downscaled_for_playback_only": False}}]}
            _json(case / "results" / label / options["datasets"][0] / "fixture/metrics.json",
                {"label": label, "modes": entries, "evaluated_frames": 17, "source_fps": 24.0})
        summary = {"status": "completed", "protocol_sha256": "b" * 64}
        _json(case / "summary.json", summary)
        return summary

    def execute(self):
        with ExitStack() as stack:
            stack.enter_context(patch("vsr.native_qa_reader.NativeQAReader", SyntheticReader))
            stack.enter_context(patch("vsr.native_qa_restore.AuditedNativeRestore", lambda plans: object()))
            stack.enter_context(patch("vsr.native_operator_qa.run_frozen_evaluation", self.engine))
            return run_native_operator_qa(self.root, self.paths, self.plans, run_dir=self.run,
                device="cpu", factories={"preview": self.preview})

    def test_pre_pixel_contract_both_cases_all_modes_and_encoded_video(self):
        summary = self.execute()
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(len(self.events), 2)
        self.assertFalse(summary["quality_hypotheses_verified"])
        self.assertEqual(summary["training_updates"], 0)
        for dataset, result in summary["cases"].items():
            audit = json.loads((Path(result["run"]) / "native_operator_case_audit.json").read_text())
            self.assertEqual(len(audit["playback"]), 3)
            self.assertEqual({entry["observed_container"]["frame_count"] for entry in audit["playback"]}, {17})
            self.assertEqual({entry["observed_container"]["fps"] for entry in audit["playback"]}, {24.0})
            self.assertFalse(audit["quality_scores_used_for_acceptance"])

    def test_observed_noise_mismatch_preserves_failure_without_crop_fallback(self):
        self.mismatch_noise = True
        with self.assertRaisesRegex(ValueError, "same observed global noise"):
            self.execute()
        failed = json.loads((self.run / "qa_status.json").read_text())
        self.assertEqual(failed["status"], "failed")
        self.assertTrue(failed["no_silent_fallback"])
        self.assertEqual(len(self.events), 1)
        self.assertFalse((self.run / "qa_summary.json").exists())

    def test_checkpoint_change_cannot_resume_same_outer_protocol(self):
        self.execute()
        before = (self.run / "qa_protocol.json").read_bytes()
        self.paths["RefBase"].write_bytes(b"changed-synthetic-checkpoint")
        with self.assertRaisesRegex(ValueError, "Existing QA protocol changed"):
            self.execute()
        self.assertEqual((self.run / "qa_protocol.json").read_bytes(), before)
        self.assertEqual(len(self.events), 2)

    def test_sampling_change_rejected_before_any_qa_pixel_entry(self):
        self.plans[1]["restoration"]["steps"] = 21
        with self.assertRaisesRegex(ValueError, "fixed predeclaration"):
            self.execute()
        self.assertEqual(self.events, [])
        self.assertFalse(self.run.exists())

    def test_concurrent_training_guard_precedes_reader_and_qa_outputs(self):
        with patch("vsr.reference_diagnostics.assert_gpu_idle", side_effect=RuntimeError("active training")):
            for options in ({}, {"device": "cpu", "metric_device": "cuda"}):
                with self.assertRaisesRegex(RuntimeError, "active training"):
                    run_native_operator_qa(self.root, self.paths, self.plans, run_dir=self.run, **options)
        self.assertEqual(self.events, [])
        self.assertFalse(self.run.exists())

    def test_failed_actual_restore_observations_are_persisted(self):
        error = RuntimeError("synthetic observed sampler failure")
        error.audit = {"status": "failed", "observed_tile_count": 1, "full_noise_sha256": "c" * 64}
        self.restore_failure = error
        with self.assertRaisesRegex(RuntimeError, "observed sampler failure"):
            self.execute()
        status = json.loads((self.run / "qa_status.json").read_text())
        artifact = status["restore_observations"]
        path = self.run / artifact["path"]
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"])
        self.assertEqual(json.loads(path.read_text())["last_observed_restore"], error.audit)
        # A same-protocol retry archives the original failure/cost record.
        self.restore_failure = None
        self.execute()
        history = list((self.run / "qa_attempt_history").glob("*_qa_status.json"))
        self.assertEqual(len(history), 1)
        self.assertEqual(json.loads(history[0].read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
