"""CPU fixtures only: structural resource orchestration, no real GPU/models."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from vsr.native_qa_restore import NativeQAAuditError
from vsr.native_resource_probe import _verify_musiq_assets, fixed_plans, run_resource_probe


class FakeMeasurements:
    """CPU-only measurement substitute when testing declared CUDA routing."""
    def __init__(self, run, devices):
        self.run, self.devices, self.phases = run, devices, {}

    def call(self, name, action):
        try:
            result = action()
            self.phases[name] = {"status": "completed", "synthetic_fixture": True}
            return result
        finally:
            (self.run/"phase_observations.json").write_text(json.dumps(self.phases))


class ResourceProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root/"runs/probe_fixture"
        self.paths = {}
        self.events, self.synthetic_shapes, self.native_calls = [], [], []
        self.metric_devices = []
        self.fail_second_restore = False
        self.bad_shape = False
        for role in ("no_reference", "reference"):
            p = self.root/"runs"/role/"Both/checkpoint_008192.pt"
            p.parent.mkdir(parents=True)
            p.write_bytes(("synthetic-checkpoint-"+role).encode())
            self.paths[role] = p

    def tearDown(self):
        self.temp.cleanup()

    def preview(self, path, **kwargs):
        return {"path": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "step": 8192, "variant": "Both", "config": {"model": {"synthetic_fixture": True},
            "reference": {"enabled": "reference" in path.parts and "no_reference" not in path.parts}}}

    def frozen(self, event):
        protocol = json.loads((self.run/"protocol.json").read_text())
        self.assertFalse(protocol["dataset_pixels_opened"])
        self.assertFalse(protocol["quality_scores_retained"])
        self.assertEqual(set(protocol["checkpoints"]), {"no_reference", "reference"})
        self.events.append(event)

    def caption(self, *args, **kwargs):
        self.frozen("caption_load")
        return torch.zeros(1,2,4), {"synthetic_fixture": True}

    def model(self, path, **kwargs):
        self.frozen("model_load")
        return nn.Linear(1,1), self.preview(path)

    def vae(self, *args, **kwargs):
        self.frozen("VAE_load")
        return nn.Linear(1,1)

    def synthetic(self, shape):
        self.frozen("synthetic_allocation")
        self.synthetic_shapes.append(list(shape))
        actual = [1,3,1,1,1] if self.bad_shape else shape
        # Expanded scalar avoids allocating gigabytes in orchestration tests.
        value = torch.full((1,3,1,1,1), .5).expand(tuple(actual))
        return value, {"shape": list(actual), "constant_rgb": .5,
            "source": "CPU lazy synthetic fixture", "sha256": hashlib.sha256(str(actual).encode()).hexdigest()}

    def restore(self, model, vae, lrv, text, config, **kwargs):
        self.frozen("restore")
        self.assertEqual((kwargs["steps"], kwargs["seed"], kwargs["tile_hw"], kwargs["overlap"]),
                         (3,42,(256,448),64))
        self.native_calls.append({"shape": list(lrv.shape), "reference_index": kwargs["reference_index"],
            "HRI_shape": list(kwargs["reference_image"].shape) if kwargs["reference_image"] is not None else None,
            "reference_trained": config["reference"]["enabled"]})
        if self.fail_second_restore and len(self.native_calls) == 2:
            raise NativeQAAuditError("synthetic observed OOM", {"status":"failed","observed_tiles":1})
        output = torch.empty((1,3,17,lrv.shape[-2]*4,lrv.shape[-1]*4), device="meta")
        return output, {"native_operator_audit": {"status":"passed", "fixture": True}}

    def musiq(self, root, *, device):
        self.frozen("MUSIQ_load")
        self.metric_devices.append(str(device))
        parent = self
        class FakeMUSIQ:
            def evaluate_video(self, video, *, batch_size):
                parent.assertEqual(list(video.shape), [1,3,1,2304,2880])
                parent.assertEqual(batch_size, 1)
                return {"frame_count":1,"per_frame_scores":[17.123456789],"mean":17.123456789,
                    "provenance": {"backend":"synthetic_fixture", "no_real_asset_claim":True}}
        return FakeMUSIQ()

    def factories(self):
        return {"preview":self.preview,"caption":self.caption,"model":self.model,
            "vae":self.vae,"synthetic":self.synthetic,"restore":self.restore,"musiq":self.musiq}

    def execute(self, **kwargs):
        return run_resource_probe(self.root,self.paths["no_reference"],self.paths["reference"],
            device="cpu", metric_devices=("cpu",), run_dir=self.run, factories=self.factories(), **kwargs)

    def test_fixed_maximum_shapes_tiles_and_three_step_scope(self):
        native, paired = fixed_plans()
        self.assertEqual(native["input_preparation"]["LR_shape"], [1,3,17,576,720])
        self.assertEqual(native["input_preparation"]["HR_shape"], [1,3,17,2304,2880])
        self.assertEqual(native["restoration"]["tile_count"], 96)
        self.assertEqual(paired["input_preparation"]["LR_shape"], [1,3,17,270,480])
        self.assertEqual(paired["restoration"]["padded_HR_T_HW"], [17,1088,1920])
        self.assertEqual(paired["restoration"]["tile_count"], 30)
        self.assertEqual({p["restoration"]["steps"] for p in (native,paired)}, {3})

    def test_protocol_precedes_all_loads_and_exact_modes_score_discarded(self):
        with patch("torch.cuda.synchronize", side_effect=AssertionError("CPU fixture must not touch GPU")):
            result = self.execute()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(self.native_calls, [
            {"shape":[1,3,17,576,720],"reference_index":None,"HRI_shape":None,"reference_trained":False},
            {"shape":[1,3,17,270,480],"reference_index":0,"HRI_shape":[1,3,1,1080,1920],"reference_trained":True},
            {"shape":[1,3,17,576,720],"reference_index":None,"HRI_shape":None,"reference_trained":True}])
        self.assertFalse(result["quality_scores_retained"])
        self.assertFalse(result["capacity_upperbound_claimed"])
        protocol=json.loads((self.run/"protocol.json").read_text())
        self.assertTrue(protocol["MUSIQ_assets"]["synthetic_fixture"])
        self.assertEqual(result["MUSIQ_resource_forwards"]["cpu"]["metric_provenance"]["backend"],"synthetic_fixture")
        text = "\n".join(p.read_text() for p in self.run.glob("*.json"))
        self.assertNotIn("17.123456789", text)
        self.assertNotIn('"per_frame_scores"', text)
        self.assertIn("whole process lifetime high-water", text)
        self.assertIn("20step", result["scope"])

    def test_guard_covers_inference_and_metric_CUDA_before_any_load(self):
        for device, metric_devices in (("cuda",("cpu",)),("cpu",("cuda",))):
            with patch("vsr.native_resource_probe.assert_gpu_idle", side_effect=RuntimeError("active training")):
                with self.assertRaisesRegex(RuntimeError,"active training"):
                    run_resource_probe(self.root,self.paths["no_reference"],self.paths["reference"],
                        device=device,metric_devices=metric_devices,run_dir=self.run,factories=self.factories())
        self.assertEqual(self.events, [])
        self.assertFalse(self.run.exists())

    def test_failure_preserves_partial_observations_cost_and_never_falls_back(self):
        self.fail_second_restore = True
        with self.assertRaisesRegex(NativeQAAuditError,"observed OOM"):
            self.execute()
        status = json.loads((self.run/"status.json").read_text())
        obs = self.run/"failure_observations.json"
        self.assertEqual(status["failure_observations_sha256"], hashlib.sha256(obs.read_bytes()).hexdigest())
        partial = json.loads(obs.read_text())
        self.assertEqual(partial["partial_restore_audit"]["observed_tiles"],1)
        self.assertEqual(len(partial["completed_native_cases"]),1)
        self.assertEqual(status["status"],"failed")
        self.assertGreater(status["wall_seconds"],0)
        self.assertTrue(status["no_silent_fallback"])
        self.assertEqual(len(self.native_calls),2)
        self.assertEqual(self.metric_devices,[])
        phases=json.loads((self.run/"phase_observations.json").read_text())
        self.assertEqual(phases["reference_maximum_paired_on"]["status"],"failed")
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertEqual(json.loads((self.run/"status.json").read_text())["status"],"failed")

    def test_changed_synthetic_shape_fails_before_restoration(self):
        self.bad_shape=True
        with self.assertRaisesRegex(ValueError,"fixed resource geometry"):
            self.execute()
        self.assertEqual(self.native_calls,[])
        self.assertEqual(json.loads((self.run/"status.json").read_text())["status"],"failed")

    def test_explicit_both_metric_devices_are_retained_without_scores(self):
        with ExitStack() as stack:
            guard=stack.enter_context(patch("vsr.native_resource_probe.assert_gpu_idle", return_value={"fixture":True}))
            stack.enter_context(patch("vsr.native_resource_probe._Measurements", FakeMeasurements))
            stack.enter_context(patch("torch.cuda.empty_cache"))
            result=run_resource_probe(self.root,self.paths["no_reference"],self.paths["reference"],
                device="cpu",metric_devices=("cpu","cuda"),run_dir=self.run,factories=self.factories())
        self.assertEqual(guard.call_count,2)
        self.assertEqual(self.metric_devices,["cpu","cuda:0"])
        self.assertEqual(set(result["MUSIQ_resource_forwards"]),{"cpu","cuda:0"})
        for value in result["MUSIQ_resource_forwards"].values():
            self.assertTrue(value["SR_model_and_VAE_co_resident"])
            self.assertTrue(value["synthetic_quality_value_discarded"])

    def test_real_MUSIQ_assets_require_CPU_verification_before_any_load(self):
        factories=self.factories();del factories["musiq"]
        with patch("vsr.native_resource_probe._verify_musiq_assets", side_effect=ValueError("unverified MUSIQ assets")):
            with self.assertRaisesRegex(ValueError,"unverified MUSIQ assets"):
                run_resource_probe(self.root,self.paths["no_reference"],self.paths["reference"],
                    device="cpu",run_dir=self.run,factories=factories)
        self.assertEqual(self.events,[])
        self.assertFalse(self.run.exists())

    def test_metric_weight_full_SHA_and_source_manifest_are_bound_and_tamper_fails(self):
        from vsr.no_reference import KONIQ_CHECKPOINT
        weight=self.root/"weights"/KONIQ_CHECKPOINT;weight.parent.mkdir();weight.write_bytes(b"synthetic-pinned-weights")
        manifest=self.root/"docs/musiq-provenance.json";manifest.parent.mkdir();manifest.write_text('{"fixture":true}')
        digest=hashlib.sha256(weight.read_bytes()).hexdigest()
        source={"source_repo":"synthetic_fixture", "files":{"musiq_arch.py":{"vendored_sha256":"a"*64}}}
        with patch("vsr.no_reference.verify_musiq_sources",return_value=source) as verified:
            with patch("vsr.no_reference.KONIQ_SHA256",digest):
                record=_verify_musiq_assets(self.root)
                verified.assert_called_with(self.root)
                self.assertEqual(record["weights_sha256"],digest)
                self.assertEqual(record["provenance_manifest_sha256"],hashlib.sha256(manifest.read_bytes()).hexdigest())
                self.assertEqual(record["vendor_file_sha256"]["musiq_arch.py"],"a"*64)
                weight.write_bytes(b"changed-synthetic-weights")
                with self.assertRaisesRegex(ValueError,"pinned KonIQ SHA256"):
                    _verify_musiq_assets(self.root)

    def test_second_guard_failure_logs_without_initializing_CUDA(self):
        with ExitStack() as stack:
            stack.enter_context(patch("vsr.native_resource_probe.assert_gpu_idle",
                side_effect=[{"idle":True},RuntimeError("another GPU job appeared")]))
            stack.enter_context(patch("torch.cuda.is_initialized",return_value=False))
            query=stack.enter_context(patch("torch.cuda.mem_get_info",side_effect=AssertionError("must not initialize CUDA for failure logging")))
            cache=stack.enter_context(patch("torch.cuda.empty_cache"))
            with self.assertRaisesRegex(RuntimeError,"another GPU job appeared"):
                run_resource_probe(self.root,self.paths["no_reference"],self.paths["reference"],
                    device="cuda",run_dir=self.run,factories=self.factories())
        self.assertTrue((self.run/"protocol.json").exists())
        self.assertEqual(self.events,[])
        query.assert_not_called();cache.assert_not_called()
        failure=json.loads((self.run/"failure_observations.json").read_text())
        self.assertTrue(failure["cuda_memory"]["cuda:0"]["CUDA_not_initialized"])
        self.assertEqual(json.loads((self.run/"status.json").read_text())["status"],"failed")


if __name__ == "__main__":
    unittest.main()
