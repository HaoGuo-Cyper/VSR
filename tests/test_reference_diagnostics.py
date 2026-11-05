"""Zero-update diagnostics use tiny/fake CPU data, never real test pixels."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from diffusers import WanTransformer3DModel

from vsr.alignment import latent_geometry
from vsr.audit import file_hash
from vsr.checkpoints import OFFICIAL_REPO_ID, OFFICIAL_REVISION
from vsr.data import _digest, build_manifest
from vsr.reference_model import ReferenceConditionalWan
from vsr.reference_diagnostics import (assert_gpu_idle, collect_reference_diagnostics,
                                       load_reference_inputs, main, seal_reference_supervision)

SOURCE = Path(__file__).resolve().parent.parent


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


class RecordingReference(ReferenceConditionalWan):
    def __init__(self, native, **kwargs):
        super().__init__(native, **kwargs)
        self.observed = []
        self.fail = False

    def forward(self, noisy, timestep, text, low, *args, **kwargs):
        self.observed.append({"noisy": noisy.detach().clone(), "text_requires_grad": text.requires_grad,
            "low_requires_grad": low.requires_grad, "ref_requires_grad": kwargs.get("ref_latent").requires_grad if kwargs.get("ref_latent") is not None else None})
        flow, student = super().forward(noisy, timestep, text, low, *args, **kwargs)
        return flow, student*float("nan") if self.fail else student


def model():
    native = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
        in_channels=4, out_channels=4, text_dim=16, freq_dim=16, ffn_dim=48, num_layers=3, rope_max_seq_len=16)
    result = RecordingReference(native, reference_layer=1, reference_stem_dim=8, rank=2, alpha=2,
        injection_layers=(0, 1), alignment_layer=1, stem_channels=8, projector_hidden=32, teacher_dim=16,
        gradient_checkpointing=True)
    # Stand in for an already trained checkpoint without optimizing in this
    # test: open the zero residual maps so auxiliary-path gradients can flow.
    with torch.no_grad():
        for layer in result.lrv_residuals.values():
            layer.weight.normal_(std=.03)
        result.reference_adapter.output_projection.weight.normal_(std=.03)
    return result


def sample(record, split, offset=0):
    return {"metadata": {"record_id": record, "source_id": record+"_source", "split": split,
                          "source_indices": [offset+4*i for i in range(5)]},
        "clean": torch.randn(1, 4, 2, 4, 4, requires_grad=True),
        "low": torch.randn(1, 4, 2, 4, 4, requires_grad=True),
        "text": torch.randn(1, 4, 16, requires_grad=True),
        "teacher": torch.randn(1, 8, 16, requires_grad=True),
        "flows": {key: torch.zeros(1, 1, 2, 32, 32, requires_grad=True) for key in ("forward", "backward")},
        "reference_latents": {index: torch.randn(1, 4, 1, 4, 4, requires_grad=True) for index in (0, 4)}}


def inputs():
    return {"training": sample("train0", "train", 20), "validation": [sample("val0", "val", 40), sample("val1", "val", 80)],
            "geometry": {"latent_grid": [2, 4, 4], "token_grid": [2, 2, 2], "anchors": [0, 4]}}


CONFIG = {"seed": 20261102, "data": {"num_frames": 5, "height": 32, "width": 32, "temporal_stride": 4}, "training": {"lambda_align": .01, "lambda_relation": 1.},
          "reproducibility": {"mode": "strict"}, "flow": {"fb_alpha": .01, "fb_beta": .5}}


def loader_fixture(directory):
    root = directory/"VSR"; raw = directory/"raw"; raw.mkdir()
    for index in range(5):
        (raw/f"source{index:05d}_0_0to99.mp4").write_bytes(f"synthetic raw bytes{index}".encode())
    manifest = build_manifest(raw, seed=42)
    write(root/"manifests/hq.json", manifest)
    hashes = {}
    for name in ("vsr/data.py", "vsr/preflight.py"):
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((SOURCE/name).read_bytes())
        hashes[name] = file_hash(path)
    official = root/"official"
    vae = {"temperal_downsample": [False, True, True], "dim_mult": [1, 2, 4, 4], "z_dim": 4}
    dit = {"patch_size": [1, 2, 2], "in_channels": 4}
    write(official/"vae/config.json", vae); write(official/"transformer/config.json", dit)
    source, run = root/"runs/frozen_source", root/"runs/reference_formal"
    source.mkdir(parents=True); run.mkdir()
    data = {"manifest": "manifests/hq.json", "selection": "train", "num_frames": 5, "height": 32, "width": 32,
            "scale": 4, "fps": "preserve_sampled_input_fps", "temporal_stride": 4, "val_limit": 1}
    source_config = {"seed": 42, "model": {"repo_id": OFFICIAL_REPO_ID, "revision": OFFICIAL_REVISION, "local_dir": str(official)},
        "data": data, "reproducibility": {"mode": "strict"}}
    source_config["_sha256"] = "source_origin_config"
    config = {**copy.deepcopy(source_config), "data": {**data, "cache_from": str(source)},
        "reference": {"enabled": True}, "training": {"variants": ["Base", "Align", "Relation", "Both"], "updates": 8192},
        "teacher": {"weights_sha256": "f"*64, "revision": "teacher_revision"}, "flow": {"weights_sha256": "e"*64, "updates": 12}}
    config["_sha256"] = "reference_origin_config"
    training = sorted([r for r in manifest["records"] if r["split"] == "train"],
                      key=lambda r: hashlib.sha256(f"train:42:{r['id']}".encode()).hexdigest())
    validation = sorted([r for r in manifest["records"] if r["split"] == "val"],
                        key=lambda r: hashlib.sha256(f"val:42:{r['id']}".encode()).hexdigest())[:1]
    selected = training+validation
    selection = {"manifest_sha256": manifest["manifest_sha256"], "training_ids": [r["id"] for r in training],
                 "validation_ids": [r["id"] for r in validation], "test_accessed": False}
    geometry = latent_geometry(vae, dit, 5, 32, 32)
    for folder, cfg in ((source, source_config), (run, config)):
        write(folder/"config.json", cfg); write(folder/"selection.json", selection); write(folder/"correspondence.json", geometry)
    write(source/"environment.json", {"code_sha256": hashes})
    producer = {"vsr/reference_training.py": "a"*64, "vsr/preflight.py": hashes["vsr/preflight.py"]}
    write(run/"environment.json", {"code_sha256": {**hashes, **producer, "vsr/alignment.py": "b"*64,
        "vsr/reference_diagnostics.py": file_hash(SOURCE/"vsr/reference_diagnostics.py")}})
    write(run/"teacher_cache_contract.json", {"weights_sha256": "f"*64, "source_revision": "teacher_revision",
        "alignment_source_sha256": "b"*64, "frozen": True})
    write(run/"flow_provenance.json", {"anchors": geometry["anchors"], "clip_count": len(selected),
        "checkpoint_sha256": "e"*64, "num_flow_updates": 12, "frozen": True})
    assets, ref_rows = {}, []
    for index, record in enumerate(selected):
        metadata = {"data_version": "hq-vsr-paired-v2", "record_id": record["id"], "source_id": record["source_id"],
            "source_sha256": record["sha256"], "split": record["split"], "seed": 42, "caption": "",
            "source_indices": [0, 4, 8, 12, 16], "temporal_stride": 4, "crop_tlhw": [0, 0, 32, 32],
            "original_fps": 24., "output_fps": 6., "fps_policy": "preserve_sampled_input_fps",
            "valid_region": "entire_crop_no_padding", "degradation": {"scale": 4, "kernel": "torch_bicubic", "antialias": True,
                "align_corners": False, "clamp_01": True, "temporal_parameters": "shared_for_all_frames"}}
        binding = {"data_cache_key": _digest(metadata), "manifest_sha256": manifest["manifest_sha256"],
            "official_revision": OFFICIAL_REVISION, "config_sha256": source_config["_sha256"], "code_sha256": hashes}
        clip = source/f"clip_{index}"; clip.mkdir()
        write(clip/"metadata.json", {**metadata, "binding": binding})
        torch.save({"hr": torch.rand(1, 3, 5, 32, 32), "lr": torch.rand(1, 3, 5, 8, 8), "lr_up": torch.rand(1, 3, 5, 32, 32)}, clip/"paired_rgb.pt")
        torch.save({"clean": torch.rand(1, 4, 2, 4, 4), "low": torch.rand(1, 4, 2, 4, 4), "binding": binding}, clip/"fresh_latents.pt")
        for name in ("metadata.json", "paired_rgb.pt", "fresh_latents.pt"):
            assets[f"clip_{index}/{name}"] = file_hash(clip/name)
        directory = run/f"reference_cache/clip_{index}"; directory.mkdir(parents=True)
        refs = {i: torch.rand(1, 4, 1, 4, 4) for i in range(5)}
        ref_binding = {"source_binding": binding, "clip_metadata": metadata, "config_sha256": config["_sha256"],
            "official_revision": OFFICIAL_REVISION, "encoder_source_sha256": producer,
            "vae_config_sha256": file_hash(official/"vae/config.json"), "encoding": "one separate T=1 VAE encode per original frame",
            "posterior": "mode", "normalization": "official (latent-mean)/std", "vae_frozen": True, "frame_indices": list(range(5))}
        path = directory/"reference_latents.pt"; torch.save({"reference_latents": refs, "binding": ref_binding}, path)
        write(directory/"reference_binding.json", {"binding": ref_binding, "reference_latents_sha256": file_hash(path)})
        ref_rows.append({"clip_index": index, "record_id": record["id"], "frame_count": 5, "sha256": file_hash(path)})
    torch.save({"": torch.rand(1, 4, 16)}, source/"fresh_text_embeddings.pt")
    assets["fresh_text_embeddings.pt"] = file_hash(source/"fresh_text_embeddings.pt")
    write(source/"cache_asset_seal.json", {"artifact_sha256": assets})
    write(run/"cache_reuse.json", {"source_run": str(source), "source_asset_seal_sha256": file_hash(source/"cache_asset_seal.json")})
    torch.save([torch.rand(1, 8, 16) for _ in selected], run/"fresh_teacher_features.pt")
    torch.save([{key: torch.zeros(1, 1, 2, 32, 32) for key in ("forward", "backward")} for _ in selected], run/"fresh_anchor_flows.pt")
    write(run/"reference_cache/reference_cache_seal.json", {"single_frame_only": True, "video_latent_slicing": False,
        "new_cache": True, "legacy_state_loaded": False, "clip_count": len(selected), "encoder_source_sha256": producer, "per_clip": ref_rows})
    seal_reference_supervision(run, config, project_root=root)
    write(run/"status.json", {"status": "completed", "reference_trained": True})
    return root, run, source, config


class ReferenceDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(22); torch.set_num_threads(2)

    def test_separate_A_R_reach_generator_and_restore_state_rng_flags(self):
        net, data = model(), inputs(); net.train(); net.transformer.blocks[-1].eval()
        before = {name: value.clone() for name, value in net.state_dict().items()}
        flags = [m.training for m in net.modules()]; rng = torch.get_rng_state().clone()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        with patch("torch.cuda.init", side_effect=AssertionError("GPU forbidden")), patch("torch.optim.AdamW", side_effect=AssertionError("No optimizer permitted")), torch.no_grad():
            report = collect_reference_diagnostics(net, data, CONFIG, "Both")
        self.assertEqual(report["protocol"]["optimizer_updates"], 0)
        self.assertFalse(report["protocol"]["restoration_quality_evidence"])
        self.assertTrue(all(report["cleanup"].values()))
        self.assertEqual(report["reference_cases"]["0"]["reference_role"], "first_frame_main")
        self.assertEqual(report["reference_cases"]["4"]["reference_role"], "known_time_additional")
        for case in report["reference_cases"].values():
            for mode, result in case["modes"].items():
                self.assertEqual([r["objective"] for r in result["auxiliary_only_gradients"]], ["A", "R"])
                for row in result["auxiliary_only_gradients"]:
                    self.assertFalse(row["FM_in_backward"])
                    self.assertTrue(row["all_generator_groups_reached"])
                    self.assertTrue(row["reached"]["projector"])
                    self.assertEqual(row["gradient_groups"]["frozen"]["elements_with_grad"], 0)
                    self.assertEqual(row["gradient_groups"]["hri"]["elements_with_grad"] > 0, mode == "hri_on")
                self.assertEqual([row["record_id"] for row in result["validation"]], ["val0", "val1"])
                self.assertIn("student_effective_rank", result["validation_macro"])
                self.assertEqual(result["auxiliary_only_gradients"][0]["noise_seed"], result["auxiliary_only_gradients"][1]["noise_seed"])
        self.assertEqual(flags, [m.training for m in net.modules()]); torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(python_rng, random.getstate()); np.testing.assert_equal(numpy_rng, np.random.get_state())
        for name, value in net.state_dict().items():
            torch.testing.assert_close(before[name], value, rtol=0, atol=0)
        for entry in [data["training"], *data["validation"]]:
            tensors = [entry[k] for k in ("clean", "low", "text", "teacher")]+list(entry["flows"].values())+list(entry["reference_latents"].values())
            self.assertTrue(all(tensor.grad is None for tensor in tensors))
        self.assertTrue(all(not row["text_requires_grad"] and not row["low_requires_grad"] and row["ref_requires_grad"] is not True for row in net.observed))

    def test_variant_objectives_and_off_does_not_read_reference(self):
        class ForbiddenMapping:
            def __getitem__(self, _):
                raise AssertionError("Missing HRI must not read references")
        net, data = model(), inputs()
        for entry in [data["training"], *data["validation"]]:
            entry["reference_latents"] = ForbiddenMapping()
        for variant, expected in (("Align", ["A"]), ("Relation", ["R"])):
            report = collect_reference_diagnostics(net, data, CONFIG, variant, reference_indices=(0,), modes=("hri_off",))
            self.assertEqual(report["applicable_objectives"], expected)
            self.assertFalse(report["reference_cases"]["0"]["modes"]["hri_off"]["auxiliary_only_gradients"][0]["reached"]["hri"])
        base = collect_reference_diagnostics(net, data, CONFIG, "Base")
        self.assertEqual(base["reference_cases"], {})

    def test_failure_cleans_gradients_reference_and_training_flags(self):
        net, data = model(), inputs(); net.fail = True; net.eval(); net.transformer.blocks[-1].train()
        flags = [m.training for m in net.modules()]; rng = torch.get_rng_state().clone()
        with self.assertRaises((ValueError, RuntimeError)):
            collect_reference_diagnostics(net, data, CONFIG, "Align")
        self.assertTrue(all(p.grad is None for p in net.parameters()))
        self.assertIsNone(net._reference_context); self.assertIsNone(net._condition_tokens)
        self.assertEqual(flags, [m.training for m in net.modules()]); torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_bound_T1_cache_loader_and_changed_reference_binding(self):
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.init", side_effect=AssertionError("GPU forbidden")):
            root, run, source, config = loader_fixture(Path(directory))
            real_load, opened = torch.load, []
            def observed(path, **kwargs):
                opened.append(Path(path)); return real_load(path, **kwargs)
            with patch("vsr.reference_diagnostics.torch.load", side_effect=observed):
                loaded = load_reference_inputs(run, config, project_root=root, expected_validation_count=1)
            self.assertEqual(loaded["training"]["metadata"]["split"], "train")
            self.assertEqual(len(loaded["validation"]), 1)
            self.assertFalse(loaded["source_audit"]["video_latent_sliced_for_HRI"])
            self.assertEqual(loaded["training"]["reference_latents"][4].shape, (1, 4, 1, 4, 4))
            self.assertTrue(all(path.is_relative_to(root/"runs") and "checkpoint" not in path.name for path in opened))
            binding = run/"reference_cache/clip_0/reference_binding.json"
            row = json.loads(binding.read_text()); row["binding"]["frame_indices"] = [0, 1]
            write(binding, row)
            with self.assertRaisesRegex(ValueError, "supervision asset changed"):
                load_reference_inputs(run, config, project_root=root, expected_validation_count=1)

    def test_cache_loader_refuses_legacy_origin_and_wrong_val_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run, source, config = loader_fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "validation32"):
                load_reference_inputs(run, config, project_root=root)
            bad = {**config, "data": {**config["data"], "cache_from": "/home/hugo/Projects/ReprVSR-pretrained/runs/old"}}
            write(run/"config.json", bad)
            write(run/"cache_reuse.json", {"source_run": bad["data"]["cache_from"]})
            with patch("vsr.reference_diagnostics.torch.load", side_effect=AssertionError("No old state may be read")):
                with self.assertRaises(ValueError):
                    load_reference_inputs(run, bad, project_root=root, expected_validation_count=1)

    def test_pretraining_supervision_seal_rejects_changed_targets_and_late_reconstruction(self):
        for filename in ("fresh_teacher_features.pt", "fresh_anchor_flows.pt"):
            with tempfile.TemporaryDirectory() as directory:
                root, run, source, config = loader_fixture(Path(directory))
                path = run/filename; values = torch.load(path, weights_only=True)
                if filename.startswith("fresh_teacher"):
                    values[0] += 5
                else:
                    values[0]["forward"] += 3
                torch.save(values, path)
                with self.assertRaisesRegex(ValueError, "supervision asset changed"):
                    load_reference_inputs(run, config, project_root=root, expected_validation_count=1)
                with self.assertRaisesRegex(ValueError, "after training"):
                    seal_reference_supervision(run, config, project_root=root)
        with tempfile.TemporaryDirectory() as directory:
            root, run, source, config = loader_fixture(Path(directory))
            (run/"status.json").unlink()
            old = (run/"reference_supervision_seal.json").read_bytes()
            sealed = seal_reference_supervision(run, config, project_root=root)
            self.assertEqual(sealed["optimizer_updates_before_seal"], 0)
            self.assertEqual(old, (run/"reference_supervision_seal.json").read_bytes())
            (run/"Align").mkdir()
            with self.assertRaisesRegex(ValueError, "before any variant"):
                seal_reference_supervision(run, config, project_root=root)

    def test_idle_guard_refuses_markers_units_gpu_and_unverifiable_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/"runs").mkdir()
            def runner(args, **kwargs):
                return SimpleNamespace(returncode=0, stdout="LoadState=loaded\nActiveState=inactive\nMainPID=0\n" if args[0] == "systemctl" else "99, gjs\n")
            self.assertTrue(assert_gpu_idle(root, runner=runner, own_pid=100)["GPU_idle_verified"])
            for name in ("active_training_service.json", "active_reference_training.json", "active_training.json"):
                write(root/"runs"/name, {"status": "running"})
                with self.assertRaisesRegex(RuntimeError, "Active"):
                    assert_gpu_idle(root, runner=runner, own_pid=100)
                (root/"runs"/name).unlink()
            for stdout, message in (("LoadState=loaded\nActiveState=active\nMainPID=101\n", "service"),):
                with self.assertRaisesRegex(RuntimeError, message):
                    assert_gpu_idle(root, runner=lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout))
            def gpu_runner(args, **kwargs):
                return runner(args, **kwargs) if args[0] == "systemctl" else SimpleNamespace(returncode=0, stdout="101, /usr/bin/python\n")
            with self.assertRaisesRegex(RuntimeError, "compute"):
                assert_gpu_idle(root, runner=gpu_runner, own_pid=100)
            with self.assertRaisesRegex(RuntimeError, "Cannot verify"):
                assert_gpu_idle(root, runner=lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
            write(root/"runs/active_training_service.json", {"status": "running", "unit": "vsr-training.service"})
            stale = assert_gpu_idle(root, runner=runner, own_pid=100)
            self.assertTrue(stale["service_marker_running_but_unit_inactive"])
            def own_runner(args, **kwargs):
                return SimpleNamespace(returncode=0, stdout="LoadState=loaded\nActiveState=active\nMainPID=100\n" if args[0] == "systemctl" else "100, /usr/bin/python\n")
            own = assert_gpu_idle(root, runner=own_runner, own_pid=100)
            self.assertEqual(own["own_diagnostic_service_units"], ["vsr-training.service"])

    def test_CLI_rejects_legacy_checkpoint_before_model_or_task_loading(self):
        with patch("sys.argv", ["reference_diagnostics", "--checkpoint", "/home/hugo/Projects/ReprVSR-pretrained/runs/old.pt"]), \
                patch("vsr.reference_diagnostics.assert_gpu_idle", return_value={"synthetic": True}), \
                patch("vsr.reference_diagnostics.load_training_checkpoint", side_effect=AssertionError("No old task loading")):
            with self.assertRaisesRegex(ValueError, "Legacy"):
                main()


if __name__ == "__main__":
    unittest.main()
