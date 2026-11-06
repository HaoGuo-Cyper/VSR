"""Interrupted cache recovery uses fresh temporary data, never task state."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from vsr.audit import file_hash
from vsr.data import DATA_VERSION, _digest, build_manifest
from vsr.alignment import latent_geometry
from vsr.prepared_cache import ENCODER_FILES, _STRICT_FLAGS, _selected_ids, seal_prepared_cache

SOURCE = Path(__file__).resolve().parent.parent


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def fixture(root):
    project = root / "VSR"
    run = project / "runs/interrupted_formal"
    run.mkdir(parents=True)
    raw = root / "raw"
    raw.mkdir()
    for i in range(5):
        (raw / f"source{i:05d}_0_0to99.mp4").write_bytes(f"raw video bytes {i}".encode())
    manifest = build_manifest(raw, seed=42)
    dump(project / "manifests/hq.json", manifest)
    for name in ENCODER_FILES:
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((SOURCE / name).read_bytes())
    hashes = {name: file_hash(project / name) for name in ENCODER_FILES}
    model = project / "official_model"
    official = {
        "vae": {"temperal_downsample": [False, True, True], "dim_mult": [1, 2, 4, 4], "z_dim": 16,
                "latents_mean": [0.] * 16, "latents_std": [1.] * 16},
        "transformer": {"patch_size": [1, 2, 2], "in_channels": 16},
        "text_encoder": {"d_model": 4},
    }
    siblings, audited = [], []
    for name, value in official.items():
        path = model / name / "config.json"
        dump(path, value)
        content = path.read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        siblings.append({"rfilename": f"{name}/config.json", "blobId": digest})
        audited.append({"path": str(path), "algorithm": "git_blob_sha1", "digest": digest})
    model_id, revision = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "0fad780a534b6463e45facd96134c9f345acfa5b"
    dump(project / "docs/official-model-manifest.json", {"id": model_id, "sha": revision, "siblings": siblings})
    config = {"seed": 42, "model": {"repo_id": model_id, "revision": revision, "local_dir": str(model)},
              "data": {"manifest": "manifests/hq.json", "num_frames": 9, "height": 16, "width": 16,
                       "scale": 4, "temporal_stride": 4, "fps": "preserve_sampled_input_fps",
                       "selection": "train", "val_limit": 1},
              "reproducibility": {"mode": "strict"}}
    dump(project / "configs/formal.json", config)
    config["_sha256"] = file_hash(project / "configs/formal.json")
    hashes["configs/formal.json"] = config["_sha256"]
    dump(run / "config.json", config)
    dump(run / "environment.json", {"code_sha256": hashes})
    dump(run / "reproducibility.json", {"mode": "strict", "flags": _STRICT_FLAGS})
    dump(run / "official_artifacts.json", {"status": "passed", "repo_id": model_id, "revision": revision,
         "official_weights_only": True, "legacy_state_loaded": False, "files": audited})
    train_ids, val_ids = _selected_ids(manifest, config)
    records = {r["id"]: r for r in manifest["records"]}
    training = [records[i] for i in train_ids]
    validation = [records[i] for i in val_ids]
    selected = training + validation
    dump(run / "selection.json", {"manifest_sha256": manifest["manifest_sha256"],
         "training_ids": [r["id"] for r in training], "validation_ids": [r["id"] for r in validation], "test_accessed": False})
    geometry = latent_geometry(official["vae"], official["transformer"], 9, 16, 16)
    dump(run / "correspondence.json", geometry)
    latents, binding0 = [], None
    for i, record in enumerate(selected):
        directory = run / f"clip_{i}"
        directory.mkdir()
        metadata = {"data_version": DATA_VERSION, "record_id": record["id"], "source_id": record["source_id"],
                    "source_path": record["path"], "source_sha256": record["sha256"], "split": record["split"],
                    "seed": 42, "source_indices": list(range(0, 33, 4)), "temporal_stride": 4,
                    "crop_tlhw": [0, 0, 16, 16], "original_fps": 24., "output_fps": 6.,
                    "fps_policy": "preserve_sampled_input_fps", "valid_region": "entire_crop_no_padding",
                    "caption": record["caption"], "caption_status": record["caption_status"],
                    "degradation": {"scale": 4, "kernel": "torch_bicubic", "antialias": True,
                                    "align_corners": False, "clamp_01": True, "temporal_parameters": "shared_for_all_frames"}}
        binding = {"data_cache_key": _digest(metadata), "manifest_sha256": manifest["manifest_sha256"],
                   "official_revision": revision, "code_sha256": hashes, "config_sha256": config["_sha256"]}
        if i == 0:
            binding0 = binding
        dump(directory / "metadata.json", {**metadata, "binding": binding})
        torch.save({"hr": torch.full((1, 3, 9, 16, 16), .5), "lr": torch.full((1, 3, 9, 4, 4), .5),
                    "lr_up": torch.full((1, 3, 9, 16, 16), .5)}, directory / "paired_rgb.pt")
        pair = {"clean": torch.full((1, 16, 3, 2, 2), .25 + i), "low": torch.full((1, 16, 3, 2, 2), .125 + i)}
        latents.append(pair)
        torch.save({**pair, "binding": binding}, directory / "fresh_latents.pt")
        dump(directory / "vae_roundtrip.json", {"posterior": "mode", "vae_frozen": True,
             "vae_has_grad": False, "shape": [1, 3, 9, 16, 16]})
    torch.save({"latents": latents, "binding": binding0, "teacher_features": None}, run / "fresh_training_cache.pt")
    torch.save({"A high quality video.": torch.ones(1, 512, 4)}, run / "fresh_text_embeddings.pt")
    dump(run / "text_audit.json", {"prompts": ["A high quality video."], "encoder_frozen": True,
         "encoder_has_grad": False, "text_max_length": 512})
    dump(run / "status.json", {"status": "interrupted", "reason": "synthetic SIGTERM"})
    # Deliberately unreadable task bytes: a successful seal must never open them.
    (run / "Align").mkdir()
    (run / "Align/checkpoint.pt").write_bytes(b"MUST NOT READ TASK STATE")
    return project, run


class PreparedCacheTests(unittest.TestCase):
    def seal(self, project, run, **kwargs):
        return seal_prepared_cache(run, project_root=project, allowed_root=project / "runs", quiet_seconds=0, **kwargs)

    def test_complete_interrupted_prep_seals_compatible_allowlist_without_task_state(self):
        with tempfile.TemporaryDirectory() as directory:
            project, run = fixture(Path(directory))
            original_load = torch.load
            with patch("vsr.prepared_cache.torch.load", wraps=original_load) as loader:
                seal = self.seal(project, run)
            self.assertTrue(seal["preparation_completed"])
            self.assertEqual(seal["origin_training_status"], "interrupted")
            self.assertFalse(seal["entire_experiment_completed"])
            self.assertFalse(seal["task_checkpoint_read"])
            self.assertFalse(seal["optimizer_state_read"])
            self.assertTrue(seal["validation"]["raw_source_bytes_rehashed"])
            for call in loader.call_args_list:
                self.assertNotIn("Align", str(call.args[0]))
                self.assertEqual(call.kwargs, {"map_location": "cpu", "weights_only": True})
            for name, expected in seal["artifact_sha256"].items():
                self.assertEqual(file_hash(run / name), expected)
            self.assertEqual(json.loads((run / "status.json").read_text())["status"], "interrupted")
            self.assertEqual(self.seal(project, run), seal)

    def test_status_active_incomplete_and_nonown_roots_rejected_without_seal(self):
        for kind in ("running", "failed", "completed", "active", "prep_failed", "outside"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                project, run = fixture(Path(directory))
                if kind in ("running", "failed", "completed"):
                    dump(run / "status.json", {"status": kind})
                elif kind == "active":
                    dump(project / "runs/active_training.json", {"status": "running", "run": str(run)})
                elif kind == "prep_failed":
                    dump(run / "status.json", {"status": "interrupted", "preparation_status": "failed"})
                with self.assertRaises(ValueError):
                    if kind == "outside":
                        seal_prepared_cache(run, project_root=project, allowed_root=Path(directory), quiet_seconds=0)
                    else:
                        self.seal(project, run)
                self.assertFalse((run / "cache_asset_seal.json").exists())

    def test_source_split_stride_encoder_geometry_and_missing_files_rejected(self):
        for kind in ("raw", "split", "stride", "encoder", "geometry", "official", "missing", "strict"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                project, run = fixture(Path(directory))
                path = run / "clip_0/metadata.json"
                metadata = json.loads(path.read_text())
                if kind == "raw":
                    Path(metadata["source_path"]).write_bytes(b"changed source")
                elif kind in ("split", "stride"):
                    metadata["split" if kind == "split" else "temporal_stride"] = "val" if kind == "split" else 1
                    metadata["binding"]["data_cache_key"] = _digest({k:v for k,v in metadata.items() if k != "binding"})
                    dump(path, metadata)
                elif kind == "encoder":
                    (project / "vsr/preflight.py").write_text("changed encoder")
                elif kind == "geometry":
                    g = json.loads((run / "correspondence.json").read_text())
                    g["latent_grid"][1] += 1
                    dump(run / "correspondence.json", g)
                elif kind == "official":
                    (project / "official_model/vae/config.json").write_text("{}")
                elif kind == "missing":
                    (run / "clip_0/fresh_latents.pt").unlink()
                else:
                    repro = json.loads((run / "reproducibility.json").read_text())
                    repro["flags"]["sdp_flash"] = True
                    dump(run / "reproducibility.json", repro)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.seal(project, run)
                self.assertFalse((run / "cache_asset_seal.json").exists())

    def test_invalid_rgb_latents_caption_and_latent_binding_rejected(self):
        for kind in ("rgb_nan", "rgb_range", "latent_nan", "latent_shape", "latent_equal", "latent_binding", "top_binding", "caption", "text_nan", "text_shape"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                project, run = fixture(Path(directory))
                if kind.startswith("rgb"):
                    path = run / "clip_0/paired_rgb.pt"
                    payload = torch.load(path, weights_only=True)
                    payload["hr"][0,0,0,0,0] = float("nan") if kind == "rgb_nan" else 1.5
                elif kind.startswith("latent"):
                    path = run / "clip_0/fresh_latents.pt"
                    payload = torch.load(path, weights_only=True)
                    if kind == "latent_nan": payload["clean"].flatten()[0] = float("nan")
                    elif kind == "latent_shape": payload["clean"] = torch.zeros(1)
                    elif kind == "latent_equal": payload["clean"].add_(.1)
                    else: payload["binding"]["data_cache_key"] = "wrong"
                elif kind == "top_binding":
                    path = run / "fresh_training_cache.pt"
                    payload = torch.load(path, weights_only=True)
                    payload["binding"]["data_cache_key"] = "wrong"
                else:
                    path = run / "fresh_text_embeddings.pt"
                    payload = torch.load(path, weights_only=True)
                    if kind == "caption": payload["wrong caption"] = payload.pop("A high quality video.")
                    elif kind == "text_nan": payload["A high quality video."].flatten()[0] = float("nan")
                    else: payload["A high quality video."] = torch.zeros(1)
                torch.save(payload, path)
                with self.assertRaises(ValueError): self.seal(project, run)
                self.assertFalse((run / "cache_asset_seal.json").exists())

    def test_existing_different_seal_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            project, run = fixture(Path(directory))
            target = run / "cache_asset_seal.json"
            target.write_text('{"different": true}\n')
            before = target.read_bytes()
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                self.seal(project, run)
            self.assertEqual(target.read_bytes(), before)

    def test_concurrent_preparation_change_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            project, run = fixture(Path(directory))
            def mutate(message):
                path = run / "clip_0/vae_roundtrip.json"
                path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "changed during validation"):
                self.seal(project, run, progress=mutate)
            self.assertFalse((run / "cache_asset_seal.json").exists())

    def test_status_or_active_pointer_change_during_validation_refused(self):
        for kind in ("status", "active"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                project, run = fixture(Path(directory))
                def mutate(message):
                    if kind == "status":
                        dump(run / "status.json", {"status": "interrupted", "preparation_status": "failed"})
                    else:
                        dump(project / "runs/active_training.json", {"status": "running", "run": str(run)})
                with self.assertRaisesRegex(ValueError, "changed|running"):
                    self.seal(project, run, progress=mutate)
                self.assertFalse((run / "cache_asset_seal.json").exists())


if __name__ == "__main__":
    unittest.main()
