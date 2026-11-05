"""Synthetic PNG/ZIP/ffprobe fixtures only; never decode real external data."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import io
import json
import tempfile
import unittest
from unittest import mock
import zipfile

from PIL import Image
import torch

from vsr.external_data import ExternalReader, REDS4_IDS, probe_external_streams


def digest(data):
    return hashlib.sha256(data).hexdigest()


def png(width, height, value):
    stream = io.BytesIO()
    Image.new("RGB", (width, height), (value, value, value)).save(stream, format="PNG")
    return stream.getvalue()


class ExternalDataTest(unittest.TestCase):
    def fixture(self, root):
        datasets = {}
        # A byte-only synthetic archive is sufficient to check the provenance
        # seal; REDS pixel payloads are separately verified on disk.
        red_archive = root / "red-source.tar.gz"
        red_archive.write_bytes(b"synthetic-source-provenance-fixture")
        records = []
        for sequence in REDS4_IDS:
            row = {"id": sequence, "frame_count": 3}
            for role, width, height in [("lq", 3, 2), ("gt", 12, 8)]:
                frames = []
                for number, value in enumerate((0, 127, 255)):
                    data = png(width, height, value)
                    relative = f"data/external/REDS4/{role}/{sequence}/{number:08d}.png"
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    frames.append({"path": relative, "bytes": len(data), "sha256": digest(data)})
                row[role] = {"width": width, "height": height, "frame_count": 3, "frames": frames}
            records.append(row)
        datasets["REDS4"] = {"dataset": "REDS4", "training_allowed": False, "tuning_allowed": False,
            "sequence_ids": list(REDS4_IDS), "sequence_count": 4, "records": records,
            "source": {"archive_path": str(red_archive), "archive_bytes": red_archive.stat().st_size,
                       "archive_sha256": digest(red_archive.read_bytes())}}
        for name in ("YouHQ40", "VideoLQ"):
            archive = root / f"{name}.zip"
            row = {"id": "000"}
            roles = [("lq", "LQ", 3, 2), ("gt", "GT", 12, 8)] if name == "YouHQ40" else [("lq", "LQ", 5, 4)]
            with zipfile.ZipFile(archive, "w") as output:
                for role, branch, width, height in roles:
                    for number, value in enumerate((0, 127, 255)):
                        output.writestr(f"{name}/{branch}/000/{number:04d}.png", png(width, height, value))
                    relative = f"data/external/{name}/{branch}-Video/000.mkv"
                    container = root / relative
                    container.parent.mkdir(parents=True, exist_ok=True)
                    container.write_bytes(f"synthetic-container-{name}-{role}".encode())
                    row[role] = {"path": relative, "bytes": container.stat().st_size,
                                 "sha256": digest(container.read_bytes())}
            datasets[name] = {"dataset": name, "training_allowed": False, "tuning_allowed": False,
                "sequence_count": 1, "records": [row],
                "source": {"archive_path": str(archive), "archive_bytes": archive.stat().st_size,
                           "archive_sha256": digest(archive.read_bytes())}}
        manifest = {"schema_version": 1, "datasets": datasets}
        path = root / "manifests/external.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest))
        return manifest

    def ffprobe(self, command, **kwargs):
        self.commands.append(command)
        path = command[-1]
        width, height = (12, 8) if "GT-Video" in path else ((5, 4) if "VideoLQ" in path else (3, 2))
        rate = "30/1" if self.probe_error != "fps" or "GT-Video" not in path else "24/1"
        if self.probe_error == "geometry" and "GT-Video" in path:
            width += 1
        stream = {"width": width, "height": height, "avg_frame_rate": rate, "r_frame_rate": rate,
                  "pix_fmt": "rgb24", "codec_name": "synthetic", "nb_frames": "N/A"}
        return SimpleNamespace(stdout=json.dumps({"streams": [stream]}))

    def setUp(self):
        self.commands, self.probe_error = [], None
        torch.set_num_threads(2)

    def test_probe_has_no_count_decode_and_checks_pair_geometry_and_fps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.fixture(root)
            with mock.patch("vsr.external_data.subprocess.run", side_effect=self.ffprobe), \
                 mock.patch("vsr.external_data.Image.open", side_effect=AssertionError("No pixel decode during probe")):
                result = probe_external_streams(root, manifest, preview_fps=25)
                self.assertEqual(result["status"], "passed")
                self.assertEqual(len(self.commands), 3)
                self.assertTrue(all("-count_frames" not in command and "-show_frames" not in command for command in self.commands))
                self.assertTrue(all("stream=" in " ".join(command) for command in self.commands))
                red = result["datasets"]["REDS4"]["records"][0]
                self.assertIsNone(red["original_fps"])
                self.assertEqual(red["preview_fps"], 25)
                self.assertIsNone(result["datasets"]["YouHQ40"]["records"][0]["lq"]["nb_frames"])
                for error in ("geometry", "fps"):
                    self.probe_error = error
                    rejected = probe_external_streams(root, manifest)
                    self.assertEqual(rejected["status"], "failed")
                    self.assertTrue(rejected["issues"])

    def test_native_png_reads_prefix_pairing_and_declared_reds_preview_fps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            with mock.patch("vsr.external_data.subprocess.run", side_effect=self.ffprobe):
                with ExternalReader(root, preview_fps=25) as reader:
                    red = reader.load("REDS4", "011", num_frames=2)
                    self.assertEqual(tuple(red["lr"].shape), (1, 3, 2, 2, 3))
                    self.assertEqual(tuple(red["hr"].shape), (1, 3, 2, 8, 12))
                    self.assertEqual(red["fps"], 25)
                    self.assertIsNone(red["metadata"]["original_fps"])
                    self.assertIn("not_original", red["metadata"]["fps_source"])
                    self.assertEqual(red["metadata"]["source_frame_indices"], [0, 1])
                    self.assertTrue(red["metadata"]["benchmark_prefix_explicit"])
                    paired = reader.read("YouHQ40", "000", num_frames=2, start=1)
                    self.assertEqual(tuple(paired["lr"].shape), (1, 3, 2, 2, 3))
                    self.assertEqual(tuple(paired["hr"].shape), (1, 3, 2, 8, 12))
                    self.assertEqual(paired["fps"], 30)
                    self.assertEqual(paired["metadata"]["source_frame_names"], ["0001.png", "0002.png"])
                    torch.testing.assert_close(paired["lr"][:, :, 0], torch.full((1, 3, 2, 3), 127/255))
                    self.assertEqual(paired["metadata"]["pixel_source"], "original_PNG_members_from_supplied_ZIP")
                    self.assertFalse(paired["metadata"]["spatial_resize"])
                    self.assertEqual(paired["metadata"]["paired_scale"], 4)
                    real = reader.load("VideoLQ", "000", num_frames=3)
                    self.assertNotIn("hr", real)
                    self.assertEqual(tuple(real["lr"].shape), (1, 3, 3, 4, 5))
                    self.assertIsNone(real["metadata"]["paired_scale"])
                    self.assertTrue(real["metadata"]["benchmark_interval_explicit"]["full_source"])
                    self.assertEqual(real["lr"].device.type, "cpu")
                    self.assertEqual(real["lr"].dtype, torch.float32)

    def test_archive_sha_once_per_reader_no_pixels_until_load_and_png_tamper(self):
        from vsr import external_data
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.fixture(root)
            source_paths = [Path(row["source"]["archive_path"]) for row in manifest["datasets"].values()]
            original = external_data._sha256_file
            with mock.patch("vsr.external_data.subprocess.run", side_effect=self.ffprobe), \
                 mock.patch("vsr.external_data._sha256_file", wraps=original) as hashed:
                with mock.patch("vsr.external_data.Image.open", side_effect=AssertionError("Constructor must not decode pixels")):
                    reader = ExternalReader(root)
                reader.load("YouHQ40", "000", num_frames=1)
                reader.load("YouHQ40", "000", num_frames=1)
                for source in source_paths:
                    self.assertEqual(sum(Path(call.args[0]) == source for call in hashed.call_args_list), 1)
                frame = root / manifest["datasets"]["REDS4"]["records"][0]["lq"]["frames"][0]["path"]
                frame.write_bytes(frame.read_bytes() + b"tampered")
                with self.assertRaisesRegex(ValueError, "PNG size/SHA256 mismatch"):
                    reader.load("REDS4", "000", num_frames=1)
                reader.close()

    def test_project_owned_archive_copy_is_portable_and_still_hash_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.fixture(root)
            recorded = Path(manifest["datasets"]["REDS4"]["source"]["archive_path"])
            owned = root / "data/external/archives" / recorded.name
            owned.parent.mkdir(parents=True, exist_ok=True)
            owned.write_bytes(recorded.read_bytes())
            recorded.unlink()
            with mock.patch("vsr.external_data.subprocess.run", side_effect=self.ffprobe):
                with ExternalReader(root) as reader:
                    result = reader.load("REDS4", "000", num_frames=1)
                    self.assertEqual(result["metadata"]["source_archive"], str(owned))
                    self.assertEqual(result["metadata"]["recorded_source_archive"], str(recorded))
                owned.write_bytes(b"wrong-project-owned-copy")
                with self.assertRaisesRegex(ValueError, "archive size/SHA256 mismatch"):
                    ExternalReader(root)

    def test_rejects_insufficient_frames_unsafe_zip_archive_seal_and_reds_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.fixture(root)
            path = root / "manifests/external.json"
            with mock.patch("vsr.external_data.subprocess.run", side_effect=self.ffprobe):
                with ExternalReader(root) as reader:
                    with self.assertRaisesRegex(ValueError, "exceeds source"):
                        reader.load("YouHQ40", "000")
                    with self.assertRaises(ValueError):
                        reader.load("VideoLQ", "000", num_frames=2, start=2)
                    with self.assertRaises(KeyError):
                        reader.load("REDS4", "001", num_frames=1)
                seal = root / "docs/external-data-provenance.json"
                seal.parent.mkdir(exist_ok=True)
                seal.write_text(json.dumps({"manifest_sha256": "0"*64}))
                with self.assertRaisesRegex(ValueError, "provenance seal"):
                    ExternalReader(root)
                seal.unlink()
                manifest["datasets"]["REDS4"]["sequence_ids"][0] = "001"
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "standard REDS4"):
                    ExternalReader(root)
                manifest["datasets"]["REDS4"]["sequence_ids"][0] = "000"
                archive = Path(manifest["datasets"]["YouHQ40"]["source"]["archive_path"])
                with zipfile.ZipFile(archive, "a") as output:
                    output.writestr("../escape.png", png(2, 2, 0))
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "archive size/SHA256 mismatch"):
                    ExternalReader(root)
                manifest["datasets"]["YouHQ40"]["source"].update(archive_bytes=archive.stat().st_size,
                    archive_sha256=digest(archive.read_bytes()))
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "traversal"):
                    ExternalReader(root)


if __name__ == "__main__":
    unittest.main()
