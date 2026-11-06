"""External-test metadata probing and deterministic native PNG sequence reads.

The probe CLI requests ffprobe stream metadata only, without frame counting,
frame output or scoring. Pixel reads occur only through ExternalReader.load.
Original PNG members are preferred to converting the supplied MKV containers.
No source files are written, no images are resized, and no frames are selected
using model results. REDS4 PNG playback FPS is an explicitly declared preview
value because the archive does not establish original capture timestamps.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import zipfile

import numpy as np
from PIL import Image
import torch


REDS4_IDS = ("000", "011", "015", "020")
DATASETS = {"REDS4", "YouHQ40", "VideoLQ"}
PROBE_FIELDS = "width,height,avg_frame_rate,r_frame_rate,pix_fmt,codec_name,nb_frames"


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _safe_name(name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name or re.match(r"^[A-Za-z]:", name):
        raise ValueError("Invalid local/archive relative path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Absolute paths and traversal are forbidden for dataset members")
    return path


def _local(root, relative):
    path = Path(root).resolve().joinpath(*_safe_name(relative).parts)
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Dataset member escapes project root")
    return path


def _manifest(root, manifest):
    if isinstance(manifest, dict):
        data, file_hash = manifest, None
    else:
        path = Path(manifest)
        path = path if path.is_absolute() else _local(root, str(path))
        if not path.resolve().is_relative_to(Path(root).resolve()):
            raise ValueError("External manifest must be inside the selected project")
        raw = path.read_bytes()
        data, file_hash = json.loads(raw), hashlib.sha256(raw).hexdigest()
    if data.get("schema_version") != 1 or not isinstance(data.get("datasets"), dict) or not data["datasets"]:
        raise ValueError("Unsupported / empty external manifest")
    if set(data["datasets"]) - DATASETS:
        raise ValueError("Unexpected external dataset")
    for name, dataset in data["datasets"].items():
        if dataset.get("dataset") != name or dataset.get("training_allowed") is not False or dataset.get("tuning_allowed") is not False:
            raise ValueError("External dataset identity / evaluation-only contract mismatch")
        records = dataset.get("records", [])
        ids = [record.get("id") for record in records]
        if (not ids or len(set(ids)) != len(ids) or len(ids) != dataset.get("sequence_count")
                or any(not isinstance(value, str) or not re.fullmatch(r"\d{3}", value) for value in ids)):
            raise ValueError("External sequence IDs/count must be distinct and preserved")
        if name == "REDS4" and (dataset.get("sequence_ids") != list(REDS4_IDS) or set(ids) != set(REDS4_IDS)):
            raise ValueError("User-selected standard REDS4 IDs must be exactly 000/011/015/020")
        source = dataset.get("source", {})
        if not isinstance(source.get("archive_path"), str) or not re.fullmatch(r"[0-9a-f]{64}", source.get("archive_sha256", "")):
            raise ValueError("Source archive needs a full recorded SHA256 and path")
        for record in records:
            roles = ("gt", "lq") if name in ("REDS4", "YouHQ40") else ("lq",)
            if any(role not in record for role in roles):
                raise ValueError("Missing declared external GT/LQ source")
            for role in roles:
                item = record[role]
                if name == "REDS4":
                    frames = item.get("frames", [])
                    if not frames or len(frames) != item.get("frame_count") or len(frames) != record.get("frame_count"):
                        raise ValueError("REDS4 frame lists/counts differ")
                    for frame in frames:
                        _local(root, frame["path"])
                        if not re.fullmatch(r"[0-9a-f]{64}", frame.get("sha256", "")):
                            raise ValueError("Each REDS4 PNG requires a recorded SHA256")
                else:
                    _local(root, item["path"])
    return data, file_hash


def _positive_fps(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError("preview_fps must be explicitly positive and finite")
    return float(value)


def _rate(value):
    try:
        fraction = Fraction(str(value))
        return fraction if fraction > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def _probe_one(path):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=" + PROBE_FIELDS,
               "-of", "json", str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected exactly one selected video stream: {path}")
    stream = streams[0]
    if min(int(stream.get("height", 0)), int(stream.get("width", 0))) <= 0:
        raise ValueError("ffprobe did not establish native spatial dimensions")
    chosen = _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate"))
    count = stream.get("nb_frames")
    return {"path": str(path), "width": int(stream["width"]), "height": int(stream["height"]),
            "avg_frame_rate": stream.get("avg_frame_rate"), "r_frame_rate": stream.get("r_frame_rate"),
            "fps": float(chosen) if chosen is not None else None,
            "fps_fraction": str(chosen) if chosen is not None else None,
            "fps_field": "avg_frame_rate" if _rate(stream.get("avg_frame_rate")) is not None else "r_frame_rate",
            "pix_fmt": stream.get("pix_fmt"), "codec_name": stream.get("codec_name"),
            "nb_frames": int(count) if str(count).isdigit() else None,
            "nb_frames_source": "container metadata when present; never counted/decoded"}


def probe_external_streams(project_root, manifest, *, preview_fps=30.0):
    """Probe supplied MKV stream headers only; validate paired x4/FPS contracts."""
    root = Path(project_root).resolve()
    data, manifest_hash = _manifest(root, manifest)
    preview_fps = _positive_fps(preview_fps)
    jobs = []
    for name, dataset in data["datasets"].items():
        if name != "REDS4":
            for row in dataset["records"]:
                for role in (("gt", "lq") if name == "YouHQ40" else ("lq",)):
                    jobs.append((name, row["id"], role, _local(root, row[role]["path"]), row[role]))
    def verified_probe(job):
        path, expected = job[3], job[4]
        if path.stat().st_size != expected["bytes"] or _sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Prepared container size/SHA256 differs from manifest: {path}")
        return {**_probe_one(path), "bytes": expected["bytes"], "sha256": expected["sha256"]}
    with ThreadPoolExecutor(max_workers=4) as executor:
        probed = list(executor.map(verified_probe, jobs))
    index = {(job[0], job[1], job[2]): value for job, value in zip(jobs, probed)}
    datasets, issues = {}, []
    for name, dataset in data["datasets"].items():
        records = []
        for row in dataset["records"]:
            if name == "REDS4":
                wrong_scale = (row["gt"]["height"], row["gt"]["width"]) != (row["lq"]["height"]*4, row["lq"]["width"]*4)
                if wrong_scale:
                    issues.append(f"REDS4/{row['id']}: declared GT/LQ geometry is not exact x4")
                records.append({"id": row["id"], "gt": {key: row["gt"][key] for key in ("height", "width", "frame_count")},
                    "lq": {key: row["lq"][key] for key in ("height", "width", "frame_count")},
                    "original_fps": None, "preview_fps": preview_fps,
                    "fps_source": "declared_preview_only_not_original_PNG_timestamps", "status": "failed" if wrong_scale else "passed"})
                continue
            item = {"id": row["id"], "lq": index[(name, row["id"], "lq")], "issues": []}
            if item["lq"]["fps"] is None:
                item["issues"].append("LQ container FPS unavailable")
            if name == "YouHQ40":
                item["gt"] = index[(name, row["id"], "gt")]
                gt, lq = item["gt"], item["lq"]
                if (gt["height"], gt["width"]) != (lq["height"] * 4, lq["width"] * 4):
                    item["issues"].append("GT/LQ container geometry is not exact x4")
                if gt["fps_fraction"] is None or gt["fps_fraction"] != lq["fps_fraction"]:
                    item["issues"].append("Paired GT/LQ container FPS differs or is unavailable")
                if gt["nb_frames"] is not None and lq["nb_frames"] is not None and gt["nb_frames"] != lq["nb_frames"]:
                    item["issues"].append("Paired container frame-count metadata differs")
            item["status"] = "failed" if item["issues"] else "passed"
            issues.extend(f"{name}/{row['id']}: {issue}" for issue in item["issues"])
            records.append(item)
        datasets[name] = {"records": records, "sequence_count": len(records)}
    return {"status": "failed" if issues else "passed", "issues": issues, "datasets": datasets,
        "manifest_sha256": manifest_hash, "manifest_content_sha256": _content_hash(data),
        "probe": "ffprobe selected video stream metadata only", "probe_stream_fields": PROBE_FIELDS.split(","),
        "count_frames_requested": False, "frame_decode_requested": False, "gpu_used": False,
        "metrics_computed": False, "tuning_performed": False,
        "red_png_fps": "unknown; preview FPS is explicitly declared and is not capture FPS"}


def _zip_index(archive, dataset):
    mapping, seen = {}, set()
    for entry in archive.infolist():
        relative = _safe_name(entry.filename)
        key = relative.as_posix()
        if key in seen:
            raise ValueError("Duplicate normalized ZIP member")
        seen.add(key)
        mode = entry.external_attr >> 16
        if stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR) or entry.flag_bits & 1:
            raise ValueError("ZIP links/special/encrypted members are forbidden")
        parts = relative.parts
        roles = {"LQ": "lr", "GT": "hr"} if dataset == "YouHQ40" else {"LQ": "lr"}
        if (not entry.is_dir() and len(parts) == 4 and parts[0] == dataset and parts[1] in roles
                and re.fullmatch(r"\d{3}", parts[2]) and re.fullmatch(r"\d+\.png", parts[3])):
            mapping.setdefault((parts[2], roles[parts[1]]), {})[parts[3]] = entry
    return mapping


class ExternalReader:
    """Verified read-only external assets; explicit load() performs pixel decode."""

    def __init__(self, root, manifest_path="manifests/external.json", *, preview_fps=30.0):
        self.root = Path(root).resolve()
        self.manifest, self.manifest_sha256 = _manifest(self.root, manifest_path)
        self.preview_fps = _positive_fps(preview_fps)
        seal_path = self.root / "docs/external-data-provenance.json"
        if seal_path.is_file():
            seal = json.loads(seal_path.read_text())
            if seal.get("manifest_sha256") != self.manifest_sha256:
                raise ValueError("External manifest differs from its provenance seal")
        self.records = {name: {row["id"]: row for row in value["records"]} for name, value in self.manifest["datasets"].items()}
        self.verified_archives, self.zip_files, self.zip_members, self.archive_locations = {}, {}, {}, {}
        try:
            for name, value in self.manifest["datasets"].items():
                source = value["source"]
                recorded = Path(source["archive_path"])
                owned = _local(self.root, "data/external/archives/" + recorded.name)
                # Prepared REDS4 archives move with the independent project;
                # the immutable manifest still records the original download
                # location. Either location must match the same full SHA256.
                path = (owned if owned.is_file() else recorded).resolve()
                self.archive_locations[name] = path
                if path not in self.verified_archives:
                    if not path.is_file() or path.stat().st_size != source["archive_bytes"] or _sha256_file(path) != source["archive_sha256"]:
                        raise ValueError(f"Source archive size/SHA256 mismatch: {name}")
                    self.verified_archives[path] = source["archive_sha256"]
                elif self.verified_archives[path] != source["archive_sha256"]:
                    raise ValueError("Conflicting expected digests for a shared archive")
                if name != "REDS4":
                    archive = zipfile.ZipFile(path)
                    self.zip_files[name] = archive
                    self.zip_members[name] = _zip_index(archive, name)
            probe_path = self.root / "docs/external-streams.json"
            cached = json.loads(probe_path.read_text()) if probe_path.is_file() else None
            if cached is not None and cached.get("manifest_content_sha256") == _content_hash(self.manifest):
                self.streams = cached
            else:
                self.streams = probe_external_streams(self.root, manifest_path, preview_fps=self.preview_fps)
            self.stream_index = {name: {row["id"]: row for row in dataset["records"]}
                                 for name, dataset in self.streams["datasets"].items()}
        except Exception:
            self.close()
            raise

    def close(self):
        for archive in self.zip_files.values():
            archive.close()
        self.zip_files.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _frame_sources(self, dataset, sequence):
        row = self.records[dataset][sequence]
        if dataset == "REDS4":
            result = {"lr": row["lq"]["frames"], "hr": row["gt"]["frames"]}
            for frames in result.values():
                names = [Path(item["path"]).name for item in frames]
                if len(names) != len(set(names)) or names != sorted(names, key=lambda value: int(Path(value).stem)):
                    raise ValueError("REDS4 frame files must be uniquely ordered by original frame number")
            if [Path(item["path"]).name for item in result["lr"]] != [Path(item["path"]).name for item in result["hr"]]:
                raise ValueError("REDS4 GT/LQ frame names differ")
            return result
        result = {}
        for role in (("lr", "hr") if dataset == "YouHQ40" else ("lr",)):
            members = self.zip_members[dataset].get((sequence, role), {})
            if not members:
                raise ValueError(f"Original PNG members missing for {dataset}/{sequence}/{role}; MKV conversion is not used")
            result[role] = [members[name] for name in sorted(members, key=lambda value: int(Path(value).stem))]
        if "hr" in result and [Path(item.filename).name for item in result["lr"]] != [Path(item.filename).name for item in result["hr"]]:
            raise ValueError("YouHQ40 original GT/LQ PNG frame names do not match exactly")
        return result

    def _read_png(self, dataset, source):
        if dataset == "REDS4":
            path = _local(self.root, source["path"])
            payload = path.read_bytes()
            if len(payload) != source["bytes"] or hashlib.sha256(payload).hexdigest() != source["sha256"]:
                raise ValueError("REDS4 PNG size/SHA256 mismatch")
            origin = {"path": source["path"]}
        else:
            payload = self.zip_files[dataset].read(source)  # ZIP CRC is verified by ZipFile.read.
            origin = {"member": source.filename, "zip_crc32": f"{source.CRC:08x}"}
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "PNG":
                raise ValueError("Expected an original PNG payload")
            pixels = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
        return tensor, {**origin, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

    def load(self, dataset, sequence_id, num_frames=17, start=0):
        """Deterministic contiguous ordinal frames; native RGB size, no resize."""
        if dataset not in self.records or sequence_id not in self.records[dataset]:
            raise KeyError("Dataset / original sequence ID not present in frozen manifest")
        if (not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames <= 0
                or not isinstance(start, int) or isinstance(start, bool) or start < 0):
            raise ValueError("num_frames must be positive and start a nonnegative integer")
        frame_sources = self._frame_sources(dataset, sequence_id)
        count = len(frame_sources["lr"])
        if start + num_frames > count:
            raise ValueError(f"Requested ordinal interval [{start},{start+num_frames}) exceeds source's {count} frames; no padding/truncation is applied")
        stream = self.stream_index[dataset][sequence_id]
        if stream.get("status") != "passed":
            raise ValueError("External sequence failed paired stream metadata validation")
        fps = self.preview_fps if dataset == "REDS4" else stream["lq"]["fps"]
        result, provenance = {}, {}
        for role, sources in frame_sources.items():
            tensors, entries = [], []
            for source in sources[start:start+num_frames]:
                frame, entry = self._read_png(dataset, source)
                tensors.append(frame)
                entries.append(entry)
            shapes = {tuple(frame.shape) for frame in tensors}
            if len(shapes) != 1:
                raise ValueError("Native spatial dimensions changed within the selected source interval")
            video = torch.stack(tensors, dim=1).unsqueeze(0)
            declaration = self.records[dataset][sequence_id]["lq" if role == "lr" else "gt"] if dataset == "REDS4" else stream["lq" if role == "lr" else "gt"]
            if tuple(video.shape[-2:]) != (declaration["height"], declaration["width"]):
                raise ValueError("Original PNG geometry differs from declared PNG/container geometry; no silent resize")
            result[role], provenance[role] = video, entries
        if "hr" in result and tuple(result["hr"].shape[-2:]) != tuple(size * 4 for size in result["lr"].shape[-2:]):
            raise ValueError("Original paired PNG dimensions are not exact x4")
        selected_sources = frame_sources["lr"][start:start+num_frames]
        names = [Path(item["path"] if dataset == "REDS4" else item.filename).name for item in selected_sources]
        source = self.manifest["datasets"][dataset]["source"]
        result.update(fps=float(fps), metadata={
            "dataset": dataset, "sequence_id": sequence_id, "manifest_sha256": self.manifest_sha256,
            "source_archive": str(self.archive_locations[dataset]), "recorded_source_archive": source["archive_path"],
            "source_archive_sha256": source["archive_sha256"],
            "pixel_source": "original_disk_PNG_pairs" if dataset == "REDS4" else "original_PNG_members_from_supplied_ZIP",
            "frame_provenance": provenance, "source_frame_indices": list(range(start, start+num_frames)),
            "source_frame_numbers": [int(Path(name).stem) for name in names], "source_frame_names": names,
            "source_frame_count": count, "requested_num_frames": num_frames, "start": start,
            "benchmark_prefix_explicit": start == 0 and num_frames < count,
            "benchmark_interval_explicit": {"start": start, "length": num_frames, "full_source": start == 0 and num_frames == count},
            "frame_selection": "deterministic contiguous ordinal interval; no result-dependent selection",
            "paired_scale": 4 if "hr" in result else None, "native_lr_hw": list(result["lr"].shape[-2:]),
            "native_hr_hw": list(result["hr"].shape[-2:]) if "hr" in result else None,
            "spatial_resize": False, "fps": float(fps),
            "original_fps": None if dataset == "REDS4" else float(fps),
            "original_capture_fps": None,
            "fps_source": "declared_preview_fps_not_original_PNG_timestamps" if dataset == "REDS4" else "corresponding_supplied_MKV_" + stream["lq"]["fps_field"],
            "fps_caveat": "REDS4 original capture FPS unknown" if dataset == "REDS4" else "PNG timing inherited from supplied corresponding MKV; original capture timing not independently established",
            "color": "Pillow PNG RGB to float32 /255; no MKV pixel conversion",
            "gpu_used": False, "metrics_computed": False, "tuning_performed": False})
        return result

    read = load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-only", action="store_true", required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", default="manifests/external.json")
    parser.add_argument("--preview-fps", type=float, default=30.0)
    args = parser.parse_args()
    report = probe_external_streams(args.project_root, args.manifest, preview_fps=args.preview_fps)
    output = args.project_root / "docs/external-streams.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "datasets": {name: row["sequence_count"] for name, row in report["datasets"].items()},
                      "frame_decode_requested": False, "count_frames_requested": False, "metrics_computed": False}, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
