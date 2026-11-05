#!/usr/bin/env python3
"""Prepare external-test assets without decoding images/videos or running models.

PNG verification reads only the signature and IHDR chunk. MKV verification
reads only the EBML magic; FPS, frame count and geometry remain uninspected.
Archive paths and links are validated before any payload is written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import tarfile
from urllib.request import Request, urlopen
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[1]
REDS4_URL = "https://github.com/JingyunLiang/VRT/releases/download/v0.0/testset_REDS4.tar.gz"
REDS4_IDS = ("000", "011", "015", "020")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name):
    if not name or "\\" in name or "\x00" in name or re.match(r"^[A-Za-z]:", name):
        raise ValueError(f"Unsupported archive name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Archive member escapes destination: {name!r}")
    if not path.parts:
        raise ValueError("Empty archive member name")
    return path


def destination(root, member_name):
    relative = safe_name(member_name)
    root = Path(root).resolve()
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError("Extraction destination escapes root")
    for ancestor in (path, *path.parents):
        if ancestor == root:
            break
        if ancestor.is_symlink():
            raise ValueError("Existing symlink in extraction destination")
    return path


def copy_payload(source, target, expected_bytes):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if partial.is_symlink():
        raise ValueError("Temporary output must not be a symlink")
    digest, count = hashlib.sha256(), 0
    with partial.open("wb") as out:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            out.write(block)
            count += len(block)
            digest.update(block)
    if count != expected_bytes:
        raise ValueError(f"Extracted byte count mismatch for {target}: {count} != {expected_bytes}")
    value = digest.hexdigest()
    if target.exists() and (target.stat().st_size != count or sha256_file(target) != value):
        raise ValueError(f"Existing prepared file has different content: {target}")
    partial.replace(target)
    return {"path": target.relative_to(ROOT).as_posix(), "bytes": count, "sha256": value}


def png_ihdr(path):
    with Path(path).open("rb") as stream:
        header = stream.read(33)
    if (len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n"
            or header[8:12] != b"\x00\x00\x00\r" or header[12:16] != b"IHDR"):
        raise ValueError(f"Invalid PNG signature / IHDR: {path}")
    if zlib.crc32(header[12:29]) != struct.unpack(">I", header[29:33])[0]:
        raise ValueError(f"Invalid PNG IHDR CRC: {path}")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", header[16:29])
    if min(width, height) <= 0:
        raise ValueError("Invalid PNG dimensions")
    return {"width": width, "height": height, "bit_depth": depth, "color_type": color,
            "compression_method": compression, "filter_method": filtering,
            "interlace_method": interlace, "verification": "signature_and_IHDR_only_no_pixel_decode"}


def source_record(path, *, url=None, origin):
    path = Path(path).resolve()
    return {"archive_path": str(path), "archive_bytes": path.stat().st_size,
            "archive_sha256": sha256_file(path), "download_url": url,
            "origin": origin}


def prepare_mkv_zip(archive_path, dataset, external_root):
    expected_count = 40 if dataset == "YouHQ40" else 50
    expected_branches = ("GT-Video", "LQ-Video") if dataset == "YouHQ40" else ("LQ-Video",)
    groups = {branch: {} for branch in expected_branches}
    source = source_record(archive_path, origin="user-provided local Downloads archive; upstream download URL not independently established")
    with zipfile.ZipFile(archive_path) as archive:
        entries, names, chosen = archive.infolist(), set(), []
        for entry in entries:
            relative = safe_name(entry.filename)
            normalized = relative.as_posix()
            if normalized in names:
                raise ValueError(f"Duplicate ZIP member: {normalized}")
            names.add(normalized)
            mode = entry.external_attr >> 16
            if stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"ZIP links/special files are rejected: {normalized}")
            if entry.flag_bits & 1:
                raise ValueError("Encrypted ZIP members are unsupported")
            parts = relative.parts
            if (not entry.is_dir() and len(parts) == 3 and parts[0] == dataset
                    and parts[1] in expected_branches and parts[2].lower().endswith(".mkv")):
                clip_id = Path(parts[2]).stem
                if not re.fullmatch(r"\d{3}", clip_id) or clip_id in groups[parts[1]]:
                    raise ValueError("MKV IDs must be distinct original three-digit identifiers")
                groups[parts[1]][clip_id] = entry
                chosen.append((entry, parts[1], clip_id))
        ids = {f"{index:03d}" for index in range(expected_count)}
        if any(set(group) != ids for group in groups.values()):
            raise ValueError(f"{dataset} branches do not contain the expected original IDs")
        outputs = {branch: {} for branch in expected_branches}
        for entry, branch, clip_id in chosen:
            target = destination(external_root, entry.filename)
            with archive.open(entry) as payload:
                info = copy_payload(payload, target, entry.file_size)
            with target.open("rb") as stream:
                if stream.read(4) != b"\x1aE\xdf\xa3":
                    raise ValueError(f"MKV EBML header missing: {target}")
            outputs[branch][clip_id] = {**info, "container": "Matroska", "header_check": "EBML_magic_only",
                "fps": None, "frame_count": None, "width": None, "height": None,
                "stream_metadata_status": "not_probed; no video decode or stream inspection"}
        records = []
        for clip_id in sorted(ids):
            row = {"id": clip_id, "lq": outputs["LQ-Video"][clip_id]}
            if dataset == "YouHQ40":
                row["gt"] = outputs["GT-Video"][clip_id]
            records.append(row)
        source.update(total_archive_members=len(entries), extracted_mkv_count=len(chosen),
                      ignored_member_count=len(entries) - len(chosen),
                      extraction_policy="only original MKV GT-Video/LQ-Video; no PNG or macOS metadata extraction",
                      archive_path_validation="all members; absolute/traversal/links/special/encrypted rejected")
    return {"dataset": dataset, "role": "external_paired_test" if dataset == "YouHQ40" else "real_world_unpaired_test",
            "training_allowed": False, "tuning_allowed": False, "sequence_count": expected_count,
            "paired": dataset == "YouHQ40", "source": source, "records": records,
            "pixel_decode_performed": False, "metrics_computed": False, "fps_status": "pending_stream_inspection"}


def prepare_reds4(archive_path, external_root):
    source = source_record(archive_path, url=REDS4_URL,
        origin="VRT authors' curated REDS4 release, GitHub JingyunLiang/VRT release v0.0")
    groups, selected, png_branch_counts = {}, [], {}
    selected_branch_roles = {"GT": "gt", "sharp_bicubic": "lq"}
    with tarfile.open(archive_path, "r:gz") as archive:
        entries, names = archive.getmembers(), set()
        for entry in entries:
            relative = safe_name(entry.name)
            normalized = relative.as_posix()
            if normalized in names:
                raise ValueError(f"Duplicate TAR member: {normalized}")
            names.add(normalized)
            if not (entry.isdir() or entry.isfile()):
                raise ValueError(f"TAR links/special members are rejected: {normalized}")
            parts = relative.parts
            if entry.isfile() and relative.suffix.lower() == ".png":
                if (len(parts) != 4 or parts[0] != "REDS4" or parts[2] not in REDS4_IDS
                        or not re.fullmatch(r"\d{8}\.png", parts[3])):
                    raise ValueError(f"Unexpected REDS4 PNG path: {normalized}")
                png_branch_counts[parts[1]] = png_branch_counts.get(parts[1], 0) + 1
                if parts[1] in selected_branch_roles:
                    selected.append((entry, parts[1], parts[2], parts[3]))
        for entry, branch, sequence, frame in selected:
            target = destination(external_root, entry.name)
            payload = archive.extractfile(entry)
            if payload is None:
                raise ValueError("TAR regular payload missing")
            with payload:
                info = copy_payload(payload, target, entry.size)
            info["png_ihdr"] = png_ihdr(target)
            groups.setdefault(branch, {}).setdefault(sequence, {})[frame] = info
        expected_frames = {f"{index:08d}.png" for index in range(100)}
        branch_roles = {}
        for branch, sequences in groups.items():
            if set(sequences) != set(REDS4_IDS) or any(set(frames) != expected_frames for frames in sequences.values()):
                raise ValueError("REDS4 requires exactly the selected four IDs and frames 00000000..00000099 in every branch")
            geometry = {(entry["png_ihdr"]["height"], entry["png_ihdr"]["width"])
                        for frames in sequences.values() for entry in frames.values()}
            role = selected_branch_roles[branch]
            expected_geometry = {(720, 1280)} if role == "gt" else {(180, 320)}
            if geometry != expected_geometry:
                raise ValueError(f"Unexpected REDS4 PNG header geometry: {geometry}")
            if role in branch_roles:
                raise ValueError("Ambiguous REDS4 GT/LQ branches")
            branch_roles[role] = branch
        if set(branch_roles) != {"gt", "lq"} or len(selected) != 800:
            raise ValueError("REDS4 must contain 400 GT + 400 x4 LQ frames")
        records = []
        for sequence in REDS4_IDS:
            record = {"id": sequence, "frame_count": 100, "scale": 4, "fps": None,
                      "fps_status": "not provided by PNG archive; pending explicit timing protocol"}
            for role, branch in branch_roles.items():
                frames = groups[branch][sequence]
                record[role] = {"directory": f"data/external/REDS4/{branch}/{sequence}",
                    "frame_count": len(frames), "height": frames["00000000.png"]["png_ihdr"]["height"],
                    "width": frames["00000000.png"]["png_ihdr"]["width"],
                    "frames": [frames[name] for name in sorted(frames)]}
            records.append(record)
        source.update(total_archive_members=len(entries), extracted_png_count=len(selected),
            available_archive_png_branches=png_branch_counts,
            selected_tracks={role: branch for branch, role in selected_branch_roles.items()},
            ignored_png_tracks=sorted(set(png_branch_counts) - set(selected_branch_roles)),
            archive_path_validation="all members; absolute/traversal/links/special rejected",
            extraction_policy="PNG payload bytes only; signature/IHDR including CRC checked; no pixel decode")
    return {"dataset": "REDS4", "role": "external_paired_test", "training_allowed": False,
            "tuning_allowed": False, "sequence_count": 4, "sequence_ids": list(REDS4_IDS),
            "paired": True, "scale": 4, "source": source, "records": records,
            "pixel_decode_performed": False, "metrics_computed": False,
            "fps_status": "PNG archive has no inspected playback timestamps"}


def download_reds4(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return
    partial = path.with_name(path.name + ".partial")
    with urlopen(Request(REDS4_URL, headers={"User-Agent": "VSR-external-data-preparation/1.0"}), timeout=60) as source:
        with partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    partial.replace(path)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--youhq40-archive", type=Path, default=Path("/home/hugo/Downloads/YouHQ40.zip"))
    parser.add_argument("--videolq-archive", type=Path, default=Path("/home/hugo/Downloads/VideoLQ.zip"))
    parser.add_argument("--datasets", nargs="+", choices=("YouHQ40", "VideoLQ", "REDS4"), default=("YouHQ40", "VideoLQ", "REDS4"))
    parser.add_argument("--download-reds4", action="store_true")
    args = parser.parse_args()
    external = ROOT / "data/external"
    external.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "manifests/external.json"
    document = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"schema_version": 1, "datasets": {}}
    for dataset in args.datasets:
        print(json.dumps({"phase": "preparing", "dataset": dataset}), flush=True)
        if dataset == "REDS4":
            archive = external / "archives/testset_REDS4.tar.gz"
            if args.download_reds4:
                download_reds4(archive)
            result = prepare_reds4(archive, external)
        else:
            archive = args.youhq40_archive if dataset == "YouHQ40" else args.videolq_archive
            result = prepare_mkv_zip(archive, dataset, external)
        document["datasets"][dataset] = result
        print(json.dumps({"phase": "prepared", "dataset": dataset, "sequences": result["sequence_count"],
                          "archive_sha256": result["source"]["archive_sha256"]}), flush=True)
        document.update(prepared_at_utc=datetime.now(timezone.utc).isoformat(),
            purpose="external evaluation only; no tuning or training", gpu_used=False,
            pixel_decode_performed=False, metrics_computed=False,
            script_sha256=sha256_file(__file__), complete=all(name in document["datasets"] for name in ("REDS4", "YouHQ40", "VideoLQ")))
        write_json(manifest_path, document)
        provenance = {key: value for key, value in document.items() if key != "datasets"}
        provenance["datasets"] = {name: {key: value for key, value in row.items() if key != "records"}
                                  for name, row in document["datasets"].items()}
        provenance["manifest_path"] = "manifests/external.json"
        provenance["manifest_sha256"] = sha256_file(manifest_path)
        write_json(ROOT / "docs/external-data-provenance.json", provenance)


if __name__ == "__main__":
    main()
