"""Read-only official artifact verification; no task checkpoint loaders."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def file_hash(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def verify_official(root, config):
    """Compare all used Wan files to the online manifest at an immutable revision."""
    metadata = json.loads((root / "docs/official-model-manifest.json").read_text())
    model = config["model"]
    if metadata["sha"] != model["revision"] or metadata["id"] != model["repo_id"]:
        raise ValueError("Official manifest identity mismatch")
    base = Path(model["local_dir"])
    checks = []
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if name != "model_index.json" and name.split("/")[0] not in {"transformer", "vae", "text_encoder", "tokenizer", "scheduler"}:
            continue
        path = base / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if item.get("lfs"):
            expected = item["lfs"]["sha256"]
            actual = file_hash(path)
            kind = "sha256"
        else:
            value = path.read_bytes()
            actual = hashlib.sha1(b"blob " + str(len(value)).encode() + b"\0" + value).hexdigest()
            expected = item["blobId"]
            kind = "git_blob_sha1"
        if actual != expected:
            raise ValueError(f"Official artifact checksum mismatch: {name}")
        checks.append({"path": str(path), "algorithm": kind, "digest": actual, "bytes": path.stat().st_size})
        print(f"verified {name}", flush=True)
    if len([x for x in checks if x["path"].endswith(".safetensors")]) != 8:
        raise ValueError("Expected all eight official safetensors files")
    teacher = config["teacher"]
    source = root / teacher["source_dir"]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != teacher["revision"]:
        raise ValueError("DINOv2 source commit differs from pinned configuration")
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], text=True)
    if dirty:
        raise ValueError("DINOv2 pinned source checkout has local changes")
    weight = root / teacher["weights"]
    teacher_sha = file_hash(weight)
    expected_teacher_sha = teacher["weights_sha256"]
    if teacher_sha != expected_teacher_sha:
        raise ValueError("DINOv2 teacher checksum mismatch")
    return {"status": "passed", "repo_id": model["repo_id"], "revision": model["revision"], "files": checks,
            "teacher": {"source_commit": commit, "url": teacher["weights_url"], "sha256": teacher_sha},
            "legacy_state_loaded": False, "official_weights_only": True}


def environment(root):
    import torch
    packages = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "peft", "safetensors", "numpy", "Pillow", "timm", "av", "scikit-image", "lpips", "tifffile", "lazy-loader"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    code = {str(p.relative_to(root)): file_hash(p) for folder in ("vsr", "configs", "tests")
            for p in sorted((root / folder).rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "packages": packages, "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
              "git_commit": commit, "code_sha256": code,
              "git_status": subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True),
              "nvidia_smi": subprocess.run(["nvidia-smi"], text=True, capture_output=True).stdout}
    if result["cuda_available"]:
        result.update(gpu=torch.cuda.get_device_name(), bf16=torch.cuda.is_bf16_supported(),
                      free_total_bytes=torch.cuda.mem_get_info())
    # Pin source files actually implementing the native architecture/scheduler.
    import diffusers.models.transformers.transformer_wan as wan_impl
    import diffusers.models.autoencoders.autoencoder_kl_wan as vae_impl
    import diffusers.schedulers.scheduling_unipc_multistep as scheduler_impl
    result["dependency_source_sha256"] = {m.__file__: file_hash(m.__file__) for m in (wan_impl, vae_impl, scheduler_impl)}
    return result
