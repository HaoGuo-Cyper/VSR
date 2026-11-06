"""Local MUSIQ-KonIQ frame evaluation using pinned IQA-PyTorch code.

This is Chaofeng Chen's PyTorch reimplementation of Google Research MUSIQ,
not the original Google JAX/Flax implementation. Its unchanged preprocessing
uses native resolution and aspect-preserved 224/384 longer-side scales. It is
an image-quality complement; mean frame scores do not measure video motion,
content fidelity or temporal consistency, and are not a recovery conclusion.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn

SOURCE_COMMIT = "259b9161125194d69b31a35defe48cef2244b294"
KONIQ_CHECKPOINT = "musiq_koniq_ckpt-e95806b9.pth"
KONIQ_SHA256 = "e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa"
KONIQ_WEIGHTS_URL = "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/" + KONIQ_CHECKPOINT
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_musiq_sources(project_root=PROJECT_ROOT):
    """Require the pinned source manifest and all adapted/unchanged file hashes."""
    root = Path(project_root).resolve()
    manifest_path = root / "docs/musiq-provenance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("MUSIQ source revision differs from pinned IQA-PyTorch commit")
    if manifest.get("weights", {}).get("sha256") != KONIQ_SHA256 or manifest["weights"].get("training_dataset") != "KonIQ10k":
        raise ValueError("MUSIQ manifest must select the pinned KonIQ10k checkpoint, not AVA")
    source_root = (root / "vendor/musiq_pytorch").resolve()
    for name, item in manifest["files"].items():
        path = (source_root / name).resolve()
        if not path.is_relative_to(source_root) or not path.is_file():
            raise ValueError("MUSIQ vendored source path is missing or escapes its namespace")
        if _sha256_file(path) != item["vendored_sha256"]:
            raise ValueError(f"MUSIQ pinned source fingerprint mismatch: {name}")
    return manifest


def _vendor_modules(project_root=PROJECT_ROOT):
    verify_musiq_sources(project_root)
    folder = Path(project_root).resolve() / "vendor/musiq_pytorch"
    # A private namespace avoids importing the full pyiqa distribution, registry
    # plugins, unrelated models, dependency installers or download helpers.
    namespace = "_vsr_musiq_pytorch_" + hashlib.sha256(str(folder).encode()).hexdigest()[:12]
    if namespace not in sys.modules:
        spec = importlib.util.spec_from_file_location(namespace, folder / "__init__.py", submodule_search_locations=[str(folder)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[namespace] = module
        spec.loader.exec_module(module)
    return (importlib.import_module(namespace + ".musiq_arch"),
            importlib.import_module(namespace + ".multiscale_trans_util"))


def _validate_frames(frames):
    if frames.ndim != 4 or frames.shape[1] != 3 or min(frames.shape) < 1:
        raise ValueError("MUSIQ frames must be nonempty B,3,H,W RGB")
    if not frames.is_floating_point() or not bool(torch.isfinite(frames).all()):
        raise ValueError("MUSIQ input must be finite floating RGB in [0,1]")
    if float(frames.min()) < 0 or float(frames.max()) > 1:
        raise ValueError("MUSIQ input RGB must lie in [0,1]")


@torch.inference_mode()
def preprocess_frames(frames, *, project_root=PROJECT_ROOT):
    """Expose exact pinned patch preprocessing for inspection, without weights.

    Returns B,N,(3*32*32+3): raw patch values, spatial/scale IDs, validity mask.
    The last native-resolution scale keeps ALL input patches; no square crop or
    arbitrary native-token cap is introduced. Do not feed these patches back
    into the eval-mode architecture, which performs this transform internally.
    """
    _validate_frames(frames)
    _, transforms = _vendor_modules(project_root)
    return transforms.get_multiscale_patches(
        (frames.float() - 0.5) * 2, patch_size=32, patch_stride=32,
        hse_grid_size=10, longer_side_lengths=[224, 384],
        max_seq_len_from_original_res=-1)


class MUSIQEvaluator(nn.Module):
    """Frozen KonIQ MOS regression; strict local assets and no implicit network."""

    def __init__(self, localweights, *, device="cpu", project_root=PROJECT_ROOT,
                 backend_factory=None):
        super().__init__()
        self.project_root = Path(project_root).resolve()
        manifest = verify_musiq_sources(self.project_root)
        path = Path(localweights).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Pinned local MUSIQ-KonIQ checkpoint missing: {path}")
        digest = _sha256_file(path)
        if not digest.startswith("e95806b9") or digest != KONIQ_SHA256:
            raise ValueError("MUSIQ weights must match pinned KonIQ checkpoint SHA256, not AVA/unknown weights")
        if backend_factory is None:
            architecture, _ = _vendor_modules(self.project_root)
            backend_factory = architecture.MUSIQ
        # Direct MUSIQ(pretrained=True) defaults to AVA aesthetics upstream;
        # disable automatic loading and construct the single-score KonIQ head.
        self.backend = backend_factory(pretrained=False, num_class=1,
                                       longer_side_lengths=[224, 384],
                                       max_seq_len_from_original_res=-1)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            raise ValueError("MUSIQ checkpoint must contain a nonempty tensor state dictionary")
        if "head.weight" not in state or tuple(state["head.weight"].shape) != (1, 384):
            raise ValueError("MUSIQ checkpoint must have the KonIQ scalar regression head")
        if any(not isinstance(value, torch.Tensor) or not value.is_floating_point() or not bool(torch.isfinite(value).all()) for value in state.values()):
            raise ValueError("MUSIQ checkpoint contains an invalid/nonfinite tensor")
        self.backend.load_state_dict(state, strict=True)
        self.backend.requires_grad_(False).eval().to(device)
        self.provenance = {
            "metric": "MUSIQ-KonIQ10k (IQA-PyTorch reimplementation)",
            "implementation_repo": manifest["source_repo"], "source_commit": SOURCE_COMMIT,
            "original_method_repo": "https://github.com/google-research/google-research/tree/master/musiq",
            "weights_url": KONIQ_WEIGHTS_URL, "weights_path": str(path), "weights_sha256": digest,
            "training_dataset": "KonIQ10k", "output": "unclamped scalar MOS, approximately0–100; higher is better",
            "preprocessing": manifest["preprocess"], "license": manifest["license"],
            "source_hashes": {name: item["vendored_sha256"] for name, item in manifest["files"].items()},
            "implicit_network": False, "frozen": True,
            "parity_with_google_jax_claimed": False,
        }
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.inference_mode()
    def evaluate_video(self, video, *, batch_size=1):
        """Evaluate every input frame and return arithmetic mean per video.

        video is float RGB [1,3,T,H,W] in [0,1] at its native output size.
        This method neither changes video frame rate nor selects favorable
        frames; FPS/playback metadata remain with the caller's video artifact.
        """
        if video.ndim != 5 or video.shape[:2] != (1, 3) or video.shape[2] < 1:
            raise ValueError("MUSIQ video must have nonempty [1,3,T,H,W] shape")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("MUSIQ batch_size must be a positive integer")
        frames = video[0].permute(1, 0, 2, 3)
        _validate_frames(frames)
        self.eval()
        parameter = next(self.backend.parameters(), None)
        device = parameter.device if parameter is not None else video.device
        scores = []
        for start in range(0, len(frames), batch_size):
            batch = frames[start:start + batch_size].to(device=device, dtype=torch.float32)
            with torch.autocast(device_type=device.type, enabled=False):
                prediction = self.backend(batch)
            if not isinstance(prediction, torch.Tensor) or prediction.numel() != len(batch) or not bool(torch.isfinite(prediction).all()):
                raise ValueError("MUSIQ backend must return one finite scalar score per frame")
            scores.extend(prediction.detach().cpu().reshape(-1).double().tolist())
        return {"metric": "musiq_koniq10k_pyiqa", "per_frame_scores": scores,
                "frame_indices": list(range(len(scores))), "frame_count": len(scores),
                "mean": sum(scores) / len(scores), "aggregation": "all frames arithmetic mean per video",
                "input_hw": list(video.shape[-2:]), "native_spatial_size_preserved": True,
                "quality_hypothesis_verified": False,
                "interpretation": "No-reference image quality complement; not temporal consistency or content-fidelity proof",
                "provenance": self.provenance}

    def forward(self, video, *, batch_size=1):
        return self.evaluate_video(video, batch_size=batch_size)
