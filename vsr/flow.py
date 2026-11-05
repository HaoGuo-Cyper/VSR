"""Frozen official torchvision RAFT-large for direct HR-anchor motion.

This module loads one local checkpoint, never downloads assets, never reads old
experiment artifacts, and never resizes or pads HR frames. Flow values retain
the same original HR-pixel coordinate system as the input crop.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from pathlib import Path

import torch
from torch import nn
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

WEIGHTS_ENUM = "Raft_Large_Weights.C_T_SKHT_V2"
OFFICIAL_WEIGHTS_URL = "https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
OFFICIAL_SHA256_PREFIX = "ff5fadd5"
# PyTorch publishes an eight-character SHA prefix in this filename. A full
# expected hash may additionally be supplied after the first verified download.


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(obj):
    source = inspect.getsourcefile(obj)
    if source is None:
        raise ValueError("Cannot fingerprint installed optical-flow source")
    return {"path": source, "sha256": sha256_file(source)}


class FlowTeacher(nn.Module):
    """Strict local RAFT-large loader; call .to(device) explicitly afterwards."""

    def __init__(self, checkpoint_path, num_flow_updates=12, expected_sha256=None):
        super().__init__()
        if not isinstance(num_flow_updates, int) or num_flow_updates < 1:
            raise ValueError("num_flow_updates must be a positive integer")
        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Local official RAFT checkpoint missing: {path}")
        actual_sha = sha256_file(path)
        if not actual_sha.startswith(OFFICIAL_SHA256_PREFIX):
            raise ValueError("RAFT checkpoint SHA256 does not match official URL hash prefix")
        if expected_sha256 is not None and actual_sha != expected_sha256:
            raise ValueError("RAFT checkpoint does not match pinned full SHA256")
        weights = Raft_Large_Weights.C_T_SKHT_V2
        if weights.url != OFFICIAL_WEIGHTS_URL:
            raise ValueError("Installed torchvision RAFT weights URL changed from pinned contract")
        self.encoder = raft_large(weights=None, progress=False)
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(state, strict=True)
        self.encoder.requires_grad_(False).eval()
        self.preprocess = weights.transforms()
        self.num_flow_updates = num_flow_updates
        self.provenance = {
            "implementation": "torchvision.models.optical_flow.raft_large",
            "torchvision_version": importlib.metadata.version("torchvision"),
            "weights_enum": WEIGHTS_ENUM, "official_url": OFFICIAL_WEIGHTS_URL,
            "checkpoint_path": str(path), "checkpoint_sha256": actual_sha,
            "published_sha256_prefix": OFFICIAL_SHA256_PREFIX,
            "full_expected_sha256": expected_sha256,
            "full_digest_verification": "pinned_full_sha256" if expected_sha256 is not None else "published_8_hex_prefix_and_local_full_hash_recorded",
            "encoder_source": _fingerprint(type(self.encoder)),
            "transform_source": _fingerprint(type(self.preprocess)),
            "preprocessing": "official C_T_SKHT_V2.transforms(): RGB float [0,1] to [-1,1]; contiguous; no resize/crop",
            "geometry": "input H,W>=128 and divisible by8; no resize/pad; outputs original HR pixel units",
            "pairing": "direct flow between actual consecutive supervision anchors, not adjacent raw frames",
            "num_flow_updates": num_flow_updates, "frozen": True,
        }
        self.train(False)

    def train(self, mode=True):
        # A parent training module must never activate stochastic flow behavior.
        super().train(False)
        return self

    @torch.inference_mode()
    def estimate_anchor_flows(self, hr, anchors):
        """hr: B,3,T,H,W RGB[0,1]; returns forward,backward B,Ta-1,2,H,W.

        For anchors [0,4,8], forward motion is estimated directly for 0->4 and
        4->8; backward motion is separately estimated for 4->0 and 8->4.
        No temporal composition or interpolation approximates these pairs.
        """
        if hr.ndim != 5 or hr.shape[1] != 3 or hr.shape[0] < 1:
            raise ValueError("HR input must have B,3,T,H,W shape")
        if not hr.is_floating_point() or not bool(torch.isfinite(hr).all()):
            raise ValueError("HR input must be finite floating RGB in [0,1]")
        if float(hr.min()) < 0 or float(hr.max()) > 1:
            raise ValueError("HR RGB input must lie in [0,1]")
        b, _, t, h, w = hr.shape
        if min(h, w) < 128 or h % 8 or w % 8:
            raise ValueError("Official RAFT requires H,W>=128 divisible by8; resizing/padding is disabled")
        anchors = tuple(anchors)
        if len(anchors) < 2 or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= t for i in anchors):
            raise ValueError("At least two valid integer HR supervision anchors are required")
        if any(left >= right for left, right in zip(anchors, anchors[1:])):
            raise ValueError("HR supervision anchors must be strictly increasing")
        parameter = next(self.encoder.parameters(), None)
        if parameter is not None and parameter.device != hr.device:
            raise ValueError("Move FlowTeacher and HR inputs to the same device explicitly")
        self.eval()
        forward, backward = [], []
        for first, second in zip(anchors, anchors[1:]):
            image1, image2 = self.preprocess(hr[:, :, first].float(), hr[:, :, second].float())
            # Keep official RAFT calculations FP32 even if called inside the
            # generator's mixed-precision context.
            with torch.autocast(device_type=hr.device.type, enabled=False):
                f = self.encoder(image1, image2, num_flow_updates=self.num_flow_updates)[-1]
                r = self.encoder(image2, image1, num_flow_updates=self.num_flow_updates)[-1]
            if tuple(f.shape) != (b, 2, h, w) or r.shape != f.shape:
                raise ValueError("RAFT output changed original HR flow geometry")
            if not bool(torch.isfinite(f).all()) or not bool(torch.isfinite(r).all()):
                raise ValueError("RAFT produced nonfinite motion")
            forward.append(f.float())
            backward.append(r.float())
        return torch.stack(forward, dim=1), torch.stack(backward, dim=1)

    def forward(self, hr, anchors):
        return self.estimate_anchor_flows(hr, anchors)
