"""Pinned LPIPS and explicitly versioned TecoGAN temporal evaluation.

Primary definitions: thunil/TecoGAN commit
bc747a4651c1e08bd0a73bed08c4cb9fac99b4d5, metrics.py lines 75-90,
112-115 and 132-197; DistModel defaults to LPIPS version 0.1. The original
calibration weights are byte-identical to lpips 0.1.4's v0.1/alex.pth.

Spatial LPIPS uses *all* lossless float RGB frames without border shaving.
Temporal numbers follow the author's different protocol: uint8 RGB, exclude
the first and last two frames, and crop to the central multiples of 32 with
at least 16 total pixels removed from each dimension. tOF estimates flows
on full frames, and crops only the resulting flows. Report these protocols
separately; do not compare their numbers to an all-frame temporal variant.
"""
from __future__ import annotations

from importlib import metadata
import hashlib
from pathlib import Path

import numpy as np
import torch


LPIPS_KEY = "lpips_alex01_rgb_allframes"
TLP_KEY = "tLP_alex_v0_1_TecoGAN_crop"
TLP100_KEY = "tLP100_alex_v0_1_TecoGAN_crop"
TOF_KEY = "tOF_Farneback_TecoGAN_crop"
TECOGAN_REVISION = "bc747a4651c1e08bd0a73bed08c4cb9fac99b4d5"
ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
LINEAR_SHA256 = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
FARNEBACK_PARAMETERS = (0.5, 3, 15, 3, 5, 1.2, 0)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _video_frames(video):
    if video.ndim != 5 or tuple(video.shape[:2]) != (1, 3) or not all(video.shape):
        raise ValueError("Expected one nonempty RGB [1,3,T,H,W] video")
    if not bool(torch.isfinite(video).all()):
        raise ValueError("Metric pixels must be finite")
    if float(video.min()) < 0 or float(video.max()) > 1:
        raise ValueError("Metric RGB input must be in [0,1]")
    return video.detach().float().cpu()[0].permute(1, 2, 3, 0).contiguous()


def tecogan_crop(image):
    """Author's ``crop_8x8`` geometry, despite its misleading function name."""
    if image.ndim < 2:
        raise ValueError("Expected image / optical-flow array with H,W axes")
    original_h, original_w = image.shape[:2]
    height, width = (original_h // 32) * 32, (original_w // 32) * 32
    while height > original_h - 16:
        height -= 32
    while width > original_w - 16:
        width -= 32
    if min(height, width) <= 0:
        raise ValueError("TecoGAN crop requires at least 48 pixels per dimension")
    top, left = (original_h - height) // 2, (original_w - width) // 2
    return image[top:top + height, left:left + width], top, left


def _uint8_frames(frames):
    # Equivalent to first materializing 8-bit PNGs with round-to-nearest.
    # Never decode the lossy playback MP4 for measurement.
    return np.rint(frames.numpy() * 255).clip(0, 255).astype(np.uint8)


class PerceptualMetrics:
    """Frozen official LPIPS-Alex 0.1 with strictly local audited weights.

    No package install or network access occurs here. Install lpips==0.1.4
    into the project environment / .deps first, and pass both weight paths.
    The temporary random trunk is fully replaced by the official ImageNet
    state before any call; this avoids torchvision's implicit cache download.
    """

    def __init__(self, alexnet_weights, linear_weights, *, device="cpu", batch_size=8):
        if metadata.version("lpips") != "0.1.4":
            raise RuntimeError("This protocol is pinned to lpips==0.1.4")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        alexnet_weights, linear_weights = Path(alexnet_weights), Path(linear_weights)
        if _sha256(alexnet_weights) != ALEXNET_SHA256 or _sha256(linear_weights) != LINEAR_SHA256:
            raise ValueError("LPIPS weights differ from the pinned official artifacts")
        import lpips
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model = lpips.LPIPS(net="alex", version="0.1", pretrained=False,
                                 pnet_rand=True, pnet_tune=False, spatial=False,
                                 eval_mode=True, use_dropout=True, verbose=False)
        backbone = torch.load(alexnet_weights, map_location="cpu", weights_only=True)
        # The official LPIPS wrapper preserves torchvision's original indices
        # in five slices. Strictly replace every trunk tensor, with no network.
        slices = {i: group for group, indices in enumerate(((0, 1), (2, 3, 4),
                    (5, 6, 7), (8, 9), (10, 11)), start=1) for i in indices}
        trunk = {}
        for name, value in backbone.items():
            if name.startswith("features."):
                _, index, suffix = name.split(".", 2)
                trunk[f"slice{slices[int(index)]}.{index}.{suffix}"] = value
        self.model.net.load_state_dict(trunk, strict=True)
        calibrated = torch.load(linear_weights, map_location="cpu", weights_only=True)
        expected = {f"lin{i}.model.1.weight" for i in range(5)}
        if set(calibrated) != expected:
            raise ValueError("Unexpected official LPIPS linear-layer checkpoint structure")
        for index in range(5):
            getattr(self.model, f"lin{index}").load_state_dict(
                {"model.1.weight": calibrated[f"lin{index}.model.1.weight"]}, strict=True)
        self.model.pnet_rand = False
        self.model.requires_grad_(False).eval().to(device=self.device, dtype=torch.float32)
        self.audit = {
            "lpips_package": "0.1.4", "lpips_network": "alex", "lpips_version": "0.1",
            "alexnet_sha256": ALEXNET_SHA256, "linear_sha256": LINEAR_SHA256,
            "tecogan_revision": TECOGAN_REVISION,
            "tecogan_source": f"https://github.com/thunil/TecoGAN/blob/{TECOGAN_REVISION}/metrics.py",
            "frozen": True, "network_access": False,
        }

    @torch.no_grad()
    def _distances(self, first, second, *, normalized=False):
        """N,H,W,3 RGB [0,1] -> N scalar official LPIPS distances."""
        if first.shape != second.shape or first.ndim != 4 or first.shape[-1] != 3:
            raise ValueError("LPIPS batches must have matching N,H,W,3 shapes")
        if min(first.shape[1:3]) < 32:
            raise ValueError("AlexNet LPIPS requires at least 32 pixels per dimension")
        values = []
        self.model.eval()
        # Do not inherit an outer mixed-precision training context.
        with torch.autocast(device_type=self.device.type, enabled=False):
            for start in range(0, len(first), self.batch_size):
                a = first[start:start+self.batch_size].permute(0, 3, 1, 2).to(self.device).float()
                b = second[start:start+self.batch_size].permute(0, 3, 1, 2).to(self.device).float()
                if not normalized:
                    a, b = a * 2 - 1, b * 2 - 1
                result = self.model(a, b, normalize=False)
                values.append(result.reshape(-1).float().cpu())
        return torch.cat(values).tolist()

    def evaluate(self, prediction, target, *, include_temporal=True):
        """One scalar per video, then aggregate videos with equal weights."""
        predicted, truth = _video_frames(prediction), _video_frames(target)
        if predicted.shape != truth.shape:
            raise ValueError("Paired metric inputs must have identical shapes")
        distances = self._distances(predicted, truth)
        result = {
            LPIPS_KEY: float(np.mean(distances)), "lpips_per_frame": distances,
            "frame_count": len(distances), "audit": self.audit,
            "lpips_protocol": {"frames": "all", "border_crop": 0,
                "input": "lossless float32 RGB [0,1] converted to [-1,1]", "aggregation": "frame arithmetic mean"},
        }
        if include_temporal:
            result["tecogan"] = self._temporal(predicted, truth)
        return result

    def _temporal(self, predicted, truth):
        import cv2
        if len(predicted) < 6:
            raise ValueError("TecoGAN cutfr=2 needs at least six frames for a temporal pair")
        pred8, gt8 = _uint8_frames(predicted), _uint8_frames(truth)
        indices = list(range(2, len(pred8) - 2))
        pred_crops, gt_crops = [], []
        for i in indices:
            p, top, left = tecogan_crop(pred8[i])
            g, _, _ = tecogan_crop(gt8[i])
            pred_crops.append(p)
            gt_crops.append(g)
        # Reproduce util.im2tensor: NumPy float64 division by 127.5 and
        # subtraction before torch's float32 conversion.
        pred_tensor = torch.from_numpy(np.stack(pred_crops) / (255.0 / 2) - 1).float()
        gt_tensor = torch.from_numpy(np.stack(gt_crops) / (255.0 / 2) - 1).float()
        pred_distances = self._distances(pred_tensor[:-1], pred_tensor[1:], normalized=True)
        gt_distances = self._distances(gt_tensor[:-1], gt_tensor[1:], normalized=True)
        pairs = []
        for position, (before, after) in enumerate(zip(indices, indices[1:])):
            def flow(frames):
                first = cv2.cvtColor(frames[before], cv2.COLOR_RGB2GRAY)
                second = cv2.cvtColor(frames[after], cv2.COLOR_RGB2GRAY)
                return cv2.calcOpticalFlowFarneback(first, second, None, *FARNEBACK_PARAMETERS)
            # Estimate full-frame flows first; cropping images before the flow
            # call would change the author's definition at the crop boundary.
            pred_flow, _, _ = tecogan_crop(flow(pred8))
            gt_flow, _, _ = tecogan_crop(flow(gt8))
            flow_difference = np.absolute(gt_flow - pred_flow)
            tof = float(np.sqrt(np.sum(flow_difference * flow_difference, axis=-1)).mean())
            tlp = abs(gt_distances[position] - pred_distances[position])
            pairs.append({"indices": [before, after], "lpips_target": gt_distances[position],
                          "lpips_prediction": pred_distances[position], TLP_KEY: tlp,
                          TLP100_KEY: tlp * 100.0, TOF_KEY: tof})
        return {
            **{key: float(np.mean([pair[key] for pair in pairs])) for key in (TLP_KEY, TLP100_KEY, TOF_KEY)},
            "per_pair": pairs, "frame_indices": indices,
            "protocol": {"source_revision": TECOGAN_REVISION, "cutfr": 2,
                "frame_rule": "range(2,T-2); pairs only within retained frames",
                "spatial_crop": "author crop_8x8; central multiples of 32, remove >=16 total pixels per axis",
                "crop_tlhw": [top, left, pred_crops[0].shape[0], pred_crops[0].shape[1]],
                "pixel_input": "lossless RGB float -> round-to-nearest uint8 PNG-equivalent; no MP4 decode",
                "tLP": "mean(abs(LPIPS(GTprev,GTnext)-LPIPS(SRprev,SRnext)))",
                "tLP100": "100*tLP; author's reported temporal LPIPS scale",
                "tOF": "mean spatial L2 norm of Farneback flow difference; flow on full uint8 grayscale frames before crop",
                "farneback_parameters": list(FARNEBACK_PARAMETERS), "opencv_version": cv2.__version__,
                "lpips": "lpips==0.1.4, AlexNet, v0.1 calibration byte-identical to author's checkpoint",
                "pair_aggregation": "arithmetic mean per video"},
        }
