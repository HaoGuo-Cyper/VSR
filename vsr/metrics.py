"""Restoration metrics and scientific summaries shared by every evaluation.

RGB/LPIPS/TecoGAN quantization, original-pair pools, G2 statistics and detached
feature diagnostics retain the archived numerical definitions. No model is
loaded, downloaded or placed on a GPU by importing this module.
"""
from __future__ import annotations
from importlib import metadata
import hashlib
import math
from pathlib import Path
import re
import statistics
import numpy as np
import torch
import torch.nn.functional as F


# vsr/evaluation.py — consolidated implementation
METRIC_KEYS = ("psnr_rgb_db", "ssim_rgb", "mse_rgb_01")


PSNR_MSE_FLOOR = 1e-12


def _validate_video(video):
    if video.ndim != 5 or video.shape[:2] != (1, 3) or not all(video.shape):
        raise ValueError("Expected nonempty RGB [1,3,T,H,W] video")
    if not bool(torch.isfinite(video).all()):
        raise ValueError("Metrics require finite RGB pixels")
    if float(video.min()) < -1e-6 or float(video.max()) > 1 + 1e-6:
        raise ValueError("Metrics require RGB values in [0,1]")


def rgb_metrics(prediction, target):
    """Frame-mean RGB PSNR and skimage SSIM; no border shave or Y transform.

    PSNR's MSE floor is 1e-12 (120 dB maximum) so exact matches remain valid
    finite JSON. The number of exact frames and the floor are recorded.
    SSIM uses skimage defaults: 7x7 uniform window, sample covariance and
    channel_axis=-1, data_range=1. Small windows are explicitly rejected.
    """
    from skimage.metrics import structural_similarity
    _validate_video(prediction)
    _validate_video(target)
    if prediction.shape != target.shape:
        raise ValueError("Metric inputs must have exactly equal frame/spatial shapes")
    if min(prediction.shape[-2:]) < 7:
        raise ValueError("SSIM requires at least 7 pixels in each spatial dimension")
    pred = prediction.detach().float().cpu()[0].permute(1, 2, 3, 0).numpy()
    truth = target.detach().float().cpu()[0].permute(1, 2, 3, 0).numpy()
    per_frame = []
    for p, t in zip(pred, truth):
        mse = float(((p.astype("float64") - t.astype("float64")) ** 2).mean())
        per_frame.append({"mse_rgb_01": mse,
                          "psnr_rgb_db": -10 * math.log10(max(mse, PSNR_MSE_FLOOR)),
                          "ssim_rgb": float(structural_similarity(p, t, channel_axis=-1, data_range=1.0))})
    return {**{key: sum(frame[key] for frame in per_frame) / len(per_frame) for key in METRIC_KEYS},
            "frame_count": len(per_frame), "exact_match_frames": sum(x["mse_rgb_01"] == 0 for x in per_frame),
            "per_frame": per_frame}


def _mean_metrics(rows):
    if not rows:
        raise ValueError("Cannot aggregate an empty metric group")
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in METRIC_KEYS}


def aggregate_per_video(per_video, modes):
    """Give every video equal weight after averaging its sampling seeds."""
    if not per_video:
        raise ValueError("No videos to aggregate")
    return {"video_count": len(per_video),
            "bicubic": _mean_metrics([row["bicubic"] for row in per_video]),
            "modes": {mode: _mean_metrics([row["modes"][mode]["mean"] for row in per_video]) for mode in modes}}



# vsr/perceptual_metrics.py — consolidated implementation
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



# vsr/metric_pools.py — consolidated implementation
PAIRED_METRIC_KEYS = ("psnr_rgb_db", "ssim_rgb", "mse_rgb_01", "lpips_alex_v0_1", TLP_KEY, TLP100_KEY, TOF_KEY)


POOL_NAMES = ("all_frames", "matched_nonreference", "reference_frame_separate")


@torch.no_grad()
def paired_metric_pools(prediction, target, perceptual, *, reference_index=0):
    """Return all / identical-nonref / reference-only RGB+LPIPS+time pools.

    Inputs are full RGB [1,3,T,H,W] in [0,1]. ``perceptual`` supplies the
    project's pinned PerceptualMetrics.evaluate interface. An empty temporal
    pool returns None, never a fabricated zero. Frame and pair indices always
    refer to the complete original input sequence.
    """
    _validate_video(prediction)
    _validate_video(target)
    if prediction.shape != target.shape or prediction.shape[2] < 6:
        raise ValueError("Paired full-sequence temporal evaluation needs matching videos with >=6 frames")
    if (not isinstance(reference_index, int) or isinstance(reference_index, bool)
            or not 0 <= reference_index < prediction.shape[2]):
        raise ValueError("Reference index is outside the original evaluated sequence")
    full = perceptual.evaluate(prediction, target, include_temporal=True)
    lpips = full["lpips_per_frame"]
    if len(lpips) != prediction.shape[2]:
        raise ValueError("Official LPIPS must return every original frame")
    pairs = full["tecogan"]["per_pair"]
    frame_count = prediction.shape[2]
    pair_ids = []
    for pair in pairs:
        endpoints = pair["indices"]
        if (len(endpoints) != 2 or any(not isinstance(value, int) or isinstance(value, bool) for value in endpoints)
                or not 0 <= endpoints[0] < endpoints[1] < frame_count or endpoints[1] != endpoints[0]+1):
            raise ValueError("Perceptual evaluator must retain original adjacent temporal pairs")
        pair_ids.append(tuple(endpoints))
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("Duplicate temporal metric pairs")
    pools = {"all_frames": list(range(frame_count)),
             "matched_nonreference": [index for index in range(frame_count) if index != reference_index],
             "reference_frame_separate": [reference_index]}
    result = {"kind": "paired", "primary_pool": "matched_nonreference", "excluded_reference_index": reference_index,
              "perceptual_audit": full.get("audit"), "lpips_protocol": full.get("lpips_protocol"),
              "temporal_protocol": full["tecogan"].get("protocol"),
              "input_pixel_contract": "caller-provided full RGB; spatial scores not requantized by metric pool helper"}
    for name, indices in pools.items():
        rgb = rgb_metrics(prediction[:, :, indices], target[:, :, indices])
        for original, row in zip(indices, rgb["per_frame"]):
            row["original_frame_index"] = original
        selected = [pair for pair in pairs if all(endpoint in indices for endpoint in pair["indices"])]
        result[name] = {**rgb, "frame_indices": indices,
            "lpips_alex_v0_1": sum(float(lpips[index]) for index in indices)/len(indices),
            "lpips_per_frame": [float(lpips[index]) for index in indices],
            **{key: sum(float(pair[key]) for pair in selected)/len(selected) if selected else None
               for key in (TLP_KEY, TLP100_KEY, TOF_KEY)},
            "temporal_pair_indices": [pair["indices"] for pair in selected],
            "temporal_pair_count": len(selected), "per_temporal_pair": selected}
    return result


def paired_metrics(prediction, target, perceptual, *, excluded_reference=0):
    """Compatibility alias; prefer reference_index on paired_metric_pools."""
    return paired_metric_pools(prediction, target, perceptual, reference_index=excluded_reference)



# vsr/paired_statistics.py — consolidated implementation
def _observations(rows):
    result = {}
    for row in rows:
        record, source, seed = row["record_id"], row["source_id"], row["seed"]
        if not isinstance(record, str) or not record or not isinstance(source, str) or not source:
            raise ValueError("Every video requires an explicit record_id and source_id")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("Sampling seeds must be nonnegative integers")
        key = record, seed
        if key in result:
            raise ValueError(f"Duplicate video/seed: {key}")
        values = row["metrics"]
        if not values or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                             for v in values.values()):
            raise ValueError("Metric values must be finite numbers")
        result[key] = {"source_id": source, "metrics": {k: float(v) for k, v in values.items()}}
    if not result:
        raise ValueError("No observations")
    metrics = set(next(iter(result.values()))["metrics"])
    records = {record for record, _ in result}
    first = next(iter(records))
    seeds = {seed for record, seed in result if record == first}
    for record in records:
        if {seed for r, seed in result if r == record} != seeds:
            raise ValueError("Every video must include the same complete sampling seed set")
        if len({row["source_id"] for (r, _), row in result.items() if r == record}) != 1:
            raise ValueError("Source identity changed across sampling seeds")
    if any(set(row["metrics"]) != metrics for row in result.values()):
        raise ValueError("Metric keys differ between observations")
    return result


def paired_summary(groups, reference, *, replicates=10000, seed=20261102):
    """Compare matched experiments; delta always means candidate minus reference.

    The primary mean gives every video equal weight. For its percentile 95%
    interval, resample source IDs, retaining all videos of each drawn source.
    Equal-source means are a disclosed supplementary estimand. Confidence
    intervals are pointwise descriptive intervals, without multiplicity
    correction, training-repeat uncertainty, or a significance claim.
    """
    if reference not in groups or not groups:
        raise ValueError("Reference experiment missing")
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    observed = {label: _observations(rows) for label, rows in groups.items()}
    base = observed[reference]
    keys = set(base)
    metric_keys = sorted(next(iter(base.values()))["metrics"])
    for label, rows in observed.items():
        if set(rows) != keys:
            raise ValueError(f"{label}: video/seed sets differ; paired comparisons cannot drop observations")
        for key in keys:
            if rows[key]["source_id"] != base[key]["source_id"]:
                raise ValueError(f"{label}: source identities differ")
            if sorted(rows[key]["metrics"]) != metric_keys:
                raise ValueError(f"{label}: metric sets differ")
    records = sorted({record for record, _ in keys})
    seeds = sorted({s for _, s in keys})
    sources = [base[(record, seeds[0])]["source_id"] for record in records]
    unique_sources = sorted(set(sources))
    membership = [np.array([i for i, value in enumerate(sources) if value == source], dtype=int)
                  for source in unique_sources]
    source_counts = np.array([len(indices) for indices in membership], dtype=float)
    rng = np.random.default_rng(seed)
    # Common cluster draws preserve pairing and make comparisons order invariant.
    draws = rng.integers(0, len(membership), size=(replicates, len(membership)))
    result = {"protocol": {
        "mean": "all sampling seeds per video, then equal video weights",
        "interval": "paired source-cluster percentile bootstrap, 95%, pointwise descriptive",
        "supplementary": "clips averaged within source, then equal source weights",
        "delta": "candidate minus reference; direction depends on metric",
        "bootstrap_replicates": replicates, "bootstrap_seed": seed,
        "training_repeat_uncertainty_included": False, "multiplicity_correction": False,
        "single_source_interval_available": False,
        "limitation": "sampling repeats and source bootstrap do not establish reproducibility across training seeds"},
        "reference": reference, "video_count": len(records), "source_count": len(membership),
        "sampling_seeds": seeds, "experiments": {}, "comparisons": {}}
    arrays = {}
    for label, rows in observed.items():
        array = np.array([[np.mean([rows[(record, s)]["metrics"][metric] for s in seeds])
                           for metric in metric_keys] for record in records])
        source_array = np.stack([array[indices].mean(axis=0) for indices in membership])
        arrays[label] = array
        result["experiments"][label] = {
            "video_macro": dict(zip(metric_keys, array.mean(axis=0).tolist())),
            "source_macro": dict(zip(metric_keys, source_array.mean(axis=0).tolist())),
            "per_video": [{"record_id": record, "source_id": source,
                           "metrics": dict(zip(metric_keys, values.tolist()))}
                          for record, source, values in zip(records, sources, array)]}
    for label, array in arrays.items():
        if label == reference:
            continue
        delta = array - arrays[reference]
        source_sums = np.stack([delta[indices].sum(axis=0) for indices in membership])
        source_means = source_sums / source_counts[:, None]
        if len(membership) > 1:
            samples_video = source_sums[draws].sum(axis=1) / source_counts[draws].sum(axis=1)[:, None]
            samples_source = source_means[draws].mean(axis=1)
            video_interval = np.quantile(samples_video, [.025, .975], axis=0)
            source_interval = np.quantile(samples_source, [.025, .975], axis=0)
        result["comparisons"][label] = {
            metric: {"delta_video_macro": float(delta[:, m].mean()),
                     "ci95_source_cluster_video_macro": video_interval[:, m].tolist() if len(membership) > 1 else None,
                     "delta_source_macro": float(source_means[:, m].mean()),
                     "ci95_source_macro": source_interval[:, m].tolist() if len(membership) > 1 else None,
                     "video_deltas": dict(zip(records, delta[:, m].tolist())),
                     "positive_videos": int((delta[:, m] > 0).sum()),
                     "negative_videos": int((delta[:, m] < 0).sum()),
                     "zero_videos": int((delta[:, m] == 0).sum())}
            for m, metric in enumerate(metric_keys)}
    return result



# vsr/feature_diagnostics.py — consolidated implementation
NORMALIZATION_EPS = 1e-12


def _inputs(student, teacher, validmask):
    if student.ndim != 3 or student.shape != teacher.shape or min(student.shape) <= 0:
        raise ValueError("Student and teacher must share nonempty B,N,D shapes")
    if validmask.dtype != torch.bool or validmask.shape != student.shape[:2]:
        raise ValueError("validmask must be boolean with shape B,N")
    # CPU float64 avoids retaining training graphs or solver workspaces on the
    # GPU and makes constant-feature centering stable. Invalid values are masked
    # before normalization; invalid padding may therefore contain NaNs.
    student = student.detach().to(device="cpu", dtype=torch.float64)
    teacher = teacher.detach().to(device="cpu", dtype=torch.float64)
    validmask = validmask.detach().to(device="cpu")
    if not bool(validmask.any()):
        raise ValueError("Feature diagnostics require at least one valid token")
    if not bool(torch.isfinite(student[validmask]).all()) or not bool(torch.isfinite(teacher[validmask]).all()):
        raise ValueError("Valid feature vectors must be finite")
    return student, teacher, validmask


def _statistics(normalized):
    # Subtracting a common origin before computing the mean is algebraically
    # equivalent centering and gives exact zero for any repeated vector, even
    # when an irrational normalized component would accumulate mean roundoff.
    offset = normalized - normalized[:1]
    centered = offset - offset.mean(dim=0, keepdim=True)
    variance = float(centered.square().mean(dim=0).mean().clamp_min(0))
    singular_values = torch.linalg.svdvals(centered).clamp_min(0)
    total = float(singular_values.sum())
    maximum_rank = min(normalized.shape[0] - 1, normalized.shape[1])
    if total == 0 or maximum_rank <= 0:
        effective_rank = 0.0
    else:
        probabilities = singular_values / singular_values.sum()
        positive = probabilities[probabilities > 0]
        entropy = float(-(positive * positive.log()).sum())
        # Roundoff can place exp(H) infinitesimally beyond the algebraic bound.
        effective_rank = min(float(maximum_rank), max(0.0, math.exp(entropy)))
    return {"mean_channel_variance": variance, "effective_rank": effective_rank,
            "centered_singular_value_sum": total, "maximum_centered_rank": maximum_rank}


@torch.no_grad()
def feature_statistics(student, teacher, validmask):
    """Cosine error, population channel variance and centered effective rank.

    Select valid tokens, L2-normalize each with eps=1e-12, then center the M,D
    matrix across tokens. For singular values s, p=s/s.sum and effective rank
    is exp(-sum(p*log(p))). An exactly zero centered matrix has rank 0.
    Similarities are clipped to [-1,1]; negative similarities are retained.
    """
    student, teacher, mask = _inputs(student, teacher, validmask)
    selected_student, selected_teacher = student[mask], teacher[mask]
    student_zero = int((selected_student.norm(dim=-1) <= NORMALIZATION_EPS).sum())
    teacher_zero = int((selected_teacher.norm(dim=-1) <= NORMALIZATION_EPS).sum())
    u = F.normalize(selected_student, dim=-1, eps=NORMALIZATION_EPS)
    v = F.normalize(selected_teacher, dim=-1, eps=NORMALIZATION_EPS)
    cosine = (u * v).sum(dim=-1).clamp(-1, 1)
    student_stats, teacher_stats = _statistics(u), _statistics(v)
    return {"valid_token_count": int(mask.sum()), "feature_dimension": student.shape[-1],
            "valid_point_cosine_error": float((1 - cosine).mean().clamp(0, 2)),
            "student_mean_channel_variance": student_stats["mean_channel_variance"],
            "teacher_mean_channel_variance": teacher_stats["mean_channel_variance"],
            "student_effective_rank": student_stats["effective_rank"],
            "teacher_effective_rank": teacher_stats["effective_rank"],
            "student_centered_singular_value_sum": student_stats["centered_singular_value_sum"],
            "teacher_centered_singular_value_sum": teacher_stats["centered_singular_value_sum"],
            "maximum_centered_rank": student_stats["maximum_centered_rank"],
            "student_zero_norm_tokens": student_zero, "teacher_zero_norm_tokens": teacher_zero,
            "protocol": {"features": "valid token L2 normalization before all statistics", "normalization_eps": NORMALIZATION_EPS,
                         "cosine_clip": [-1, 1], "negative_cosine": "retained", "variance": "population variance per channel, then channel mean",
                         "centering": "valid token mean subtracted per channel", "svd_dtype": "float64", "svd_device": "CPU",
                         "effective_rank": "p=s/s.sum; exp(-sum(p*log(p))); zero centered matrix -> 0",
                         "numerical_rank_bound": "clipped to min(valid_token_count-1,feature_dimension)",
                         "autograd": "student and teacher detached", "restoration_quality_metric": False}}


@torch.no_grad()
def same_anchor_patch_identity_retrieval(student, teacher, validmask, token_grid):
    """Same-anchor spatial-patch top1 identity rate, without motion mapping.

    token_grid is (anchor_count,patch_height,patch_width), flattened T,H,W.
    Each student query searches valid teacher patches only in its own batch
    video and anchor. The correct answer is the same spatial patch index.
    Ties use torch.argmax's first valid patch; tied-query counts are reported.
    This checks the chosen spatial correspondence, not cross-frame tracking.
    """
    student, teacher, mask = _inputs(student, teacher, validmask)
    if len(token_grid) != 3 or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in token_grid):
        raise ValueError("token_grid must contain positive anchor,height,width integers")
    anchors, height, width = token_grid
    patches = height * width
    if anchors * patches != student.shape[1]:
        raise ValueError("token_grid does not match the feature token dimension")
    batch, _, channels = student.shape
    student = student.reshape(batch, anchors, patches, channels)
    teacher = teacher.reshape(batch, anchors, patches, channels)
    mask = mask.reshape(batch, anchors, patches)
    rows, hits, queries, empty = [], 0, 0, 0
    for b in range(batch):
        for anchor in range(anchors):
            valid = mask[b, anchor]
            count = int(valid.sum())
            if count == 0:
                empty += 1
                continue
            u = F.normalize(student[b, anchor, valid], dim=-1, eps=NORMALIZATION_EPS)
            v = F.normalize(teacher[b, anchor, valid], dim=-1, eps=NORMALIZATION_EPS)
            similarity = (u @ v.T).clamp(-1, 1)
            maxima = similarity.max(dim=-1).values
            selected = similarity.argmax(dim=-1)
            correct = selected == torch.arange(count)
            tied = (similarity == maxima[:, None]).sum(dim=-1) > 1
            anchor_hits = int(correct.sum())
            hits += anchor_hits
            queries += count
            rows.append({"batch_index": b, "anchor_index": anchor, "valid_patch_count": count,
                         "top1_identity_rate": anchor_hits / count,
                         "mean_corresponding_cosine": float(similarity.diag().mean()),
                         "tied_top1_query_count": int(tied.sum())})
    return {"same_anchor_spatial_patch_top1_identity_rate": hits / queries,
            "anchor_macro_top1_identity_rate": sum(row["top1_identity_rate"] for row in rows) / len(rows),
            "correct_query_count": hits, "valid_query_count": queries, "nonempty_anchor_count": len(rows),
            "empty_anchor_count": empty, "per_anchor": rows,
            "protocol": {"name": "same-anchor spatial-patch top1 identity retrieval", "flatten_order": "T,H,W",
                         "candidate_set": "valid spatial teacher patches in the same batch video and same anchor",
                         "target": "identical spatial patch index", "motion_compensated": False, "cross_frame": False,
                         "similarity": "L2-normalized cosine clipped[-1,1]; negatives retained",
                         "tie_policy": "first valid spatial patch via torch.argmax; exact-equality ties counted",
                         "autograd": "student and teacher detached", "restoration_quality_metric": False}}



# vsr/method_diagnostics.py — consolidated implementation
@torch.no_grad()
def analyze_feature_snapshot(snapshot):
    fields = ('student_raw','teacher_raw','student_projected','teacher_normalized')
    features = snapshot['features']
    if not isinstance(features,dict) or set(features) != set(fields):
        raise ValueError('Complete raw/projected student and raw/LayerNorm teacher snapshots required')
    result = {}
    counts = set()
    for name in fields:
        value = features[name]
        if (not isinstance(value,torch.Tensor) or value.device.type != 'cpu' or value.requires_grad
                or value.ndim != 2 or min(value.shape) <= 0 or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Feature diagnostics accept finite detached nonempty CPU [tokens,channels] only')
        counts.add(value.shape[0])
        value = value.to(dtype=torch.float64)
        offset = value - value[:1]
        centered = offset - offset.mean(dim=0,keepdim=True)
        channel_variance = centered.square().mean(dim=0).clamp_min(0)
        normalized = F.normalize(value,dim=-1,eps=NORMALIZATION_EPS)
        normalized_stats = _statistics(normalized)
        result[name] = {'tokens':value.shape[0],'channels':value.shape[1],
            'mean_population_channel_std':float(channel_variance.sqrt().mean()),
            'rms_population_channel_std':float(channel_variance.mean().sqrt()),
            'normalized_rms_channel_std':math.sqrt(normalized_stats['mean_channel_variance']),
            'normalized_centered_effective_rank':normalized_stats['effective_rank'],
            'maximum_centered_rank':normalized_stats['maximum_centered_rank'],
            'normalized_centered_singular_value_sum':normalized_stats['centered_singular_value_sum']}
    if len(counts) != 1 or snapshot.get('selected_tokens') != next(iter(counts)):
        raise ValueError('All feature snapshots must describe the same selected tokens')
    return {key:value for key,value in snapshot.items() if key != 'features'} | {
        'statistics':result,'scope':{'quality_metric':False,'selection':'corrupted tokens in sigma>0.3 only',
            'feature_state':'features used in this step before its optimizer update',
            'dtype':'CPU float64','SVD_frequency':'only explicitly drained first-active/every-N snapshots',
            'raw_std':'population variance across selected tokens per channel; report mean SD and RMS SD',
            'effective_rank':'L2 normalize tokens eps1e-12, center tokens, p=s/s.sum, exp(-sum(p*log(p))); zero ->0',
            'timing':'CPU statistics/SVD outside Step; owned snapshot copy is inside the complete Step',
            'model_optimizer_gradients_RNG_accessed':False}}



# scripts/stage2_metrics.py — consolidated implementation
TLP = "tLP100_alex_v0_1_TecoGAN_crop"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def score_pools(rgb, perceptual, mask):
    need(len(mask) == 33 and all(type(x) is bool for x in mask) and sum(mask) == 8,
         "Require33 boolean mask entries with8 selected frames")
    selected = [i for i, flag in enumerate(mask) if flag]
    need(selected == list(range(selected[0], selected[0]+8)) and selected[0] in (4, 8, 12, 16, 20),
         "Require the frozen interior8-frame burst")
    neighbors = [selected[0]-1, selected[-1]+1]
    frames = rgb["per_frame"]
    lpips = perceptual["lpips_per_frame"]
    need(len(frames) == len(lpips) == 33 and all(finite(x) for x in lpips), "Incomplete/nonfinite per-frame metrics")
    for row in frames:
        need(all(finite(row[k]) for k in ("psnr_rgb_db", "ssim_rgb")), "Nonfinite frame fidelity metric")
    pairs = perceptual["tecogan"]["per_pair"]
    pairmap = {tuple(row["indices"]): row for row in pairs}
    expected = {(i, i+1) for i in range(2, 30)}
    need(len(pairmap) == len(pairs) == 28 and set(pairmap) == expected
         and all(finite(row[TLP]) for row in pairs), "Expected28 complete original TecoGAN adjacent pairs")
    touch = sorted(p for p in expected if any(mask[i] for i in p))
    boundary = [p for p in touch if mask[p[0]] != mask[p[1]]]
    need(len(touch) == 9 and len(boundary) == 2, "Unexpected temporal mask support")
    pools = {}
    for name, indices in (("masked", selected), ("neighbors", neighbors), ("all_frames", list(range(33)))):
        pools[name] = {"indices": indices, "psnr_rgb_db": statistics.mean(frames[i]["psnr_rgb_db"] for i in indices),
                       "ssim_rgb": statistics.mean(frames[i]["ssim_rgb"] for i in indices),
                       "lpips_alex_v0_1": statistics.mean(lpips[i] for i in indices)}
    return {"mask": mask, "pools": pools,
            "raw_harm_gap": {"psnr_rgb_db": pools["neighbors"]["psnr_rgb_db"]-pools["masked"]["psnr_rgb_db"],
                             "lpips_alex_v0_1": pools["masked"]["lpips_alex_v0_1"]-pools["neighbors"]["lpips_alex_v0_1"]},
            "temporal": {"mask_touch_original_pairs": [list(p) for p in touch],
                "boundary_original_pairs": [list(p) for p in boundary],
                "mask_touch_tLP100": statistics.mean(pairmap[p][TLP] for p in touch),
                "all28pairs_tLP100": statistics.mean(row[TLP] for row in pairs)}}


def paired_excess(clean, corrupt):
    need(clean["mask"] == corrupt["mask"], "Clean/corrupt masks differ")
    return {metric: {"clean_gap": clean["raw_harm_gap"][metric],
                    "corrupt_gap": corrupt["raw_harm_gap"][metric],
                    "paired_excess_gap": corrupt["raw_harm_gap"][metric]-clean["raw_harm_gap"][metric]}
            for metric in ("psnr_rgb_db", "lpips_alex_v0_1")}


def summarize_g2(rows, seal):
    cfg = seal["config"]
    cohort = {r["record"]["id"]: r for r in seal["cohort"]}
    seeds, samples, conditions = cfg["training_seeds"], cfg["inference"]["sampling_seeds"], cfg["conditions"]
    need(len(seeds) == 3 and len(samples) == 2 and len(conditions) == 5 and len(cohort) == 32,
         "G2 requires3 matched training seeds,2 sampler seeds,all5 types and val32")
    seen, checkpoint_hashes = {}, {}
    for row in rows:
        key = row["training_seed"], row["record_id"], row["sampling_seed"], row["condition"]
        need(key not in seen and key[0] in seeds and key[1] in cohort and key[2] in samples
             and key[3] in ["clean", *conditions], "Duplicate/out-of-cohort score observation")
        need(row["source_id"] == cohort[key[1]]["record"]["source_id"]
             and row["cohort_seal_digest"] == seal["seal_digest"]
             and row["score"]["mask"] == cohort[key[1]]["mask"], "Source/seal/mask score binding mismatch")
        need(isinstance(row["checkpoint_sha256"], str) and re.fullmatch('[0-9a-f]{64}',row["checkpoint_sha256"]),
             "A checkpoint digest declaration is required")
        checkpoint_hashes.setdefault(key[0], row["checkpoint_sha256"])
        need(checkpoint_hashes[key[0]] == row["checkpoint_sha256"], "Checkpoint changed within training seed")
        seen[key] = row["score"]
    need(len(seen) == 3*32*2*6, "Missing score rows; cannot drop clips/seeds/types")
    appendix, per_seed = [], {}
    metrics = ("psnr_rgb_db", "lpips_alex_v0_1")
    for seed in seeds:
        type_rows = {}
        for condition in conditions:
            clips = []
            for record in cohort:
                comparisons = [paired_excess(seen[seed, record, sample, "clean"], seen[seed, record, sample, condition])
                               for sample in samples]
                values = {metric: {field: statistics.mean(c[metric][field] for c in comparisons)
                                   for field in ("clean_gap", "corrupt_gap", "paired_excess_gap")} for metric in metrics}
                clips.append(values)
                appendix.append({"training_seed": seed, "record_id": record, "condition": condition, "metrics": values})
            type_rows[condition] = {metric: {field: statistics.mean(c[metric][field] for c in clips)
                                            for field in ("clean_gap", "corrupt_gap", "paired_excess_gap")} for metric in metrics}
        per_seed[str(seed)] = {"per_type": type_rows, "equal_type_macro": {
            metric: {field: statistics.mean(type_rows[c][metric][field] for c in conditions)
                     for field in ("clean_gap", "corrupt_gap", "paired_excess_gap")} for metric in metrics}}
    gate = {}
    for metric in metrics:
        clean = [per_seed[str(s)]["equal_type_macro"][metric]["clean_gap"] for s in seeds]
        excess = [per_seed[str(s)]["equal_type_macro"][metric]["paired_excess_gap"] for s in seeds]
        spread, mean = statistics.stdev(clean), statistics.mean(excess)
        need(all(finite(x) for x in [*clean, *excess]), "Nonfinite G2 values")
        gate[metric] = {"clean_gap_by_training_seed": dict(zip(map(str, seeds), clean)),
                        "paired_excess_gap_by_training_seed": dict(zip(map(str, seeds), excess)),
                        "mean_paired_excess_gap": mean, "paired_excess_gap_sample_std": statistics.stdev(excess),
                        "clean_gap_sample_std_ddof1_threshold": spread, "passes": mean > spread}
    return {"status": "passed_G2" if all(x["passes"] for x in gate.values()) else "failed_G2_stop",
            "gate_metrics": gate, "per_training_seed_and_type": per_seed, "per_clip_appendix": appendix,
            "cohort_seal_digest": seal["seal_digest"], "checkpoint_digest_declarations": checkpoint_hashes,
            "observations": len(rows), "p_values_computed": False, "Stage5_significance_decision": False,
            "underlying_prediction_payloads_revalidated_by_this_summarizer": False,
            "failure_policy": "stop/report; no in-place severity or cohort change"}


def quantize_rgb(video):
    """Common PNG-equivalent q/255 pixels; retain torch round-to-nearest."""
    _validate_video(video)
    return torch.round(video.detach().float().clamp(0, 1) * 255) / 255


def score_masked(full, mask):
    """Preserve original Stage2 pixel-frame and temporal-pair score pools."""
    pool = full['all_frames']
    return score_pools(pool, {'lpips_per_frame': pool['lpips_per_frame'],
                              'tecogan': {'per_pair': pool['per_temporal_pair']}}, mask)


def _score_vector(score):
    """Prespecified scalar pools; no row/type/metric selection from results."""
    result = {}
    for pool in ('masked', 'neighbors', 'all_frames'):
        for key in ('psnr_rgb_db', 'ssim_rgb', 'lpips_alex_v0_1'):
            value = score['pools'][pool][key]
            need(finite(value), 'Nonfinite pilot metric')
            result[pool + '/' + key] = value
    for key in ('mask_touch_tLP100', 'all28pairs_tLP100'):
        value = score['temporal'][key]
        need(finite(value), 'Nonfinite original-pair temporal score')
        result['temporal/' + key] = value
    for key in ('psnr_rgb_db', 'lpips_alex_v0_1'):
        value = score['raw_harm_gap'][key]
        need(finite(value), 'Nonfinite recovery gap')
        result['raw_harm_gap/' + key] = value
    return result


def index_observations(rows, seal, role_hashes, *, kind):
    """Require every role x val32 x six conditions x sampler42/43 once."""
    need(kind in ('bases', 'pilot') and len(role_hashes) == 3, 'Three declared roles required')
    cohort = {item['record']['id']:item for item in seal['cohort']}
    need(len(cohort) == 32, 'Require the complete sealed val32')
    conditions = ['clean', *seal['config']['conditions']]
    need(len(conditions) == 6 and len(set(conditions)) == 6, 'Require all five corrupt types plus clean')
    found = {}
    for row in rows:
        role = str(row['training_seed']) if kind == 'bases' else row['arm']
        key = role, row['record_id'], row['condition'], row['sampling_seed']
        need(role in role_hashes and key[1] in cohort and key[2] in conditions and key[3] in (42, 43)
             and key not in found, 'Duplicate or out-of-protocol observation')
        item = cohort[key[1]]
        need(row['checkpoint_sha256'] == role_hashes[role]
             and row['source_id'] == item['record']['source_id']
             and row['cohort_seal_digest'] == seal['seal_digest']
             and row['score']['mask'] == item['mask'], 'Observation checkpoint/source/seal/mask mismatch')
        if kind == 'pilot':
            need(row['training_seed'] == 20261102, 'Pilot has exactly one matched training seed')
        found[key] = _score_vector(row['score'])
    expected = {(role, record, condition, sample) for role in role_hashes for record in cohort
                for condition in conditions for sample in (42, 43)}
    need(set(found) == expected and len(rows) == 1152, 'Missing observations; full1152 matrix required')
    return found


def summarize_pilot(rows, base_rows, seal, role_hashes, base_hashes, *, draws=10000, seed=20261102):
    """Frozen G4 point gate plus descriptive paired-clip PCG64 intervals.

    This does not run training, select a setting, compute a p-value or authorize
    an automatic4096-step continuation. Definitions must be bound by the caller.
    """
    need(draws == 10000 and seed == 20261102, 'Use the prospectively frozen bootstrap draws and seed')
    need(set(role_hashes) == {'flow_only', 'C1', 'A2'} and set(base_hashes) == {'20261102', '20261103', '20261104'},
         'Unexpected pilot or Base roles')
    method = index_observations(rows, seal, role_hashes, kind='pilot')
    base = index_observations(base_rows, seal, base_hashes, kind='bases')
    records = [item['record']['id'] for item in seal['cohort']]
    corrupt = seal['config']['conditions']
    conditions = ['clean', *corrupt]
    names = list(next(iter(method.values())))
    def by_clip(index, role, condition, metric):
        return np.asarray([statistics.mean(index[role, r, condition, s][metric] for s in (42,43))
                           for r in records], dtype=np.float64)
    def macro(index, role, metric):
        return np.mean([by_clip(index, role, condition, metric) for condition in corrupt], axis=0)
    def stats(values):
        return {'mean':float(np.mean(values)), 'between_clip_sample_SD':float(np.std(values, ddof=1)), 'n_clips':32}
    def table(index, role):
        result = {condition:{m:stats(by_clip(index, role, condition, m)) for m in names} for condition in conditions}
        result['equal_five_type_macro'] = {m:stats(macro(index, role, m)) for m in names}
        return result
    method_table = {r:table(method,r) for r in role_hashes}
    base_table = {r:table(base,r) for r in base_hashes}
    key = 'masked/lpips_alex_v0_1'
    base_means = {r:float(macro(base,r,key).mean()) for r in base_hashes}
    spread = statistics.stdev(base_means.values())
    recovery = macro(method,'C1',key)-macro(method,'A2',key)
    fidelity = {m:by_clip(method,'A2','clean','all_frames/'+m)-by_clip(method,'C1','clean','all_frames/'+m)
                for m in ('psnr_rgb_db','ssim_rgb')}
    gain = float(recovery.mean())
    guards = {m:{'A2_minus_C1_mean':float(v.mean()), 'passes':bool(v.mean() >= 0)} for m,v in fidelity.items()}
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0,32,size=(draws,32),dtype=np.int64)
    indices_hash = hashlib.sha256(indices.astype('<i8',copy=False).tobytes()).hexdigest()
    def interval(values):
        sampled = values[indices].mean(axis=1)
        return list(map(float,np.quantile(sampled,[.025,.975],method='linear')))
    comparisons = {}
    for left,right in [('A2','C1'),('A2','flow_only'),('C1','flow_only'),
                       ('flow_only','matched_Base'),('C1','matched_Base'),('A2','matched_Base')]:
        rhs,role = (base,'20261102') if right == 'matched_Base' else (method,right)
        result = {}
        for condition in [*conditions,'equal_five_type_macro']:
            result[condition] = {}
            for metric in names:
                if condition == 'equal_five_type_macro':
                    values = macro(method,left,metric)-macro(rhs,role,metric)
                else:
                    values = by_clip(method,left,condition,metric)-by_clip(rhs,role,condition,metric)
                result[condition][metric] = stats(values) | {'paired_clip_95pct_CI':interval(values)}
        comparisons[left+'_minus_'+right] = result
    passes = gain > spread and all(g['passes'] for g in guards.values())
    return {'status':'passed_G4' if passes else 'failed_G4_stop','method_observations':1152,'Base_observations':1152,
            'method_training_seeds':[20261102], 'per_arm':method_table,'Base_per_training_seed':base_table,
            'recovery':{'C1_minus_A2_masked_LPIPS':gain,'Base_corrupt_masked_LPIPS_by_seed':base_means,
                        'Base_seed_sample_SD_ddof1_threshold':spread,'passes':gain>spread,
                        'paired_clip_95pct_CI':interval(recovery)},
            'clean_guards':guards,'paired_comparisons':comparisons,
            'uncertainty':{'draws':draws,'seed':seed,'generator':'PCG64','draw_indices_sha256':indices_hash,
                           'scope':'descriptive paired clips conditional on one trained method seed',
                           'CI_is_additional_gate':False,'p_values_computed':False},
            'automatic_retry':False,'automatic_4096_continuation':False,'Stage5_significance_decision':False}
