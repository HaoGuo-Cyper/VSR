# Next audit execution plan — 2026-11-05

Status: static feasibility review only. No GPU task, checkpoint deserialization, model load or image decode was started by this review. The user's instruction to continue authorizes progress on Stages 0/1; it does not mark D1–D4 SETTLED or authorize Stage 2 training. This document is an execution plan, not an execution result.

## 1. T0.2: reproduce one completed NoRef Base external row

### Fixed target and numerical protocol

Use `NoRef_Base_seed20261102_step8192 / REDS4 / 000`, selected now for being the first declared REDS4 sequence. Reproduce both sampler seeds 42 and 43; do not select a seed after inspecting its quality. Original checkpoint:

`/home/hugo/Projects/VSR/runs/formal_hq_v2_noref_20261102T142214.588571Z/Base/checkpoint_008192.pt`

Original reference row:

`/home/hugo/Projects/VSR/runs/external_paired_frozen_20261103T125240.867596Z/external/results/NoRef_Base_seed20261102_step8192/REDS4/000/`

Retain the original first-17-frame prefix (indices 0–16), stride 1, native LR 180×320 / HR 720×1280, UniPC 20 steps, CFG 1, HR tiles 256×448, overlap 64, original global CPU noise construction, exact cached generic text, original PNG source and q/255 metric pixels. No HRI is supplied. This is a historical protocol reproduction, not a proposal to adopt stride 1 for subsequent training.

There are 16 spatial tiles per restore: padded HR 720×1280, y starts `[0,192,384,464]` and x starts `[0,384,768,832]`. Each of the two seeds therefore makes 16 original VAE encode delegates and 16×20 DiT sampler iterations. The T1-specific VAE route is unused for this NoRef video; preserve the historical instance wrapper and per-restore remove/reinstall lifecycle, whose ledger should contain 16 original delegates, zero T1 events and no dropped entries.

### Why a dedicated bounded harness is needed

The unchanged historical outer worker cannot run a single row: `phase_metadata()` requires all 44 sequences and its final `verify_totals()` requires all eight model roles. It also binds the old exact worker path, predecessor review and all-eight execution seal. Do not edit its scope checks or give a one-row job the old phase's completion label.

The ordinary `run_frozen_evaluation()` API also needs care in the new checkout. It resolves checkpoints inside `project_root/runs` (`vsr/frozen_evaluation.py:329–344`), whereas original checkpoints live under the old root. `ExternalReader._local()` requires resolved data paths inside the supplied read root (`vsr/external_data.py:56–60`); the new checkout's `data` symlink resolves to the old project and is therefore not an automatically valid reader root. The new checkout correctly keeps its own `runs/` directory separate.

Implement a new script, proposed name `scripts/reproduce_external_row.py`, with separate explicit `--read-root /home/hugo/Projects/VSR` and `--output-dir /home/hugo/Projects/VSR_/runs/reproduce_NoRefBase_REDS4_000_<UTC_TIMESTAMP>`. It should have one fixed checkpoint/sequence and the two declared seeds, not a broad search interface. The implementation can call the existing APIs directly:

1. Freeze a new subset manifest and compare the original root numerical protocol, source files, checkpoint file, official model/scheduler config and text-cache SHA256 bindings. Hashing the checkpoint file is read-only; actual deserialization belongs only to the later execution phase.
2. Construct `ExternalReader(read_root)`, retain its full immutable manifest, and call `load('REDS4', '000', num_frames=17, start=0)`. Never rewrite the manifest to a fake one-sequence REDS4 dataset: its validation requires the standard four IDs. Record the original 17 LR/GT PNG hashes and source frame numbers, and verify they equal the reference row's provenance.
3. Load the checkpoint with `load_training_checkpoint(checkpoint, allowed_root=read_root/'runs')`, recording the exact origin. Use the already accepted cached text via the original read root; enforce that the text comes from its pinned cache and fail instead of falling back to T5. The default caption function has a T5 fallback (`vsr/frozen_evaluation.py:208–230`), which this bounded reproduction must disable explicitly.
4. Load only the real FP32 frozen VAE alongside the DiT required for this reproduction. Apply the historical `t1_vae_cache_policy.install` and `RotatingExternalRestore` instance wrapper, using the original shared `validate_ledger`. A single-row harness must replace the outer all-eight aggregate check with explicit subset counts: one NoRef role, two restores, 32 original delegates, zero T1 events, two rotations, no retained policy tensors or model feature contexts. The original numerical restore delegate remains `vsr.inference.restore_video`.
5. Use the original `quantize_rgb`, PNG saver and `paired_metrics`/`PerceptualMetrics` implementations. Keep all-frames, matched-nonreference and reference-frame-separate pools; the latter is a reporting pool even for NoRef. Compute metrics on CPU exactly as before.
6. Write all new protocols, images, metrics, resource records and comparisons only below the new output directory. Do not use the old child run as `run_dir`; do not call its generic resume path. On interruption, preserve the new partial attempt and start any later attempt in another dated directory.

A read-root/output-root split also avoids copying a 107 MB adapter checkpoint merely to satisfy an output-path guard. Do not bypass that guard by widening it in the unchanged legacy evaluator. The new script owns and validates its own narrower read/write roots.

### Evidence and seals

Reuse the following as immutable references, not as authorization or a completion certificate for the new subset:

- `docs/final_evaluation_v1/protocol.json`, original raw SHA `418e9e36cf75b8929b63aaf0eaa5222c38fb4ad485b55f012a4d558aa7b2f746`.
- The original checkpoint cohort acceptance, text-cache preflight, VAE policy parity and producer environment records.
- `external_paired_execution_seal_desktop_v2.json`, the HQ predecessor completion review and existing row's complete/metrics/artifact seals.
- The archived seven-file helper closure and its `scripts/frozen_external/provenance.json` hashes.

The new subset needs its own pre-execution manifest binding its new harness SHA, exact imported module paths/hashes, old numerical protocol SHA, read-root/output-root, checkpoint/hash, one sequence, two sampler seeds, input catalogue, cache/hash, device/dtypes, backend flags and comparison rules. A new completion record must say `single_row_reproduction`, not full external-phase pass. It must not pretend the old all-eight/predecessor path bindings authorize a relocated script.

The archived modules insert the old project root into `sys.path`. If importing them verbatim for the wrapper classes, record `vsr.__file__` and every numerical module path/hash after imports and require their bytes equal the frozen producer. Avoid accidentally mixing a new module with a stale old module simply because both have the same package name. No archived `main()` is needed.

Run a fresh process with the old environment before CUDA initialization:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTORCH_ALLOC_CONF=expandable_segments:True
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Use `/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python` and the existing project extras. Apply `configure_reproducibility()` from the checkpoint's exact config before model construction; preserve math-SDPA/TF32/deterministic settings and native FP32 VAE. Record actual versions. Never reduce the steps, frames, precision, tile size or seed set to make the reproduction pass.

### Time, memory and pass criteria

The original REDS4/000 NoRef Base row records the following costs, read from the stored row without recomputing quality:

| Sampler seed | Restore seconds | PNG-save seconds | Metric seconds | Restore CUDA peak allocated |
| --- | ---: | ---: | ---: | ---: |
| 42 | 120.398 | 3.205 | 10.977 | 5,695,114,752 bytes |
| 43 | 120.089 | 3.233 | 10.954 | 5,695,114,752 bytes |

The two restores took approximately four minutes in that run; both restores plus PNG/metric work totaled about 269 seconds. This excludes cold-start model loading, input verification, checkpoint/base-file hashing, baseline metrics, process startup and final comparison. The recorded peak includes already-resident model/VAE/metric weights and is not a measurement of a new process. Allow additional startup time rather than calling 269 seconds a complete-job guarantee.

Before execution, fix these checks:

- Input PNG hashes, ordered indices, checkpoint hash, cached text hash and numerical settings must match exactly.
- For each seed, compare all 17 decoded uint8 output frames exactly. Also compare compressed PNG SHA values separately; differing file encoding with identical decoded pixels must be reported distinctly.
- Compare every historical metric key and original frame/pair pool. Propose an absolute tolerance of `1e-10` for finite JSON scalar comparisons in the same software environment, recorded before the run. Report actual deltas and any changed environment. Do not widen tolerances after seeing a mismatch.
- Any pixel mismatch, nonfinite value, missing artifact or changed numerical contract leaves T0.2 unpassed. Preserve diagnostics and identify the cause; a successful single row does not verify the entire historical external cohort.

## 2. T1.1 / T1.2: VAE-only audit

### Reuse functions, not the old preflight entry point

`vsr.preflight` includes transformer construction and training diagnostics in its broader workflow; do not run its CLI for a VAE audit. Reuse the small `normalize_latent()` helper (`vsr/preflight.py:46–49`), `decode_clip()`, `rgb_metrics()`, `PerceptualMetrics` and the frozen VAE loader. Implement a separate proposed `scripts/audit_vae_protocol.py` which loads `AutoencoderKLWan` alone from the pinned local VAE safetensors, uses FP32/eval/no_grad, and never loads the DiT, LoRA checkpoint, DINOv2, T5 or an optimizer.

For each selected clip/condition, compute:

```python
raw_hr = vae.encode(hr * 2 - 1).latent_dist.mode()
reconstruction = (vae.decode(raw_hr).sample.float() / 2 + 0.5).clamp(0, 1)
raw_lr = vae.encode(upsampled_lr * 2 - 1).latent_dist.mode()
normalized_hr = (raw_hr - official_mean) / official_std
normalized_lr = (raw_lr - official_mean) / official_std
```

Use posterior mode consistently; do not sample. Decode raw latent directly, or explicitly inverse-normalize a normalized latent before decode. Save geometry assertions and per-channel raw/normalized mean, std and LR–HR distances. This obtains T1.2 from the same inputs without introducing a teacher or training run. Save reference to the exact official normalization constants and whether std uses population or sample convention.

Wan's official `_encode` processes the first input frame and subsequent four-frame chunks with temporal caches. It yields latent counts 3/5/9 for input lengths 9/17/33. These are internal causal chunks, not independently encoded temporal windows. Never turn the main audit into per-frame T=1 encoding or discard the temporal cache between internal chunks.

### Frame eligibility and paired sample design

A clip with T sampled frames and source stride s needs `1+(T-1)*s` source frames:

| T | Stride 1 source frames | Stride 4 source frames | Latent time positions |
| --- | ---: | ---: | ---: |
| 9 | 9 | 33 | 3 |
| 17 | 17 | 65 | 5 |
| 33 | 33 | 129 | 9 |

The existing HQ header inventory has 201 val clips. All 201 have header eligibility for the first five cells; only 110 have at least 129 frames and sufficient crop dimensions for T33/stride4. These are metadata-only counts, not a new decode/SHA validation. Preserve the existing train/val/test assignments. Freeze the 110-clip common intersection for paired six-cell comparisons; optionally report the other 91 val clips separately for the five supported cells. Do not pool unmatched condition means as if they came from identical clips.

Within each clip, derive one deterministic shared source start from the maximum 129-frame span and the record ID, then take each condition's explicit indices from that start. The existing `decode_clip()` hashes its start modulo a condition-dependent eligible span, so calling it blindly for every T/s changes the start and confounds temporal-condition comparisons. A new read-only explicit-index decoder or a verified selection adapter is required. Retain the same centered 128×224 spatial crop, interpolation and bicubic degradation in every condition. No original source/split manifest is rewritten.

Start with a deterministic, predeclared eight-clip subset of the 110 common-intersection val clips (48 cases) for resource/time measurement and numerical checks. This bounded pilot is not completion of the full validation audit. Report measured seconds/case and extrapolate the remaining cost before running the full set; no reliable whole-audit ETA is available from static tensor sizes alone.

For REDS4, all four sources contain only 100 frames. T33/stride4 requires 129 and is impossible without changing the source. Mark this cell `not_available_source_too_short`; do not pad, wrap, repeat frames or silently replace T33 with T25. REDS4/000 can support the other five cells. For a deterministic prefix comparison across those five, use start 0 and explicit source indices, with no resizing in the full-resolution branch. The common HQ intersection covers the complete six-cell design.

### Memory feasibility

The local VAE safetensors file is 507,591,892 bytes. VAE-only 128×224 FP32 inference should be a practical 16 GB candidate because the DiT, T5, DINOv2 and optimizer are absent, and native temporal execution uses bounded causal chunks. This is an engineering expectation, not a measured capacity pass.

For the largest crop case T33, raw RGB storage is 11,354,112 bytes and a `[1,16,9,16,28]` latent is 258,048 bytes. Activations, convolution workspaces, temporal caches and decoder concatenations dominate the actual peak and must be measured. Process one condition at a time, transfer latents/reconstructions to CPU, release GPU references/caches between cases, and measure synchronized peak allocated/reserved memory and elapsed encode/decode seconds separately. Run LPIPS and RAFT in a later phase after releasing the VAE; do not keep every metric network resident simultaneously.

The historical T1 cache policy only changes T=1 encodes and explicitly delegates all multi-frame clips unchanged. It does not solve general full-frame 9/17/33-frame VAE memory pressure. No new precision or caching behavior should be silently added to the primary FP32 audit.

### Tiled versus untiled full-resolution comparison

The actual pinned VAE source is `/home/hugo/miniconda3/envs/reprvsr-pretrained/lib/python3.12/site-packages/diffusers/models/autoencoders/autoencoder_kl_wan.py`, SHA recorded in the frozen protocol. `enable_tiling()` is implemented at lines 1097–1125; defaults are tile 256×256, spatial strides 192×192 (64-pixel overlap). It affects both encode and decode. Native `_encode` and `_decode` branch on `use_tiling` at lines 1144–1145 and 1196–1197.

128×224 crops are below the native tiling threshold, so toggling the flag there is not a valid spatial-tiling test. Use native REDS4/000 HR 720×1280, first nine frames, as an initial bounded full-resolution trial. Keep T9/stride1 fixed for this spatial diagnostic. Untiled full-resolution VAE memory is not established by the 5.7 GB tiled DiT reproduction measurement and may exceed 16 GB.

To isolate encoding effects, compare raw latent tensors from untiled and tiled encoding, then reconstruct both through one identical decoder mode. A common tiled decoder is acceptable if full-frame untiled decoding does not fit, provided the report clearly states this. If both encoder and decoder are toggled together, label the result an end-to-end tiled-VAE comparison, not an encoding-only effect. Record all four choices explicitly if expanding to encoder{tiled,untiled} × decoder{tiled,untiled}.

On OOM, preserve the failed geometry, dtype, allocated/reserved peak and stage. Record the untiled cell as unmeasured/OOM; do not claim a tiled-versus-untiled full-resolution quality comparison from unmatched spatial crops. CPU untiled execution would be a separate expensive fallback with its own recorded device contract, not an automatic substitute for CUDA capacity evidence.

## 3. Metric support and exact names

| Requested/possible measure | Existing implementation | Use in the audit |
| --- | --- | --- |
| RGB PSNR/SSIM, per frame | `vsr.evaluation.rgb_metrics`, lines 34–59 | Ready; no Y conversion, no border shave; frame-mean PSNR and skimage SSIM |
| LPIPS per frame | `vsr.perceptual_metrics.PerceptualMetrics.evaluate`, lines 146–160 | Ready; pinned lpips 0.1.4 / Alex v0.1 and local weights |
| TecoGAN tLP / tLP100 | Same module lines 162–209 | Ready; cutfr=2 and author's crop; keep these names, not a generic tLPIPS alias |
| TecoGAN tOF | Same module, Farneback | Ready; retain `tOF_Farneback_TecoGAN_crop`; not RAFT E*warp |
| RAFT flows | `vsr.flow.FlowTeacher`, lines 40–126 | Flow estimator is available; FP32, 12 updates, local pinned weights; no complete image E*warp metric implemented |
| Flow-based warp error | `vsr.relation.temporal_correspondence`, lines 21–77 | Grid and forward/backward reliability helper exists for token supervision; it is not a published image-level E*warp implementation |

Existing temporal tLP is `abs(LPIPS(GT_i,GT_j)-LPIPS(recon_i,recon_j))`, averaged over retained original adjacent pairs. For T9/T17/T33, cutfr=2 leaves 4/12/28 pairs respectively. Their different durations and stride conditions must be labeled separately. Report per-frame reconstruction errors on every original sampled index, including first/last frames; do not use cutfr to hide VAE boundary effects in spatial tables.

A precise published E*warp definition is not present in the repository. If adding a diagnostic now, give it its own formula/name before measuring, for example `raft12_gtflow_temporal_residual_l1`: estimate forward/backward flow on HR adjacent sampled frames; warp the next reconstruction and next HR with the same HR-derived grid; compare their temporal residuals under the same in-bounds/forward-backward-consistent mask. Record alpha=0.01, beta=0.5, mask weighting, RGB normalization, denominator and empty-mask handling. This can reuse RAFT and grid helpers after an image-resolution adapter and meaningful fixtures. It must not be called E*warp without establishing that definition from its intended reference, and it does not settle D4's future primary metric.

Use full FP32 RGB reconstructions for the primary VAE reconstruction audit and explicitly declare this pixel contract. If comparing to frozen external metrics, produce a separate q/255 view using the historical quantizer; never combine float and q/255 values without labels. Temporal Farneback/TecoGAN internals perform their documented uint8 conversion regardless of the spatial float contract.

## 4. Proposed outputs and gates

Each execution creates a fresh dated directory, for example:

```text
/home/hugo/Projects/VSR_/runs/reproduce_NoRefBase_REDS4_000_<UTC_TIMESTAMP>/
/home/hugo/Projects/VSR_/runs/vae_protocol_audit_<UTC_TIMESTAMP>/
```

Use the user's local date in prose and an explicit timezone for every timestamp; retain UTC in path stamps for consistency with existing runs. Required audit files: frozen selection/config, source/module/weight hashes, environment, source indices and crop geometry, per-case status/cost/memory, per-frame CSV, per-channel latent CSV, per-pair temporal CSV, paired summary and failed/unavailable cell ledger. Preserve local images/latents under the dated run only; commit concise configurations and numerical report tables when ready.

T0.2 passes only after the declared row reproduces and its comparison record is reviewed. The T1 pilot establishes measurement code and capacity, not G1. A complete T1.1 report must state which REDS cells are impossible and which full-resolution comparisons actually fit; missing E*warp implementation cannot be hidden under the Farneback tOF label. T1.2 can share VAE inputs and latents. T1.3's Base-plus-teacher training-step profile is a separate later diagnostic and is not performed by this VAE-only plan. G0 remains open until the remaining Stage 0 work is complete, and G1 remains open until the human settles D1–D4.
