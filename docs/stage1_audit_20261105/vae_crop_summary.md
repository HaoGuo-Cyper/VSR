# VAE crop audit — 2026-11-05

The frozen 58-case CUDA FP32 VAE-only crop audit passed its execution checks. This report verified all 58 result hashes against the run summary, complete markers, case statuses and frozen protocol, using JSON only. It did not load pixels, checkpoints or models and did not recompute metrics. The original run remains unchanged.

## Scope and uncertainty

HQ-VSR: eight distinct validation sources × lengths 9/17/33 × strides 1/4 = 48 cases. REDS4: sequences 000 and 011, five valid conditions each = 10 cases. Both REDS 33/stride4 cells are unsupported: 129 source frames are required but each source has 100; no padding or wrapping was used. HQ and REDS results are kept separate.

Each entry below is an equal-clip mean ± **sample SD across clips** (HQ n=8, REDS n=2), not variation over training seeds. This is a deterministic, frozen VAE with no training or sampler seeds. The selected sources are a small, fixed diagnostic sample; no p-values or generalization confidence intervals are claimed. PSNR is a mean of per-frame dB values, not PSNR of pooled MSE.

## HQ-VSR reconstruction (n=8 per cell)

| Frames | Stride | Latent positions | RGB PSNR dB ↑ | RGB SSIM ↑ | LPIPS Alex0.1 ↓ | Normalized LR–HR latent RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| 9 | 1 | 3 | 29.993 ± 2.802 | 0.9099 ± 0.0570 | 0.01535 ± 0.00565 | 0.4975 ± 0.0506 |
| 9 | 4 | 3 | 28.585 ± 2.511 | 0.8898 ± 0.0548 | 0.02235 ± 0.00639 | 0.4760 ± 0.0553 |
| 17 | 1 | 5 | 30.273 ± 2.880 | 0.9135 ± 0.0545 | 0.01420 ± 0.00496 | 0.5427 ± 0.0569 |
| 17 | 4 | 5 | 28.488 ± 2.339 | 0.8894 ± 0.0547 | 0.02222 ± 0.00586 | 0.5149 ± 0.0646 |
| 33 | 1 | 9 | 30.160 ± 2.920 | 0.9123 ± 0.0540 | 0.01481 ± 0.00518 | 0.5691 ± 0.0632 |
| 33 | 4 | 9 | 28.314 ± 2.061 | 0.8857 ± 0.0625 | 0.02354 ± 0.00791 | 0.5374 ± 0.0708 |

## REDS4 crop reconstruction (n=2 per cell)

These are 128×224 aspect-cover/center crops of GT frames, with new bicubic ×4 synthetic LR for condition-gap measurement. They are not full-resolution restoration scores or measurements using the supplied benchmark LQ.

| Frames | Stride | RGB PSNR dB ↑ | RGB SSIM ↑ | LPIPS Alex0.1 ↓ | Normalized LR–HR latent RMSE |
| --- | --- | --- | --- | --- | --- |
| 9 | 1 | 25.790 ± 1.415 | 0.8439 ± 0.0612 | 0.03673 ± 0.02243 | 0.5044 ± 0.0139 |
| 9 | 4 | 23.136 ± 0.462 | 0.7279 ± 0.0586 | 0.07911 ± 0.03231 | 0.4454 ± 0.0007 |
| 17 | 1 | 25.608 ± 1.649 | 0.8372 ± 0.0704 | 0.04008 ± 0.02614 | 0.5407 ± 0.0218 |
| 17 | 4 | 23.046 ± 0.544 | 0.7187 ± 0.0513 | 0.08089 ± 0.03388 | 0.4727 ± 0.0182 |
| 33 | 1 | 25.295 ± 1.086 | 0.8251 ± 0.0510 | 0.04260 ± 0.01941 | 0.5565 ± 0.0119 |

## Matched-source stride comparison

Each delta is stride 1 minus stride 4 for the same clip, shared starting source frame and fixed spatial crop. **The two conditions cover different source frames and elapsed time.** These deltas do not isolate temporal spacing from content, and do not establish an SR model improvement.

| Dataset | Frames | n | ΔPSNR dB | ΔLPIPS | PSNR positive / LPIPS negative |
| --- | --- | --- | --- | --- | --- |
| HQ-VSR | 9 | 8 | 1.409 ± 0.858 | -0.00700 ± 0.00323 | 8/8 / 8/8 |
| HQ-VSR | 17 | 8 | 1.785 ± 0.847 | -0.00801 ± 0.00346 | 8/8 / 8/8 |
| HQ-VSR | 33 | 8 | 1.846 ± 1.256 | -0.00873 ± 0.00539 | 8/8 / 8/8 |
| REDS4 | 9 | 2 | 2.655 ± 0.953 | -0.04237 ± 0.00988 | 2/2 / 2/2 |
| REDS4 | 17 | 2 | 2.562 ± 1.105 | -0.04082 ± 0.00773 | 2/2 / 2/2 |

All eight HQ sources have higher recorded PSNR and lower LPIPS at stride 1 for each clip length; both REDS sequences show the same direction at lengths 9 and 17. In HQ, 17 and 33 contiguous frames have similar aggregate reconstruction quality, while producing five and nine latent time positions. Because longer clips contain additional frames, their all-frame averages are not a pure length/context effect. This supports carrying contiguous protocols into method testing; shared-prefix checks below separate coverage changes from any changes to scores of already-covered frames.

## Shared-prefix checks and D1 candidates

Shared-prefix spatial metric comparison (same source indices and spatial crop; no pixel access):

| Dataset | Stride | Short / long length | Matched clips | RGB PSNR/SSIM/MSE exact | Max absolute LPIPS difference |
| --- | --- | --- | --- | --- | --- |
| HQ-VSR | 1 | 9 / 17 | 8 | all | 5.59e-09 |
| HQ-VSR | 1 | 9 / 33 | 8 | all | 5.59e-09 |
| HQ-VSR | 1 | 17 / 33 | 8 | all | 5.59e-09 |
| HQ-VSR | 4 | 9 / 17 | 8 | all | 3.73e-09 |
| HQ-VSR | 4 | 9 / 33 | 8 | all | 3.73e-09 |
| HQ-VSR | 4 | 17 / 33 | 8 | all | 1.12e-08 |
| REDS4 | 1 | 9 / 17 | 2 | all | 1.86e-09 |
| REDS4 | 1 | 9 / 33 | 2 | all | 1.86e-09 |
| REDS4 | 1 | 17 / 33 | 2 | all | 5.59e-09 |
| REDS4 | 4 | 9 / 17 | 2 | all | 7.45e-09 |

Every shared-prefix RGB PSNR/SSIM/MSE value is exactly equal. LPIPS differs by at most 1.12e−8 across these checks. Those tiny differences are consistent with the LPIPS evaluator using a different final CPU batch size at different clip lengths; their cause was not independently retested. The longer conditions therefore must not be ranked by their different all-frame mean alone. In particular, T17 and T33 at stride 1 share the first17 input indices; the table quantifies only their recorded metrics on those matching frames. It does not assert reconstructed pixel equality.

For D1, 33 contiguous frames are a useful leading candidate when the separate resource profile permits: they preserve the original 33-source-frame temporal span while yielding nine latent positions, whereas 17 gives five. A two-latent-frame corruption span occupies 2/9=22.2% versus 2/5=40.0%; a nominal 25% mask ratio needs an explicit integer/span quantization policy. Optimizer/EMA memory and end-to-end step time remain separate measurements, and the human must settle D1–D4. The present VAE-only result does not authorize retraining or silently change any baseline.

### First sampled frame and later errors

For each source, the stored PSNR/SSIM/LPIPS of sampled index 0 are exactly equal across all supported length/stride conditions. This checks the shared first-frame/crop handling at the level of recorded scores, not equality of unseen reconstructed pixels. Per-index means and sample SDs are in `per_frame.csv`; source offset is index × stride from the common source start.

HQ sampled index 0 averages 29.142 dB in every cell. Later-frame means are 30.100/30.343/30.191 dB at stride 1 for lengths 9/17/33, versus 28.515/28.447/28.288 dB at stride 4. REDS index 0 averages 25.945 dB; later means at lengths 9/17 are 25.771/25.587 dB for stride 1 and 22.785/22.865 dB for stride 4. Thus the first-frame pattern differs by condition; temporal cutfr=2 scores must not replace the all-frame spatial/per-index audit.

## T1.2 condition gap and channel distributions

`channel_gap.csv` contains equal-clip means and sample SDs for every channel, clip length, stride and latent time position, in raw and officially normalized units. Each channel records HR/up-LR mean, within-clip population std, signed mean gap, RMSE and MAE. Time-position rows include nominal anchor offsets; these are not claims that a causal latent has a four-frame-only receptive field.

The summary latent gap is computed within each clip as sqrt(mean of the 16 channel RMSE² values), then averaged over clips. Channel distribution stds are separately averaged as **means of within-clip population stds**; they are not a pooled global std. Wan constants define an affine normalization, not a requirement that these selected data have zero mean or unit std; no moment-matching correction was applied. Stride 4 has a smaller normalized LR–HR latent RMSE in these cells despite worse VAE reconstruction scores, so raw distance in this representation is not by itself a reconstruction-quality proxy.

## Execution and repeated-case check

Recorded full-run wall time: 81.419 s for all 58 cases. Maximum recorded CUDA allocated memory: 1,112,336,896 bytes (1.036 GiB); this covers the VAE and current crop tensors, with no DiT, teacher, T5 or optimizer. It establishes this VAE-only crop workload, not the capacity of Base training or Base+EMA teacher.

Pilot case `HQVSR_00_T9_s1` matches the full run exactly in its case definition, input hashes/geometry, all stored metric values and latent statistics. Timing/resource values and timestamps were excluded. This is one deterministic repeated-case check, not a multi-seed experiment or a guarantee across hardware.

## Remaining scope

The crop audit covers RGB PSNR/SSIM, LPIPS and the existing TecoGAN tLP/tLP100/tOF (Farneback) definitions. Their full condition values are in the JSON/CSV. TecoGAN uses cutfr=2, different temporal pair counts and uint8/crop preprocessing; changes across stride/length are not duration-normalized. These numbers are not E*warp or an independently specified tLPIPS. Both requested definitions/implementations remain pending. The separate full-resolution attempt stopped during common untiled decoding with OOM, before tiled encoding or any difference measurement; see `full_resolution_attempt.md`. The full-resolution encode-tiling comparison remains incomplete. T1.3 training/teacher resource profiling is a separate task. T1.1 remains partial; D1–D4 remain for the human to settle.

## Provenance

- Full run: `/home/hugo/Projects/VSR_/runs/vae_crop_full_20261105T114559Z`
- Full summary SHA256: `c8923195f4989d3f9346cef63cce5c9a5d2d55cf7e08d3677b125cd50782f7a8`
- Full protocol SHA256: `3055426bf39b14ec01d053cac7d6c861ab97e9963d66bee5d704bd8f3ace4cfe`
- Pilot summary SHA256: `1897e631a6c9fbeeb3463f193cc881573c671e42f8f797f0adcbdda366c4e4f7`
- `vae_crop_summary.json` preserves all case seal hashes, aggregation definitions, paired deltas, per-condition temporal scores, first-versus-later statistics and the pilot/full equality checks.
- This summarization read only JSON. Original pixel/weight/environment bytes were not independently reverified by this pass.
