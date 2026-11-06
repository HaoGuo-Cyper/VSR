# Results

Append-only. Do not edit or delete earlier entries; add corrections as new entries.

## Inference protocol (fill in during T0.2; frozen for each comparison)

| Setting | Value |
| --- | --- |
| Sampler | |
| Sampling steps | |
| Sampler seed | |
| Spatial tiling (tile size, overlap) | |
| Temporal window (frames, stride, overlap) | |
| Metric implementations (PSNR/SSIM on Y, LPIPS version, RAFT model for E*warp) | |

## Run log

| Run id | Date | Stage / task | Git commit | Config | Seed | Start checkpoint | Updates | Wall-clock | Peak VRAM | Final L_flow | Final L_sd | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Stage summaries

<!-- For each stage: what was run, the results table (mean ± std over seeds, bootstrap p-values), the gate verdict, and questions for the human. -->


## 2026-11-05 — Stage 0 code map and frozen-protocol inspection (T0.1 / T0.2)

This entry records read-only code, configuration, protocol and completion-marker inspection. No training, GPU inference, VAE reconstruction audit or new quality measurement was run. Existing checkpoints, configurations, results and partial outputs remain unchanged. Detailed evidence is in `docs/stage0-code-audit-20261105.md`; the source code was copied from `/home/hugo/Projects/VSR` at producer commit `04228644605f5a3d96834ba05dde636636400817` together with the separately inventoried working-copy documentation.

### Frozen external inference protocol recovered from existing files

| Setting | Existing protocol |
| --- | --- |
| Cohort | Eight original seed-20261102 checkpoints: NoRef/HRI × Base/Align/Relation/Both, each at step 8192 |
| Datasets | REDS4 IDs 000/011/015/020 and YouHQ40 IDs 000–039 |
| Sampler | Local official-config `UniPCMultistepScheduler`; CFG 1.0 |
| Sampling steps / seeds | 20 steps; both sampling seeds 42 and 43 |
| Input interval | First 17 contiguous ordinal PNG frames, indices 0–16, stride 1; these are prefix results |
| Temporal window / overlap | One 17-frame clip; no temporal windowing or temporal overlap |
| Spatial operator | Native ×4; one whole-input bicubic interpolation before spatial tiling |
| Spatial tiles / overlap | HR 256×448; overlap 64; positive raised-cosine blend with an axis-wise 1e-3 floor |
| Noise | One CPU float32 Gaussian field per video/seed; corresponding spatial crop reused by each tile |
| Text | `A high quality video.` from the existing accepted text-cache path |
| Execution | CUDA restoration, CPU metrics; original PNG source pixels; no metric resize |
| Metric pixels | Saved uint8 PNG after `round(clamp(RGB01)*255)`; metric input is exactly q/255 |
| Spatial metrics | `psnr_rgb_db`, `ssim_rgb`, `mse_rgb_01`, `lpips_alex_v0_1`; zero border shaving |
| Temporal metrics | `tLP_alex_v0_1_TecoGAN_crop`, `tLP100_alex_v0_1_TecoGAN_crop`, `tOF_Farneback_TecoGAN_crop`; cutfr=2 and original adjacent pairs |
| Reference / pools | NoRef no-reference mode; HRI planned on/off with supplied GT frame 0; paired primary pool `matched_nonreference` for every model/mode |
| FPS | REDS4 30 is declared playback FPS only; YouHQ40 PNG timing inherits supplied MKV metadata; original capture timing is not independently established |

This records the historical protocol; it is not a reproduction certificate. T0.2 remains in progress because no checkpoint/sequence was rerun. The original blank template above mentions Y-channel PSNR/SSIM and RAFT E*warp; those names do not describe these historical frozen metrics. Existing tOF must not be relabeled as RAFT E*warp, and TecoGAN tLP must retain its implementation name.

### Checkpoint inventory and actual external progress

There are 16 preserved final checkpoints: the eight original seed-20261102 models, plus Base/Align repeats for both families at training seeds 20261103 and 20261104. The frozen external cohort contains only the original eight. Sampling seeds 42/43 are not independent training replicates. T0.3's requested all-16 coverage is therefore broader than the existing frozen eight-model cohort and needs an explicit compatible evaluation plan.

| Original checkpoint | REDS4 completion markers | YouHQ40 completion markers | Total markers |
| --- | --- | --- | --- |
| NoRef Base, training seed 20261102 | 4/4 | 40/40 | 44/44 |
| NoRef Align, training seed 20261102 | 4/4 | 40/40 | 44/44 |
| NoRef Relation, training seed 20261102 | 4/4 | 24/40 | 28/44 |
| NoRef Both, training seed 20261102 | 0/4 | 0/40 | 0/44 |
| HRI Base / Align / Relation / Both, each | 0/4 | 0/40 | 0/44 each |

The 116 `complete.json` markers were counted in the existing run; they are not a new strict artifact/semantic acceptance or a quality comparison. The pause record states that Base/Align execution artifacts were reviewed earlier, while Relation markers still need strict verification. NoRef Relation last completed YouHQ40/023; YouHQ40/024 has preserved seed-42 PNGs but no complete marker and is excluded from the 116 count. The original phase planned 352 model/sequence rows and 1056 restores, owing to HRI on/off modes. No quality values were opened for new benefit claims during this audit.

Existing run: `/home/hugo/Projects/VSR/runs/external_paired_frozen_20261103T125240.867596Z`; rows under `external/results/`; log `/home/hugo/Projects/VSR/runs/console_external_paired_frozen.log`. Preservation receipt: `/home/hugo/Documents/Codex/2026-11-02/files-pasted-by-the-user-wan2/outputs/VSR_paused_by_user_20261104T050113Z/PAUSE_AND_RESUME.txt`.

### Gate status and next decision

T0.1 is complete once the code map and D1–D3 evidence are recorded. T0.2 is partially documented and not reproduced; T0.3 is blocked pending an explicit fresh-output continuation/re-evaluation plan. G0 has not passed: external coverage, statistics and loss diagnostics remain unfinished. G1 has not passed: D1–D4 remain human decisions and Stage 1 measurements have not been performed.

The audit found conflicts with the handoff assumptions: training uses 9 stride-4 frames while the frozen external protocol uses 17 contiguous frames; historical external testing covers prefixes, not full sequences; the frozen cohort has eight models rather than all 16 checkpoints; and HRI external Base/Align have not run. The original worker is fresh-only/no-resume and binds original paths, source bytes and protocol seals. Under AGENTS.md rule 8, report these discrepancies and wait for the human's next GPU evaluation/profiling plan. Do not resume the old partial run, overwrite its outputs, or silently change its protocol. No new training or GPU evaluation was launched.

### 2026-11-05 — Source import validation

The current source snapshot and original handoff were committed before the audit edits (`c95495b`). The existing CPU suite ran with `CUDA_VISIBLE_DEVICES=''`: 230 tests in 9.729 seconds, no failures/errors, one skipped because Node is unavailable (inline viewer JavaScript parse-only check). See `docs/cpu-tests-import-20261105.txt` and `docs/import-validation-20261105.json`. All 212 inventoried original files still match their pre-import SHA256 values and the old repository's Git status is unchanged. Copied code, configs, manifests and tests are byte-identical; README and ignore rules have new checkout instructions. This test run does not fulfill the T0.2 GPU reproduction criterion.

The seven actual frozen external workers/helpers and four directly referenced JSON records were archived byte-for-byte under `scripts/frozen_external/`, with SHA256 provenance. Historical execution and model/data bindings remain in force; no archived worker was run. Local binaries, checkpoints and datasets are excluded from the public code import. GitHub synchronization remains pending write authentication: the connected integration returned HTTP 403, and the local Git client had no authenticated username. No remote branch was changed by these attempts.

## 2026-11-05 — GitHub publication completed

The human explicitly authorized publishing the 239 text files, including local paths, experiment metadata and dataset manifests, to the public repository `HaoGuo-Cyper/VSR`. Weights, checkpoints, original videos, cached tensors, credentials and local symlinks remain excluded.

The GitHub connection published the two prepared snapshots as `2a697995d24483c66c38ab67b8d72793eaf2abf6` and `fe8b581e57a662028a9ac56a7246b7ce04a1ce36`, preserving the initial remote commit `468db7a927a528f2bf8a80b9dee5e0336d955743`. The connection generated new commit identities; their file trees are byte-identical to local `c95495b` and `dbedaaf` respectively. Verified tree SHAs are `b72c6e473343d426420369d67b14628de4fbcad0` and `ddc2a8f264de985cb97c4c2c673270bb5697d422`. The local pre-publication history is preserved in an archive ref and the earlier Git bundle.

A recursive, non-truncated GitHub tree read and independent agent review confirmed 239 regular UTF-8 text blobs, no symlink/gitlink or excluded binary asset, and the original six handoff files in the first snapshot. This append-only entry records the completed publication; the earlier pending-authentication entry remains historical. No training, GPU evaluation or numerical protocol change was performed during synchronization. T0.2 reproduction and the open research gates remain unfinished.

## 2026-11-05 — Continued Stage 0: existing statistics and loss diagnostics

The human instructed "continue". The following work reads existing JSON/JSONL only; it does not train, rescore saved images or alter any original artifact. Full tables and provenance are in `docs/stage0_analysis_20261105/{statistics.json,statistics.md,statistics.csv,seed_uncertainty.json,seed_uncertainty.md,loss_diagnostics.json,loss_diagnostics.md}`. Reusable scripts have the corresponding names under `scripts/`.

### Internal validation: three matched training seeds

Each number is the mean over the fixed 32 held-out validation clips within a training seed, then mean ± sample SD over training seeds 20261102/03/04. Sampling seeds 42/43 are averaged within each clip, not counted as training replicates. The float32 validation protocol is distinct from frozen PNG/255 test211 evaluation. NoRef uses all frames; HRI ref0/on uses its predeclared nonreference pool, so these family rows must not be compared directly.

| Family / context | Variant | Training seeds | PSNR RGB dB | LPIPS Alex0.1 |
| --- | --- | ---: | --- | --- |
| NoRef / all frames | Base | 3 | 21.918281 ± 0.139593 | 0.134595 ± 0.003537 |
| NoRef / all frames | Align | 3 | 21.970403 ± 0.220101 | 0.135256 ± 0.003720 |
| HRI / ref0 on, nonreference | Base | 3 | 21.870963 ± 0.231003 | 0.139322 ± 0.009634 |
| HRI / ref0 on, nonreference | Align | 3 | 21.876967 ± 0.256260 | 0.137264 ± 0.007132 |

The main statistics table reports 10,000-draw paired source-cluster bootstrap intervals conditioned on the existing trained instances. A separate supplement simultaneously resamples the three matched training seeds and source groups (10,000 draws, seed 20261105) to show seed uncertainty. All five metric intervals in the two primary validation contexts cross zero under that crossed bootstrap. Examples: NoRef Align−Base PSNR = +0.052122 ± 0.093152 dB over seeds, crossed 95% interval [−0.038588, +0.135569]; HRI ref0/on Align−Base LPIPS = −0.002058 ± 0.003074, crossed 95% interval [−0.005079, +0.001232]. Three training seeds give limited information; these are exploratory descriptive intervals with no p-value, multiplicity correction or Stage 5 decision. They do not prove a zero effect.

Frozen internal HQ test211 results are tabulated separately for each of the five fixed degradation conditions and HRI on/off modes. They contain only training seed 20261102, so every model there, including Relation/Both, is labeled "single training seed, indicative". Validation Relation/Both are likewise single-seed. No best context/seed was selected after looking at quality. These existing results support retaining the narrower statement that a stable overall auxiliary-loss benefit has not been established. T0.4 remains partial because the external evaluation is incomplete.

### Training losses (T0.5 complete)

All 16 final-checkpoint runs have 8192 consecutive logged updates. The diagnostic window was fixed at the first 512 records (steps 1–512) and last 512 records (7681–8192). Flow loss decreased in 16/16 runs (about 49.5–55.9%); active alignment loss decreased in 8/8 (about 38.7–40.1%); active relation loss decreased in 4/4 (about 60.6–61.8%). Inactive zero placeholders are excluded. An earlier interrupted 3849-record Align attempt without a final checkpoint is explicitly excluded, never spliced into a final run. Loss decrease shows objective optimization; it does not establish restoration or temporal-quality improvement.

### T0.2 protocol fixed before the fresh reproduction

Use exactly NoRef Base training seed 20261102 / REDS4 / 000, selected as the first declared REDS4 sequence. Both sampler seeds 42 and 43 are mandatory. Retain 17 consecutive original PNG frames from index 0, native LR 180×320 / HR 720×1280, official-config UniPC 20 steps, CFG=1, HR tile 256×448, overlap 64, one global CPU float32 Gaussian field, original cached generic text and real CUDA FP32 VAE. Restoration is on CUDA, metrics on CPU, using the original PNG q/255 pixel contract and all original metric pools. Preserve the historical per-restore VAE policy remove/reinstall lifecycle. No T5 fallback, changed precision/steps or old-output resume is allowed.

The new single-row harness separates the old read-only assets from a new dated output directory and creates its own pre-execution subset protocol, binding harness/core/helper/assets/runtime hashes. It does not reuse an all-eight completion certificate for this subset. Before execution, freeze a pass rule requiring all 34 saved PNG files to have identical byte hashes, all pool scalar metrics (including bicubic baseline and seed mean) to match with atol=1e−10 and rtol=1e−8, two restores/32 original VAE delegates/zero T1 events, and seed42 failure stopping before seed43. Preserve any failed attempt without automatic retry or changed tolerances. A passed single row fulfills T0.2's bounded reproduction criterion, not T0.3's full external coverage. Actual execution status will be appended separately.

### T1.1 / T1.2 VAE crop protocol fixed before measurement

The VAE-only crop audit will use the pinned official FP32 frozen VAE, posterior mode, eval/no_grad, no DiT/T5/DINO/optimizer and zero parameter updates. Source eligibility and deterministic selection are fixed in `configs/audits_20261105/vae_crop_full_v1.json` (raw SHA256 `3055426bf39b14ec01d053cac7d6c861ab97e9963d66bee5d704bd8f3ace4cfe`). Eight distinct HQ validation sources with at least 129 header frames are selected by a fixed SHA256 ordering. Each source has one fixed maximum-span start shared across clip lengths 9/17/33 and source strides 1/4. Spatial processing is the existing 128×224 aspect-cover/center crop and ×4 bicubic degradation. REDS4/000 and /011 use start 0; their 100 source frames make the 33-frame/stride4 cells ineligible, recorded as unsupported rather than padded or wrapped. This gives 58 supported cases and two unsupported cells.

The separate first-case capacity pilot is fixed in `vae_crop_pilot_v1.json` (raw SHA256 `d852a450342c9c59f434a72460727e112f2bb938cadf6e83cae9309a63bb4a7d`). The full grid, sources and definitions are already frozen, so pilot values will not be used to change selection. Script SHA256 is `81f156b71f4ab98a62d3658758b10fc1aeb9ad8fbe06bca5caa9c48690efce36`. Both scopes use new dated output directories and stop on any failure/OOM without changing precision, tiling or inputs.

Measure per-frame RGB PSNR/SSIM, LPIPS Alex0.1 and the existing precisely named TecoGAN tLP/tLP100/tOF. Record raw/normalized per-channel and per-latent-time HR versus upsampled-LR statistics and gaps; normalization uses official Wan constants in FP32 before FP64 population summaries. No binary latents are saved. This is a deterministic VAE measurement rather than a multi-training-seed experiment.

T1.1 remains partial until its remaining metric definitions and full-resolution tiling comparison are addressed. The planned names E*warp and tLPIPS are not silently assigned to existing tOF/tLP. The inspected [DOVE implementation](https://github.com/zhengchen1999/DOVE/blob/1a499a84bb6733f71ba5a9cf20394f2373f635ba/metrics/raft/ewarp.py) and [adapter](https://github.com/zhengchen1999/DOVE/blob/1a499a84bb6733f71ba5a9cf20394f2373f635ba/metrics/ewarp.py) use a different RAFT checkpoint, half-resolution processing and a CUDA resampling extension; the adapter requires frames at least 256×256. They do not match the available torchvision RAFT weights or 128×224 crop protocol. No new metric package/weight is installed, and no literature-comparable E*warp score is claimed by the first crop batch. D1–D4 remain OPEN.

### T0.2 execution — PASS

Fresh run: `/home/hugo/Projects/VSR_/runs/stage0_reproduction_20261105T113721Z`; raw subset-protocol SHA256 `5940a7d4f5264a9df179be5b4bcd8bfd8154d02810be32d8b21edb1ce9e3c623`. Harness source is committed in local preparation commit `8f2a2a3` and bound by SHA256 `d4f026054cf384447e154600776a85e16610ae52c5877f92bfa930c22ac922ed`. The exact original checkpoint SHA256 is `8f3d1e19edc24b8ad72219ef207b85c4c798f241f6f204fac57e57c11960806c`.

Both seeds passed: **34/34 saved PNG files are byte-identical** to the historical row. All pool scalar metrics, the bicubic baseline and sampling-seed mean passed the predeclared atol=1e−10 / rtol=1e−8 comparisons. The retained VAE policy recorded two restores, two rotations, 32 original encode delegates and zero T1 events, with no model-context or VAE-reference leak. Restore times were 118.864 s (seed42) and 119.103 s (seed43); total execution wall time was 295.721 s including pre-execution binding checks, model load, baseline/metrics and source verification. These timings describe this reproduction, not a general throughput claim.

T0.2's one-sequence reproduction criterion is satisfied. This does not validate all historical Relation artifacts, expand the external cohort, remove its training/test spacing mismatch or complete T0.3. No old output was resumed or changed. Training updates: 0. The protocol and compact summary are archived in `docs/stage0_analysis_20261105/`; image payloads stay local in the new ignored run directory.

### T1.3 resource-profile scope fixed before execution

The new `scripts/profile_protocol_resources.py` is a **zero-parameter-update synthetic forward/backward resource diagnostic**, not a training run or a complete optimizer-step benchmark. It uses one unchanged NoRef Base seed20261102 model, the existing cached text and fixed random latent inputs at 128×224 for 9/17/33 input-frame geometries (3/5/9 latent positions). Each geometry has Base-only and Base preceded by one same-parameter no-grad forward, with one excluded warmup plus three timed cycles. The detached extra-forward output remains resident until student backward finishes. No optimizer is constructed or stepped; no VAE, T5, DINO, EMA rule or new alignment objective is loaded or selected. All parameter identities/versions and trainable-byte SHA256 must remain unchanged, with gradients/features cleared.

Prepared run: `/home/hugo/Projects/VSR_/runs/profile_protocol_resources_20261105T114931Z`; raw protocol SHA256 `0ef963d980e3208916b79ac84d68cb918f850a458fdaff9b30aefc64b634c4dc`. Script SHA256 `6a715dbaf047d1f718c2469ed3d81e11985e344e8f0810ef0890fe22f1f36b7f`. Preserve the strict bf16/checkpointed Base path and fixed shapes on failure/OOM, stop without retry. Timings are seconds per forward/backward/zero-grad cycle; any 8192-cycle extrapolation excludes optimizer, data, logging and actual EMA/projector costs and must not be called the formal RE cost. The extra forward uses the same train-mode Base with zero dropout, not an implemented eval-mode EMA teacher. T1.3 remains partial until the requested full-update/teacher training configuration is authorized and measured.

### Full-resolution VAE encoding comparison fixed before execution

The independent protocol `configs/audits_20261105/vae_full_resolution_v1.json` (SHA256 `31dc4f55199794f06fea8658722bf741a9caa3cc3d0bca50b9e672fdde9c6420`) fixes REDS4/000, its first nine native 720×1280 frames, stride1, official frozen FP32 VAE and posterior mode. Compare untiled encoding against native `enable_tiling` encoding (256×448 tiles, 192×384 strides). Disable tiling before both decodes so the same untiled decoder isolates the encoding difference. Save latent/reconstruction differences, per-frame metrics, boundary-region errors, phase peak VRAM and time. Script SHA256 is `a21f32175bec1191c4b9ac786087c2cbede26411cf75a64d40d7713b2f7755cd`. Any OOM/failure is preserved and stops this attempt without precision, size, decoder or tiling fallback. This is one native-resolution setting, not evidence for all frame lengths/resolutions.

## 2026-11-05 — Stage 1 bounded execution and current gate disposition

### VAE reconstruction and condition gap

The first-case capacity pilot passed, followed by **58/58 supported crop cases** under the previously frozen full protocol. Full run: `/home/hugo/Projects/VSR_/runs/vae_crop_full_20261105T114559Z`, summary SHA256 `c8923195f4989d3f9346cef63cce5c9a5d2d55cf7e08d3677b125cd50782f7a8`, execution wall 81.419 s. Independent JSON-only review verified all result/complete/status seals. The pilot and full run's first case have identical input records, stored metrics and latent statistics. There were zero parameter updates, no DiT/T5/DINO model and no binary latent output.

Each HQ cell below is mean ± sample SD across the eight selected sources, not training-seed variation. The two REDS sources are reported separately in `docs/stage1_audit_20261105/vae_crop_summary.md` and CSVs.

| HQ frames / stride | Latent positions | RGB PSNR dB | LPIPS Alex0.1 | Normalized LR−HR latent RMSE |
| --- | ---: | --- | --- | --- |
| 9 / 1 | 3 | 29.993 ± 2.802 | 0.01535 ± 0.00565 | 0.4975 ± 0.0506 |
| 9 / 4 | 3 | 28.585 ± 2.511 | 0.02235 ± 0.00639 | 0.4760 ± 0.0553 |
| 17 / 1 | 5 | 30.273 ± 2.880 | 0.01420 ± 0.00496 | 0.5427 ± 0.0569 |
| 17 / 4 | 5 | 28.488 ± 2.339 | 0.02222 ± 0.00586 | 0.5149 ± 0.0646 |
| 33 / 1 | 9 | 30.160 ± 2.920 | 0.01481 ± 0.00518 | 0.5691 ± 0.0632 |
| 33 / 4 | 9 | 28.314 ± 2.061 | 0.02354 ± 0.00791 | 0.5374 ± 0.0708 |

Stride1 covers different source frames/time spans from stride4, despite sharing the start and crop. These differences are VAE diagnostic observations on a fixed small subset, not isolated causal stride effects or SR-model improvements. Shared-prefix RGB PSNR/SSIM/MSE values are exactly equal for matched short/long cases; maximum per-frame LPIPS difference is about 1.12e−8. Different clip-wide means must not be presented as degradation of already-covered frames. Pixel tensors were not retained for an independent bitwise-prefix comparison.

T1.2 is complete within this declared sample: raw and normalized per-channel/per-latent-time statistics and distances are in `channel_gap.csv`; per-frame errors are in `per_frame.csv`. Mean within-clip population SD is labeled accordingly, not treated as a pooled dataset SD. Wan's normalization constants are retained; this audit neither changes them nor treats the HR VAE as the method's teacher. T1.1 remains partial because the named E*warp/tLPIPS implementations and the encoding-tiling comparison are incomplete.

### Zero-update resource diagnostic

Run `/home/hugo/Projects/VSR_/runs/profile_protocol_resources_20261105T114931Z` passed all declared checks; summary SHA256 `0678eb7b152779b721c4da24d846bb6c31471967c891df10bd5feddcc669d36b`. The six conditions ran 24 forward/backward cycles (six excluded warmups and 18 timed cycles), with 12 additional no-grad forwards in the extra-forward conditions. Parameters retained identical identities/versions and trainable SHA256 before/after, and gradients/features were cleared. Optimizer steps and parameter updates: **0**.

| Nominal frames / latent positions | Base cycle seconds, mean ± SD | Extra-forward + Base cycle seconds, mean ± SD | Extra-forward ratio | Max allocated GiB including extra forward |
| --- | --- | --- | ---: | ---: |
| 9 / 3 | 0.143702 ± 0.001235 | 0.224949 ± 0.081070 | 1.5654× | 2.8469 |
| 17 / 5 | 0.162595 ± 0.000042 | 0.214272 ± 0.000686 | 1.3178× | 2.9085 |
| 33 / 9 | 0.288841 ± 0.000633 | 0.379086 ± 0.000193 | 1.3124× | 3.0600 |

These are synthetic synchronized forward/backward/zero-grad cycles in the same train-mode Base, with one no-grad output retained through backward. They omit optimizer, gradient clipping, data/metric overhead, actual EMA state swapping, feature projector and alignment loss. The 9-frame second timed cycle (0.318560 s) is retained and accounts for its higher SD; no outlier was removed. No formal RE time, full-training-memory guarantee, actual EMA overhead or G3 verdict follows. T1.3 remains partial. Independent details are in `resource_profile.{json,md,csv}`.

### Native-resolution attempt — STOPPED on capacity failure

Run `/home/hugo/Projects/VSR_/runs/vae_full_resolution_20261105T115318Z` preserved its failure; SHA256 `6f8f09a09eee4fff204a7d87e3d05a57d559af2c9dd5024ad13eb476f44c3dc1`. The FP32 native720p untiled encode completed (3.526 s, peak allocated 12.682 GiB). The first common **untiled decoder** then failed in the native later-temporal-chunk path at `F.pad`, requesting a further 1.99 GiB with only about 660 MiB free. Failed-phase peak allocated was 13.592 GiB. No tiled encode, finished reconstruction, metric or difference table was reached; `completed_encode_decode_paths=0` is correct.

Read-only traceback/source review found the expected native encoder and decoder cache cleanup, no accidental second model/autograd graph, and no prior retained reconstruction. This is a capacity boundary of this specific untiled FP32 execution path, not a rejection of A2 or proof that every decoder policy fails. No precision/size/tiling fallback or automatic retry was attempted. Further GPU work is stopped under the declared failure rule. Details are in `full_resolution_attempt.{json,md}`.

### Proposal and gates

`DECISIONS.md` now marks D1–D4 **PROPOSED**, never SETTLED: prefer 33 contiguous training frames/9 latent positions, matched future stride1, initial zero-based student9/teacher21, NoRef and trainable-only EMA0.999, with protocol-matched Base seeds before method continuations. Explicit LPIPS/TecoGAN tLP100 metric candidates are proposed for human acceptance. Integer mask ratios/support still require prospective implementation tests; the crop/resource results do not select a best A2 method.

Current completion: T0.1/T0.2/T0.5 and sample-scoped T1.2/T1.4 are done. T0.4 is partial (internal statistics done, external coverage missing); T0.3 is unfinished; T1.1 and T1.3 are partial as labeled. **G0 and G1 have not passed. No new training run or parameter update has occurred.** Stop and await the human's D1–D4 decisions rather than beginning Stage 2 or retrying the failed protocol.

### External completion coverage and cost — plan only, no queue started

The new read-only `external_completion_plan.{json,md,csv}` inventories all 16 checkpoints × 44 original external sequences = 704 model/sequence rows. NoRef rows have two restores; HRI rows have four (two seeds × reference on/off), for 2112 restores total. The existing 116 completion markers correspond to 232 restores and are not newly accepted by this metadata-only plan. If those artifacts later pass strict verification, at least 588 rows / 1880 restores / 52,488 tile-restores remain, plus 704 HRI reference encodes. Otherwise more work is needed.

The two successfully reproduced NoRef REDS4/000 restores calibrate about 7.43646 seconds per tile at the fixed20-step protocol. Applying that **NoRef rate** to all remaining tile-restores gives a recovery-compute-only optimistic extrapolation of about **108.424 hours**. It excludes or cannot model HRI attention/reference-encoding differences, CPU metrics, PNG/I/O, cold starts, prior-artifact acceptance and possible fresh reruns. It is not a forecast interval or completion-date guarantee. It must not be shortened by silently changing steps, precision, frame spacing or resolution.

This plan does not start an evaluator, GPU queue or automation. A future Stage0 continuation must first review the existing artifacts, freeze a new all-16 cohort contract distinct from the old all-eight outer seal, and use fresh output directories while preserving the original prefix protocol. The pending human decision should consider this substantial cost and its priority relative to the proposed NoRef self-distillation line.


## Public text synchronization authorization — 2026-11-05

The human explicitly authorized: “允许同步本次和后续新增和更新的文本文件” (allow synchronization of this and future new/updated text files). This extends the previous 239-file approval to project source, configs, experiment statistics, audit logs and repository documentation, including project-local paths, experiment metadata and data manifests. Weights, checkpoints, raw media, tensor caches and credentials remain excluded. The standing scope is recorded in AGENTS.md rule 11; publication within this scope needs no repeated confirmation.

The four previously pending source/audit commits (35 added text files and 5 updated documents) have been published to main at `2e243f0c1355a371cb9f75890659cdca194d0129` with tree `4db09f639c62fc8d8d4e6d1ed639543352d0af44`. Each uploaded Git tree exactly matches its corresponding native local tree. GitHub creates its own commit metadata, so native and GitHub commit IDs differ while their file trees match:

| Native local commit | GitHub commit | Matching tree |
| --- | --- | --- |
| `8f2a2a36dd641d3cb88fcdfef564f646540c141c` | `6f0f86516cb7137e08b6a06c853dc2b3e6d778e3` | `956f84854710d3fcdbdaf94251ba257a4e3bdb72` |
| `1fb9fdda5ad66c84a758af9cdb38ca29e90ac729` | `834978a711fbfdea3dddbf0c6d7c7443a86078dc` | `7d7fa83eea56f930ad0cb0d77836393044612c88` |
| `f1aef3d935abaf1ebb6f4e2fdaaca74e918d25be` | `f4ff8a00640f52feda267f6e8f13ff91b5907550` | `3b3f2d9fde04e840dea7549a4282f798e323f095` |
| `4e838c6c66012f4aa50e4637a888ded19e4495dd` | `2e243f0c1355a371cb9f75890659cdca194d0129` | `4db09f639c62fc8d8d4e6d1ed639543352d0af44` |

Publication used a normal fast-forward from `880b7738c6b4f73019a97df5662357b562344086`; no branch history was overwritten. The final tree at this point contains 274 regular UTF-8 text files. A separate read-only review of all four snapshots found no binary, symlink, checkpoint/media file or obvious credential pattern. This authorization and the corresponding working rule are recorded in a following documentation commit. Native local history is retained in an archive ref before local/remote commit identities are aligned.

No training or GPU experiment was started by this synchronization. D1–D4 remain PROPOSED; G0/G1 and all research failure dispositions retain their previous status. Original assets/results and all 16 checkpoints remain outside this publication.


## Stage 2 authorized protocol — 2026-11-05 (before execution)

The human accepted the33-frame proposal, settled D1–D4 and authorized three Base seeds plus normal planned continuation after successful checks. G1 therefore passes; G0 and the explicitly partial audit items remain open. No native-resolution untiled retry is scheduled.

Frozen T2.1 design: NoRef Base only, fresh official Wan revision0fad780a534b6463e45facd96134c9f345acfa5b, no task checkpoint/optimizer reuse; training seeds20261102/03/04, identical data/window seed20261102; source-grouped train1641/val32 from the unchanged manifest; 33 consecutive frames, stride1, crop128×224, unchanged ×4 bicubic, posterior-mode full-clip FP32 VAE encoding with9 latent positions. Rank8 LoRA and LR residual injections0/5/9; batch1, AdamW lr1e-4/weight_decay.01, norm clip1, bf16/native FP32 exceptions, gradient checkpointing and the existing strict reproducibility policy. Each formal seed gets8192 optimizer updates (3RE total), checkpoints2048/4096/8192, validation at0/2048/4096/8192; fixed20-step UniPC CFG1, monitoring seed42 and final seeds42/43 with correct/disabled/shuffled LR controls. Scores during training retain the existing float32 RGB tensor protocol, separate from the later prospective PNG-equivalent main metrics. Internal test and external test pixels are not read for preparation/model selection.

Before formal initialization, a separate8-update train-only resource diagnostic will use two deterministic training clips, the same full AdamW/gradient path and a resident frozen FP32 evaluation VAE; its adapters/checkpoint are never reused. Finite loss/gradients, no frozen gradients, changed trainable state, correct9-position geometry and observed peak allocation below16GiB must pass. Failures stop the queue. The complete actual EMA/A2 update overhead remains future Stage3 work. Diagnostic updates are reported separately from3RE.

Preparation uses a SHA-bound existing generic text embedding without loading T5; paired RGB is decoded/encoded one clip at a time and retained on disk, with a two-clip lazy reader. A fresh33-frame cache is mandatory; the old9-frame latent cache cannot be reused. Base never constructs DINO or RAFT. Later seeds reuse only the verified/sealed new data/text/VAE cache and independently initialize all task parameters and optimizers. Original configs, checkpoints and historical outputs remain unchanged. Run IDs, git/config hashes, timings, peaks and results will be appended when observed.


## Human-approved independent review scope — 2026-11-05

The human supplied a read-only review of GitHub2dc229a and authorized following its approach with continuous progression after successful checks. Retain the explicit earlier choice of **three** NoRef Base seeds. Adopt latent-level A2 at the LR-branch input, excluding latent position0; defer pixel-level training corruption to a possible later ablation after a crop VAE leakage test. Pixel-degraded restoration tests remain distinct.

Descope the old108+hour external backlog from the required new line (including the optional7-hour slice, which is not launched). Close G0 in this revised internal-evidence scope using completed internal statistics/loss diagnostics and reproduced historical row; incomplete external coverage remains labeled. Reserve external evaluation for final models, HRI for optionalStage8. Close the native full-resolution untiled and unavailable E*warp/separate tLPIPS requirements out of T1.1, retain the58-case crop audit and its limited sample, and use explicitly named existing TecoGAN tLP100/tOF. These are human-approved scope changes, not fabricated successful measurements.

The review's training-cost and memory statements remain estimates: the synthetic3.06GiB/1.31× result excludes optimizer/actual EMA/projector, and does **not** pass G3. The separately recorded8-update full-optimizer Base diagnostic remains the first bounded GPU check. A33-frame external tile restore and crop leakage test are not required before Base training and are not launched here. All runtime failures still stop progression.


### Stage2 prelaunch CPU checks — PASS

The streaming-preparation, queue/config/lease, unchanged data/model/evaluation/cache/checkpoint/temporal-stride/reproducibility checks passed40/40 in1.007s, with no GPU experiment. A real subprocess fixture verifies that the worker GPU lease survives release of the parent queue lock. Effective diagnostic/formal/reuse configs pass the existing checkpoint loader SHA contract, including tamper rejection. The independent window audit verifies1673/1673 unchanged source windows. See docs/stage2_20261105/. These CPU results do not pass the upcoming full-optimizer resource gate or G3. Runtime numerical source is frozen during the three-seed queue; development and publication of new logs remain in the separate staging checkout.


## Stage2 startup — STOPPED before optimizer/resource measurement

Service vsr-protocol33-base-20261105-a1 started the resource diagnostic from frozen commit c70fd6056607a97e1fc477b2a7ae707ce8ca3c9b. Run runs/stage2_protocol33_optimizer_probe_20261105T132535.476959Z passed official artifact verification, then stopped at the exact environment persistence guard. audit.environment returns free_total_bytes as a tuple; JSON stores the same integers as a list. Comparing the unnormalized in-memory dictionary to the loaded JSON incorrectly rejected this representation change. The guard fires before text/cache preparation, source decoding, VAE/task-model construction or optimizer updates. There are no clip files, training logs or checkpoints in this failed run. **Formal Base runs:0; diagnostic optimizer updates:0.** This is a startup serialization defect, not an observed memory/learning failure or a passed resource gate.

The new worker now canonicalizes the **entire** environment dictionary through JSON before persistence/use; no field or check is removed. A CPU regression first reproduces the original tuple rejection, then completes the strict mock-VAE preparation with preserved code hashes and exact byte counts after canonicalization. The focused suite passes41/41 in1.198s. Failure/config/artifact/console records and a redacted-to-relevant-fields environment contract are retained in docs/stage2_20261105/startup_20261105T132535Z/. Original failed output is never overwritten. The service exited with MainPID0; no training process remains. No automatic GPU retry was performed; after reporting this startup problem, await confirmation for a fresh run directory/service.

Independent CPU preparation of the already-authorized T2.2/T2.3 protocol continues in staging: fixed val32, clean+blur/noise/single-frame all-intra/drop-repeat/drop-black,8-frame masks,20step seeds42/43, original-frame/pair metric pools and a prospectively defined three-seed G2 rule. The metadata-only seal predates any new validation predictions; no actual G2 score or verdict exists. Internal test211/external pixels remain outside pilot tuning.


### T2.2 CPU validation assets — COMPLETE, no predictions

The frozen val32 cohort was materialized in new staging/runs/stage2_val_assets_20261105T132924Z:224 float32 NPY files,499609600bytes;32HR+32cleanLR+160degradedLR. All192 reader views passed file/pixelSHA, source/window/stride/FPS, stable HQ input-contract and target/unmasked pixel identity checks. Each of the five predeclared types changed exactly8 selected frames in all32 clips;256 independent all-intra codec frames completed. Scripts/config/seal remain byte-identical. Dataset SHA256355060708dd3ac3203801764a202ac124999cfa07bb317a45db2a493fa669004; cohort seal9c5e938d8b2c9b5e7d965fca2f6b3925557848235d6b9d404b6231bffcbc36e8. Arrays stay ignored/local. No checkpoint, restoration prediction, GPU or G2 quality verdict was produced. See docs/stage2_eval_20261105/materialization_summary.json and protocol.md.

### Stage3 CPU primitives — PARTIAL

Trainable-only EMA, exception-safe teacher swap/owned detached outputs, position0-excluding latent-span masks, nonmutating replacement, T/H/W token mapping and masked FP32 student-projector/teacher-LayerNorm cosine are implemented separately in vsr/self_distillation.py. Seventeen CPU tests passed with no CUDA or optimizer update. The teacher must run before the student graph because swap copy operations increment parameter versions. Default rho0.25/span2 realizes2/9; rho0.1 rounds to0 under the explicit whole-span rule and must not be silently executed as a distinct nonzero sweep. No block9/21 hook integration, actual continuation trainer, real rho0 trajectory comparison or actual EMA overhead benchmark exists yet. G3 remains pending. This independent CPU work does not restart the failed Base queue.


## Confirmed restart and full-optimizer check — 2026-11-06

The human explicitly confirmed the repaired restart. Fresh service `vsr-protocol33-base-20261106-a2.service` started a new queue under frozen runtime commit `41407b0e62748d879d2d9e1d8c5b58fa5202d09f`; the previous failed startup remains untouched.

The new independent diagnostic `stage2_protocol33_optimizer_probe_20261106T025808.644791Z` **PASSED** eight actual full-AdamW updates on two fixed training clips,33frames/stride1,latent[9,16,28], with the evaluation FP32 VAE resident. Peak allocation was3845157888bytes (3.581GiB). Excluding the declared first update, seven observed updates averaged0.342009±0.057118seconds (sampleSD); the slower eighth update0.470781s is retained. The update-only extrapolation is0.7783hours per8192updates, excluding data preparation, initialization, validation, checkpoint I/O and actualEMA/projector work. This is not a completion-time guarantee or a G3/A2 result.

All losses/gradients are finite, frozen gradients remain absent, trainable state changes, and independent inspection verifies fresh official initialization, exact source/config bindings, gradient groups and AdamW-state size. Diagnostic8updates are recorded separately from the planned three formal RE; their trained parameters are never reused. Raw text records and independent review are under `docs/stage2_20261106/`.

The queue automatically entered fresh Base seed20261102 run `stage2_protocol33_base_seed20261102_20261106T025845.129456Z`, preparing the new33-frame train1641+val32 cache. It will train8192updates, complete/seal that cache, then independently initialize seeds20261103/04 with fresh optimizers and only verified data/text/VAE cache reuse. At this report's first snapshot, formal optimizer updates have not begun. The live text snapshot explicitly distinguishes preparation from training.

Normal planned continuation is authorized. A15-minute heartbeat on this thread checks progress and continues after gates pass, without duplicate GPU jobs or unchanged-state reports. Runtime checkout/source/HEAD remain fixed during all three seeds; ongoing development and public text-log synchronization use the separate staging checkout. T2.4 and real Stage3/G3 remain pending, and any failed scientific/runtime gate stops for a report rather than automatic retry or protocol changes.


## Stage3 staged CPU continuation integration — 2026-11-06

The new step module wraps the real ConditionalWan interface with a tiny random30-block native Wan for29 CPU cases. Root also ran those cases with16 preliminary evaluator cases in one process:45 passed in31.547s. Stage3 validates rho0 three-step tiny AdamW parameters/moments against core Base, gamma0+corruption against C1, same-layer EMA equality, teacher-first/gradient-checkpoint hook lifecycle, FP32 sigma>.3 and exception cleanup. Trainable source/shadow/gradient finite checks are combined per device, with complete structure checks and failure-before-mutation retained;128 values require one normal-path host verdict. Details and exact source hashes are in docs/stage3_20261106/cpu_integration.json.

These are synthetic CPU optimizer trajectories, not official-model updates or formal RE. No real GPU cost/equivalence benchmark, full continuation queue/writer or G3 pass exists. Future comparison arms deliberately share foreach=False; no CUDA equivalence to the historical optimizer default is inferred. The actual three-Base queue remains on41407b0 and continues independently; candidate evaluator code is under separate review before publication/execution.


## Stage2 evaluation worker — CPU/source review complete,2026-11-06

The staged fresh-only worker and17 CPU fixtures passed root(0.394s) and independent(0.384s) checks. Its complete1152-restore/three-Base matrix preserves original native numerical restore/UniPC/quantization/LPIPS/TecoGAN and frozen mask/pair pools. Exact effective configs, completed worker statuses, new first-cache5024-entry seal/reuse, shared actual GPU lease, observed gjs identity and explicit local dependency/native-source bindings are enforced. Source/config/lock/dependency/combined-suite fixture issues were found and fixed before any real evaluator run. Source hashes and boundaries are in docs/stage2_20261106/evaluation_worker_review.json.

Actual prepare/execute must wait for all three Base8192 runs and the cache seal. No real checkpoint/evaluation prediction was read/generated by these fixtures, and G2/G3 have not passed. Preparing freezes bound staging code for the entire comparison; further development then uses an unbound scratch area. The Base queue remains on actual41407b0 and continues independently.


## Protocol33 first formal optimizer updates — 2026-11-06

All1673 paired33-frame crops/latents were generated in2861.237seconds before the new Base model initialized. Its initial trainable SHA2568b809c66d1c7bfc5e938397b7eec0c8383e3078e5a47027cb0465d5a47d9d299 matches the diagnostic **untrained** initial state and differs from the diagnostic trained state. The first two formal input IDs and FM scalars exactly match that diagnostic prefix before their data orders diverge. Fresh official initialization and no diagnostic-weight reuse are confirmed; this two-step prefix is not a broader repeatability or quality claim.

Initial validation step0 completed and the first Base entered its8192-update formal budget. Live JSONL and metadata/history snapshots are exported separately from the diagnostic8updates, with all earlier log lines preserved. The cache is generated but is not reusable across seeds until this first run completes and writes its seal. The other two Base workers remain queued.


## First protocol33 Base completed — 2026-11-06

Seed20261102 completed8192 updates, its full final validation/probes, final checkpoint and5024-entry frozen-data seal. Independent text audit confirms all8192 sequential rows with finite losses/gradient norms; the19 declared detailed gradient records also show finite groups and absent frozen gradients. Fresh official initialization/AdamW and unchanged clean runtime41407b0 are confirmed. The final checkpoint SHA256 is `28356e22e4e718d3c378588793f60732dbc93825ec5d78b8a85c8150060da63f`; only its hash/metadata are published. Optimization took2458.877s, the training/validation phase3378.519s, and the whole run including fresh preparation6293.882s. Peak allocation4100571648bytes. Last loss0.06674346; last1024 mean±within-run sampleSD=0.08753347±0.10500010, not training-seed statistics.

The single-seed final float-RGB monitoring PSNR/SSIM are22.37118dB/0.707213. These monitoring numbers do not use the sealed T2.4 quantized metric protocol and do not decide G2. All3 training-seed means±SD and the complete1152-restore stress comparison remain pending.

The existing queue automatically started seed20261103 with independent official initialization/fresh optimizer and verified reuse of only seed02's sealed data/text/VAE cache; it did not load seed02's task checkpoint. The producer seal and consumer reuse receipt agree. This audit reads the seal/receipt without repeating the consumer's full cache rehash. Details are in `docs/stage2_20261106/base_seed20261102_completion.json`. No additional GPU job or runtime deployment was started.


## Stage3 resource worker, logging and method checkpoints — CPU ready,2026-11-06

The final staged source passed89 combined CPU cases in34.356s:19 G3-worker,29 primitive/real tinyWan Step,19 new logging/architecture/checkpoint and22 historical checkpoint/CLI/model/DINO/HRI cases. Independent source review passed; the initial17 new tests plus revised10 writer tests cover19 distinct cases. Two writer findings (ancestor-root destination bypass and same-shape conditioning drift) were fixed before any GPU execution. Exact final source hashes and validation scope are in `docs/stage3_20261106/resource_worker_review.json` and `method_compatibility_review.json`.

New method arms explicitly use the NoRef Base restoration architecture. Generic inference restores only student LoRA/LR parameters, while EMA/projector/optimizer/RNG remain training-only. New checkpoints use canonical configs and atomic exclusive publication. Historical unmarked variants/HRI retain their behavior. Active-step cosine and first-active/every512 feature std/effective-rank diagnostics are implemented without altering the CPU training/RNG trajectory; snapshot copies count in ordinary Step timing, while explicitly drained CPU std/SVD does not.

The prospective G3 worker requires real passedG2 and70 fresh diagnostic updates, with mean-all20 A2/flow<=1.5 fixed before measurement. It holds the shared lease through failure receipts and final cleanup. No real G3 protocol/run or GPU update exists; G3 is pending. A full T4.1 pilot runner is still separate CPU work. Actual Base training continues on frozen41407b0 and receives none of this staged source.


## Explicit checkpoint root in the inference CLI — 2026-11-06

Follow-up source review found that the CLI still used the historical checkpoint root, although its preview and loader supported explicit new run roots. The new optional `--allowed-checkpoint-root` now reaches both calls and keeps the historical default. This lets the actual method-arm path use the generic student-only inference route without changing restoration numerics. Owner33 compatibility cases passed(1.910s); root independently reviewed the diff and passed5 CLI fixtures(0.122s), including new-root A2 and both default-root calls. These are CPU/mock pipeline checks, not official-model inference. Details are in `docs/stage3_20261106/cli_checkpoint_root_review.json`. No GPU/actual source was changed and no gate verdict was inferred.


## Future pilot training/evaluation controllers — CPU ready,2026-11-06

The two new controllers passed47 combined CPU cases (17pilot+11evaluation+19G3 regression,2.586s). Independent pilot17 passed(1.214s); root separately passed17pilot(1.197s) and11evaluation(0.760s) after source review. Root also checked all1641 real completed seed02 training metadata/latent seals (3282file hashes,1.328s), without Torch, tensor-payload loading, val consumption or real prepare. Source hashes and exact boundaries are in `docs/stage4_20261106/worker_review.json` and `real_cache_metadata_audit.json`.

The future three-arm pilot uses fresh matched Base01/freshAdamW and exactly3×2048=6144 scientific updates(0.75RE), no diagnostic-weight reuse/warmup. The new evaluator uses the unchanged full1152 method matrix and reuses all1152 original Base observations; it produces descriptive paired statistics only. Before method scoring, explicit nonplaceholder G4 definitions must be frozen and bound by rawSHA. No new G4 thresholds, sweep decision, p-value or pass verdict was chosen in this implementation.

Neither real controller was prepared or executed. Real G2/G3 still gate any pilot GPU work. Both preserve the shared GPU lease through failure receipts/cleanup and publish new checkpoints without overwrite. The active Base queue remains on41407b0; only staged code and allowlisted text are synchronized.


## Pilot G4 definitions recorded prospectively — 2026-11-06

Before method training or scores, the agent fixed the accepted G4 gate's operational pools/reference: masked LPIPS C1-minus-A2 gain > same-corrupt-pool three-Base seed sampleSD, with clean all33 RGB PSNR/SSIM A2-minus-C1 means>=0 at zero tolerance. Complete temporal/other-control diagnostics and descriptive paired-clip bootstrap intervals are retained. This is explicitly agent operationalization of the authorized plan, not a claim of separate human selection of each formula. Definitions/rawSHA and reasoning are recorded in `DECISIONS.md` and `configs/stage4_g4_definitions_20261106.json`. Actual T2 protocol binding and all method/G4 scores remain pending.


## Stage2 evaluator after staged method integration — CPU regression,2026-11-06

The17 evaluator fixtures passed again(0.394s) after related model-factory/checkpoint/CLI source changed during method integration. This repeat checks the final candidate sources before real T2 preparation; the evaluator script itself remains SHA298e8f6f. Exact source bindings are in `docs/stage2_20261106/post_integration_cpu_review.json`. No real prepare, checkpoint inference or G2 decision occurred.


## Second protocol33 Base completed — 2026-11-06

Seed20261103 completed all8192 optimizer updates, final validation/probes and its final checkpoint. Independent component audit checks8192 finite scalar gradient norms/losses with the declared fixed train1641 order and19 detailed gradient-group/frozen checks, exact effective config, fresh official initialization/AdamW, all119 runtime source files and3 native implementation files. The complete cache-reuse receipt agrees with producer seed02. Checkpoint SHA256 `87f60ce812797d73b27b9d2d0bcb41abb45ea4659ca2b6eb6092d02f3b58061a` is published as metadata only; tensor payload was not deserialized or exported.

Optimization took2459.561s, whole run including verified cache reuse3449.091s; peak allocation4100571648bytes. Single-seed final float-RGB monitoring PSNR/SSIM are22.19982dB/0.701893. These are monitoring observations, not the quantized T2.4/G2 verdict. Three-seed statistics wait for the final seed.

The original queue automatically started seed20261104 in `stage2_protocol33_base_seed20261104_20261106T054135.722171Z`, with another fresh official initialization/AdamW and verified reuse of only producer data/text/VAE caches. All three initial trainable hashes differ as expected across independent training seeds. No new GPU queue or runtime deployment was started; actual remains41407b0. The full three-seed audit/cache rehash has not run early.


## Completion auditor prepared before final seed ends — 2026-11-06

The independent full three-Base auditor now has24 focused CPU schema/rejection/atomic-publication checks and an exact archived source snapshot. It distinguishes8192 scalar gradient norms from19 detailed records, rederives data/order/timing/config/init/cache provenance, and waits for the full completed queue before its5024-file cache/three-checkpoint audit. Review corrected an overly strict helper-only SSIM bound to the valid[-1,1] interval before the final audit; existing seed02/03 monitoring values were valid and no experimental failure occurred. The full auditor main/cache rehash has not run early. Source hash and boundaries are recorded in `docs/stage2_20261106/three_base_auditor_review.json`.


## Three protocol33 Bases complete; full T2 evaluation declared — 2026-11-06

All three fresh official NoRef Bases completed8192 updates, full final monitoring/probes and final checkpoints. The independent full audit and a second read-only report/statistics review passed. It checks24576 finite scalar-loss/gradient-norm rows,57 declared detailed group/frozen checks, identical fixed train1641/val32 data/order/33stride1 geometry, independent initialization, full cache reuse/source/config provenance,119 runtime source files and3 native implementation files. All5024 producer artifacts(40963007497bytes) and final checkpoint files were rehashed; checkpoint tensors were not deserialized by this audit. Diagnostic8 updates remain separate from3 formal RE.

| Training seed | Final FM loss | Optimizer seconds | Peak GiB | Float RGB monitor PSNR | Float RGB monitor SSIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20261102 | 0.06674346 | 2458.877 | 3.818955 | 22.37118 | 0.707213 |
| 20261103 | 0.05190755 | 2459.561 | 3.818955 | 22.19982 | 0.701893 |
| 20261104 | 0.04724322 | 2456.388 | 3.818955 | 22.15527 | 0.708878 |

Training-seed mean±sampleSD: monitoring RGB PSNR22.24209±0.11399dB, SSIM0.705995±0.003648; last1024 FM mean0.08598861±0.00456177. These unquantized monitoring scores do not replace T2.4 or decide G2. Full records are in `docs/stage2_20261106/three_base_completion_audit.json`. The original queue service exited normally(code0). Actual source remains41407b0.

The prepared T2 run is `stage2_base_eval_20261106T064330Z`, protocol raw SHA256 `ddebf3768f55d0c9f33b31c1bb03bed0ad9e8bbdddfff56af08970dd408eef7e`. Before any restore, it fixes3trainingseeds×32clips×6conditions×2samplingseeds=1152 observations,38016 PNGs,33 stride1 frames,128×224 HR,32×56 LR, original native UniPC20/CFG1, CPU Gaussian42/43, one crop tile and no temporal window/HRI. Preserve original quantize_rgb/LPIPS/TecoGAN and the sealed8-frame masks, two neighbors and original temporal-pair pools. All5 types are retained with equal weights. G2 uses the previously frozen paired-excess-gap and clean-gap training-seed SD rule, without p-values or post-hoc severity changes.

Metadata preparation initially could not see/prove the desktop process from the restricted sandbox. A read-only host comparison verified the same original gjs process; unchanged host metadata preparation passed. No GPU job, data/protocol change or relaxed guard occurred in that failed restricted preflight. The separate receipt preserves both observations.

The prepared protocol now freezes every bound numeric worker/core/reader/metric/native/dependency source through the complete matrix. Documents/logs and unbound workspace scratch may change. Execution follows independent protocol review under the shared actual GPU lease and desktop guard; no duplicate GPU job, fallback, retry or Stage3 work is authorized by a partial matrix. Real G2/G3 remain pending.


## T2 evaluator stopped before model loading — 2026-11-06

The unique host service `vsr-stage2-eval-20261106t064330.service` exited1 at its fresh GPU startup guard, before Torch/model/VAE loading and before any restore:0/1152 observations,0 training updates. The passed three-Base audit is unchanged. G2 is unmeasured, rather than a measured quality failure; G3 was not started.

The raw inventory contained bound system gjs4816(64MiB) and system Nautilus1652088(72MiB). The unchanged classifier independently verified both same-user system desktop processes and confirmed no concurrent model job; the original evaluator's outer guard nevertheless rejected its pinned-match vector[true,false], because it restricts the inventory to the bound gjs. Full guard/failure/console/host-launch receipts are preserved beside the frozen protocol under `docs/stage2_20261106/stage2_base_eval_20261106T064330Z/`.

Per the human's runtime-failure stop rule, no automatic retry/resume, desktop termination or deployed guard/protocol change occurred. The original-protocol option is for the user to exit Nautilus and authorize a new run. A separate unbound scratch compatibility candidate may be CPU-reviewed for the user's choice; the failed prepared source stays unchanged. All dependent GPU work stops awaiting that decision.


## Desktop-guard compatibility candidate — CPU only, awaiting human choice,2026-11-06

A separate unbound scratch candidate passed9 owner and9 independent new CPU fixtures plus source review. Captured startup inventory reproduces the original classifier exactly; the proposed outer policy permits its already verified Nautilus/Ptyxis categories alongside exactly one fully pinned gjs. Original UID/system-exe/inode/mode/cgroup and256MiB checks remain. The candidate also explicitly rejects an empty inventory, missing/second gjs and duplicate PIDs. These policy differences are disclosed in `docs/stage2_20261106/desktop_guard_proposal/proposal.md`; source/fixtures/hashes are archived there. Nothing was deployed and no GPU/live-proc query/desktop termination/prepare/retry occurred. Human choice is required before adopting the candidate and creating a new full1152 run; retaining the original guard and exiting Nautilus is the alternative.


## Human-authorized desktop guard adoption and fresh evaluation — 2026-11-06

The human explicitly selected the reviewed desktop compatibility correction and a fresh full1152 run. Only the outer desktop admission guard and its CPU fixtures will change; the original strict classifier and scientific protocol remain unchanged. Original failed run/protocol/status remain preserved. Implementation/regression and a new metadata protocol review precede the unique new GPU execution; no in-place resume is permitted.

The adopted implementation passed26 T2 CPU fixtures(original17+new9,0.410s) and19 related G3 cases(0.509s). Root and independent source/AST reviews confirm only the guard and two category constants changed; the policy equals the approved scratch candidate, with an adopted label. Worker SHA73f8499c7ec4017021cd4d680742afb4baceb7a2e5edd77cb4c62ca7f90f1980; the original classifier and all scientific/numerical paths are unchanged. Exact review is in `docs/stage2_20261106/desktop_guard_adoption_20261106/adoption_review.json`.

The new metadata run `stage2_base_eval_20261106T081159Z` prepared successfully on the host. Protocol rawSHA `2b6156ca40cdfc953e709f5d4da6feb3de610445fcb1f554b0c8c47881f8da29`. Independent recursive comparison to the original failed protocol confirms exactly two changed leaves: worker.bytes47130→48620 and worker.sha256298e8f→73f8499c. All checkpoint/data/mask/sampler/native/metric/dependency/runtime/desktop fields are identical. Original failure directory remains preserved. The new protocol is prospectively frozen and independently reviewed before any restore; execution must perform the fresh guard again.


## Approved fresh T2 execution is running — 2026-11-06

The unique service `vsr-stage2-eval-20261106t081159.service` is active/running after the fresh adopted guard passed. It has completed134/1152 restores at this snapshot, with0 training updates. The pinned gjs is unique and the original strict classifier admitted the verified desktop extras; no desktop process was killed. Native restore/PNG/metric/row completion checks have produced real completed rows. Partial scores do not decide G2.

The full matrix continues under the shared actual GPU lease,Restart=no and frozen numeric sources. Complete committed row JSON/markers and an append-only canonical observation JSONL are exported incrementally, with no weights/tensors/PNGs/media. Future new failures stop for a report; this human approval is not an automatic-retry waiver. Original failed run remains intact. The current protocol/source and startup receipts are under `docs/stage2_20261106/stage2_base_eval_20261106T081159Z/`.


## Human-requested execution stop — 2026-11-06

The human requested stopping first and receiving a timeline/current-stage report. The unique evaluation service was stopped; host confirmation shows inactive/dead/MainPID0 and no VSR compute process in GPU inventory. Automationvsr is PAUSED. The final committed prefix is228/1152 restores(19.79%), matching228 complete row markers. All completed and uncommitted partial outputs are retained. The prior worker status was archived before marking operational status stopped_by_user. This is user cancellation, not a scientific failure; G2 remains unmeasured because the full matrix is incomplete. G3 and later GPU stages have not run. No experiment or autonomous development is to restart without new human instruction. The protocol is fresh-only and does not support in-place resume of this stopped run.
