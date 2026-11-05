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
