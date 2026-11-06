# Tasks

Status: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked · `[-]` descoped by an explicit scope decision.
Add a date and a one-line note whenever a status changes.

Training budget is in RE (run-equivalents): 1 RE = one 8,192-update run.

## Current execution order — H100 planning update,2026-11-06

Read `docs/H100_MAIN_EXPERIMENT_PLAN_20261106.md` first for current status, machine roles, migration inventory and dependency order. Remote H10080GB is the future main experiment platform; RTX5080 16GB handles development/validation. The human requests **planning only, no experiment launch**. All GPU work remains stopped and automationvsr PAUSED. Historical tasks below retain evidence and superseded-status chronology; the latest three-Base audit and user-stop records determine current completion.

- [x] **P0 Reorganize the plan.** (2026-11-06: completed work, H100 migration and E2–E6 documented; no transfer or experiment.)
- [x] **R0 Plan the10-module code structure and README.** (2026-11-06: responsibilities, existing-source mapping, all151Python file index, two-host workflow and oldREADME archive; numeric code unchanged.)
- [ ] **R1–R4 Consolidate and validate the mainline source.** Follow `docs/CODE_ORGANIZATION.md`: VAE/model factory boundaries, EMA/losses separation, shared train/infer/evaluate, relevant CPU parity and historical compatibility. No claim that target10-file CLI is already implemented.
- [ ] **R5 Validate the new structure on both devices.** Planned small5080 GPU loop and H100 compatibility cases require prospective protocols and a later start instruction. OldBase success is not validation of unimplemented new structure.
- [ ] **M1 Resolve H100 access/storage/paths and select the deployment commit.** No remote host verified.
- [ ] **M2 Migrate and hash-verify private assets.** Existing3Bases, official Wan/VAE, fixed data/cache/text, val32 and metrics; no Base retraining.
- [ ] **M3 Review H100 environment and runner.** Explicit paths/roots/device/host admission/locks and source/environment bindings, related CPU checks. GPU smoke/resource tests await a later start instruction.
- [ ] **E2 Return to T2.4 on H100 in a fresh run.** Full1152 observations; preserve stopped5080 prefix separately, no cross-host row stitching. Complete recomputed G2 is required before G3.
- [ ] **E3–E6 Follow G3/pilot/G4/formal/ablation/final stages after their gates.** Budgets and mappings are in the new plan; no experiment started.

## Stage 0. Consolidate existing results (start now, no training)

- [x] **T0.1 Map the codebase.** (2026-11-05: code/assets mapped; D1–D3 documented with code evidence; see `docs/stage0-code-audit-20261105.md`.) Locate the training code, evaluation code, configs, checkpoints, data and logs, and fill in "Where things are" in `AGENTS.md`. From the code, answer:
  - how training frames are encoded by the VAE (D1);
  - how test sequences are fed at inference (D2);
  - which blocks the LR branch injects into, and whether it mixes frames before injection (D3).

  *Done when:* all paths are filled in, and D1–D3 in `DECISIONS.md` have code-level evidence (file and line references).
- [x] **T0.2 Freeze the inference protocol.** (2026-11-05: fresh NoRef Base / REDS4 / 000 reproduction passed for seeds42/43; all 34 PNG files byte-identical and all original metric pools, bicubic baseline and seed mean match fixed tolerances; see `reproduction_summary.json` in `docs/stage0_analysis_20261105/`.) Record the protocol used for the finished REDS4 and YouHQ40 runs: sampler, steps, sampler seed, tiling, temporal window and overlap. Put it in `RESULTS.md`.
  *Done when:* re-running one sequence with one checkpoint reproduces the stored numbers.
- [-] **T0.3 Finish external evaluation — DESCOPED by human-approved review,2026-11-05.** Old evaluation backlog is not required for the new research line; keep all historical outputs/coverage limits. Reserve new-protocol external evaluation for final models, and HRI for optional Stage8. No part of the old backlog is launched. Historical scope/cost follows: (2026-11-05: read-only 16-checkpoint/704-row completion plan prepared; at least 1880 restores remain if 116 old markers later pass strict review. NoRef-rate recovery-only extrapolation is about108.4h, with HRI/metrics/I/O costs additional or unknown. No queue started; a new cohort/fresh-output plan needs human review, and GPU work is stopped after the separate Stage1 OOM.) Run the remaining Relation sequences and all Both sequences, in both families, with the frozen protocol.
  *Done when:* all 16 checkpoints have per-clip REDS4 and YouHQ40 metrics saved as CSV.
- [x] **T0.4 Statistics — complete for revised internal-evidence scope.** External missingness remains explicitly reported; no external multi-seed claim is made. Historical scope follows: (2026-11-05: internal val32 three-seed and frozen test211 single-seed tables completed, with paired source-cluster and separate crossed seed/source bootstrap; external coverage remains incomplete.) For each family:
  - mean ± std over seeds for Base and Align;
  - a paired per-clip bootstrap, Base vs Align;
  - Relation and Both reported as single-seed.

  Include the internal HQ results.
  *Done when:* `RESULTS.md` has one consolidated table and a verdict paragraph.
- [x] **T0.5 Loss diagnostics.** (2026-11-05: all 16 final logs have 8192 consecutive observations; first/last 512-record flow/alignment/relation summaries saved under `docs/stage0_analysis_20261105/`; unfinished Align excluded, no quality inference from loss decrease.) From the training logs, check whether the alignment and relation losses actually decreased. Optional: CKNNA between block features and DINOv2 features, Base vs Align.
  *Done when:* a plot or table plus a one-sentence conclusion.

**Gate G0:** report to the human. Stage 1 can proceed without waiting.

2026-11-05 audit: **not passed**; T0.2 reproduction and T0.3–T0.5 remain unfinished. Contradictions between the handoff and the frozen external protocol are recorded in `RESULTS.md`. Per AGENTS.md rule 8, report them and await the next GPU evaluation/profiling plan; no GPU job was launched.

2026-11-05 continuation: the human instructed "continue" after publication. Proceed with existing-score/loss consolidation and one fresh, bounded T0.2 reproduction (NoRef Base / REDS4 / 000, seeds 42/43) under the unchanged historical numerical protocol. Do not resume the old partial run. This continuation does not settle D1–D4 or start a training stage. T0.5 is now complete; T0.4 remains partial.

Latest continuation disposition: T0.2's reproduction now passed; T0.5 is complete, T0.4 remains partial and the new T0.3 coverage/cost plan is not an execution result. **G0 remains unclosed.** No old marker has been newly accepted merely because it exists.

## Stage 1. Protocol and VAE audit (start now, no training)

- [x] **T1.1 VAE reconstruction audit — closed in revised crop scope,2026-11-05.** The human-approved review removes native720p untiled comparison and unavailable E*warp/separate tLPIPS implementations from the required scope. Keep58 crop cases and sampled evidence limitations; use explicitly named existing TecoGAN tLP100 and tOF. The stopped full-resolution attempt remains a capacity failure, not a passed measurement, and is not retried. Historical scope follows: (2026-11-05: all 58 supported crop cases passed; two REDS33/stride4 cells unsupported. Native720p attempt stopped during untiled decoding with OOM, so tiling differences remain unmeasured; E*warp/tLPIPS definitions remain pending. See `docs/stage1_audit_20261105/`.) Use the HQ-VSR validation split and a REDS4 subset. Encode and decode at stride 1 vs stride 4, with clip lengths 9, 17 and 33. Measure PSNR, SSIM, LPIPS, E*warp, tLPIPS and error per frame index. Also compare tiled vs untiled encoding at full resolution.
  *Done when:* the results table is in `RESULTS.md`.
- [x] **T1.2 Condition gap.** (2026-11-05: raw/normalized per-channel and per-latent-time LR–HR distances and mean/population-SD comparisons measured on the fixed 58 crop cases; scope is the selected 8HQ+2REDS sources, not a dataset-wide estimate.) Measure the per-channel distance between E(upsampled LR) and E(HR). Compare latent mean and std on your data with Wan's normalization constants.
- [~] **T1.3 Resource profile.** (2026-11-05: six synthetic no-update cases passed with parameter hashes unchanged; forward/backward-cycle peaks/time and extra-forward ratios recorded. Full optimizer-update/actual-EMA costs remain unmeasured; no new training was started.) At 128×224, profile the current protocol (9 frames, stride 4) and contiguous 17 and 33 frames. For each, record peak VRAM and seconds per update for Base training, and for Base plus one no-gradient teacher forward pass.
  *Done when:* hours per RE are known for each protocol.
- [x] **T1.4 Protocol proposal.** (2026-11-05: D1–D4 marked PROPOSED with explicit partial-audit limits, preferring 33 contiguous frames; GPU work stopped after the native-resolution OOM, awaiting human decisions. No proposal is SETTLED.) Write a recommendation for D1 and D2 with the evidence, and mark them PROPOSED in `DECISIONS.md`. Then **stop and wait** for the human to settle D1–D4.

**Gate G1:** the human has settled D1–D4.

2026-11-05 audit: **not passed**; Stage 1 measurements are not run, and D1–D4 have not been settled by the human.

2026-11-05 continuation: bounded crop/gap and no-update resource measurements are now recorded; T1.1 and T1.3 remain partial as described above. **G1 is still not passed.** Native untiled decoding exceeded capacity and was not retried. Stop further GPU work and await human settlement of the proposals in `DECISIONS.md`; T2.1 and all later training tasks remain unstarted.

**Latest G1 disposition, 2026-11-05:** PASSED by explicit human settlement of D1–D4. Proceed with T2.1 and later planned tasks after their checks pass, without repeated permission. Failed gates and exceptional protocol changes still require reporting and a human decision. Remaining audit limitations are retained; G0 is separately closed in the revised scope below. The old untiled OOM protocol is not retried.

**Latest G0 disposition,2026-11-05:** CLOSED for the human-approved reduced scope. Internal seed statistics and loss diagnostics support the limited motivation statement; T0.3 is descoped, not executed. External coverage is incomplete and no external conclusion is inferred. T1.1 is likewise closed only in its revised crop scope. Full optimizer/EMA costs remain measured separately; the old synthetic profile does not pass G3.

## Stage 2. Baseline and test sets

- [x] **T2.1 Base under the final protocol.** (2026-11-06: all3 fresh official NoRef seeds completed8192 each; independent full provenance/log/5024-cache/three-checkpoint audit and seed statistics passed. Diagnostic8 is separate from3RE. Runtime41407b0 preserved.)
- [x] **T2.2 Frame-varying test set.** (Latest2026-11-05: CPU materialization complete,32val clips/224NPYs;192 reader views passed SHA/shape/window/FPS/source and unmasked identity checks. No predictions or G2 result.) (2026-11-05: fixed val32 metadata cohort/masks sealed before pixels/new scores; CPU materialization underway; no test/external tuning.) Start from REDS4, YouHQ40 or held-out HQ-VSR clips. Inject per-frame spikes into selected frames: extra blur, noise, heavy codec compression, dropped frames (repeated or blank). Save a mask of which frames are corrupted.
- [~] **T2.3 Corrupted-frame recovery metric.** (2026-11-05: original-frame/pair pool and three-seed G2 summarizer implemented; five-corruption CPU synthetic checks passed; actual checkpoint scores remain unmeasured.) Compute PSNR and LPIPS on the corrupted frames, compared with the same frames restored from clean LR. Also compute the gap between corrupted frames and their neighbours.
- [~] **T2.4 Evaluate all three Bases.** (USER STOP2026-11-06: service stopped and automationvsr PAUSED at228/1152 committed restores; outputs retained. G2 cannot be decided from this incomplete matrix. No restart without new human instruction; original protocol is fresh-only.)
  *Done when:* Base mean ± std is reported on every set and the headroom is measured.

**Gate G2:** corrupted frames are worse than their neighbours by more than the seed spread. If not, report and propose harder settings (longer clips, larger motion, stronger per-frame degradation).

## Stage 3. Self-distillation code

- [~] **T3.1 EMA.** (2026-11-05: standalone CPU primitives/17 tests pass; actual Wan hooks/trainer/GPU integration and G3 remain pending.) Keep an EMA over trainable parameters (LoRA and LR branch), initialized from the starting checkpoint, with configurable decay (default 0.999).
- [~] **T3.2 Teacher pass.** (2026-11-05: standalone CPU primitives/17 tests pass; actual Wan hooks/trainer/GPU integration and G3 remain pending.) Run the frozen backbone with the EMA adapters under `no_grad`, either by swapping adapter weights or by holding a second adapter set. Check memory use.
- [~] **T3.3 Frame-selective corruption at the LR-branch input.** (2026-11-05: standalone CPU primitives/17 tests pass; actual Wan hooks/trainer/GPU integration and G3 remain pending.) The accepted pilot default is latent-level corruption, excluding latent position0. Pixel-level is deferred to an optional later ablation after a crop leakage test; implement required pilot behavior first. Original alternatives:
  - pixel-level: blur, downscale or codec applied to the LR frames behind the chosen latent frames;
  - latent-level: replace them with matched-variance noise or a learned null token.

  Add a mask sampler with ratio ρ and span length.
- [~] **T3.4 Alignment loss.** (2026-11-05: standalone CPU primitives/17 tests pass; actual Wan hooks/trainer/GPU integration and G3 remain pending.)
  - Forward hooks on student block l and teacher block k.
  - An MLP projector on the student features; LayerNorm without affine on the teacher features.
  - Cosine loss on corrupted-frame tokens, weighted by γ.
  - A timestep window controlling which t the loss applies to.
- [~] **T3.5 Unit tests.** (2026-11-06:89 combined CPU cases passed, including29 primitive/real tinyWan Step,19 G3-worker,19 logging/architecture/checkpoint and22 historical regressions; independent review passed. Actual GPU equivalence/cost gate remains pending.)
  - With EMA equal to the student and no corruption, teacher features equal student features.
  - Corruption changes only the chosen frames' condition.
  - With the auxiliary loss disabled, training numerics match Base training.
  - Measure step-time overhead.
- [x] **T3.6 Logging.** (2026-11-06: staged active cosine/layer IDs and first-active/every512 detached feature std/entropy-rank logging pass CPU trajectory/RNG tests. Full Step includes snapshot copies; explicit CPU statistics drain is outside timing. Real GPU/G3 remains pending.) Log L_flow, L_sd, cosine per layer, feature std and effective rank.

**Gate G3:** all tests pass and the overhead is at most 1.5×.

## Stage 4. Pilot A2 (one seed, about 3 RE)

- [~] **T4.1 Three arms.** (2026-11-06: complete staged training controller is CPU/source reviewed; actual prepare/execution waits for real G2/G3. Exact3×2048=6144 scientific updates,0.75RE; no warmup or diagnostic-state reuse.) From NoRef Base seed1, continue for K=2,048 updates in three arms: flow-only,C1 and A2. A2 defaults:
  - ρ = 0.25, in spans of 2 latent frames;
  - latent-level LR-branch corruption, excluding latent position0;
  - γ = 0.8, loss applied for t > 0.3;
  - EMA decay 0.999.
- [ ] **T4.2 A2 sweep, one factor at a time.** ρ ∈ {0.1, 0.4}; γ ∈ {0.25, 0.5}; latent-level corruption.
- [~] **T4.3 Evaluate.** (2026-11-06: staged full1152 method evaluator is CPU/source reviewed; actual scoring waits for completed pilot and prospectively frozen G4 definitions. Outputs descriptive paired metrics, no automatic G4 verdict.) Under the human-approved review, run every pilot arm on the fixed clean internal validation crops and the frame-varying validation set. Reserve REDS4/YouHQ40 external evaluation for final models; do not launch the old external backlog.

**Gate G4:** A2 beats C1 on corrupted-frame recovery by more than the Base seed spread, with no loss on the clean sets. If it fails: check the diagnostics and try K = 4,096. If it still fails, report and stop.

## Stage 5. Formal comparison (three seeds, about 4 RE)

- [ ] **T5.0 Fix the primary metrics** and the significance test in `DECISIONS.md` before launching any run.
- [ ] **T5.1 Five arms.** Continue from each of the three Base seeds in five arms: flow-only, C1, A2 (best pilot setting), DINOv2-Align continuation, VAE-latent alignment continuation.
- [ ] **T5.2 Evaluate.** Use a paired bootstrap across clips and seeds.

**Gate G5:** A2 beats C1 and flow-only on the primary metrics at p < 0.05, with fidelity unchanged.

## Stage 6. Ablations and mechanism (about 7 RE)

- [ ] **T6.1 One-seed screens.** ρ; spans vs single frames; pixel vs latent corruption; student and teacher blocks; γ; EMA decay; C4 frozen teacher; C2 random-token mask; C3 no mask; LoRA rank 8 vs 32.
- [ ] **T6.2 Three seeds** for C2 and C4.
- [ ] **T6.3 From scratch.** One A2 run from the pretrained model for 8,192 updates, with the alignment loss switched on after an SR warm-up.
- [ ] **T6.4 Mechanism.**
  - C5: evaluate with shuffled or unrelated neighbouring frames.
  - Attention mass from corrupted-frame queries to other frames.
  - Qualitative recovery figures and failure cases (large motion, scene cuts).
- [ ] **T6.5 Per-frame timesteps (only with 5 or more latent frames).** Implement per-frame timesteps. Unit test: equal timesteps on all frames reproduce the original outputs. Then run A1 and A3.

## Stage 7. Final evaluation

- [ ] **T7.1 Full evaluation.** All final models on REDS4, YouHQ40, internal HQ and the frame-varying set: full sequences, overlapping temporal windows, spatial tiling.
- [ ] **T7.2 Sampling-step sweep.** 4, 8, 25 and 50 steps.
- [ ] **T7.3 Published models for context.** Released FlashVSR and DOVE weights. They train on real-world degradation, so report them as context only.

## Stage 8. Optional extensions (only if G5 passed)

- [ ] HRI family with the best A2 setting, 3 seeds.
- [ ] Real-world degradation pipeline.
- [ ] Few-step or one-step distillation, if it fits in 16 GB.

## Stage 9. Write-up support

- [ ] Final tables, figures and a method description, built from `RESULTS.md`.

2026-11-05 accepted review: wherever the earlier T4.1 description chooses pixel-level A2 first, use latent-level LR-branch corruption excluding position0 instead. Pixel-level evaluation stress tests remain required in T2.2 and are not the pilot training intervention. Keep three seeds as explicitly authorized; five seeds and a full33-frame external tile probe remain possible future decisions, not silently added runs.
