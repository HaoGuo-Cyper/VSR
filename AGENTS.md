# Agent brief: temporal self-distillation for diffusion VSR

Read this file first, then `DECISIONS.md`, then `TASKS.md`. `RESEARCH_PLAN.md` is the technical reference for the method, baselines, metrics and controls.

**Wan TRD delivery completed,2026-11-08:** read `docs/ema_teacher_targets_20261108/final_report.md`. Full6144/41/640/192,12CP,75CI,2112PNG regression,6fixedvideo/contact/all33ROI review andreproducibletexts complete; serviceexit0/GPUreleased. EMA-vs-baseline testLPIPS evidenceinsufficient andPSNRcost; noacceleration. This iscompletedresearchdelivery,notmethodsuccess. Latesthuman separatelyauthorizes nextVideoMAEv2 plan afterthisclosure. Preserveall11currentnumericfiles andresults; implementonlyinthenewisolatedworktree. See `docs/videorepa_videomaev2_20261108/plan.md` (4096science/26diagnostic/448val/128test,onlinefixedteacher,noEMA).

**Latest human follow-on authorization, 2026-11-08:** after the current Wan TRD round has fully finished evaluation, analysis and delivery, the human instructs: “帮我整理一份保留VideoMAEv2教师、尽量忠于原版VideoREPA的Wan2.1视频超分实验方案，标清必须修改的部分，然后启动实验”. Read-only method/resource research may proceed now in unbound workspace scratch. The independently reviewed prospective plan is `docs/videorepa_videomaev2_20261108/plan.md`; it fixes online frozen VideoMAEv2, baseline/method each2048,26diagnostic updates,448val/128test. Real CPU weight loading has passed; no new teacher forward or GPU experiment has started. Complete current P6 first; then prospectively document the official VideoMAEv2 checkpoint, preprocessing, projection and temporal/spatial mapping, distinguish unavoidable Wan/VSR changes from optional simplifications, and implement/test/launch a separate bounded study. This authorizes normal implementation and passed-gate launch without another permission question, but does not authorize changing current frozen sources/helpers, overlapping GPU jobs, an unreviewed resource failure fallback, or claims of exact paper reproduction. Preserve current/previous experimental records. The earlier statement that every next study is only a proposal is superseded only for this explicitly requested VideoMAEv2 study.

**Latest human authorization, 2026-11-08: execute the accepted Wan teacher-target plan.** After the planning-only delivery, the human explicitly instructed: “好的，按你这个思路开始吧”. This authorizes implementation, relevant CPU tests, bounded GPU preflight/calibration, and normal continuation through the accepted three-arm study and delivery. Read `docs/ema_teacher_targets_20261108/plan.md` for unchanged science and budgets: three2048-step arms (6144 science updates),41 separate diagnostic updates,validation640/test192. Develop only in this new isolated worktree `/home/hugo/.codex/worktrees/vsr-wan-trd/VSR-sync`; preserve oldgamma18e6582 numerical sources/results, oldcanonicala91 andactual414. This latest human instruction supersedes the earlier planning-only execution prohibition below. Stop on failed target viability/nonfinite/OOM/scientific gates; do not silently change margin, weight grid, data or budget. Normal passed-gate continuation is authorized without repeated questions.

**Latest human request, 2026-11-08: planning only for the next teacher-target comparison.** Read `docs/ema_teacher_targets_20261108/plan.md`. Preserve the existing Wan framework and compare baseline, frozen Base teacher and EMA Base teacher using one common VideoREPA-inspired TRD objective. This is a Wan adaptation, not a CogVideoX/VideoMAEv2 paper reproduction. Proposed budget: three trajectories of2048 updates,640 validation and192 test restores. The human explicitly said not to start training: no new GPU preflight, experiment, evaluation or automation is authorized now. This restriction supersedes earlier continuous-execution permissions for this new study. Only the plan/tracking documents are updated; all completed sources and results remain preserved.

**Latest human scope,2026-11-08: baseline + HR-VAE + HR-DINOv2 fixed-teacher study. Follow docs/fixed_teacher_20261108/plan.md and method_references.md; it supersedes previous EMA/gamma and HR-VAE-only drafts, both0GPU. One trainingseed,atmost2weights/method,atmost5trajectories; sameBase/block7/commonFM/RNG,meanSmoothL1VAE/channel-L2cosineDINO. Realpreflight/calibrationbeforeweights/budgetfreeze; planned5x2048=1.25RE withinpriorStage4budget. Completeval1024 andselection-boundtest192,all33frames; independenttestsourcewithhistoricalexposuredisclosure. Onlycompleteimplementation/correctness/budgettraining/nativeinference/analysisdeliveriesfinishthisround. Recoverablefaultresumeauthorized. EMA/SelfFlowround2suggestionsonly.**

**Superseded draft,2026-11-08: HR VAE-only latent study (0GPU; do not execute).** The human explicitly replaces EMA/gamma work with frozen clean HR VAE latent supervision and requests implementation, correctness tests, small paired training, full inference evaluation and analysis. Follow docs/hr_vae_latent_20261108/plan.md and configs/hr_vae_latent_20261108.json. This overrides older rules saying HR latent is not the teacher and forbidding same-study checkpoint recovery: this independent study uses the requested target and permits diagnosed recoverable-fault resume with complete state and preserved records. Budget baseline/.01/.1each2048=.75RE; complete640clean restores incl shared0/intermediateCPs, singletrainingseed; no multi-seed/formal extension. Gamma prototype archived locally, never launched. Preserve oldcanonical a91 numericalsources/HEAD and actual41407b0; develop only thisisolatedworktree.

**Latest human authorization,2026-11-07:** the human instructed “继续在本地完成这个任务…继续，不用问我，你可以自行决定继续”. RTX5080 is the active experiment platform;H100 migration is deferred. Read the current plan and active run declarations before acting;normal gated continuation and routine engineering decisions are authorized without repeated permission. Preserve all old stopped/failed runs,never resume or reuse their rows. Scientific failure/nonfinite/OOM stops dependent GPU work for reporting,no silent protocol change or automatic retry. Actual41407b0 remains untouched.

**Latest gate state,2026-11-07: G4 FAILED; dependentGPU work STOPPED.** The full1152pilot matrix completed and38016PNG/source/hash chains passed nativeverify;independentpoint/547CI audits reproduce the result. Recovery C1-minus-A2 maskedLPIPS=.009195653052302077 exceeds BaseSD=.004334114461900804, but clean A2-minus-C1 PSNR=-.13303764684337582dB andSSIM=-.00555444686588915 fail the two zero-tolerance guards. Allthreeconditionswerepredeclared;no rules/toleranceschanged. No4096/sweep/Stage5 launched. AllservicesMainPID0,GPUreleased. Automationvsr wasdeleted after the failedgate because there is no active job left to monitor. Read docs/local_continuation_20261107/G4_report.md andnext_protocol_options_not_executed.md;the latter is a proposal only. Preserve numericalsource/dependencies/results for reproducibility;do not restart or alter this completedrun. Overallresearchobjective is not marked achieved.

**Fixed-teacher round completed,2026-11-08:** all5×2048 training updates, nativeval1024/test192, independent stats, exact2046PNG regression, two curves andsix fixed video comparisons plusactual contact/ROI review are complete. Read docs/fixed_teacher_20261108/final_report.md. Single-seed testVAE λ.11 LPIPS gain.0012671 (descriptive95%CI .0002218.. .0024026) withSSIMcost; DINO λ.094 evidenceinsufficient; noacceleration. AllservicesMainPID0/sharedGPUlockreleased. No automaticround2; preservefrozennumericalsource/results andoldcanonical/actualHEAD. Currentmonitor is beingclosed afterfinaltextpublication because noexperimentremains.

## Current goal

The Wan TRD baseline/frozen/EMA study is complete. After its final text publication, implement and execute the separately authorized VideoMAEv2/Wan VSR adaptation in a new isolated worktree using `docs/videorepa_videomaev2_20261108/plan.md`: baseline/method each2048,26diagnostic updates,448validation/128test. Keep this completed checkout's numerical sources/results unchanged. Complete correctness and target/resource preflight before its sole GPU training job; no new weights, third arm, EMA,4096 continuation or H100 migration.

## Historical EMA goal

Test whether temporally structured self-distillation improves a Wan2.1-T2V-1.3B video super-resolution (VSR) model. An EMA copy of the model (the teacher) sees frames the student only sees in corrupted form, and the student learns to rebuild those frames from neighbouring frames. The target gains are better restoration of badly degraded frames and better temporal consistency, at zero inference cost.

The comparison that matters most is against **C1**: the same corruption used as plain data augmentation, without the alignment loss. A gain that C1 also gets is not a gain from self-distillation.

## Current state (as of 2026-11-05)

**Hardware and assets**
- One NVIDIA RTX 5080, 16 GB VRAM.
- HQ-VSR dataset stored locally; 1,641 training clips used so far.
- Wan2.1-T2V-1.3B weights and its VAE stored locally.

**Existing models: 8 configurations, 16 final checkpoints**
- Backbone frozen. Trained parts: rank-8 attention LoRA plus a learned LR residual branch injected into several transformer blocks. There is no widened 32-channel input embedding.
- Every run started from the official pretrained weights and ran 8,192 optimizer updates.
- Training data: 4× bicubic degradation, HR crops of 128×224, 9 frames sampled at temporal stride 4 (spanning 33 original frames), one fixed generic text condition, and the same degradation on every frame.
- Two families:
  - **NoRef**: conditioned on the LR video only.
  - **HRI**: LR video plus a high-resolution reference image, through an extra trained reference adapter.
- Four variants per family:

| Variant | Training objective | Seeds per family |
| --- | --- | --- |
| Base | Flow-matching restoration loss only | 3 |
| Align | Flow matching + pointwise alignment to frozen DINOv2-B/14 features of HR frames | 3 |
| Relation | Flow matching + matching of spatial and motion-corresponding temporal relations between DINOv2 features | 1 |
| Both | Flow matching + Align and Relation losses | 1 |

**Evaluation status**
- Internal HQ evaluation is done for all 8 configurations. Results were mixed, with no stable overall gain from the auxiliary losses.
- External REDS4 + YouHQ40 testing is paused. Only NoRef training seed 20261102 has external outputs: Base and Align each have all 44 sequence summaries; Relation has 28/44; Both has none. These are 17-frame prefixes with sampler seeds 42/43, not full-source-video evaluation. HRI and the other training seeds have no external sequence summaries in this run. Presence of an output is not a fresh reproduction or a passed quality gate.

**Not done yet:** EMA self-distillation, per-frame-timestep models, VAE-latent alignment, one-step distillation.

**Continuation audit, 2026-11-05.** T0.2 now passed a fresh NoRef Base / REDS4 / 000 reproduction: 34 saved PNG files byte-identical across both sampler seeds. Internal statistics (including a separate crossed training-seed/source bootstrap) and all 16 final-run loss diagnostics are recorded in `docs/stage0_analysis_20261105/`; external coverage is still incomplete. Stage 1 completed 58 supported VAE crop cases and condition-gap measurements plus six synthetic no-parameter-update resource cases. The independent native720p VAE attempt stopped with OOM in its common untiled decoder, before a tiled-path comparison. Reports are in `docs/stage1_audit_20261105/`. **D1–D4 are PROPOSED, not SETTLED. GPU work is stopped awaiting the human; no training run or optimizer update has occurred.**

**Latest human decision, 2026-11-05.** D1–D4 are now **SETTLED**: 33 contiguous frames / stride1 / 9 latent positions; unchanged 128×224 ×4 bicubic and source split; future student9/teacher21 and trainable-only EMA0.999. G1 passed by explicit human acceptance. T2.1 is authorized: three independently initialized NoRef Base runs, seeds20261102/03/04, 8192 updates each. Cached text only and streaming RGB preparation are required. Normal planned continuation after passing gates is authorized without repeated confirmation. Historical audit statements above describe the earlier pause; The subsequent human-approved review closes G0 in a reduced internal-evidence scope, descopes the old external backlog and native untiled/E*warp audit requirements, and selects latent-level A2 masking excluding position0; the failed native untiled audit is not retried.

**Latest execution,2026-11-06.** All three protocol33 NoRef Bases completed8192 updates each, with diagnostic8 updates separate. Read `docs/stage2_20261106/three_base_completion_audit.json` for the complete data/source/cache/checkpoint/seed-statistics audit. Actual source remains41407b0. Current evaluation declaration is staging `runs/active_stage2_evaluation.json`; read its run status/progress and latest `docs/stage2_20261106/live/progress.json` before any action. Never duplicate a planned/running matrix. Once prepared, every numeric source/dependency bound by that protocol is frozen through all1152 restores; only documentation/logs and unbound workspace scratch may change. G2 and real G3 require their measured gate verdicts. The human explicitly requested stopping execution and a timeline on2026-11-06. The evaluator is stopped and automationvsr is PAUSED; preserve all completed/partial artifacts and do not restart experiments or autonomous development without a new human instruction. Read the user_stop receipt for the final committed-row count.

## Where things are

**Current planning entry,2026-11-06:** read `docs/H100_MAIN_EXPERIMENT_PLAN_20261106.md`. The human assigned a remote H10080GB to main experiments and RTX5080 16GB to development/validation, requesting a new plan **without starting experiments**. H100 access/environment/assets are not yet verified or migrated. Experiments remain stopped and automation PAUSED; this planning instruction does not authorize GPU execution or restart. Keep D1–D4, existing three Bases and scientific gates; prepare a new prospective H100 protocol before future execution.

**Latest source refactor,2026-11-06:** the human explicitly authorized actual10-module consolidation, preservation of old code/results, relatedCPU tests, README update and GitHub sync, while forbidding GPU experiments/H100 migration. The staging `vsr/` now has exactly data/vae/model/teacher/losses/train/infer/evaluate/metrics/checkpoints plus __init__.py, with real implementations and no runtime legacy import. Oldvsr/scripts/tests144 files are byte-identical under `legacy/pre_refactor_20261106/`, manifest/source commit60e2b4b. All130CPU tests passed,0 failures/0 skips,CUDA never initialized; see `docs/code_refactor_20261106/cpu_validation.json`. Read README and `docs/CLI_GUIDE.md` for the real unifiedCLI. Current scientific gates and GPU/H100 validation remain pending; automation stays PAUSED. Old stopped protocol sources are preserved in the archive/original commit and actual41407b0; never resume that oldrun using new code.

**Latest boundedGPU result,2026-11-07:** the human separately authorized exactly one local5080 development closure, not full experiments/H100 migration. It PASSED:3A2 diagnosticupdates(teacher2/EMA3), atomicstep3 save, engine release and byte-exact studentreload,1nativeUniPC20/33frame restore,33PNG pixelreadback andMP4 checks. Run `runs/development_smoke_5080_20261107T035103Z`, protocol117d2bfddccfd993294a681083380e445b822d354fa09e90e173a51cadda88f5, summary/audit in `docs/development_smoke_20261107/`. PyTorchpeak3.746GiB allocated/3.887GiB reserved; no quality orG3overhead inference.19targetedCPU checks passed beforeGPU. Service exited0/MainPID0,GPU freed. This grants no continuingGPU work: automationPAUSED, fullBase evaluationstill228/1152 stopped, G2/G3/G4 andH100pending. Do not reuse the smokecheckpoint or automatically rerun.

Verified by read-only inspection on 2026-11-05. The new source root is `/home/hugo/Projects/VSR_`; the original `/home/hugo/Projects/VSR` remains the read-only asset and historical-results source. Existing configs retain their original paths and hashes. Do not redirect outputs into the historical runs.

| Item | Path |
| --- | --- |
| Training code | Current10-module staging: `/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/VSR-sync/vsr/train.py`; original actual runtime stays `/home/hugo/Projects/VSR_/` at41407b0. Historical HRI/DINO/phase sources are archived in staging `legacy/pre_refactor_20261106/`. |
| Evaluation code | Current staging `vsr/evaluate.py`, `vsr/infer.py`, `vsr/metrics.py`; old actual/original implementations remain unchanged in `/home/hugo/Projects/VSR_/` and `/home/hugo/Projects/VSR/`. |
| Configs for the 8 existing configurations | `/home/hugo/Projects/VSR_/configs/formal_hq_v2_noref.json`, `formal_hq_v2_remaining_noref.json`, `formal_hq_v2_reference.json`; Base/Align repeats in `formal_hq_v2_{noref,reference}_repeat_seed2026110{3,4}.json`. Variants share family configs; original immutable copies are in `/home/hugo/Projects/VSR/configs/`. |
| The 16 checkpoints | `/home/hugo/Projects/VSR/runs/` — exact run directories and variant mapping below; each filename is `checkpoint_008192.pt`. Eight per family: Base ×3, Align ×3, Relation ×1, Both ×1. |
| HQ-VSR data (train / val split) | Raw `/home/hugo/Projects/ReprVSR-pretrained/data/HQ-VSR/`; split `/home/hugo/Projects/VSR/manifests/hq_vsr.json`; stride-4 eligibility `/home/hugo/Projects/VSR/manifests/hq_temporal_stride4_inventory.json`. Manifest copies are also in the new source root. |
| REDS4, YouHQ40 | `/home/hugo/Projects/VSR/data/external/{REDS4,YouHQ40}/`, source archives `/home/hugo/Projects/VSR/data/external/archives/`; manifest `/home/hugo/Projects/VSR/manifests/external.json`; provenance `/home/hugo/Projects/VSR/docs/external-data-provenance.json`. |
| Wan2.1-T2V-1.3B weights and VAE | `/home/hugo/Projects/ReprVSR-pretrained/models/Wan2.1-T2V-1.3B-Diffusers/{transformer,vae}/` (official revision recorded in configs); cached tokenizer/text encoder are in the same model root. |
| Cached text embedding, cached DINOv2 features | `/home/hugo/Projects/VSR/runs/formal_hq_v2_noref_20261102T142214.588571Z/{fresh_text_embeddings.pt,fresh_teacher_features.pt,fresh_training_cache.pt}`. DINO source `/home/hugo/Projects/VSR/vendor/dinov2/`, weights `/home/hugo/Projects/VSR/weights/dinov2_vitb14_pretrain.pth`; shared frozen-cache reuse is validated by `vsr/train.py:97–139`. |
| Training logs | `/home/hugo/Projects/VSR/runs/<formal-run>/<variant>/training.jsonl`, sibling `summary.json`, `initialization.json` and `evaluation_history.json`; console logs `/home/hugo/Projects/VSR/runs/console_*.log`. |
| Existing evaluation outputs | Internal `/home/hugo/Projects/VSR/runs/HQ_test_frozen_T1_policy_20261103T022832.538880Z/`; external `/home/hugo/Projects/VSR/runs/external_paired_frozen_20261103T125240.867596Z/external/`; supporting reports `/home/hugo/Projects/VSR/docs/final_evaluation_v1/` and `/home/hugo/Documents/Codex/2026-11-02/files-pasted-by-the-user-wan2/outputs/`. |

All checkpoint paths below are relative to `/home/hugo/Projects/VSR/runs/`; append `/<variant>/checkpoint_008192.pt`. These files remain outside Git.

| Family / training seed | Run directory | Final variants |
| --- | --- | --- |
| NoRef / 20261102 | `formal_hq_v2_noref_20261102T142214.588571Z` | Base |
| NoRef / 20261102 | `formal_hq_v2_remaining_noref_20261102T153108.569879Z` | Align, Relation, Both |
| NoRef / 20261103 | `formal_hq_v2_noref_repeat_seed20261103_20261102T194730.270735Z` | Base, Align |
| NoRef / 20261104 | `formal_hq_v2_noref_repeat_seed20261104_20261102T204607.424405Z` | Base, Align |
| HRI / 20261102 | `formal_hq_v2_reference_20261102T171545.936007Z` | Base, Align, Relation, Both |
| HRI / 20261103 | `formal_hq_v2_reference_repeat_seed20261103_20261102T214631.884279Z` | Base, Align |
| HRI / 20261104 | `formal_hq_v2_reference_repeat_seed20261104_20261102T225815.432804Z` | Base, Align |

The 2026-11-05 T0.1 evidence, bounded Stage 1 proposals and subsequent human settlement of D1–D4 are in `DECISIONS.md`. The earlier read-only-audit outputs are under `/home/hugo/Projects/VSR_/runs/{stage0_reproduction_20261105T113721Z,vae_crop_full_20261105T114559Z,profile_protocol_resources_20261105T114931Z,vae_full_resolution_20261105T115318Z}/`. New Stage2 runs use separate dated directories and the accepted protocol33 configs.

## Working rules

1. **No new training before the open decisions are settled.** Stages 0 and 1 need no decisions and can start immediately. Stage 2 onward waits until the human marks D1–D4 in `DECISIONS.md` as SETTLED.
2. **Never modify existing checkpoints, configs or result files.** Write new outputs to new, dated directories.
3. **One fixed inference protocol per comparison.** Fix the sampler, number of steps, sampler seed, spatial tiling and temporal windowing. Record the protocol in `RESULTS.md` before evaluating, and never change it partway through a comparison.
4. **Every claim needs seed statistics.** Report mean ± std over seeds and a paired bootstrap over clips. Label single-seed results "single seed, indicative".
5. **Primary metrics are fixed before Stage 5** and recorded in `DECISIONS.md`. Never pick metrics after seeing results.
6. **Log every training run** in `RESULTS.md`: run id, git commit, config path, seed, starting checkpoint, number of updates, wall-clock time, peak VRAM, final loss values, evaluation numbers.
7. **Keep `TASKS.md` current.** Mark each task `[~]` when started and `[x]` when done (or `[!]` when blocked), with a date and a one-line note.
8. **Stop at failed gates.** If a gate fails or a result contradicts the plan, report it and wait. Do not improvise a new research direction.
9. **Ask the human first** before you delete anything, change the data split, change the degradation used by existing baselines, or spend more than 2 run-equivalents on anything not in `TASKS.md`.
10. **Hardware roles.** The original16GB limit remains the RTX5080 development/validation target; the remote H10080GB is the future main experiment platform. Preserve first-round scientific settings and review device assumptions before migration. Use precomputed text embedding, never keep T5 loaded during training, and retain declared bf16/checkpointing settings. Historical DINO targets are cached offline. Neither hardware availability nor this planning update authorizes GPU startup.
11. **Public GitHub synchronization is authorized.** On 2026-11-05 the human authorized this and future new/updated project text files to be synchronized to the public `HaoGuo-Cyper/VSR` repository, including project-local paths, experiment metadata and data manifests. No repeated publication confirmation is needed for that scope. Keep weights, checkpoints, raw media, tensor caches and credentials outside Git. This permission does not settle D1–D4, waive failed gates or authorize training.

## Vocabulary

- **RE (run-equivalent):** one 8,192-update training run. A 2,048-update continuation is 0.25 RE.
- **A2:** the main method. Frame-selective corruption of the LR condition, plus alignment of student features to EMA-teacher features on the corrupted frames.
- **C1:** the same corruption with no alignment loss (augmentation-only control).
- **Base / Align / Relation / Both:** the existing variants above.
- **A0–A4, B0–B4, C1–C5:** defined in `RESEARCH_PLAN.md`.

## Reporting

At the end of each stage, add a short section to `RESULTS.md` containing:
- what was run;
- the results table;
- the gate verdict (pass or fail, and why);
- any questions for the human.

Human authorization on 2026-11-05 permits automatic continuation through the existing plan after successful checks. Report each stage, continue when its gate passes, and ask only for failed gates, problematic results or exceptional changes; do not repeatedly request permission for already-authorized training.
