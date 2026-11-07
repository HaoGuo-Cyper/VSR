# Decisions

Status key:
- **OPEN**: the human must settle it.
- **PROPOSED**: the agent has added evidence and a recommendation; awaiting the human.
- **SETTLED**: do not reopen without new evidence.

## Protocol decisions (D1–D4 settled by the human, 2026-11-05)

**Authoritative human settlement, 2026-11-05.** The human stated: “采纳33帧方案，确定D1–D4，按之前的规划先做三个Base，代码和日志继续同步到GitHub。” This accepts the latest concrete proposals below: 33 contiguous frames, 128×224, unchanged ×4 bicubic, whole-clip VAE → 9 latent positions; future training/inference ordinal stride1; LR injections before zero-based0/5/9, initial student output9/teacher output21; NoRef main line, mandatory C1, matched-seed Base continuations and trainable-only EMA0.999. First train three fresh official NoRef Bases (seeds20261102/03/04, 8192 updates each;3RE), with fixed data seed20261102 and identical train1641/val32 selection. Accept corrupted-frame LPIPS Alex0.1 and explicitly defined TecoGAN tLP100 as primary metric families, with clean RGB PSNR/SSIM guards; final evaluation intervals/pools/statistical rules/fidelity thresholds still freeze prospectively at T5.0. Historical proposals/evidence below retain their original wording and are superseded by this settlement where they say awaiting acceptance.

The human also authorized normal continuation to the next planned task without repeated permission. Stop and ask only on failed gates, problematic tests/results or exceptional changes requiring a new decision. This does not mark G0 complete, authorize retry of the failed native untiled protocol, change original checkpoints/configs or waive scientific gates.

### D1. Frame encoding and clip length (SETTLED)

**Question.** Are the 9 stride-4 frames encoded by the Wan VAE as one clip, giving 3 latent frames, or frame by frame, giving 9 latent frames?

**Why it matters.** With clip encoding there are only three latent time positions. Nominal VAE anchors are four sampled frames apart (16 source-frame intervals under training stride 4), but those anchors are not a measurement of disjoint VAE receptive fields. Corrupting one latent position affects a third of the available latent positions. Frame-by-frame encoding would retain nine latent positions but changes the pretrained VAE's temporal use.

**Recommendation.** Move to contiguous frames, 17 or 33 (giving 5 or 9 latent frames), if they fit in 16 GB. This requires retraining NoRef Base with 3 seeds (3 RE) but makes "neighbouring frames" meaningful.

**2026-11-05 proposal after bounded audits — awaiting the human.** Prefer **33 contiguous frames**, still 128×224 and the unchanged ×4 bicubic degradation, giving **9 latent positions**. It retains the old protocol's 33-source-frame span while sampling densely instead of retaining only nine stride-4 images. The selected HQ audit sources show similar all-frame reconstruction averages for 17 and 33 contiguous frames, and shared-prefix RGB metric values are exactly equal; the extra frames must not be mistaken for a length-induced reconstruction loss. The 33-frame synthetic no-update profile reached 3.060 GiB allocated, versus 2.908 GiB for 17 frames; cycles including the extra no-grad forward averaged 0.3791 s versus 0.2143 s. Those measurements omit optimizer/EMA/projector costs and are not full-update timings or proof of final A2 fit. Full records and limits are in `docs/stage1_audit_20261105/`.

With nine latent positions, a two-position corruption span realizes 2/9 (22.2%), near nominal ρ=0.25; with five positions it realizes 40%. Any future mask sampler must explicitly define integer rounding/span handling and log realized ratios, rather than silently claiming exact nominal ratios. The main video remains whole-clip VAE encoded. Protocol change requires the three new NoRef Base runs in T2.1 before method comparisons; do not compare A2 under the new spacing against an old stride-4 Base as its primary control.

**Evidence needed.** Code inspection (T0.1); VAE audit (T1.1); VRAM and time profile (T1.3).

**T0.1 evidence, 2026-11-05.** The existing main HR/LR video is encoded as one clip, yielding **3 latent frames**. `configs/formal_hq_v2_noref.json:13–23` fixes 9 frames and temporal stride 4; `vsr/data.py:169–204` selects a 33-source-frame span and constructs `[1,3,T,H,W]`. `vsr/preflight.py:77–100` passes the whole HR clip and whole upsampled LR clip separately to `vae.encode(...).latent_dist.mode()`. `vsr/alignment.py:35–55` derives temporal compression 4 and `(9−1)/4+1=3` latent positions from the official model config. The original run's `runs/formal_hq_v2_noref_20261102T142214.588571Z/correspondence.json:2–24` records latent grid `[3,16,28]`, token grid `[3,8,14]`, and nominal anchors `[0,4,8]`.

This is consistent across all eight `configs/formal*.json` files (including the template and remaining-variants config) and all seven actual formal-run correspondence records. HRI's independent **reference-image** cache is a separate path: `vsr/reference_training.py:98–105,142–157` encodes each possible supplied reference as T=1; it does not change main HR/LR clip encoding. Cached main-video latents are checked against the derived geometry in `vsr/train.py:97–125`. The later bounded T1.1/T1.3 measurements and their explicit remaining scope are documented above and in `RESULTS.md`; the PROPOSED future protocol still awaits human settlement.

### D2. Train/test frame spacing (SETTLED)

**Question.** How are test sequences fed at inference: consecutive frames, or stride 4 as in training?

**Why it matters.** Training at stride 4 and testing on consecutive frames changes the temporal spacing by a factor of four; the actual motion difference depends on each source. Whatever future protocol is chosen, training and test spacing must be considered together.

**Evidence needed.** Code inspection (T0.1).

**2026-11-05 proposal — awaiting the human.** Use **ordinal stride 1 in both future training and inference**. Declare clip/window lengths and intervals before each new comparison; preserve the original 17-frame-prefix external protocol as historical Stage 0 evidence. Never relabel its results as matching the old stride-4 training protocol, full-sequence scores or the future 33-frame protocol. New protocol configs/outputs must be separate. The crop VAE audit supports investigating contiguous inputs, but stride conditions cover different source frames and elapsed time; their quality difference is not an isolated causal effect of stride or an SR-model benefit.

**T0.1 evidence, 2026-11-05.** Internal HQ inference uses the training spacing: `vsr/evaluate_hq.py:24–26,49–62` sets 9 frames and temporal stride 4, and `vsr/hq_evaluation_data.py:117–135` delegates to the same strided decoder. External REDS4/YouHQ40 inference instead selects **17 consecutive ordinal frames, start 0, stride 1**: `vsr/evaluate_external.py:26–34` sets the 17-frame default; `vsr/external_data.py:315–340,347–360` slices `sources[start:start+num_frames]` and records consecutive source indices. The paused external run's frozen protocol and sequence metadata confirm the 17-frame prefix; this is not a full-sequence sliding-window evaluation. The external protocol therefore differs from training in both frame spacing and clip length. Preserve existing result labels and protocol; any future matched-spacing comparison must be declared separately. This evidence answers current behavior; the PROPOSED future spacing still awaits human settlement.

### D3. LR branch injection points and frame separation (SETTLED)

**Accepted review refinement, 2026-11-05.** The human authorized continuation following the attached independent review. A2 starts with **latent-level corruption at the LR branch input**, excluding latent position0 from random masks because the first causal VAE position has distinct support. Pixel-level corruption remains an optional later ablation after a crop-scale VAE leakage test; it is not on the pilot critical path. Layer9/21 choices remain unchanged. Evaluation with pixel-degraded LR remains a separate test of restoration and must not be described as an exactly local latent intervention.

**Question.** Which transformer blocks receive the LR residual branch, and does the branch mix information across frames before injection (for example through temporal convolution or attention)?

**Rules that follow.**
- The student alignment block must be at or after the first injection block, or the corruption never reaches the aligned features.
- The teacher block must be deeper than the student block.
- If the branch mixes frames before injection, corruption leaks into clean frames. Then corrupt at the branch input and keep the corruption per frame.

**Default to adjust.** Self-Flow uses student and teacher blocks at about 0.3 and 0.7 of depth (blocks 9 and 21 of 30). Choose blocks relative to the injection points instead of copying these.

**Evidence needed.** Code inspection (T0.1).

**2026-11-05 proposal — awaiting the human.** Keep LR injections before zero-based blocks 0/5/9. Start the future self-distillation pilot with **student output after block 9 and teacher output after block 21** (zero-based), so the student follows all existing injections and the teacher is deeper. This is an initial experiment setting, not an empirically selected best layer. The residual stem does not mix latent time positions, while the upstream clip VAE does; pixel-level corruption must not be asserted to affect exactly one isolated latent frame. Stage 3 must test the actual corruption support and distinguish pixel versus latent-branch masking. No such corruption or EMA implementation exists yet.

**T0.1 evidence, 2026-11-05.** LR residuals are injected **before zero-based transformer blocks 0, 5 and 9** (the 1st, 6th and 10th of 30 blocks). `configs/formal_hq_v2_noref.json:25–36` declares the points and alignment block 9; `vsr/model.py:99–119` registers forward **pre**-hooks and adds each projected condition to that block's input. Existing external-DINO alignment captures the **output after block 9**, via `vsr/model.py:101–102,122–125,149–153`, and therefore follows all three injections. All formal configs use these same points.

The latent LR stem itself does **not** mix latent time positions: `vsr/model.py:78–84` creates Conv3d with kernel/stride equal to the native DiT patch size plus SiLU, then independent token-wise Linear projections; `vsr/model.py:136–141` forms the token grid. The original official model's `transformer/config.json:13–19` declares 30 blocks and patch size `[1,2,2]`, also preserved in each formal run's `correspondence.json:4–8`, so the stem's temporal kernel and stride are both 1. The upstream clip VAE does encode across time, however: absence of stem mixing must not be described as absence of pixel-frame mixing throughout the LR path. The VAE receptive field and pixel-corruption spread need T1.1 evidence. A future student hook must follow an injection; its EMA teacher hook must be deeper than the student hook. No EMA layer choice is settled by this audit.

### D4. Start point, EMA, scope and primary metrics (SETTLED)

**Recommendation.**
- **Start:** continue from each NoRef Base checkpoint. Every arm (flow-only, C1, A2, baselines) then shares an identical start.
- **EMA:** keep the EMA over trainable parameters only (LoRA and LR branch), initialized to Base, with decay 0.999. Self-Flow's 0.9999 has a half-life of about 6,900 updates and barely moves the teacher within these short runs.
- **Scope:** the main line uses NoRef only. HRI doubles every run, so it becomes an optional extension.
- **Primary metrics, to fix before Stage 5:** LPIPS on corrupted frames from the frame-varying test set, plus one temporal metric (tLPIPS or E*warp). Significance by paired bootstrap across clips and seeds, p < 0.05. Fidelity (PSNR and SSIM on clean REDS4 and YouHQ40) must not drop.

**2026-11-05 concrete proposal — awaiting the human.** Keep the main line **NoRef only** and EMA decay **0.999** over LoRA/LR-branch trainables only, initialized from the matched Base. If D1 changes to 33 contiguous frames, first train the three protocol-matched NoRef Bases in T2.1 from the official pretrained model; each later flow-only/C1/A2/baseline arm starts from its corresponding same-seed Base checkpoint. Keep C1 mandatory. No Stage 2 training has started.

Propose the precisely implemented **LPIPS Alex0.1 on corrupted frames** plus **TecoGAN tLP100** as the temporal primary, rather than calling an unavailable implementation E*warp or silently using tLPIPS as an alias. Define tLP100 as 100× the existing TecoGAN absolute difference between predicted and target adjacent-frame LPIPS distances, retaining its cutfr=2/crop/quantization protocol. Keep clean-set RGB PSNR/SSIM (zero border shave, common PNG-equivalent q/255 inputs) as fidelity guards and label them RGB. Any additional Y-channel metrics are separate until implemented and frozen; do not mix them into historical RGB results. Final test intervals, corrupted-frame/pair pools, uncertainty tests and fidelity criteria must still be recorded prospectively in T5.0 before Stage 5. These metric choices change no existing result or settled S1–S6 decision and require human acceptance.

The primary validation crossed bootstrap currently includes zero for all five metrics in both main family contexts; this is exploratory evidence from three training seeds, not a Stage 5 significance decision or proof of no effect. The native-resolution VAE attempt stopped with OOM in its common untiled decoder. E*warp/tLPIPS definition, encoding-tiling comparison and full-update/actual-EMA costs remain explicitly incomplete. GPU work is stopped; these proposals do not settle the decisions or authorize a retry of the failed protocol.

## Settled (do not reopen without new evidence)

- **S1. Direction.** The method is temporal self-distillation with an EMA self-teacher, based on the information asymmetry idea in Self-Flow. External DINOv2 alignment (Align, Relation) and Wan VAE alignment are baselines, not the method.
- **S2. Not the HR VAE latent as the teacher.** The flow-matching velocity target already encodes the clean HR latent, so aligning to it is mostly deep supervision. It stays only as baseline B2, with B1 (deep supervision) as its control.
- **S3. A2 is tested first.** Raising latent noise on a frame (A1) leaves that frame's own LR condition intact, so it does not force cross-frame use. A1 and A3 come only after A2 shows a signal, and only with 5 or more latent frames.
- **S4. C1 is mandatory.** The augmentation-only control appears in every comparison involving A2.
- **S5. Existing DINOv2 results are kept.** They are reported with seed statistics as the paper's motivation: external alignment is unreliable at this scale.
- **S6. Hardware budget (historical; updated2026-11-06 below).** Original budget was one16GB GPU. Existing training uses128×224 crops; full-resolution evaluation uses tiling. The remote H100 role update does not itself change crop/frame/batch/scientific settings.

## Decision log

| Date | Decision | Outcome | Evidence |
| --- | --- | --- | --- |
| 2026-11-05 | S1–S6 | Settled during planning | Planning discussion; RESEARCH_PLAN.md |
| 2026-11-05 | D1–D3 / T0.1 | Existing behavior established from code and historical metadata; all remain OPEN for future protocol decisions | Whole-clip 9→3 VAE encoding; internal stride 4 vs external contiguous 17-frame prefixes; blocks 0/5/9 pre-injection and block 9 post-capture. Source paths in AGENTS.md; no training or protocol change. |
| 2026-11-05 | D1–D4 / T1.4 | PROPOSED, not SETTLED; awaiting the human after completed bounded crop/resource audits and a stopped native-resolution OOM | Prefer 33 contiguous frames/9 latent positions, matched stride1, zero-based student9/teacher21, NoRef matched Base continuations with trainable-only EMA0.999 and explicitly named LPIPS/tLP100 candidates. Full evidence and limitations in `docs/stage1_audit_20261105/`; no training run or parameter update. |

| 2026-11-05 | D1–D4 / G1 | SETTLED by explicit human acceptance of the 33-frame proposal; authorize T2.1 three fresh NoRef Bases and normal planned continuation after passing checks | Human message quoted above; bounded audit limitations and unclosed G0 remain recorded. |

| 2026-11-05 | Review refinements / G0 scope | Accepted by human: descope old external backlog; close native untiled and unavailable E*warp/tLPIPS sub-items out of scope; latent-level A2 default excluding position0; retain three Base seeds | Pasted independent review and “可以按照这个思路继续运行，没有问题就持续推进”. Existing evidence is retained and no unrun task is marked experimentally passed. |


## Pilot G4 operational definitions — frozen before method scores,2026-11-06

The human authorized the research plan and normal continuation. The agent records this prospective implementation of its pilot G4 rule before any method training or quality scores; the exact pool/reference formulas were not separately specified by the human. A separate read-only review found the mapping consistent with the minimal interpretation of the accepted plan. Definitions are frozen in `configs/stage4_g4_definitions_20261106.json`, raw SHA256 `d552dbb136223d246e05af202f25b072a2c822ad4eeaecead0aa0d6360b1ca58`.

Recovery uses masked LPIPS: Delta=mean(C1)-mean(A2), strictly greater than the sampleSD(ddof1) of the three Base seed means on the same corrupt-masked LPIPS and five-type macro pool. Average sampler42/43 within clip, then all32 clips and all five types equally. Clean fidelity uses all33 frames and requires unrounded A2-minus-C1 RGB PSNR and SSIM means each>=0, with zero tolerance. All types, temporal metrics, masked PSNR, flow-only and matched-Base comparisons are reported. Temporal improvement is not an extra pilot AND condition.

The method experiment remains one training seed. Report10000 paired-clip bootstrap draws with seed20261102 and95% percentile intervals as descriptive conditional uncertainty; no p-value or extra bootstrap gate is introduced. G4 failures/problematic results follow the existing stop/report rule;4096 updates are a subsequent planned option requiring a concrete protocol after that report. T5.0 retains its separate formal metric/statistical/fidelity freeze.

The actual T2 evaluation protocol does not yet exist. Before method scoring, a new immutable executable criteria file will bind that completed protocol SHA and these unchanged definitions/rawSHA. No placeholder criteria are executed and no new human-only approval flow is added.


## Desktop startup policy — human adopted,2026-11-06

After the original T2 startup guard stopped before model loading(0restores), the human selected: “采用已审查的桌面兼容修正，在新目录完整重启评价（推荐）”. Adopt the reviewed candidate: require one completely pinned gjs, permit additional Nautilus/Ptyxis only through the unchanged strict classifier, retain256MiB/UID/system executable/inode/mode/GNOME-cgroup checks, reject unknown/ML/duplicate/secondgjs/missinggjs inventories. Preserve the failed protocol/run and use a new prospective protocol SHA/run for the full1152 matrix. Scientific data/checkpoints/numerical restore/sampler/masks/metrics/G2 criteria are unchanged. This approval does not authorize automatic retries after a new failure.

## Hardware roles and planning-only scope — human settled,2026-11-06

The human confirmed an H10080GB on another machine and requested: “按H100做主实验、5080做开发验证…整理一份新的任务计划…先不要启动实验。” This updates the old single16GB hardware constraint: H100 is the future main training/official evaluation platform; RTX5080 remains the development/validation platform. Remote access, storage, environment and assets have not been verified or migrated. The current authorization is documentation/planning only; all experiments remain stopped and automationvsr PAUSED.

Read `docs/H100_MAIN_EXPERIMENT_PLAN_20261106.md` for completed tasks, migration inventory and M1–M3/E2–E6 dependencies. Preserve the three audited8192-step Bases; no hardware-triggered retraining. Initial scientific frame/crop/degradation/split/effective-batch/optimizer/teacher settings and G2/G3/G4 definitions remain unchanged. A future H100 run needs new reviewed source/environment/path/host bindings and a fresh full1152 Base evaluation. Retain the5080 stopped228-row prefix as history without cross-host row stitching. Any actual migration, GPU compatibility probe or experiment awaits a later execution instruction; larger inputs, distributed training and a cross-host teacher are not silently added.

## Ten-module organization and two-host development workflow — 2026-11-06

The human accepted planning the experiment around data.py,vae.py,model.py,teacher.py,losses.py,train.py,infer.py,evaluate.py,metrics.py,checkpoints.py and requested a GitHub-facing README explaining each Python file. The root answered the workflow question before editing: local5080 development/maintenance/relevant small checks, publish an identified Git commit, remoteH100 fetch that fixed commit for prospective compatibility checks and full experiments, then export authorized text records. Each host keeps private assets/run directories; active experiments do not follow moving development code.

The updated README explicitly labels the10 modules as the target organization; source consolidation has not occurred. `docs/CODE_ORGANIZATION.md` defines R0–R5 responsibilities/refactor/validation and `docs/PYTHON_FILE_INDEX.md` explains every existing tracked Python file. CPU tests, GPU smoke tests and complete scientific runs are distinct evidence. Current work remains documentation-only with experiments/automation paused; no GPU launch, H100 transfer or immediate server compatibility is claimed.

## Actual ten-module consolidation — human authorized,2026-11-06

The human then explicitly instructed actual source consolidation, preservation of old code/experiment records, relevantCPU tests, README update and GitHub sync, without GPU experiments or H100 migration. This supersedes the preceding planning-only source status. Current stagingvsr has the10 accepted business modules plusinit; implementations are direct and do not import/archive-load legacy code. Oldvsr/scripts/tests144 files were moved byte-identically underlegacy/pre_refactor_20261106 with a source-commit/hash manifest; all original configs/checkpoints/results and actualruntime41407b0 stay unchanged.

Unified data/train/infer/evaluate CLIs and CPU failure/source/complete-matrix checks are implemented. The full related130CPU suite passed with0fail/0skip,CUDAnot initialized. The three real Base task payloads were restored exactly into CPU shape-only fixtures; this is checkpoint compatibility, not fullWan/GPU quality verification. Read `docs/code_refactor_20261106/cpu_validation.json` and README. GPU numerical/resource gates, H100 environment/admission and formal science outcomes remain unmeasured. New code requires new prospective protocols/runs; no oldfailed/stopped run is resumed. AutomationremainsPAUSED.

## Local5080 bounded development smoke — authorized,2026-11-07

The human instructed a local5080 smallGPU validation of model loading, a few forward/backward updates, save and inference closure, then records/GitHub sync on success, explicitly without full experiments or H100 migration. Adopt the prospective brief `docs/development_smoke_20261107/protocol_brief.md`: matchedBase01, first3cachedtrain clips, exactly3A2 diagnosticupdates followed by save/studentreload and one33frame nativeUniPC20 restore. This is0formalRE,notG2/G3/G4 and not method quality evidence. Natural first3time draws and originalscientificsettings are fixed; no active-step selection, severity adjustment, automatic retry or diagnostic-weight reuse. Full experiments and automation remain stopped/PAUSED after this bounded task.

The singledeclared attempt completed successfully with3diagnosticupdates/0scientificupdates/1restore and independentbytehash/source/matrix-budget review. Studentreload is exact, trainingonlystate is excluded from inference,33PNG/MP4 checks pass. This closes only the localdevelopment-smoke subtask; it does not settle a quality claim, passG2/G3/G4, authorize H100 access/migration or resume fullGPU work. ServiceRestart=no/process exited, automation remainsPAUSED. Use a new human instruction for a subsequent GPU/remote task.


## Local execution resumed — human settled,2026-11-07

The human instructed “不太行，还是继续在本地完成这个任务吧…继续，不用问我，你可以自行决定继续”. Return the existing main experiment plan to localRTX5080, defer H100 migration and permit normal gated continuation plus routine engineering decisions without repeated confirmation. Three completedBases and D1–D4/scientific settings/G2/G3/G4 remain fixed. Use a fresh10-module full1152 Base matrix; preserve the oldstopped228 rows without reuse.

The metadata-only host preflight found Nautilus now uses the same-user root-owned system executable under the `org.freedesktop.FileManager1` DBus activation path, which the oldgnome-substring sharedguard rejected. Add this narrowly verified endpoint and tighten all desktop cgroups to fullsame-UID unified paths, matchingparent/childbus+endpoint, exactPtyxisscopePID andat-most-onegjs. This FileManager1 endpoint is a newly reviewed engineeringcompatibility addition, not a claim that the oldclassifier already accepted it. Fullpin/executablehash/inode/mode/start/UID/256MiB and unknown/ML/duplicatePID rejection remain.40relatedCPU cases and independentAST review passed; allrestoration/training/metric numerics unchanged. NoGPUwork happened during the rejected preflight.

Fresh localrun `base_evaluation_local_20261107T093027Z`, protocol `e63681c96413edddefb3a7077679e9d3d3077fdceb4a6f80135732a703c2aea7`, is running; currentcompletegatepending. Keepall10numericboundmodules anddependencies fixed through the matrix. Scientificfailure/nonfinite/OOM stopsdependentGPUwork forreporting; no silentprotocolchange orautomaticretry.
