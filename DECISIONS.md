# Decisions

Status key:
- **OPEN**: the human must settle it.
- **PROPOSED**: the agent has added evidence and a recommendation; awaiting the human.
- **SETTLED**: do not reopen without new evidence.

## Open: these block Stage 2 onward

### D1. Frame encoding and clip length (OPEN)

**Question.** Are the 9 stride-4 frames encoded by the Wan VAE as one clip, giving 3 latent frames, or frame by frame, giving 9 latent frames?

**Why it matters.** With clip encoding there are only three latent time positions. Nominal VAE anchors are four sampled frames apart (16 source-frame intervals under training stride 4), but those anchors are not a measurement of disjoint VAE receptive fields. Corrupting one latent position affects a third of the available latent positions. Frame-by-frame encoding would retain nine latent positions but changes the pretrained VAE's temporal use.

**Recommendation.** Move to contiguous frames, 17 or 33 (giving 5 or 9 latent frames), if they fit in 16 GB. This requires retraining NoRef Base with 3 seeds (3 RE) but makes "neighbouring frames" meaningful.

**Evidence needed.** Code inspection (T0.1); VAE audit (T1.1); VRAM and time profile (T1.3).

**T0.1 evidence, 2026-11-05.** The existing main HR/LR video is encoded as one clip, yielding **3 latent frames**. `configs/formal_hq_v2_noref.json:13–23` fixes 9 frames and temporal stride 4; `vsr/data.py:169–204` selects a 33-source-frame span and constructs `[1,3,T,H,W]`. `vsr/preflight.py:77–100` passes the whole HR clip and whole upsampled LR clip separately to `vae.encode(...).latent_dist.mode()`. `vsr/alignment.py:35–55` derives temporal compression 4 and `(9−1)/4+1=3` latent positions from the official model config. The original run's `runs/formal_hq_v2_noref_20261102T142214.588571Z/correspondence.json:2–24` records latent grid `[3,16,28]`, token grid `[3,8,14]`, and nominal anchors `[0,4,8]`.

This is consistent across all eight `configs/formal*.json` files (including the template and remaining-variants config) and all seven actual formal-run correspondence records. HRI's independent **reference-image** cache is a separate path: `vsr/reference_training.py:98–105,142–157` encodes each possible supplied reference as T=1; it does not change main HR/LR clip encoding. Cached main-video latents are checked against the derived geometry in `vsr/train.py:97–125`. T1.1/T1.3 remain unrun in this handoff; the future clip protocol is still OPEN.

### D2. Train/test frame spacing (OPEN)

**Question.** How are test sequences fed at inference: consecutive frames, or stride 4 as in training?

**Why it matters.** Training at stride 4 and testing on consecutive frames changes the temporal spacing by a factor of four; the actual motion difference depends on each source. Whatever future protocol is chosen, training and test spacing must be considered together.

**Evidence needed.** Code inspection (T0.1).

**T0.1 evidence, 2026-11-05.** Internal HQ inference uses the training spacing: `vsr/evaluate_hq.py:24–26,49–62` sets 9 frames and temporal stride 4, and `vsr/hq_evaluation_data.py:117–135` delegates to the same strided decoder. External REDS4/YouHQ40 inference instead selects **17 consecutive ordinal frames, start 0, stride 1**: `vsr/evaluate_external.py:26–34` sets the 17-frame default; `vsr/external_data.py:315–340,347–360` slices `sources[start:start+num_frames]` and records consecutive source indices. The paused external run's frozen protocol and sequence metadata confirm the 17-frame prefix; this is not a full-sequence sliding-window evaluation. The external protocol therefore differs from training in both frame spacing and clip length. Preserve existing result labels and protocol; any future matched-spacing comparison must be declared separately. This evidence answers current behavior, while the future protocol remains OPEN.

### D3. LR branch injection points and frame separation (OPEN)

**Question.** Which transformer blocks receive the LR residual branch, and does the branch mix information across frames before injection (for example through temporal convolution or attention)?

**Rules that follow.**
- The student alignment block must be at or after the first injection block, or the corruption never reaches the aligned features.
- The teacher block must be deeper than the student block.
- If the branch mixes frames before injection, corruption leaks into clean frames. Then corrupt at the branch input and keep the corruption per frame.

**Default to adjust.** Self-Flow uses student and teacher blocks at about 0.3 and 0.7 of depth (blocks 9 and 21 of 30). Choose blocks relative to the injection points instead of copying these.

**Evidence needed.** Code inspection (T0.1).

**T0.1 evidence, 2026-11-05.** LR residuals are injected **before zero-based transformer blocks 0, 5 and 9** (the 1st, 6th and 10th of 30 blocks). `configs/formal_hq_v2_noref.json:25–36` declares the points and alignment block 9; `vsr/model.py:99–119` registers forward **pre**-hooks and adds each projected condition to that block's input. Existing external-DINO alignment captures the **output after block 9**, via `vsr/model.py:101–102,122–125,149–153`, and therefore follows all three injections. All formal configs use these same points.

The latent LR stem itself does **not** mix latent time positions: `vsr/model.py:78–84` creates Conv3d with kernel/stride equal to the native DiT patch size plus SiLU, then independent token-wise Linear projections; `vsr/model.py:136–141` forms the token grid. The original official model's `transformer/config.json:13–19` declares 30 blocks and patch size `[1,2,2]`, also preserved in each formal run's `correspondence.json:4–8`, so the stem's temporal kernel and stride are both 1. The upstream clip VAE does encode across time, however: absence of stem mixing must not be described as absence of pixel-frame mixing throughout the LR path. The VAE receptive field and pixel-corruption spread need T1.1 evidence. A future student hook must follow an injection; its EMA teacher hook must be deeper than the student hook. No EMA layer choice is settled by this audit.

### D4. Start point, EMA, scope and primary metrics (OPEN)

**Recommendation.**
- **Start:** continue from each NoRef Base checkpoint. Every arm (flow-only, C1, A2, baselines) then shares an identical start.
- **EMA:** keep the EMA over trainable parameters only (LoRA and LR branch), initialized to Base, with decay 0.999. Self-Flow's 0.9999 has a half-life of about 6,900 updates and barely moves the teacher within these short runs.
- **Scope:** the main line uses NoRef only. HRI doubles every run, so it becomes an optional extension.
- **Primary metrics, to fix before Stage 5:** LPIPS on corrupted frames from the frame-varying test set, plus one temporal metric (tLPIPS or E*warp). Significance by paired bootstrap across clips and seeds, p < 0.05. Fidelity (PSNR and SSIM on clean REDS4 and YouHQ40) must not drop.

## Settled (do not reopen without new evidence)

- **S1. Direction.** The method is temporal self-distillation with an EMA self-teacher, based on the information asymmetry idea in Self-Flow. External DINOv2 alignment (Align, Relation) and Wan VAE alignment are baselines, not the method.
- **S2. Not the HR VAE latent as the teacher.** The flow-matching velocity target already encodes the clean HR latent, so aligning to it is mostly deep supervision. It stays only as baseline B2, with B1 (deep supervision) as its control.
- **S3. A2 is tested first.** Raising latent noise on a frame (A1) leaves that frame's own LR condition intact, so it does not force cross-frame use. A1 and A3 come only after A2 shows a signal, and only with 5 or more latent frames.
- **S4. C1 is mandatory.** The augmentation-only control appears in every comparison involving A2.
- **S5. Existing DINOv2 results are kept.** They are reported with seed statistics as the paper's motivation: external alignment is unreliable at this scale.
- **S6. Hardware budget.** One 16 GB GPU. Training uses 128×224 crops; full-resolution evaluation uses tiling.

## Decision log

| Date | Decision | Outcome | Evidence |
| --- | --- | --- | --- |
| 2026-11-05 | S1–S6 | Settled during planning | Planning discussion; RESEARCH_PLAN.md |
| 2026-11-05 | D1–D3 / T0.1 | Existing behavior established from code and historical metadata; all remain OPEN for future protocol decisions | Whole-clip 9→3 VAE encoding; internal stride 4 vs external contiguous 17-frame prefixes; blocks 0/5/9 pre-injection and block 9 post-capture. Source paths in AGENTS.md; no training or protocol change. |
