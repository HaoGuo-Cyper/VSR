# Decisions

Status key:
- **OPEN**: the human must settle it.
- **PROPOSED**: the agent has added evidence and a recommendation; awaiting the human.
- **SETTLED**: do not reopen without new evidence.

## Open: these block Stage 2 onward

### D1. Frame encoding and clip length (OPEN)

**Question.** Are the 9 stride-4 frames encoded by the Wan VAE as one clip, giving 3 latent frames, or frame by frame, giving 9 latent frames?

**Why it matters.** As one clip, each latent frame mixes frames 4 apart, covering 16 original frames. Corrupting one latent frame then removes a third of the clip, and its neighbours are far away, so cross-frame recovery is hard to learn or measure. Frame by frame, there are 9 latent frames, but the VAE's temporal compression is no longer used.

**Recommendation.** Move to contiguous frames, 17 or 33 (giving 5 or 9 latent frames), if they fit in 16 GB. This requires retraining NoRef Base with 3 seeds (3 RE) but makes "neighbouring frames" meaningful.

**Evidence needed.** Code inspection (T0.1); VAE audit (T1.1); VRAM and time profile (T1.3).

### D2. Train/test frame spacing (OPEN)

**Question.** How are test sequences fed at inference: consecutive frames, or stride 4 as in training?

**Why it matters.** Training at stride 4 and testing on consecutive frames means test motion is 4× smaller than training motion. Whatever protocol is chosen, training and test must use the same spacing.

**Evidence needed.** Code inspection (T0.1).

### D3. LR branch injection points and frame separation (OPEN)

**Question.** Which transformer blocks receive the LR residual branch, and does the branch mix information across frames before injection (for example through temporal convolution or attention)?

**Rules that follow.**
- The student alignment block must be at or after the first injection block, or the corruption never reaches the aligned features.
- The teacher block must be deeper than the student block.
- If the branch mixes frames before injection, corruption leaks into clean frames. Then corrupt at the branch input and keep the corruption per frame.

**Default to adjust.** Self-Flow uses student and teacher blocks at about 0.3 and 0.7 of depth (blocks 9 and 21 of 30). Choose blocks relative to the injection points instead of copying these.

**Evidence needed.** Code inspection (T0.1).

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
