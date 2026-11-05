# Research plan: temporal self-distillation for diffusion VSR

Technical reference for the agent. `DECISIONS.md` takes precedence over anything here; `TASKS.md` says what to do and when.

## 1. Hypothesis and contribution

**Hypothesis.** Frame-structured information asymmetry between a Wan2.1-1.3B VSR student and its EMA teacher strengthens cross-frame aggregation more than external feature alignment (DINOv2, Wan VAE) does.

**Why it could be new.** Self-Flow (Chefer et al., 2026) applies different noise levels to random tokens. A student predicts an EMA teacher's features from the corrupted view. In its video experiments, copying across frames is a shortcut to suppress, so masking is kept to 10% of tokens. In VSR, borrowing from neighbouring frames is exactly the wanted behaviour. Structuring the asymmetry by frame turns that shortcut into the training signal.

**What a paper needs.** The gain must beat the augmentation-only control C1, hold across 3 seeds, and ideally survive few-step or one-step sampling.

**Motivating evidence already in hand.** DINOv2 alignment (Align) and relation matching (Relation) gave mixed results and no stable gain. This matches the expectation that external alignment helps little when fine-tuning a strong pretrained model.

## 2. Setup and notation

- HR clip x of T frames; LR clip y = D(x), 4× bicubic.
- VAE latent z0 = E(x): F = 1 + (T − 1)/4 latent frames when encoded as one clip (D1 checks whether the existing code does this), 16 channels, 8× spatial compression.
- The DiT patchifies 1×2×2, giving 30 blocks of width 1536. A 128×224 crop has 112 tokens per latent frame.
- Conditioning: frozen backbone, rank-8 attention LoRA, and a learned LR residual branch injected into several blocks. The text condition is fixed (precomputed embedding).
- Flow matching, where t = 1 is pure noise:

$$z_t = (1-t)\,z_0 + t\,\epsilon, \qquad \mathcal{L}_{\text{flow}} = \mathbb{E}\,\big\| v_\theta(z_t, t, c) - (\epsilon - z_0) \big\|^2$$

The velocity target ε − z0 already contains the clean HR latent. Aligning intermediate features to the HR VAE latent is therefore mostly deep supervision, which is why the VAE latent is not the teacher (S2).

## 3. Method

### Objective

Choose a set of latent frames to corrupt. The student sees the corrupted input. The EMA teacher, under stop-gradient, sees every frame at the cleaner setting. 𝓜 is the set of tokens where the student's input is more corrupted than the teacher's; for A2, these are the tokens of the corrupted frames.

$$\mathcal{L}_{\text{sd}} = \frac{1}{|\mathcal{M}|}\sum_{n\in\mathcal{M}} \Big(1 - \cos\big(h_\phi(f^{(l)}_\theta(\tilde z, \tau, \tilde c))_n,\ \mathrm{sg}[f^{(k)}_{\bar\theta}(z_{\tau_{\min}}, \tau_{\min}, c)]_n\big)\Big)$$

$$\mathcal{L} = \mathcal{L}_{\text{flow}}(\tilde z, \tau, \tilde c) + \gamma\,\mathcal{L}_{\text{sd}}$$

- **Student block l, teacher block k.** Self-Flow uses 0.3 and 0.7 of depth (9 and 21). Here, choose them relative to the LR injection blocks (D3).
- **Projector h_φ:** a small MLP, as in REPA.
- **Similarity:** cosine. Self-Flow found an ℓ1 loss unstable as feature norms grew.
- **Loss weight:** γ = 0.8 by default.
- **EMA:** over trainable parameters only (LoRA and LR branch). Decay 0.999, initialized from the starting checkpoint (D4).
- **One student pass** carries both losses. The teacher adds one forward pass without gradients, about 1.3–1.4× total compute.
- **Inference is unchanged.**

### Asymmetry variants

| ID | Student sees on corrupted frames | Teacher sees | What it forces | Architecture change | Role |
| --- | --- | --- | --- | --- | --- |
| A2 Frame-condition | LR condition degraded harder (blur, heavy codec, or dropped); same noise level | All frames with normal LR, same noise | Rebuilding a frame's content from neighbours' LR | None | **Main method, run first** |
| A1 Frame-noise | Latent tokens at a second timestep s; own LR intact | All frames at min(t, s) | Borrowing HR detail from cleaner neighbours | Per-frame timestep | Later, needs 5+ latent frames |
| A3 Frame-both | A1 and A2 on the same frames | Clean on both | Strongest cross-frame pressure | Per-frame timestep | Later |
| A0 Token-random | Random tokens at s (Self-Flow as published) | All tokens at min(t, s) | Unstructured context use | Per-token timestep | Control C2 |
| A4 No mask | Every token at t | Every token at a lower t′ | Nothing specific | None | Control C3 |

Why A2 comes first: raising the latent noise on a frame (A1) leaves that frame's own LR condition intact. The student can then recover structure without its neighbours. Corrupting the condition removes that route.

### Design details

- **Mask:** whole latent frames, singly or in spans of 2. Ratio ρ ∈ {0.1, 0.25, 0.4}. Keep an uncorrupted neighbour next to every corrupted span.
- **Corruption, two forms:**
  - pixel-level: extra blur, downscale or a high-CRF codec pass on the LR frames behind a latent frame (realistic);
  - latent-level: replace that frame's condition with matched-variance noise or a learned null token (the cleanest test of aggregation).
- **Timestep window:** at low noise the student can read HR detail straight from z_t, which makes the target trivial. Apply the loss for t > 0.3 by default; sweep all t and t > 0.5.
- **Stability:** EMA plus stop-gradient, cosine loss, LayerNorm without affine on the teacher features. Track student feature std and effective rank to catch collapse.
- **Per-frame timesteps (A1, A3):** compute Wan's time-embedding modulation per latent frame and broadcast it over that frame's tokens, in every block and in the output head. Unit test: equal timesteps on all frames must reproduce the original outputs.

### Controls

| Control | Question it answers |
| --- | --- |
| C1 Same corruption, no alignment loss | Does the gain come from alignment, or just from harder training data? **The most important control.** |
| C2 Random-token mask (A0), same ratio | Does the temporal structure of the mask matter? |
| C3 No mask, cleaner teacher (A4) | Does masking matter at all? |
| C4 Frozen Base teacher instead of EMA | Does a co-evolving teacher matter? |
| C5 Shuffled or unrelated neighbours, at test time only | Does the model actually use its neighbours? |

## 4. Baselines

| ID | Method | Status |
| --- | --- | --- |
| B0 | Flow loss only | Exists as Base |
| B1 | Deep supervision: a head on the student block predicts z0 with MSE | Control for B2 |
| B2 | VAE-latent alignment: student block aligned to the HR VAE latent | Tests the original "VAE teacher" idea |
| B3 | DINOv2-B pointwise alignment (REPA-style) | Exists as Align |
| B4 | DINOv2 spatial and temporal relation matching | Exists as Relation |

All alignment baselines get the same weight sweep {0.25, 0.5, 0.8}, the same steps, the same data order and the same seeds.

## 5. Evaluation

| Group | Metrics | Note |
| --- | --- | --- |
| Fidelity | PSNR, SSIM (Y channel) | Paired sets |
| Perceptual | LPIPS, DISTS | Paired sets |
| No-reference | MUSIQ, CLIP-IQA, DOVER | Optional at this stage |
| Temporal | E*warp (RAFT-based warping error), tLPIPS | Reported against the VAE floor from T1.1 |
| Targeted | PSNR and LPIPS on frames with injected heavy degradation, plus the gap to their neighbours | The test that isolates the claim |
| Mechanism | Attention mass from corrupted-frame queries to other frames; performance drop under C5 | Evidence the model uses its neighbours |

- **Benchmarks:** REDS4, YouHQ40, internal HQ-VSR validation, and the frame-varying test set built in T2.2.
- **Statistics:** mean ± std over 3 seeds; paired bootstrap over clips; significance at p < 0.05 for the primary metrics fixed in D4.

## 6. Risks

| Risk | Early signal | Response |
| --- | --- | --- |
| Gain comes from augmentation, not alignment | C1 matches A2 | Report honestly; the contribution becomes temporally varying degradation training |
| Wan's 3D attention already aggregates well | Small corrupted-frame gap at G2 | Harder settings: longer clips, larger motion, stronger per-frame degradation |
| Collapse or instability | Feature std or effective rank falls; cosine saturates early | Lower γ, warm up γ, check the EMA and stop-gradient |
| Temporal gains capped by the VAE | Temporal metrics near the VAE floor | Report against the floor |
| LoRA rank 8 too small for any representation objective | No effect for A2 or the baselines | Rank 8 vs 32 ablation (T6.1) |
| Too few latent frames to aggregate | D1 shows 3 latent frames | Contiguous 17 or 33 frames |

## 7. Novelty positioning

- **vs Self-Flow:** text- and class-conditioned generation, random-token asymmetry, low masking for video. Here: conditional restoration, fine-tuning, frame-structured asymmetry in the condition, where cross-frame copying is the goal.
- **vs REPA, VAE-REPA, InstanceRSR:** external or VAE targets, for generation or single images. Here: no external encoder, video, temporal.
- **vs CREPA, VideoREPA:** cross-frame alignment with external encoders, for generation.
- **vs FlashVSR, DOVE, SeedVR2:** no representation objective; the method is compatible with all three.
- Claim "first" only after a full literature search, as "to our knowledge".

## 8. References

- Self-Flow (Chefer et al., 2026): https://arxiv.org/abs/2603.06507
- REPA (Yu et al., ICLR 2025): https://arxiv.org/abs/2410.06940
- REPA-E (Leng et al., 2025): https://arxiv.org/abs/2504.10483
- VAE-REPA (2026): https://arxiv.org/abs/2601.17830
- CREPA (Hwang et al., 2025): https://arxiv.org/abs/2506.09229
- InstanceRSR (2026): https://arxiv.org/abs/2603.24240
- Align4Gen (2025): https://arxiv.org/abs/2509.09547
- Spatial Gram Alignment, on REPA in pretrained LDMs (2026): https://arxiv.org/abs/2605.20808
- FlashVSR (Zhuang et al., CVPR 2026): https://arxiv.org/abs/2510.12747 (code: https://github.com/OpenImagingLab/FlashVSR)
- DOVE (Chen et al., NeurIPS 2025): https://arxiv.org/abs/2505.16239
