# Training-seed uncertainty supplement (2026-11-05)

Exploratory, descriptive val32 analysis only. This does not close T0.4, test Stage 5 significance, or provide a p-value.

Three matched Base/Align training-seed pairs (20261102/03/04), the same 32 validation clips/32 source groups, and sampling seeds 42/43 averaged within each clip. NoRef correct/all-frames and HRI ref0/ref4 × on/off keep their original measurement pools and remain separate. Ref0/on is HRI's predeclared primary reference context. No frozen test211 or external results enter this supplement.

Each of 10,000 replicates independently draws three training seeds and 32 source groups with replacement (NumPy default_rng seed 20261105); it evaluates their Cartesian product. All metrics, models and contexts use the same paired draws. The new 95% percentile intervals include resampling of the observed training-seed pairs and source groups. The previous conditional intervals resample sources after averaging the three fixed seeds. The previous files remain unchanged.

There are only three training-seed pairs. Their empirical distribution is sparse, so the new intervals are exploratory and coarse; neither crossing nor excluding zero is a confirmatory benefit or equivalence decision. Mean ± SD uses sample SD across the three per-training-seed paired deltas (ddof=1). Positive Δ favors Align for PSNR/SSIM; negative Δ favors Align for LPIPS/tLP100/tOF. No multiplicity correction or p-values are used.

## HQ_val32/HRI/ref0/hri_off

| Metric | seed02 Δ | seed03 Δ | seed04 Δ | Mean Δ ± SD | Crossed 95% CI | Previous conditional 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| psnr_rgb_db | -0.003516 | +0.012942 | +0.037651 | +0.015692 ± 0.020721 | [-0.013896, +0.048425] | [-0.002484, +0.034219] |
| ssim_rgb | +0.000451 | -0.000047 | +0.005511 | +0.001972 ± 0.003075 | [-0.000637, +0.005280] | [+0.000969, +0.003018] |
| lpips_alex_v0_1 | -0.001620 | +0.000507 | -0.004689 | -0.001934 ± 0.002612 | [-0.004715, +0.000875] | [-0.002970, -0.000926] |
| tLP100_alex_v0_1_TecoGAN_crop | -0.013733 | +0.014897 | -0.011607 | -0.003481 ± 0.015952 | [-0.028135, +0.023514] | [-0.018496, +0.011016] |
| tOF_Farneback_TecoGAN_crop | -0.007665 | +0.001543 | -0.000467 | -0.002196 ± 0.004841 | [-0.010638, +0.004372] | [-0.006804, +0.002593] |

## HQ_val32/HRI/ref0/hri_on

| Metric | seed02 Δ | seed03 Δ | seed04 Δ | Mean Δ ± SD | Crossed 95% CI | Previous conditional 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| psnr_rgb_db | -0.006837 | +0.040988 | -0.016138 | +0.006004 ± 0.030652 | [-0.030358, +0.045103] | [-0.011150, +0.023372] |
| ssim_rgb | -0.002212 | +0.000952 | +0.003633 | +0.000791 ± 0.002926 | [-0.002082, +0.003741] | [-0.000126, +0.001748] |
| lpips_alex_v0_1 | -0.002572 | +0.001241 | -0.004842 | -0.002058 ± 0.003074 | [-0.005079, +0.001232] | [-0.003008, -0.001137] |
| tLP100_alex_v0_1_TecoGAN_crop | -0.026435 | +0.004970 | -0.012946 | -0.011470 ± 0.015754 | [-0.039057, +0.016096] | [-0.027141, +0.003277] |
| tOF_Farneback_TecoGAN_crop | -0.009420 | +0.003846 | +0.002787 | -0.000929 ± 0.007372 | [-0.013884, +0.008059] | [-0.008340, +0.006198] |

## HQ_val32/HRI/ref4/hri_off

| Metric | seed02 Δ | seed03 Δ | seed04 Δ | Mean Δ ± SD | Crossed 95% CI | Previous conditional 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| psnr_rgb_db | -0.000118 | +0.010374 | +0.034746 | +0.015001 ± 0.017886 | [-0.013750, +0.045741] | [-0.003536, +0.033989] |
| ssim_rgb | +0.000551 | -0.000041 | +0.005518 | +0.002009 ± 0.003053 | [-0.000602, +0.005288] | [+0.001004, +0.003058] |
| lpips_alex_v0_1 | -0.001633 | +0.000661 | -0.004899 | -0.001957 ± 0.002794 | [-0.004909, +0.000965] | [-0.002973, -0.000963] |
| tLP100_alex_v0_1_TecoGAN_crop | +0.009624 | +0.050677 | -0.019185 | +0.013705 ± 0.035109 | [-0.033099, +0.059593] | [-0.007558, +0.036355] |
| tOF_Farneback_TecoGAN_crop | -0.012999 | -0.001636 | +0.000009 | -0.004875 ± 0.007083 | [-0.015305, +0.003196] | [-0.009400, -0.000925] |

## HQ_val32/HRI/ref4/hri_on

| Metric | seed02 Δ | seed03 Δ | seed04 Δ | Mean Δ ± SD | Crossed 95% CI | Previous conditional 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| psnr_rgb_db | -0.092005 | -0.041845 | +0.078075 | -0.018592 ± 0.087392 | [-0.092673, +0.068261] | [-0.033892, -0.002994] |
| ssim_rgb | -0.002777 | -0.002495 | +0.008161 | +0.000963 ± 0.006236 | [-0.003406, +0.007548] | [+0.000030, +0.001959] |
| lpips_alex_v0_1 | -0.001448 | +0.000658 | -0.005823 | -0.002204 ± 0.003306 | [-0.005582, +0.001032] | [-0.003179, -0.001268] |
| tLP100_alex_v0_1_TecoGAN_crop | -0.005167 | +0.013897 | +0.001565 | +0.003432 ± 0.009668 | [-0.035944, +0.041494] | [-0.022518, +0.028883] |
| tOF_Farneback_TecoGAN_crop | -0.011615 | -0.003167 | +0.003838 | -0.003648 ± 0.007738 | [-0.014466, +0.005869] | [-0.008712, +0.000748] |

## HQ_val32/NoRef/all_frames

| Metric | seed02 Δ | seed03 Δ | seed04 Δ | Mean Δ ± SD | Crossed 95% CI | Previous conditional 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| psnr_rgb_db | +0.140453 | +0.061113 | -0.045200 | +0.052122 ± 0.093152 | [-0.038588, +0.135569] | [+0.037157, +0.068398] |
| ssim_rgb | +0.005978 | +0.003081 | -0.001489 | +0.002523 ± 0.003765 | [-0.001170, +0.005991] | [+0.001639, +0.003566] |
| lpips_alex_v0_1 | +0.001089 | +0.000064 | +0.000830 | +0.000661 ± 0.000533 | [-0.001582, +0.002840] | [-0.001368, +0.002462] |
| tLP100_alex_v0_1_TecoGAN_crop | +0.008711 | +0.023022 | -0.022113 | +0.003207 ± 0.023066 | [-0.028323, +0.037519] | [-0.014243, +0.019232] |
| tOF_Farneback_TecoGAN_crop | +0.007655 | +0.006526 | +0.002457 | +0.005546 ± 0.002734 | [-0.000559, +0.013028] | [+0.001212, +0.010572] |

## Interpretation

NoRef Align's mean PSNR/SSIM improvement changes sign across training seeds; its LPIPS and tOF deltas are unfavorable in all three observed seeds. HRI ref0/on changes sign across seeds for every reported metric. The observed seed spread and mixed quality tradeoffs support keeping the current description as 'no stable overall auxiliary-loss benefit established'. They do not prove that the true effect is zero or that every metric worsens. Relation/Both were not included because their training-seed uncertainty cannot be estimated from one run.

The loader checked the existing validation JSON identities, windows, sampling seeds, reference exclusions and stored digest bindings. Underlying checkpoints, targets, RGB tensors, source videos and PNGs were not read or rehashed. Constant-delta and sign-flipping-seed synthetic cases passed; the latter has a zero-width conditional interval but a crossed interval spanning both signs.
