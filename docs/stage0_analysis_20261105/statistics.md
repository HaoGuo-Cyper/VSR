# Stage 0: existing internal HQ statistics (2026-11-05)

Status: partial T0.4. External evaluation coverage is incomplete; these tables do not close Stage 0.

Only existing JSON scores were read. No checkpoint, RGB tensor, video or PNG payload was loaded, and no metric was rescored. JSON bindings and candidate input-contract fields were checked; underlying stored payload digests were not revalidated.

Validation: fixed source-held-out val32, 32 sources, nine frames at stride 4, 128×224, ×4 bicubic, 20-step UniPC/CFG1, sampling seeds 42/43 averaged within each clip. Base/Align have three matched training seeds (20261102/03/04). HRI ref0/on is the predeclared primary reference context; ref4 and off remain separate supplementary contexts. HRI excludes the supplied reference frame and all temporal pairs touching it; NoRef uses all frames. Do not compare these unequal family pools directly.

Frozen internal test: all 211 held-out clips, 180 sources, only training seed 20261102 for all eight models. It uses saved uint8 PNG/255 measurement and independently frozen inference, whereas validation uses saved float32 RGB. Identity/blur/noise/compression/combined are kept separate. Test results are single training seed, indicative; there are no test211 repeated-seed statistics.

Mean ± SD uses equal clip means within each training seed, then equal training-seed weights; SD is sample SD (ddof=1), descriptive with N=3. Delta is candidate minus Base. Paired 95% percentile intervals resample source groups, retaining their clips and matched candidates. For three-seed comparisons each clip's three paired deltas is averaged before resampling sources: the intervals condition on those fixed trained instances and exclude training-seed uncertainty. There are 10,000 draws with seed 20261102, no multiplicity correction and no p-values. In val32 each source has one clip, so source-cluster bootstrap is also paired clip bootstrap.

PSNR/SSIM: higher is better. LPIPS/tLP100/tOF: lower is better. tLP100 is the TecoGAN temporal LPIPS scale, not E*warp or a newly measured temporal metric.

## HQ_test211/HRI/HQ-test-blur/hri_off

211 clips / 180 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1 (single seed, indicative) | 19.839616 | 0.513709 | 0.564071 | 2.551309 | 0.835178 |
| Align | 1 (single seed, indicative) | 19.865040 | 0.513496 | 0.568603 | 2.561720 | 0.838490 |
| Relation | 1 (single seed, indicative) | 19.805964 | 0.509908 | 0.574263 | 2.603222 | 0.845854 |
| Both | 1 (single seed, indicative) | 19.880909 | 0.514317 | 0.569291 | 2.556139 | 0.835322 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=1) | +0.025425 [+0.021340, +0.029433] | -0.000213 [-0.000646, +0.000157] | +0.004531 [+0.003441, +0.005622] | +0.010411 [+0.000390, +0.021100] | +0.003312 [+0.000755, +0.006007] |
| Relation (N=1) | -0.033652 [-0.038348, -0.029071] | -0.003801 [-0.004622, -0.003125] | +0.010192 [+0.008699, +0.011717] | +0.051913 [+0.037935, +0.066394] | +0.010676 [+0.006831, +0.014947] |
| Both (N=1) | +0.041294 [+0.036791, +0.045622] | +0.000608 [+0.000323, +0.000888] | +0.005219 [+0.003574, +0.006924] | +0.004831 [-0.009586, +0.019198] | +0.000144 [-0.002731, +0.003136] |

## HQ_test211/HRI/HQ-test-blur/hri_on

211 clips / 180 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1 (single seed, indicative) | 19.848601 | 0.514498 | 0.560832 | 2.517158 | 0.837236 |
| Align | 1 (single seed, indicative) | 19.854183 | 0.513488 | 0.562924 | 2.530065 | 0.844987 |
| Relation | 1 (single seed, indicative) | 19.783970 | 0.508770 | 0.575571 | 2.566803 | 0.857493 |
| Both | 1 (single seed, indicative) | 19.845790 | 0.512851 | 0.564666 | 2.508374 | 0.847159 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=1) | +0.005582 [+0.001408, +0.009743] | -0.001010 [-0.001484, -0.000613] | +0.002092 [+0.000961, +0.003248] | +0.012908 [+0.002902, +0.023350] | +0.007751 [+0.004550, +0.011139] |
| Relation (N=1) | -0.064631 [-0.070572, -0.058667] | -0.005728 [-0.006614, -0.004997] | +0.014739 [+0.012898, +0.016699] | +0.049645 [+0.033282, +0.066614] | +0.020257 [+0.015500, +0.025398] |
| Both (N=1) | -0.002811 [-0.007863, +0.002007] | -0.001647 [-0.002046, -0.001295] | +0.003834 [+0.002102, +0.005629] | -0.008783 [-0.024623, +0.006625] | +0.009923 [+0.006447, +0.013711] |

## HQ_test211/HRI/HQ-test-combined/hri_off

211 clips / 180 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1 (single seed, indicative) | 19.188908 | 0.463631 | 0.539067 | 2.134342 | 0.855794 |
| Align | 1 (single seed, indicative) | 19.223113 | 0.463966 | 0.542719 | 2.183743 | 0.859325 |
| Relation | 1 (single seed, indicative) | 19.214213 | 0.463506 | 0.545370 | 2.206172 | 0.856894 |
| Both | 1 (single seed, indicative) | 19.246203 | 0.464510 | 0.539081 | 2.172439 | 0.856177 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=1) | +0.034206 [+0.029132, +0.039318] | +0.000336 [-0.000246, +0.000865] | +0.003653 [+0.002098, +0.005224] | +0.049401 [+0.027387, +0.072846] | +0.003531 [+0.000291, +0.006782] |
| Relation (N=1) | +0.025306 [+0.019077, +0.031675] | -0.000125 [-0.001109, +0.000700] | +0.006303 [+0.004194, +0.008506] | +0.071830 [+0.047203, +0.097071] | +0.001100 [-0.002647, +0.005083] |
| Both (N=1) | +0.057295 [+0.051384, +0.063247] | +0.000879 [+0.000325, +0.001385] | +0.000014 [-0.001789, +0.001846] | +0.038098 [+0.012863, +0.064694] | +0.000383 [-0.003652, +0.004633] |

## HQ_test211/HRI/HQ-test-combined/hri_on

211 clips / 180 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1 (single seed, indicative) | 19.206649 | 0.465709 | 0.537813 | 2.162012 | 0.865730 |
| Align | 1 (single seed, indicative) | 19.203077 | 0.463337 | 0.538406 | 2.146532 | 0.869421 |
| Relation | 1 (single seed, indicative) | 19.218975 | 0.466874 | 0.557959 | 2.283883 | 0.884881 |
| Both | 1 (single seed, indicative) | 19.233200 | 0.465742 | 0.537490 | 2.152731 | 0.877936 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=1) | -0.003572 [-0.008225, +0.001061] | -0.002372 [-0.002910, -0.001891] | +0.000593 [-0.000868, +0.002128] | -0.015480 [-0.035417, +0.005992] | +0.003691 [+0.000236, +0.007177] |
| Relation (N=1) | +0.012326 [+0.005722, +0.018902] | +0.001165 [+0.000079, +0.002120] | +0.020145 [+0.017596, +0.022825] | +0.121871 [+0.090837, +0.154219] | +0.019151 [+0.013070, +0.025656] |
| Both (N=1) | +0.026551 [+0.020799, +0.032338] | +0.000033 [-0.000613, +0.000626] | -0.000323 [-0.002402, +0.001813] | -0.009281 [-0.034837, +0.016714] | +0.012206 [+0.007400, +0.017236] |

## HQ_test211/HRI/HQ-test-compression/hri_off

211 clips / 180 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1 (single seed, indicative) | 20.279385 | 0.570611 | 0.238460 | 0.978547 | 0.699891 |
| Align | 1 (single seed, indicative) | 20.302344 | 0.571117 | 0.237390 | 0.969348 | 0.703969 |
| Relation | 1 (single seed, indicative) | 20.337760 | 0.571527 | 0.236420 | 0.968087 | 0.716737 |
| Both | 1 (single seed, indicative) | 20.217088 | 0.567792 | 0.236236 | 0.971595 | 0.722839 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=1) | +0.022959 [+0.011207, +0.034577] | +0.000506 [-0.000347, +0.001350] | -0.001070 [-0.002461, +0.000272] | -0.009199 [-0.026319, +0.007053] | +0.004078 [-0.006609, +0.022544] |
| Relation (N=1) | +0.058374 [+0.045614, +0.072036] | +0.000916 [-0.000080, +0.001889] | -0.002040 [-0.003327, -0.000900] | -0.010460 [-0.028032, +0.007217] | +0.016846 [-0.009630, +0.068280] |
| Both (N=1) | -0.062297 [-0.074606, -0.049608] | -0.002819 [-0.003640, -0.002010] | -0.002225 [-0.003464, -0.001052] | -0.006951 [-0.024723, +0.010470] | +0.022947 [-0.001534, +0.069468] |

## HQ_test211/HRI/HQ-test-compression/hri_on

211 clips / 180 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1 (single seed, indicative) | 20.300936 | 0.574796 | 0.236057 | 0.968676 | 0.707598 |
| Align | 1 (single seed, indicative) | 20.301083 | 0.571168 | 0.238144 | 0.970474 | 0.711040 |
| Relation | 1 (single seed, indicative) | 20.436763 | 0.580533 | 0.237748 | 0.955662 | 0.729564 |
| Both | 1 (single seed, indicative) | 20.339821 | 0.574863 | 0.236506 | 0.963947 | 0.735868 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=1) | +0.000147 [-0.013041, +0.013888] | -0.003628 [-0.004537, -0.002760] | +0.002087 [+0.000648, +0.003498] | +0.001798 [-0.017683, +0.020493] | +0.003443 [-0.004983, +0.016229] |
| Relation (N=1) | +0.135827 [+0.121078, +0.150412] | +0.005737 [+0.004667, +0.006753] | +0.001691 [+0.000213, +0.003043] | -0.013014 [-0.034606, +0.008881] | +0.021967 [+0.001307, +0.060230] |
| Both (N=1) | +0.038885 [+0.025296, +0.053321] | +0.000067 [-0.000819, +0.000926] | +0.000449 [-0.000917, +0.001811] | -0.004729 [-0.024816, +0.015670] | +0.028271 [+0.010665, +0.059731] |

## HQ_test211/HRI/HQ-test-identity/hri_off

211 clips / 180 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1 (single seed, indicative) | 20.968670 | 0.641898 | 0.146722 | 0.855522 | 0.582783 |
| Align | 1 (single seed, indicative) | 20.965482 | 0.641285 | 0.145569 | 0.842174 | 0.569217 |
| Relation | 1 (single seed, indicative) | 21.003692 | 0.641160 | 0.144734 | 0.843637 | 0.580511 |
| Both | 1 (single seed, indicative) | 20.847893 | 0.636940 | 0.146875 | 0.855736 | 0.584606 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=1) | -0.003189 [-0.015031, +0.008160] | -0.000613 [-0.001382, +0.000132] | -0.001153 [-0.002095, -0.000120] | -0.013348 [-0.029093, +0.002905] | -0.013566 [-0.031499, -0.002519] |
| Relation (N=1) | +0.035021 [+0.020611, +0.049196] | -0.000738 [-0.001601, +0.000113] | -0.001988 [-0.002914, -0.001158] | -0.011885 [-0.029533, +0.004229] | -0.002273 [-0.007967, +0.004353] |
| Both (N=1) | -0.120777 [-0.134226, -0.107514] | -0.004957 [-0.005796, -0.004120] | +0.000152 [-0.000937, +0.001231] | +0.000214 [-0.019881, +0.018497] | +0.001822 [-0.002995, +0.007142] |

## HQ_test211/HRI/HQ-test-identity/hri_on

211 clips / 180 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1 (single seed, indicative) | 20.983554 | 0.645038 | 0.146049 | 0.841369 | 0.592401 |
| Align | 1 (single seed, indicative) | 20.972431 | 0.641221 | 0.145989 | 0.836083 | 0.575924 |
| Relation | 1 (single seed, indicative) | 21.103619 | 0.648189 | 0.146032 | 0.822811 | 0.594595 |
| Both | 1 (single seed, indicative) | 20.993819 | 0.642933 | 0.146560 | 0.830225 | 0.598214 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=1) | -0.011123 [-0.024826, +0.002064] | -0.003817 [-0.004697, -0.002999] | -0.000060 [-0.001094, +0.001009] | -0.005286 [-0.023496, +0.012491] | -0.016477 [-0.036860, -0.003511] |
| Relation (N=1) | +0.120065 [+0.105080, +0.135223] | +0.003151 [+0.002185, +0.004097] | -0.000016 [-0.001014, +0.000910] | -0.018559 [-0.035077, -0.002488] | +0.002194 [-0.009706, +0.012299] |
| Both (N=1) | +0.010264 [-0.003724, +0.024157] | -0.002105 [-0.003051, -0.001175] | +0.000511 [-0.000529, +0.001469] | -0.011144 [-0.030041, +0.007200] | +0.005813 [-0.000937, +0.011616] |

## HQ_test211/HRI/HQ-test-noise/hri_off

211 clips / 180 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1 (single seed, indicative) | 20.151584 | 0.498062 | 0.282347 | 1.511476 | 0.834191 |
| Align | 1 (single seed, indicative) | 20.107841 | 0.491785 | 0.288106 | 1.512748 | 0.839396 |
| Relation | 1 (single seed, indicative) | 20.230391 | 0.496066 | 0.277672 | 1.459426 | 0.839540 |
| Both | 1 (single seed, indicative) | 20.086243 | 0.491590 | 0.278467 | 1.502228 | 0.852214 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=1) | -0.043743 [-0.069470, -0.019846] | -0.006277 [-0.008067, -0.004501] | +0.005759 [+0.003377, +0.008165] | +0.001271 [-0.025351, +0.027738] | +0.005205 [-0.000555, +0.011804] |
| Relation (N=1) | +0.078807 [+0.058079, +0.100353] | -0.001996 [-0.003598, -0.000428] | -0.004675 [-0.006775, -0.002608] | -0.052050 [-0.076834, -0.027913] | +0.005349 [-0.002756, +0.014258] |
| Both (N=1) | -0.065341 [-0.090595, -0.040824] | -0.006472 [-0.008559, -0.004276] | -0.003881 [-0.007096, -0.000780] | -0.009249 [-0.039111, +0.019865] | +0.018023 [+0.008602, +0.028500] |

## HQ_test211/HRI/HQ-test-noise/hri_on

211 clips / 180 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1 (single seed, indicative) | 20.269877 | 0.513750 | 0.260554 | 1.404374 | 0.837610 |
| Align | 1 (single seed, indicative) | 20.161562 | 0.495042 | 0.283650 | 1.512346 | 0.845962 |
| Relation | 1 (single seed, indicative) | 20.468605 | 0.522836 | 0.251449 | 1.338641 | 0.854684 |
| Both | 1 (single seed, indicative) | 20.405872 | 0.515716 | 0.256137 | 1.453578 | 0.876364 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=1) | -0.108316 [-0.140826, -0.077996] | -0.018708 [-0.021127, -0.016396] | +0.023096 [+0.020113, +0.025979] | +0.107973 [+0.075521, +0.139337] | +0.008352 [+0.000581, +0.016146] |
| Relation (N=1) | +0.198727 [+0.176321, +0.221595] | +0.009086 [+0.007135, +0.011077] | -0.009105 [-0.011549, -0.006722] | -0.065733 [-0.091589, -0.040148] | +0.017073 [+0.008459, +0.027135] |
| Both (N=1) | +0.135994 [+0.115511, +0.157314] | +0.001966 [-0.000261, +0.004295] | -0.004417 [-0.007412, -0.001444] | +0.049204 [+0.019042, +0.080076] | +0.038754 [+0.031063, +0.047045] |

## HQ_test211/NoRef/HQ-test-blur/no_reference

211 clips / 180 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1 (single seed, indicative) | 19.861221 | 0.513112 | 0.577249 | 2.577531 | 0.836385 |
| Align | 1 (single seed, indicative) | 19.865459 | 0.512538 | 0.579537 | 2.581696 | 0.842524 |
| Relation | 1 (single seed, indicative) | 19.843051 | 0.511414 | 0.574955 | 2.563153 | 0.842342 |
| Both | 1 (single seed, indicative) | 19.838900 | 0.510877 | 0.574379 | 2.574140 | 0.839673 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=1) | +0.004238 [+0.000103, +0.008494] | -0.000574 [-0.000844, -0.000307] | +0.002288 [+0.001096, +0.003458] | +0.004165 [-0.007547, +0.015296] | +0.006138 [+0.002869, +0.009325] |
| Relation (N=1) | -0.018170 [-0.022954, -0.013445] | -0.001698 [-0.002124, -0.001316] | -0.002294 [-0.003464, -0.001182] | -0.014378 [-0.026850, -0.001581] | +0.005957 [+0.002795, +0.009082] |
| Both (N=1) | -0.022321 [-0.026582, -0.018225] | -0.002235 [-0.002649, -0.001884] | -0.002870 [-0.004370, -0.001460] | -0.003391 [-0.014477, +0.008444] | +0.003288 [+0.000457, +0.006221] |

## HQ_test211/NoRef/HQ-test-combined/no_reference

211 clips / 180 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1 (single seed, indicative) | 19.249013 | 0.466121 | 0.549273 | 2.233868 | 0.854326 |
| Align | 1 (single seed, indicative) | 19.255065 | 0.466596 | 0.558847 | 2.296879 | 0.866512 |
| Relation | 1 (single seed, indicative) | 19.239895 | 0.465695 | 0.552511 | 2.203948 | 0.860905 |
| Both | 1 (single seed, indicative) | 19.234917 | 0.465417 | 0.554964 | 2.250656 | 0.864013 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=1) | +0.006051 [+0.002322, +0.009880] | +0.000475 [+0.000171, +0.000770] | +0.009575 [+0.008012, +0.011116] | +0.063011 [+0.043334, +0.083304] | +0.012186 [+0.008321, +0.016169] |
| Relation (N=1) | -0.009118 [-0.013074, -0.005074] | -0.000425 [-0.000882, +0.000007] | +0.003239 [+0.001681, +0.004802] | -0.029920 [-0.048924, -0.012017] | +0.006579 [+0.003089, +0.010186] |
| Both (N=1) | -0.014096 [-0.018595, -0.009469] | -0.000704 [-0.001205, -0.000205] | +0.005691 [+0.004049, +0.007252] | +0.016787 [+0.000832, +0.034123] | +0.009687 [+0.005772, +0.014004] |

## HQ_test211/NoRef/HQ-test-compression/no_reference

211 clips / 180 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1 (single seed, indicative) | 20.334707 | 0.573144 | 0.237250 | 0.975082 | 0.703902 |
| Align | 1 (single seed, indicative) | 20.477903 | 0.579345 | 0.235536 | 0.977557 | 0.705109 |
| Relation | 1 (single seed, indicative) | 20.407327 | 0.576976 | 0.235123 | 0.961328 | 0.708664 |
| Both | 1 (single seed, indicative) | 20.437373 | 0.576164 | 0.235141 | 0.966925 | 0.705496 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=1) | +0.143196 [+0.131718, +0.155291] | +0.006201 [+0.005450, +0.006943] | -0.001714 [-0.002940, -0.000518] | +0.002475 [-0.015647, +0.021130] | +0.001208 [-0.017119, +0.012773] |
| Relation (N=1) | +0.072620 [+0.058650, +0.087146] | +0.003832 [+0.002930, +0.004665] | -0.002127 [-0.003158, -0.001073] | -0.013754 [-0.032114, +0.004784] | +0.004762 [-0.001352, +0.013199] |
| Both (N=1) | +0.102666 [+0.091132, +0.114330] | +0.003020 [+0.002181, +0.003824] | -0.002109 [-0.003013, -0.001222] | -0.008157 [-0.026420, +0.010461] | +0.001594 [-0.013818, +0.012702] |

## HQ_test211/NoRef/HQ-test-identity/no_reference

211 clips / 180 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1 (single seed, indicative) | 21.019178 | 0.642835 | 0.145733 | 0.850334 | 0.577251 |
| Align | 1 (single seed, indicative) | 21.162838 | 0.648823 | 0.144198 | 0.839896 | 0.575701 |
| Relation | 1 (single seed, indicative) | 21.082630 | 0.645988 | 0.144689 | 0.839613 | 0.585664 |
| Both | 1 (single seed, indicative) | 21.117058 | 0.644884 | 0.145293 | 0.843877 | 0.580986 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=1) | +0.143659 [+0.132719, +0.154500] | +0.005988 [+0.005280, +0.006682] | -0.001535 [-0.002475, -0.000602] | -0.010439 [-0.027298, +0.006538] | -0.001550 [-0.020184, +0.010896] |
| Relation (N=1) | +0.063452 [+0.048994, +0.077907] | +0.003153 [+0.002287, +0.003985] | -0.001044 [-0.001856, -0.000186] | -0.010721 [-0.028111, +0.005603] | +0.008413 [-0.000531, +0.022560] |
| Both (N=1) | +0.097880 [+0.086295, +0.109378] | +0.002049 [+0.001230, +0.002830] | -0.000440 [-0.001270, +0.000404] | -0.006457 [-0.022426, +0.009622] | +0.003735 [-0.008020, +0.013937] |

## HQ_test211/NoRef/HQ-test-noise/no_reference

211 clips / 180 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1 (single seed, indicative) | 20.203483 | 0.493797 | 0.282864 | 1.553937 | 0.836635 |
| Align | 1 (single seed, indicative) | 20.347501 | 0.503898 | 0.278013 | 1.453998 | 0.843114 |
| Relation | 1 (single seed, indicative) | 20.200851 | 0.494187 | 0.284100 | 1.521110 | 0.823566 |
| Both | 1 (single seed, indicative) | 20.459870 | 0.513475 | 0.261935 | 1.416315 | 0.843546 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=1) | +0.144018 [+0.119696, +0.169205] | +0.010101 [+0.008274, +0.011905] | -0.004851 [-0.007149, -0.002576] | -0.099939 [-0.128958, -0.071765] | +0.006479 [-0.001591, +0.014768] |
| Relation (N=1) | -0.002632 [-0.026822, +0.020033] | +0.000390 [-0.001498, +0.002171] | +0.001236 [-0.000937, +0.003409] | -0.032827 [-0.057169, -0.008781] | -0.013069 [-0.019857, -0.006411] |
| Both (N=1) | +0.256387 [+0.228774, +0.285918] | +0.019678 [+0.017171, +0.022434] | -0.020930 [-0.023670, -0.018239] | -0.137621 [-0.167793, -0.108136] | +0.006912 [+0.000019, +0.014447] |

## HQ_val32/HRI/ref0/hri_off

32 clips / 32 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 3 | 21.874753 ± 0.313581 | 0.674667 ± 0.021830 | 0.139145 ± 0.010527 | 0.850657 ± 0.070640 | 0.614778 ± 0.011005 |
| Align | 3 | 21.890445 ± 0.298767 | 0.676639 ± 0.018756 | 0.137211 ± 0.008113 | 0.847176 ± 0.085938 | 0.612582 ± 0.006298 |
| Relation | 1 (single seed, indicative) | 21.926668 | 0.683103 | 0.133511 | 0.810392 | 0.619668 |
| Both | 1 (single seed, indicative) | 21.802020 | 0.680618 | 0.134678 | 0.786377 | 0.627209 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=3) | +0.015692 [-0.002484, +0.034219] | +0.001972 [+0.000969, +0.003018] | -0.001934 [-0.002970, -0.000926] | -0.003481 [-0.018496, +0.011016] | -0.002196 [-0.006804, +0.002593] |
| Relation (N=1) | -0.013293 [-0.037492, +0.010281] | -0.001790 [-0.002924, -0.000576] | -0.000394 [-0.001896, +0.001121] | -0.018295 [-0.048468, +0.011498] | -0.007811 [-0.020942, +0.001954] |
| Both (N=1) | -0.137940 [-0.171141, -0.104431] | -0.004275 [-0.005737, -0.002846] | +0.000773 [-0.001049, +0.002346] | -0.042310 [-0.074490, -0.010120] | -0.000271 [-0.010942, +0.009649] |

## HQ_val32/HRI/ref0/hri_on

32 clips / 32 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 3 | 21.870963 ± 0.231003 | 0.676670 ± 0.017642 | 0.139322 ± 0.009634 | 0.857595 ± 0.061010 | 0.621044 ± 0.018895 |
| Align | 3 | 21.876967 ± 0.256260 | 0.677461 ± 0.015235 | 0.137264 ± 0.007132 | 0.846125 ± 0.073857 | 0.620114 ± 0.011558 |
| Relation | 1 (single seed, indicative) | 21.983118 | 0.687630 | 0.138312 | 0.837363 | 0.654148 |
| Both | 1 (single seed, indicative) | 21.898389 | 0.684172 | 0.137078 | 0.807603 | 0.657178 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=3) | +0.006004 [-0.011150, +0.023372] | +0.000791 [-0.000126, +0.001748] | -0.002058 [-0.003008, -0.001137] | -0.011470 [-0.027141, +0.003277] | -0.000929 [-0.008340, +0.006198] |
| Relation (N=1) | +0.041594 [+0.008283, +0.073823] | +0.000505 [-0.000899, +0.001939] | +0.003061 [+0.001410, +0.004842] | -0.000935 [-0.039968, +0.039998] | +0.011527 [-0.004819, +0.026251] |
| Both (N=1) | -0.043135 [-0.080951, -0.005392] | -0.002953 [-0.004294, -0.001600] | +0.001827 [-0.000051, +0.003499] | -0.030695 [-0.077669, +0.016262] | +0.014557 [-0.000324, +0.029862] |

## HQ_val32/HRI/ref4/hri_off

32 clips / 32 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 3 | 21.899105 ± 0.315159 | 0.675325 ± 0.021807 | 0.138952 ± 0.010365 | 0.922377 ± 0.095603 | 0.624453 ± 0.011188 |
| Align | 3 | 21.914106 ± 0.300994 | 0.677334 ± 0.018754 | 0.136995 ± 0.007806 | 0.936083 ± 0.130629 | 0.619578 ± 0.005050 |
| Relation | 1 (single seed, indicative) | 21.955710 | 0.683820 | 0.133408 | 0.926125 | 0.624585 |
| Both | 1 (single seed, indicative) | 21.830641 | 0.681373 | 0.134515 | 0.892360 | 0.633448 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=3) | +0.015001 [-0.003536, +0.033989] | +0.002009 [+0.001004, +0.003058] | -0.001957 [-0.002973, -0.000963] | +0.013705 [-0.007558, +0.036355] | -0.004875 [-0.009400, -0.000925] |
| Relation (N=1) | -0.009255 [-0.033177, +0.014253] | -0.001753 [-0.002880, -0.000551] | -0.000278 [-0.001774, +0.001261] | +0.023660 [-0.030634, +0.087611] | -0.012489 [-0.028164, -0.000118] |
| Both (N=1) | -0.134324 [-0.168385, -0.100059] | -0.004200 [-0.005666, -0.002759] | +0.000830 [-0.000998, +0.002432] | -0.010106 [-0.058761, +0.042994] | -0.003625 [-0.017958, +0.010604] |

## HQ_val32/HRI/ref4/hri_on

32 clips / 32 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 3 | 21.921665 ± 0.310486 | 0.675453 ± 0.022421 | 0.140078 ± 0.010817 | 0.915164 ± 0.082554 | 0.627866 ± 0.014038 |
| Align | 3 | 21.903073 ± 0.229048 | 0.676416 ± 0.016186 | 0.137874 ± 0.007629 | 0.918596 ± 0.090450 | 0.624218 ± 0.010599 |
| Relation | 1 (single seed, indicative) | 22.028628 | 0.687534 | 0.137307 | 0.902298 | 0.636736 |
| Both | 1 (single seed, indicative) | 21.976166 | 0.686589 | 0.136654 | 0.858367 | 0.647168 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=3) | -0.018592 [-0.033892, -0.002994] | +0.000963 [+0.000030, +0.001959] | -0.002204 [-0.003179, -0.001268] | +0.003432 [-0.022518, +0.028883] | -0.003648 [-0.008712, +0.000748] |
| Relation (N=1) | -0.064411 [-0.096043, -0.034164] | -0.001343 [-0.002572, -0.000123] | +0.002365 [+0.000958, +0.003817] | +0.008951 [-0.044906, +0.073912] | -0.006226 [-0.020178, +0.004692] |
| Both (N=1) | -0.116874 [-0.155393, -0.079277] | -0.002288 [-0.003403, -0.001165] | +0.001713 [-0.000290, +0.003443] | -0.034980 [-0.096591, +0.024832] | +0.004205 [-0.010033, +0.017486] |

## HQ_val32/NoRef/all_frames

32 clips / 32 source groups. No protocol or reference-pool averaging.

| Variant | Training seeds | PSNR | SSIM | LPIPS | tLP100 | tOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 3 | 21.918281 ± 0.139593 | 0.678208 ± 0.010485 | 0.134595 ± 0.003537 | 0.827004 ± 0.057707 | 0.610807 ± 0.011831 |
| Align | 3 | 21.970403 ± 0.220101 | 0.680731 ± 0.014088 | 0.135256 ± 0.003720 | 0.830211 ± 0.075129 | 0.616354 ± 0.014101 |
| Relation | 1 (single seed, indicative) | 22.024834 | 0.687589 | 0.132036 | 0.786471 | 0.625703 |
| Both | 1 (single seed, indicative) | 22.039691 | 0.686688 | 0.133541 | 0.798098 | 0.627953 |

| Candidate − Base | PSNR Δ [95% CI] | SSIM Δ [95% CI] | LPIPS Δ [95% CI] | tLP100 Δ [95% CI] | tOF Δ [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Align (N=3) | +0.052122 [+0.037157, +0.068398] | +0.002523 [+0.001639, +0.003566] | +0.000661 [-0.001368, +0.002462] | +0.003207 [-0.014243, +0.019232] | +0.005546 [+0.001212, +0.010572] |
| Relation (N=1) | +0.060907 [+0.033520, +0.089034] | +0.002791 [+0.001466, +0.004119] | -0.000528 [-0.002225, +0.001025] | -0.001327 [-0.051219, +0.049484] | +0.001471 [-0.005883, +0.010159] |
| Both (N=1) | +0.075765 [+0.048579, +0.101563] | +0.001890 [+0.000572, +0.003079] | +0.000977 [-0.000307, +0.002296] | +0.010300 [-0.029777, +0.059361] | +0.003721 [-0.003374, +0.011968] |

## Interpretation and limits

The matched val32 repeats show mixed effects. NoRef Align has slightly higher mean PSNR/SSIM but worse mean LPIPS and tOF; HRI ref0/on has nearly unchanged mean PSNR with favorable mean LPIPS/tLP100/tOF, but the direction changes between training seeds. Those effects do not establish a stable overall auxiliary-loss gain. Relation/Both and every frozen-test model remain single training seed, indicative. Report all metrics and contexts; do not select the best seed, reference position, corruption condition or metric.

The CSV is a compact metric-wise publication table. JSON additionally contains per-training-seed means/deltas, source-bootstrap intervals, coverage and input audit fingerprints. T0.4 remains partial until the external cohort is completed under a declared compatible protocol.
