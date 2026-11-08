# Wan TRD：原始基线、冻结 Base 教师与 EMA 教师

阶段：test；完整192次恢复。三个训练arm各2048步，共6144步（0.75RE），仅一个训练seed。
每clip先平均采样42/43，再对32个不同source等权平均。表内SD为clip sample SD，不是训练seed SD。

| Arm / λ | 更新 | 已提交step墙钟秒 | LPIPS↓ | PSNR↑ dB | SSIM↑ | tLP100↓ | tOF↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline / 0.0 | 2048 | 696.575 | 0.114820 ± 0.037786 | 22.688752 ± 2.210994 | 0.698868 ± 0.092031 | 0.374652 ± 0.267397 | 0.240373 ± 0.320291 |
| frozen / 0.3 | 2048 | 988.026 | 0.116348 ± 0.038698 | 22.673900 ± 2.184207 | 0.695714 ± 0.092658 | 0.380803 ± 0.272327 | 0.233569 ± 0.280826 |
| ema / 0.3 | 2048 | 1054.972 | 0.114524 ± 0.037694 | 22.663308 ± 2.178150 | 0.698149 ± 0.091521 | 0.372092 ± 0.260107 | 0.227111 ± 0.257914 |

## 最终2048步配对结果

LPIPS/tLP100/tOF取reference−method，PSNR/SSIM取method−reference；正值有利于method。10000次PCG64配对clip的95%linear-percentile区间仅描述当前训练seed。

| 对比（method vs reference） | LPIPS收益 [95%CI] | PSNR差 [95%CI] | SSIM差 [95%CI] | LPIPS判断 |
| --- | ---: | ---: | ---: | --- |
| ema_vs_frozen | 0.00182379 [0.00054336, 0.00335758] | -0.01059162 [-0.03448341, 0.01446121] | 0.00243467 [0.00151983, 0.00343367] | single_seed_descriptive_LPIPS_gain |
| frozen_vs_baseline | -0.00152811 [-0.00291541, -0.00022579] | -0.01485237 [-0.04529958, 0.01440299] | -0.00315382 [-0.00475288, -0.00185580] | no_observed_LPIPS_gain |
| ema_vs_baseline | 0.00029568 [-0.00046940, 0.00108461] | -0.02544399 [-0.04596427, -0.00552952] | -0.00071915 [-0.00195005, 0.00036224] | insufficient_evidence |

## 预先声明的判定

分类：**insufficient_evidence**。
进入多种子确认的候选条件：EMA相对冻结教师及baseline的LPIPS收益区间下界均严格大于0，并且相对两者的未取整PSNR、SSIM平均差均≥0。区间跨0表示证据不足；保真代价单列。时序指标不额外加入AND门槛。
本分析不会自动授权下一轮实验，也不把单seed描述性区间称作显著性证明。

## 训练效率与成本

Test只测试三组固定final2048，用val中间检查点曲线判断效率。

时间轴是成功保留步骤的同步墙钟时间，含在线教师与EMA，既非纯GPU时间也非整体实验时间。完整attempt墙钟、预检41步及评价开销另列；总量与子项有重叠，不能重复相加。
冻结/EMA教师均在线运行，没有本轮离线特征缓存。Base及VAE公共缓存是已有成本，并非总成本为零。

[逐clip五指标](per_clip_metrics.csv) · [逐clip配对差](paired_clip_differences.csv) · [全部汇总](summary.csv) · [配对区间](paired_summary.csv) · [成本与范围](costs.csv) · [完整JSON](analysis.json)

![质量随更新变化](quality_vs_updates.png)

![质量随同步训练时间变化](quality_vs_training_seconds.png)

## 限制

- One training seed: clip intervals and two sampling repeats do not establish reproducibility over training seeds or statistical significance.
- Both val and test cohorts have historical model exposure. Test is a reused, source-separated exploratory evaluation, not a pristine/blind confirmatory holdout.
- All three final2048 checkpoints are prospectively retained; no weights or intermediate checkpoints are selected using these metrics.
- Fixed33-frame128x224 crops with clean4x bicubic degradation; not full source duration, full resolution or an external benchmark.
- Full TRD supervision configurations are compared. EMA differs from a frozen Base teacher by evolving teacher weights; this is not an original CogVideoX VideoREPA replication.
- LPIPS improvements, PSNR/SSIM point guards and temporal tradeoffs are reported separately. No inherited G4 threshold, extra temporal AND gate or tolerance tuning.
- Efficiency uses sparse measured checkpoints and synchronized step wall time. Common preparation and overlapping attempt/calibration/evaluation costs are separate; unknown cost is not zero.
