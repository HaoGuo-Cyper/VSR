# Wan TRD：原始基线、冻结 Base 教师与 EMA 教师

阶段：val；完整640次恢复。三个训练arm各2048步，共6144步（0.75RE），仅一个训练seed。
每clip先平均采样42/43，再对32个不同source等权平均。表内SD为clip sample SD，不是训练seed SD。

| Arm / λ | 更新 | 已提交step墙钟秒 | LPIPS↓ | PSNR↑ dB | SSIM↑ | tLP100↓ | tOF↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline / 0.0 | 0 | 0.000 | 0.117742 ± 0.038532 | 22.352308 ± 1.949681 | 0.698361 ± 0.093031 | 0.321062 ± 0.201218 | 0.175904 ± 0.097106 |
| baseline / 0.0 | 512 | 173.724 | 0.126831 ± 0.037428 | 21.965106 ± 1.847019 | 0.687894 ± 0.094162 | 0.371490 ± 0.190714 | 0.180624 ± 0.099466 |
| baseline / 0.0 | 1024 | 348.359 | 0.117819 ± 0.036611 | 22.231045 ± 2.009904 | 0.697634 ± 0.093776 | 0.339074 ± 0.197257 | 0.175006 ± 0.096772 |
| baseline / 0.0 | 2048 | 696.575 | 0.116806 ± 0.037575 | 22.713395 ± 1.963004 | 0.712857 ± 0.089636 | 0.349240 ± 0.211280 | 0.173989 ± 0.095955 |
| frozen / 0.3 | 0 | 0.000 | 0.117742 ± 0.038532 | 22.352308 ± 1.949681 | 0.698361 ± 0.093031 | 0.321062 ± 0.201218 | 0.175904 ± 0.097106 |
| frozen / 0.3 | 512 | 248.077 | 0.125684 ± 0.036311 | 21.988614 ± 1.839470 | 0.689114 ± 0.093386 | 0.368823 ± 0.191745 | 0.180538 ± 0.099515 |
| frozen / 0.3 | 1024 | 496.108 | 0.117906 ± 0.035288 | 22.265954 ± 2.004972 | 0.698460 ± 0.093621 | 0.345347 ± 0.194631 | 0.175119 ± 0.096241 |
| frozen / 0.3 | 2048 | 988.026 | 0.117912 ± 0.037893 | 22.684548 ± 1.945991 | 0.709718 ± 0.089454 | 0.352982 ± 0.217814 | 0.175291 ± 0.095970 |
| ema / 0.3 | 0 | 0.000 | 0.117742 ± 0.038532 | 22.352308 ± 1.949681 | 0.698361 ± 0.093031 | 0.321062 ± 0.201218 | 0.175904 ± 0.097106 |
| ema / 0.3 | 512 | 262.617 | 0.126507 ± 0.037863 | 21.962567 ± 1.856625 | 0.688283 ± 0.094201 | 0.369326 ± 0.191379 | 0.180596 ± 0.099118 |
| ema / 0.3 | 1024 | 525.465 | 0.118813 ± 0.036816 | 22.211795 ± 2.018532 | 0.696954 ± 0.093902 | 0.343903 ± 0.197277 | 0.175007 ± 0.096556 |
| ema / 0.3 | 2048 | 1054.972 | 0.118201 ± 0.038279 | 22.696186 ± 1.955937 | 0.712124 ± 0.090203 | 0.347309 ± 0.215472 | 0.174958 ± 0.096727 |

## 最终2048步配对结果

LPIPS/tLP100/tOF取reference−method，PSNR/SSIM取method−reference；正值有利于method。10000次PCG64配对clip的95%linear-percentile区间仅描述当前训练seed。

| 对比（method vs reference） | LPIPS收益 [95%CI] | PSNR差 [95%CI] | SSIM差 [95%CI] | LPIPS判断 |
| --- | ---: | ---: | ---: | --- |
| ema_vs_frozen | -0.00028926 [-0.00155653, 0.00099610] | 0.01163757 [-0.00737241, 0.03079872] | 0.00240639 [0.00145547, 0.00334438] | insufficient_evidence |
| frozen_vs_baseline | -0.00110574 [-0.00222078, -0.00007509] | -0.02884638 [-0.05576678, -0.00308378] | -0.00313933 [-0.00384751, -0.00238601] | no_observed_LPIPS_gain |
| ema_vs_baseline | -0.00139500 [-0.00225162, -0.00054578] | -0.01720881 [-0.03631878, 0.00204460] | -0.00073294 [-0.00137223, -0.00008363] | no_observed_LPIPS_gain |

## 预先声明的判定

分类：**no_observed_overall_EMA_advantage**。
进入多种子确认的候选条件：EMA相对冻结教师及baseline的LPIPS收益区间下界均严格大于0，并且相对两者的未取整PSNR、SSIM平均差均≥0。区间跨0表示证据不足；保真代价单列。时序指标不额外加入AND门槛。
本分析不会自动授权下一轮实验，也不把单seed描述性区间称作显著性证明。

## 训练效率与成本

目标为baseline@2048 LPIPS=0.11680583；只使用实际测量点，不插值。
- baseline：最早观测2048步，696.575同步step秒。
- frozen：已测点未达到目标。
- ema：已测点未达到目标。

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
