# Wan VideoMAEv2：原始基线与冻结 HR 视频教师

阶段：val；完整448次恢复。两组各2048步，共4096步（0.5RE），一个训练seed。
每clip先平均采样42/43，再对32个不同source等权平均；SD是clip sample SD，不是训练seed SD。

| Arm / λ | 更新 | 完整step墙钟秒 | LPIPS↓ | PSNR↑ dB | SSIM↑ | tLP100↓ | tOF↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline / 0.0 | 0 | 0.000 | 0.117742 ± 0.038532 | 22.352308 ± 1.949681 | 0.698361 ± 0.093031 | 0.321062 ± 0.201218 | 0.175904 ± 0.097106 |
| baseline / 0.0 | 512 | 191.689 | 0.126831 ± 0.037428 | 21.965106 ± 1.847019 | 0.687894 ± 0.094162 | 0.371490 ± 0.190714 | 0.180624 ± 0.099466 |
| baseline / 0.0 | 1024 | 382.739 | 0.117819 ± 0.036611 | 22.231045 ± 2.009904 | 0.697634 ± 0.093776 | 0.339074 ± 0.197257 | 0.175006 ± 0.096772 |
| baseline / 0.0 | 2048 | 757.491 | 0.116806 ± 0.037575 | 22.713395 ± 1.963004 | 0.712857 ± 0.089636 | 0.349240 ± 0.211280 | 0.173989 ± 0.095955 |
| videomaev2 / 0.5 | 0 | 0.000 | 0.117742 ± 0.038532 | 22.352308 ± 1.949681 | 0.698361 ± 0.093031 | 0.321062 ± 0.201218 | 0.175904 ± 0.097106 |
| videomaev2 / 0.5 | 512 | 205.305 | 0.125028 ± 0.037002 | 22.012016 ± 1.849833 | 0.688149 ± 0.094270 | 0.364721 ± 0.190935 | 0.181307 ± 0.099577 |
| videomaev2 / 0.5 | 1024 | 410.790 | 0.118804 ± 0.035978 | 22.204086 ± 2.009905 | 0.695639 ± 0.094563 | 0.348068 ± 0.202351 | 0.176741 ± 0.097706 |
| videomaev2 / 0.5 | 2048 | 822.207 | 0.117484 ± 0.038504 | 22.623274 ± 1.945331 | 0.708743 ± 0.091178 | 0.347308 ± 0.210176 | 0.175362 ± 0.097295 |

## 最终2048步配对结果

LPIPS/tLP100/tOF为baseline−method，PSNR/SSIM为method−baseline；正值有利于方法。10000次PCG64配对clip的95%linear-percentile区间仅描述当前训练seed。

| 指标 | 平均收益 | 95%描述区间 |
| --- | ---: | ---: |
| lpips_alex_v0_1 | -0.00067802 | [-0.00164977, 0.00035034] |
| psnr_rgb_db | -0.09012017 | [-0.11362674, -0.06607185] |
| ssim_rgb | -0.00411418 | [-0.00538707, -0.00281124] |
| tLP100_alex_v0_1_TecoGAN_crop | 0.00193254 | [-0.00759095, 0.01160027] |
| tOF_Farneback_TecoGAN_crop | -0.00137296 | [-0.00251159, -0.00038751] |

LPIPS分类：**insufficient_evidence**。
区间严格大于0记探索性收益，严格小于0记下降，接触或跨0记证据不足，精确[0,0]记未观察到差异。PSNR、SSIM和时序代价全部另报，不附加旧EMA/G4的AND门槛。

## 训练效率与成本

目标为baseline@2048 LPIPS=0.11680583；仅使用实测点，不插值。
- baseline：最早观测2048步，757.491同步step秒。
- videomaev2：实测点未达到目标。

step时间包括HR传输、预处理、在线教师前向及学生/头更新；HR磁盘读取在step外单列并属于attempt墙钟。没有新教师特征缓存；26预检更新、15 probe cases和4重复教师前向单列。总量与子项重叠，不能相加。

[逐clip指标](per_clip_metrics.csv) · [配对差](paired_clip_differences.csv) · [汇总](summary.csv) · [描述区间](paired_summary.csv) · [成本](costs.csv) · [完整JSON](analysis.json)

![质量随更新变化](quality_vs_updates.png)

![质量随完整训练step时间变化](quality_vs_training_seconds.png)

## 限制

- One training seed: clip intervals and two sampling repeats do not establish reproducibility over training seeds or statistical significance.
- Both val and test cohorts have historical model exposure. Test is a reused, source-separated exploratory evaluation, not a pristine/blind confirmatory holdout.
- Both final2048 checkpoints are prospectively retained; no weights or intermediate checkpoints are selected using these metrics.
- Fixed33-frame128x224 crops with clean4x bicubic degradation; not full source duration, full resolution or an external benchmark.
- A frozen VideoMAEv2-B teacher and the original relation objective are adapted to Wan VSR with block17, matched native grids, online HR targets and LoRA continuation. Small crops/2048 updates differ from original CogVideoX VideoREPA.
- LPIPS improvements, PSNR/SSIM and temporal tradeoffs are reported separately. No inherited G4 threshold, extra temporal AND gate or tolerance tuning.
- Efficiency uses sparse measured checkpoints and synchronized step wall time. Common preparation and overlapping attempt/calibration/evaluation costs are separate; unknown cost is not zero.
