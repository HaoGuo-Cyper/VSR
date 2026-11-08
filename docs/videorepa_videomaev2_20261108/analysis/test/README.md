# Wan VideoMAEv2：原始基线与冻结 HR 视频教师

阶段：test；完整128次恢复。两组各2048步，共4096步（0.5RE），一个训练seed。
每clip先平均采样42/43，再对32个不同source等权平均；SD是clip sample SD，不是训练seed SD。

| Arm / λ | 更新 | 完整step墙钟秒 | LPIPS↓ | PSNR↑ dB | SSIM↑ | tLP100↓ | tOF↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline / 0.0 | 2048 | 757.491 | 0.114820 ± 0.037786 | 22.688752 ± 2.210994 | 0.698868 ± 0.092031 | 0.374652 ± 0.267397 | 0.240373 ± 0.320291 |
| videomaev2 / 0.5 | 2048 | 822.207 | 0.115637 ± 0.039283 | 22.587840 ± 2.199926 | 0.693865 ± 0.094131 | 0.383626 ± 0.282687 | 0.228034 ± 0.257061 |

## 最终2048步配对结果

LPIPS/tLP100/tOF为baseline−method，PSNR/SSIM为method−baseline；正值有利于方法。10000次PCG64配对clip的95%linear-percentile区间仅描述当前训练seed。

| 指标 | 平均收益 | 95%描述区间 |
| --- | ---: | ---: |
| lpips_alex_v0_1 | -0.00081749 | [-0.00260204, 0.00078408] |
| psnr_rgb_db | -0.10091259 | [-0.13221841, -0.06998178] |
| ssim_rgb | -0.00500307 | [-0.00617616, -0.00387927] |
| tLP100_alex_v0_1_TecoGAN_crop | -0.00897379 | [-0.02259815, 0.00171484] |
| tOF_Farneback_TecoGAN_crop | 0.01233918 | [-0.00419251, 0.04315378] |

LPIPS分类：**insufficient_evidence**。
区间严格大于0记探索性收益，严格小于0记下降，接触或跨0记证据不足，精确[0,0]记未观察到差异。PSNR、SSIM和时序代价全部另报，不附加旧EMA/G4的AND门槛。

## 训练效率与成本

Test只测双方固定final2048；效率由val中间检查点曲线描述。

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
