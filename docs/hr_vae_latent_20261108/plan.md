# 第一轮 HR VAE latent 特征监督执行计划

2026-11-08，按用户最新路线执行；取代尚未启动的 EMA gamma 筛选。未运行的 gamma 原型保存在本地 commit `5dcad5d`、`refs/archive/unexecuted-gamma-20261108`，本轮从 `a91a95f` 独立开发，旧源码、权重和记录保留。

研究问题：在同一 Base01 起点、同数据/更新预算/推理下，冻结 HR VAE 的干净 normalized latent 监督第8个学生 block，是否改善 ×4 视频超分的推理质量或达到相同质量的训练用时。允许有增益、无增益、证据不足。

| 步骤 | 内容与完成标准 | 当前状态 |
| --- | --- | --- |
| 1 | 模型/数据/源视频划分/缓存/GPU/现有任务核对 | 已完成，GPU无训练任务；见data_audit |
| 2 | 固定预算、验证、采样、LPIPS主指标、判定及效率定义 | 本文件与configs/hr_vae_latent_20261108.json在出分前固定 |
| 3 | Linear1536→64 + mean SmoothL1，FP32 frozen HR target，index7 | 实现与相关CPU检查中 |
| 4 | 关闭对齐精确基线、target冻结/梯度/patch、保存/恢复，GPU小测与开销 | CPU/真实GPU预检后记录，不复用诊断权重 |
| 5 | baseline、λ=.01、λ=.1，各fresh Base01/AdamW/RNG，共6144步 | 待预检通过，0.75RE在原授权小规模预算内 |
| 6 | shared0及三组512/1024/2048，共10roles×32clip×2samples=640完整恢复 | 训练后验证全部33帧/来源/CP/行/PNG，正常restore_video入口 |
| 7 | 指标表、质量-更新/质量-GPU秒曲线、固定3片段对比、分析与GitHub文本 | 单seed筛选结果，不能升级为正式确认 |

训练保留原33连续帧/stride1、128×224 HR、×4固定双三次、train1641/val32、rank8 LoRA/LR residual、bf16 DiT/FP32 VAE、固定text和strict math；新辅助项不使用EMA、条件破坏、sigma窗口、token mask或额外target归一化。原配套VAE posterior.mode 的 latent按原(z−mean)/std缩放，时间33→9，空间128×224→16×28，Wan patch1×2×2→1008个64维target。head只训练时使用，推理仅学生。

数据按原source隔离，val32为32独立source。评价单位是完整选定的33帧裁剪片段，输出不截帧；这不是原始长视频全长/全分辨率实验。只用验证集选择两权重；本轮不新增test评估，已有test被历史模型使用过。固定视觉片段按source_id字典序取前3，不能看效果挑片。LPIPS按PNG等价q/255像素、all33帧计算；同时报告RGB PSNR/SSIM与定义明确的原TecoGAN tLP100/tOF指标。采样保持UniPC20/CFG1、CPU noise42/43、原tiling/VAE策略。

主要判断逐权重报告baseline减method LPIPS的点估计和10000配对clip PCG64/seed20261102/95%linear-percentile描述区间：区间下界>0记为本seed下改善，区间上界≤0记为未见增益，否则证据不足。PSNR/SSIM下降须明确标为保真代价，不因辅助loss下降宣称推理改善。两权重最终2048步的验证LPIPS均公开，选择较低者，相等时选.01；不选最佳中间checkpoint冒充最终结果。

效率单独判断：画所有已测点的质量-更新次数曲线和质量-同步GPU训练步骤累计秒曲线（另报CUDA events与包括I/O的总wall time）。以baseline最终2048步LPIPS为固定质量目标，仅在实测checkpoint判断方法是否更早达到，不虚构点间精确达标时间；未达到则不声称加速。共用已有VAE缓存成本，额外teacher在线编码为0；小GPU预检记录真实吞吐/显存/均值±SD开销。

本轮用户已授权可恢复故障诊断、修复和检查点续跑。恢复须严格保留科学设置、完整学生/head/AdamW/显式RNG/步数/时间/来源；中断尾部记录保留，额外浪费计算另报。数据/权限/资源阻塞如实说明并推进独立工作。范围截至本轮小规模验证与分析，不自动多seed、改变监督目标或扩展VAE中间层。
