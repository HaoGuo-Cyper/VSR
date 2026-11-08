# Wan TRD：冻结教师与 EMA 教师对照结果

2026-11-08。**本轮完整训练、推理、独立统计和固定视觉交付已完成。** 原始结果保持不变；本报告明确区分实验完成和方法有效。

本轮完整执行了 B 原始基线、F 冻结 Wan Base 教师和 E EMA Wan 教师，各 2048 步，共 6144 科学更新；验证 640 次、测试 192 次正常恢复全部完成。**当前证据支持：EMA 比固定 Wan 教师有局部改善，但尚未证明它能比原始基线稳定提高超分质量，也没有训练加速证据。**

测试集 E 相对 F 的 LPIPS 收益为 +0.001824 [+0.000543, +0.003358]；E 相对 B 仅为 +0.000296 [-0.000469, +0.001085]，区间跨零，同时 PSNR 下降。验证集 E 相对 B 的 LPIPS 收益为 -0.001395 [-0.002252, -0.000546]，方向相反。不能只摘取测试 E 对 F 的正结果，宣称 EMA 方案已整体成功。

## 一、实际比较了什么

本轮是 VideoREPA 关系损失在 Wan VSR 上的**教师目标更新机制对照**，不等于原版 CogVideoX＋VideoMAEv2 的复现。三组从同一已训练 8192 步的 NoRef Base 开始，保持原视频隔离的 train1641 / val32 / test32、33 连续帧、128×224 HR crop、固定 ×4 bicubic、LoRA rank8 和 LR residual 分支。全组 fresh AdamW、相同数据顺序、公共训练噪声及 native 时间步，单一训练 seed20261102。

F/E 都取零起算 block9，使用相同的恒等初始化 Linear(1536,1536) 学生投影（2,360,832 参数）。目标是完整 1008×1008 token 余弦关系矩阵的 `mean(relu(abs(G_student−G_teacher)−0.1))`；FP32 normalize / Gram，eps=1e-6，包含首 latent、全部有序空间和跨帧对。共同 λ=0.3 来自训练集 15 例梯度校准，不利用验证或测试分数选权重。

教师看到相同 clean HR latent、LR 和共享噪声方向；教师 σ 由 native grid 选择最接近学生 σ 一半的可用项。F 始终固定在共同 Base；E 在每次成功的学生更新之后，以 decay0.999 更新 LoRA/LR 的 FP32 shadow，教师前向用本步更新前状态。教师不接收梯度；不对学生投影头做 EMA。所有推理均只使用原始学生权重，teacher / head 不参与推理。因此 F/E 的训练开销增加，推理结构不增加。

两教师对比保持损失、输入、头结构和 λ 一致，主要检验该配置下动态教师相对固定教师的作用。B 与 F/E 还存在额外投影参数和辅助目标差异，不能把所有差别归为 EMA 单一因素。全部三组最终2048均被保留，验证不挑 best-middle checkpoint，也没有测试集调参。

## 二、正确性与完整执行证据

| 项目 | 已完成的实际证据 |
| --- | --- |
| 预飞 | 15/15 校准例出现有限正 FM / TRD 主干梯度，覆盖 σ=.2/.5/.8；0 校准优化更新；41 次诊断 attempted/applied/returned 一致。 |
| 基线与恢复 | λ=0 公共学生轨迹精确复现；相同输入的师生特征逐字相等且 TRD=0；F/E 连续训练与完整保存恢复状态精确一致。 |
| 科学训练 | 三组各2048，合计6144；12份检查点。每步数据 / σ / native timestep 序列三组一致；四套 RNG 在512/1024/1536/2048逐字一致。 |
| 参数参与 | 每份12CP的488个学生 trainable tensor均已变化；F/E的8份CP中2个head tensor均变化且AdamW二阶矩非零。F shadow恒定，E shadow计数随step到2048。 |
| 完整推理 | val640＋test192＝832次；27456张预测PNG、832份预测MP4，逐行媒体 / checkpoint角色 / 来源 / hash链均通过原生verify。评价阶段0训练更新。 |
| 兼容性回归 | 同一共享起点、32clip×2sampler×33帧，共2112张PNG与上一轮逐字一致。 |
| 统计复核 | 独立CPU程序从832行重新汇总416个clip-role行，并复核75个配对指标区间、均值/SD、成本和判定；结果通过。 |

预飞诊断参数没有用作科学训练起点。训练公平性和参数确实更新的证据不能替代质量证据；本报告的质量判断只使用完成并验证后的正常推理输出。

证据入口：[实际预飞独立复核](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/scratch/wan_trd_real_preflight_review_20261108.json)；[完整训练审计](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/training_diagnostics/training_audit.json)；[原生验证集校验](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/pipeline_paired_training_20261108T124055Z/phases/05_val_verify.result.json)；[原生测试集校验](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/pipeline_paired_training_20261108T124055Z/phases/10_test_verify.result.json)；[2112 PNG 回归](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_shared_initial_regression_20261108.json)；[75 区间独立统计复核](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_independent_statistics_review_20261108.json)。

## 三、五指标完整结果

所有均值先在每个片段内平均推理 seed42/43，再等权汇总32个原视频来源。`±` 是 **clip sample SD（ddof=1）**，不是训练种子 SD。LPIPS、tLP100、tOF 越低越好；PSNR / SSIM 越高越好。指标沿用原定义，tLP100 使用 TecoGAN crop 的 Alex0.1 时序感知差，tOF 为同一 crop 的 Farneback 流误差；不能把它们换称成未实现的 warp 指标。

### 验证集32片段

| 组别（最终 2048） | LPIPS ↓ | PSNR dB ↑ | SSIM ↑ | tLP100 ↓ | tOF ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| B 原始基线 | 0.116806 ± 0.037575 | 22.713395 ± 1.963004 | 0.712857 ± 0.089636 | 0.349240 ± 0.211280 | 0.173989 ± 0.095955 |
| F 冻结 Wan 教师 | 0.117912 ± 0.037893 | 22.684548 ± 1.945991 | 0.709718 ± 0.089454 | 0.352982 ± 0.217814 | 0.175291 ± 0.095970 |
| E EMA Wan 教师 | 0.118201 ± 0.038279 | 22.696186 ± 1.955937 | 0.712124 ± 0.090203 | 0.347309 ± 0.215472 | 0.174958 ± 0.096727 |

### 测试集32片段

| 组别（最终 2048） | LPIPS ↓ | PSNR dB ↑ | SSIM ↑ | tLP100 ↓ | tOF ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| B 原始基线 | 0.114820 ± 0.037786 | 22.688752 ± 2.210994 | 0.698868 ± 0.092031 | 0.374652 ± 0.267397 | 0.240373 ± 0.320291 |
| F 冻结 Wan 教师 | 0.116348 ± 0.038698 | 22.673900 ± 2.184207 | 0.695714 ± 0.092658 | 0.380803 ± 0.272327 | 0.233569 ± 0.280826 |
| E EMA Wan 教师 | 0.114524 ± 0.037694 | 22.663308 ± 2.178150 | 0.698149 ± 0.091521 | 0.372092 ± 0.260107 | 0.227111 ± 0.257914 |

### 最终2048配对效应与区间

下表所有“收益”均统一为**正值有利于比较名称的第一个方法**：LPIPS / tLP100 / tOF 用参考减方法，PSNR / SSIM 用方法减参考。例如 E−F 的 LPIPS 数值实际为 LPIPS_F−LPIPS_E。区间来自10000次配对clip bootstrap，NumPy PCG64 seed20261102、95% linear percentile；每个比较共享重采样索引。本轮仅1个训练seed，区间是对这些模型与来源的条件描述，不能称为多训练种子显著性，未计算p值。

验证集：

| 比较 | 指标 | 配对收益均值 | 描述性 95% 区间 |
| --- | --- | ---: | --- |
| E − F | LPIPS ↓ | -0.000289 | [-0.001557, +0.000996] |
| E − F | PSNR dB ↑ | +0.011638 | [-0.007372, +0.030799] |
| E − F | SSIM ↑ | +0.002406 | [+0.001455, +0.003344] |
| E − F | tLP100 ↓ | +0.005673 | [+0.000887, +0.011196] |
| E − F | tOF ↓ | +0.000334 | [-0.000640, +0.001507] |
| E − B | LPIPS ↓ | -0.001395 | [-0.002252, -0.000546] |
| E − B | PSNR dB ↑ | -0.017209 | [-0.036319, +0.002045] |
| E − B | SSIM ↑ | -0.000733 | [-0.001372, -0.000084] |
| E − B | tLP100 ↓ | +0.001931 | [-0.003354, +0.007251] |
| E − B | tOF ↓ | -0.000969 | [-0.001637, -0.000360] |
| F − B | LPIPS ↓ | -0.001106 | [-0.002221, -0.000075] |
| F − B | PSNR dB ↑ | -0.028846 | [-0.055767, -0.003084] |
| F − B | SSIM ↑ | -0.003139 | [-0.003848, -0.002386] |
| F − B | tLP100 ↓ | -0.003742 | [-0.011770, +0.003268] |
| F − B | tOF ↓ | -0.001302 | [-0.002426, -0.000247] |

测试集：

| 比较 | 指标 | 配对收益均值 | 描述性 95% 区间 |
| --- | --- | ---: | --- |
| E − F | LPIPS ↓ | +0.001824 | [+0.000543, +0.003358] |
| E − F | PSNR dB ↑ | -0.010592 | [-0.034483, +0.014461] |
| E − F | SSIM ↑ | +0.002435 | [+0.001520, +0.003434] |
| E − F | tLP100 ↓ | +0.008711 | [+0.001160, +0.019041] |
| E − F | tOF ↓ | +0.006457 | [+0.000492, +0.017217] |
| E − B | LPIPS ↓ | +0.000296 | [-0.000469, +0.001085] |
| E − B | PSNR dB ↑ | -0.025444 | [-0.045964, -0.005530] |
| E − B | SSIM ↑ | -0.000719 | [-0.001950, +0.000362] |
| E − B | tLP100 ↓ | +0.002561 | [-0.004944, +0.010771] |
| E − B | tOF ↓ | +0.013261 | [-0.001784, +0.042228] |
| F − B | LPIPS ↓ | -0.001528 | [-0.002915, -0.000226] |
| F − B | PSNR dB ↑ | -0.014852 | [-0.045300, +0.014403] |
| F − B | SSIM ↑ | -0.003154 | [-0.004753, -0.001856] |
| F − B | tLP100 ↓ | -0.006150 | [-0.013513, +0.001151] |
| F − B | tOF ↓ | +0.006804 | [-0.004210, +0.026333] |

测试 E 对 F 的 LPIPS、SSIM、tLP100 和 tOF 都出现正向描述区间；PSNR点值略低、区间跨零。另一方面，F 的测试 LPIPS 和 SSIM 比 B 差，E 对 B 的主要 LPIPS 区间跨零，PSNR收益为 -0.025444 [-0.045964, -0.005530] dB。E 对 B 的两个时序指标区间也跨零。因此较合理的解读是：**EMA 在本测试上缓解了固定教师配置的部分损失，但还没有充分证据说明这套方法优于不加教师的 baseline。** 验证集 E 对 F 的 LPIPS 区间跨零、E 对 B更差，进一步限制其可泛化性。

原先约定的“E同时优于F/B且PSNR、SSIM均保持”的候选标签未满足；这不是工程执行失败，也不采用旧G4的Base种子spread门槛。不能因结果不理想重新定义容差、追加权重或改取中间checkpoint。

全部精度数字、所有5指标和逐来源差异可从以下原始分析材料复查：

- 验证：[完整分析 JSON](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/val/analysis.json) · [五指标均值及 clip SD](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/val/summary.csv) · [全部配对差与区间](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/val/paired_summary.csv) · [逐视频指标](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/val/per_clip_metrics.csv) · [逐视频配对差](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/val/paired_clip_differences.csv) · [成本明细](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/val/costs.csv)。
- 测试：[完整分析 JSON](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/test/analysis.json) · [五指标均值及 clip SD](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/test/summary.csv) · [全部配对差与区间](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/test/paired_summary.csv) · [逐视频指标](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/test/per_clip_metrics.csv) · [逐视频配对差](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/test/paired_clip_differences.csv) · [成本明细](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/test/costs.csv)。

## 四、质量随更新数及实际时间变化

验证集 LPIPS 实测点如下，所有方法共用step0；1536仅保存恢复checkpoint，没有额外评价，不补点或插值：

| 额外更新数 | B LPIPS | F LPIPS | E LPIPS |
| ---: | ---: | ---: | ---: |
| 0 | 0.117742 | 0.117742 | 0.117742 |
| 512 | 0.126831 | 0.125684 | 0.126507 |
| 1024 | 0.117819 | 0.117906 | 0.118813 |
| 2048 | 0.116806 | 0.117912 | 0.118201 |

三组512时LPIPS均较起点变差，后续再恢复；不能把稀疏点连线看成单调收敛。B最终LPIPS最好，F/E最终也未优于共享起点的LPIPS。其他保真度的变化已在完整CSV和曲线单独报告。

[按额外更新数的质量曲线](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/val/quality_vs_updates.png) · [按累计同步Step墙钟时间的质量曲线](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/val/quality_vs_training_seconds.png)。两张图均已由本次只读复核查看，横轴与实测点一致，连接线仅作视觉引导，不进行插值。测试集只评价最终2048，测试图不能用来推断训练速度。

以B最终2048的验证LPIPS **0.116805826** 为预定共同质量目标：B首次在本轮实测点2048达到，耗时696.575秒；F/E在0/512/1024/2048全部实测点均未达到，速度比保持未知（null），不能写成已加速，也不能外推到未跑的4096。

共同时间预算内，最新可用的实测checkpoint如下：

| 共同时间预算（秒） | B：实测更新 / LPIPS | F：实测更新 / LPIPS | E：实测更新 / LPIPS |
| ---: | --- | --- | --- |
| 173.724 | 512 / 0.126831 | 0 / 0.117742 | 0 / 0.117742 |
| 348.359 | 1024 / 0.117819 | 512 / 0.125684 | 512 / 0.126507 |
| 696.575 | 2048 / 0.116806 | 1024 / 0.117906 | 1024 / 0.118813 |

这些是“预算内最新观测点”，**不是严格相同实际计算量比较**，F/E仍有未使用的预算余量。在最早预算中，F/E只有step0可用；它的LPIPS优于当时B512，只表示尚未更新的共同模型未经历早期退化，不能算教师训练加速。在B最终的696.575秒预算内，F/E最新实测点都只有1024，LPIPS也比B2048差。当前支持的结论是**没有观察到步数或时间加速**。

## 五、训练与评价成本

| 组别 | 2048 步累计同步 Step 墙钟秒 | 相对 B | 在线教师前向 / EMA 更新 |
| --- | ---: | ---: | --- |
| B 原始基线 | 696.575 | 1.0000× | 0 / 0 |
| F 冻结 Wan 教师 | 988.026 | 1.4184× | 2048 / 0 |
| E EMA Wan 教师 | 1054.972 | 1.5145× | 2048 / 2048 |

F增量训练时间比B多41.84%，E多51.45%；E又比F多6.78%。这是已提交完整Step的同步墙钟时间，包含在线教师前向、学生前后向、关系损失、检查、AdamW和EMA；不等于纯GPU kernel计时。教师独占耗时没有单独测量，不能把全部差值都精确归给teacher forward。

| 时间范围 | 秒 | 口径 |
| --- | ---: | --- |
| 三组已提交Step合计 | 2739.573 | 是下项whole-attempt的子集，不再相加。 |
| 全训练attempt | 2828.384 | 含装载 / 保存等；本轮0已知discarded更新。 |
| 真实预飞 | 49.451 | 包含15校准案例和41诊断更新，不算科学训练。 |
| 验证execute | 2408.056 | 含装载、恢复、指标、媒体IO与cleanup；不含prepare / verify / analysis。 |
| 测试execute | 744.435 | 同上，与验证是独立阶段。 |

整轮全部人力、准备、hash复核、分析和历史Base成本的统一总计没有完整计量，保持未知，不将上述交叠范围乱加。F/E教师在线提取，没有本轮离线teacher cache；共同Base8192及VAE缓存是历史投入，增量表不重复计入，并不意味着原本没有成本。预飞显存峰值 allocated：B 3.581 GiB、F/E 3.643 GiB；属于该预飞，不冒充其他配置或全分辨率容量保证。

## 六、辅助损失诊断

前256到末256更新窗口，F单位TRD均值为 0.000878142 → 0.000766152，E为 0.000849842 → 0.000482050。E末窗口的active pair fraction也更低。实际梯度、head和学生更新已经验证，因此不是“辅助项完全没有进入训练”。

但两窗口的数据和噪声分布不同，EMA目标自身随训练变化；TRD下降也可能包含目标追随学生的作用，不能直接推导表征更好或加速。噪声分箱显示高σ段平均TRD更大，只是诊断现象，未据此删掉噪声段或修改权重。完整见[窗口诊断](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/training_diagnostics/arm_windows.csv)和[噪声分箱](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/training_diagnostics/arm_noise_bins.csv)。与正常推理合看，**辅助损失下降没有转化为已确认的baseline之上质量增益**。

## 七、固定视觉对比与实际检查

已实际查看预先固定的三个片段、sampler42/43共六张五时点contact sheet，并逐一查看全部33帧的动画脸部、婚纱人物及汽车前轮ROI网格；还查看了验证集质量对步数、质量对同步Step时间两图。视频完整保存为33帧、原25fps。这里是静态时序图人工检查，**不是连续播放盲评或主观闪烁打分**；ROI来自有损MP4解码，仅用于展示，不参与定量评分。

- 动画片段 `-mSvhSt_Q8g_85_92to239`：三组均比模糊LR恢复了人物与眼镜轮廓，但HR帽子的细条纹和脸部细节仍明显丢失，嘴型与眼镜附近有失真。六组时序检查中未见某一方法稳定地修复这些问题；同一方法的42/43采样外观也存在明显差别。
- 婚纱片段 `0_xW8oRPKRs_100_113to216`：三组均恢复了裙摆主体与地面方向性结构，但头脸、发丝和裙纹仍与HR不同，细密褶皱有生成性纹理。连续帧网格能追踪到运动变化；未观察到EMA在两采样seed中一致优于基线的细节或运动表现，不能由少量展示宣称消除闪烁。
- 汽车片段 `0gI38E1VsiY_18_125to238`：车身大轮廓更清晰，但前轮辐条和轮胎纹理被替换成不同的高频图案，seed42尤其明显，车头与车身细形也有偏差；这类内容不忠实在B/F/E中都存在。EMA没有在这两个固定样本中恢复真实轮毂结构。

这些固定片段展示了相对LR的恢复能力，也暴露出三组共同的细节重建与内容保真失败；它们不是从全部测试集中按分数挑出的“最好例子”，也不能代表全部32视频或独立证明方法优劣。定量结论仍取自完整矩阵。

[六套完整视频及contact sheet](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/video_compare_wan_trd_20261108)；[全33帧ROI检查与说明](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_visual_roi_20261108)。车ROI底部越界4行用灰色显示，不含字幕、未计入指标。媒体manifest SHA `4928c3cdcfaf8c4a892a6379d9d4a3aa70a91d10a004a42941ca9ca237d217fa`；ROI manifest SHA `578b4b79f0bc13099365f4cb62d1f1b1fd121626a4416ac6db877d53396e46e2`。

## 八、配置、检查点与复现

源码为[独立Wan-TRD源码目录](/home/hugo/.codex/worktrees/vsr-wan-trd/VSR-sync)，保持10业务模块加init。具体新增TRD / teacher shadow / step与checkpoint机制、真实执行及恢复命令、环境和12份checkpoint表均在[复现与检查点说明](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/reproduce_and_checkpoints.md)。最终实际共同λ配置是[study_common_lambda_20261108.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/study_common_lambda_20261108.json)，不是仍保留校准占位weight1的设计草案；所有source与依赖由协议SHA绑定。

| 组别 | 最终 checkpoint（仅本机） | SHA256 |
| --- | --- | --- |
| B 原始基线 | [checkpoint_002048.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/paired_training_20261108T124055Z/arm_runs/baseline/Base/checkpoint_002048.pt) | `5964a0646c67657fe63f3882793c5b6b2296fb03dfdafaba644231d299f3e897` |
| F 冻结 Wan 教师 | [checkpoint_002048.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/paired_training_20261108T124055Z/arm_runs/frozen/Base/checkpoint_002048.pt) | `161ccbb7fcda1b9afa318ba405890ac87571cc7daa70b8e50e149f4bf4678424` |
| E EMA Wan 教师 | [checkpoint_002048.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/paired_training_20261108T124055Z/arm_runs/ema/Base/checkpoint_002048.pt) | `1d5c29cba84190d9076ba21eb6dc05f99887b2b51f25eee73bd0d4545aa926eb` |

训练日志在[三组训练run](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/paired_training_20261108T124055Z)各`arm_runs/<arm>/training.jsonl`；全eval记录在[串行pipeline](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/pipeline_paired_training_20261108T124055Z)的`validation`和`test`。协议分别为：

- 预飞：`a74e1a44f362a569d934696ff171f71cb26a5351beab35b0b567d361ceb6c650`。
- 训练：`6d5221ce42f2264e10a78c51c04b4d833a3137e146aa7aa54d2c176151b4f072`。
- 验证：`c8dbde5e11d5e151e0673e152829e161221db1164644e766a22835573376aa05`。
- 测试：`ae82f1f39c255aeb566519a3feadbe5a36f040551f72af39b05fc39bab03ec11`。

复制或复现时使用全新运行目录并核对依赖、模型、数据、来源与允许输出根；本报告不要求重新执行已完成矩阵。权重、checkpoint、PNG / MP4和tensor缓存仅留本机；源码、配置、原始文本日志、统计及清单按已授权范围同步GitHub。完整832行评价及指标、独立统计和全部训练日志已同步GitHub，提交 `eb92f455620762f2e08b9d69df1f6e0885143908`，远端tree与本机原生tree一致。最终报告、视觉文本manifest和结束记录另行补充文本提交；媒体与checkpoint继续仅留本机。

## 九、限制与下一步

1. 只有一个训练seed。两次推理采样和32clip不能代替训练重复；描述区间反映当前来源条件下的变化，不给出多seed稳定性或显著性结论。
2. 验证 / 测试虽按原视频与训练隔离并通过已声明范围近重复筛查，但来源已被以前轮次使用、测试像素已预生成；不是全新的盲测或最终确认性数据。筛查也不穷尽所有裁剪 / 变速 / 转码重复。
3. 33帧128×224固定bicubic裁剪及LoRA续训范围，不能直接外推为长视频、原分辨率或真实复杂退化VSR效果。
4. 关系目标采用同构Wan教师、半σ噪声、block9、恒等Linear、FP32关系以及共同梯度校准λ；这些都是明确适配。当前结果不能否定原版VideoREPA、VideoMAEv2教师、不同层或其他训练尺度，也不能证明关系损失比逐token监督更好。
5. 稀疏checkpoint和没有teacher独占计时限制效率判断；只用实测点，没有填补缺失时间或外推后续收敛。

本轮不追加权重、层号、噪声比、4096步或多seed来挽救结果。当前更有根据的研究问题是：保持视频教师预训练语义和原版关系监督后，是否会改变这个观察，而非直接宣布EMA已获验证。

用户已经**单独授权**在本轮完成交付后，开展尽量忠于原版、保留冻结VideoMAEv2教师的Wan VSR适配；新方案见[VideoMAEv2后续计划](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/VideoREPA_VideoMAEv2_Wan_Plan_20261108.md)。这是新的前瞻协议与资源检查，不由本轮`insufficient_evidence`科学标签自动授权；应先固定其方法差异、预算、数据、数值精度和正确性检查，失败时记录并停止依赖执行。公开权重下载 / CPU strict-load成功也不等于新教师的真实forward、梯度或16GiB资源预检已经通过。新轮结果不得回写或更改本轮判断。

---

本报告由完整JSON自动格式化均值、SD、CI、时间与checkpointSHA；不重新运行bootstrap、原生推理或训练。原分析输入：val SHA`2235aedc94ae351af895910324d38cae9d2a4b7943d8fa63866a7cb9d2bf81e9`；test SHA`92b280d10413824819bbf25d6bfe45497b7fb0801236be3277b2b3d58c41943a`。实际视觉范围与公开文本状态见上文。
