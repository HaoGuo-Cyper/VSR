# 第一轮固定教师实验：完整结果

本轮已完成代码、正确性检查、5条训练轨迹、完整验证/测试推理及独立复核。**HR-VAE低权重在本训练seed的测试集上出现小幅LPIPS改善，同时牺牲SSIM；HR-DINOv2没有可靠的LPIPS改善证据。两者均未显示训练加速。** 不把辅助损失下降解释为超分收益，也不把本次单种子结果推广为普遍结论。

## 实际完成了什么

- RTX5080本地运行；同一33连续帧/stride1、128×224 HR、32×56 LR的固定×4双三次任务。每个已声明片段完整生成33帧，属于裁剪片段评价，不是原始长视频全长或原分辨率。
- 从同一已训练8192步的NoRef Base01继续训练；baseline、VAE λ=0.11/1.1、DINO λ=0.094/0.94各2048步，共10240步=1.25RE。一个训练seed20261102，有效batch1，20个checkpoint均保留。没有额外加跑或第二轮训练。
- 验证集1024次恢复：共享起点＋五轨迹512/1024/2048；验证最终2048 LPIPS选出VAE0.11与DINO0.094。随后才构建正式测试RGB并完成3模型×32片段×采样42/43=192次恢复。共1216次正常恢复、40128张预测PNG及1216份完整预测MP4；验证和测试的来源/源码/权重/hash/全帧链均通过核心verify和独立重算。
- 实际GPU训练和串行父服务均正常退出0/MainPID0；没有训练恢复、丢弃步骤或额外科学更新。27步预飞诊断单独计账。GPU工作已经结束。

## 测试结果

每片段先平均两个推理seed，再等权汇总32个不同source。表内“±”为clip sample SD，**不是训练seed SD**。LPIPS、tLP100及tOF越低越好。

| 配置 | LPIPS ↓ | PSNR dB ↑ | SSIM ↑ | tLP100 ↓ | tOF ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.114820 ± 0.037786 | 22.688752 ± 2.210994 | 0.698868 ± 0.092031 | 0.374652 ± 0.267397 | 0.240373 ± 0.320291 |
| HR-VAE λ0.11 | 0.113553 ± 0.037609 | 22.667386 ± 2.209287 | 0.696121 ± 0.091889 | 0.374774 ± 0.266960 | 0.225190 ± 0.250286 |
| HR-DINOv2 λ0.094 | 0.114659 ± 0.038124 | 22.620034 ± 2.248354 | 0.693349 ± 0.093544 | 0.386394 ± 0.278480 | 0.228521 ± 0.257423 |

配对收益LPIPS=baseline−method；保真差=method−baseline。区间为10000次paired-clip bootstrap、PCG64 seed20261102、95% linear-percentile，条件于本次训练seed的描述性区间，没有多种子显著性或p值。

| 方法 | LPIPS收益 [95%描述区间] | PSNR差 dB | SSIM差 |
| --- | ---: | ---: | ---: |
| HR-VAE λ0.11 | 0.0012671 [0.0002218, 0.0024026] | -0.021366 | -0.002747 |
| HR-DINOv2 λ0.094 | 0.0001608 [-0.0012579, 0.0015052] | -0.068718 | -0.005519 |

HR-VAE的LPIPS均值相对下降约1.10%，32片段中21个改善、11个变差。PSNR差的描述区间跨0，SSIM差区间为负；因此可以报告“小幅感知指标信号及保真代价”，不能报告综合质量全面改善。验证集该方法LPIPS收益仅0.0003675，描述区间跨0，跨数据一致性仍有限。

HR-DINOv2的LPIPS收益区间跨0，17片段改善、15个变差；PSNR/SSIM降低、tLP100升高（该指标越低越好，表示变差）。两方法tOF均值较低，但相对baseline的配对描述区间跨0；VAE的tLP100也没有改善证据，不能宣称解决闪烁。选定VAE配置在本次测试五指标均值上均优于选定DINO配置，但两种损失和输出头容量不同，不能将差异唯一归因于教师表示。

## 全部候选与训练效率

只用最终2048验证LPIPS选择权重，没有选最佳中间checkpoint、根据test改权重或删除失败候选。

| 轨迹 | λ | final验证LPIPS | 已提交同步Step秒 | final耗时/Baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.0 | 0.1168058 | 704.994 | 1.0000× |
| vae_low | 0.11 | 0.1164383 | 708.138 | 1.0045× |
| vae_high | 1.1 | 0.1199763 | 708.564 | 1.0051× |
| dino_low | 0.094 | 0.1195549 | 712.342 | 1.0104× |
| dino_high | 0.94 | 0.1269314 | 711.588 | 1.0094× |



两图已实际检查，保留共享0/512/1024/2048所有点。512步各轨迹LPIPS均较起点变差，之后回升，不能只展示最后一段。相同2048步下VAE-low略有质量信号，DINO没有稳定优势；同时间的精确新分数未额外测量，连线只引导阅读、不作插值证据。以预定baseline@2048验证LPIPS=0.116805826为目标，baseline和VAE-low都首次在实测2048点达到，分别704.994与708.138秒，其余候选未达。**本轮没有减少更新数或缩短达标时间的证据。**

时间轴是已提交、CUDA同步的训练Step墙钟时间，包含既有Step内检查，不是纯GPU kernel时间或端到端总成本。缓存DINO让在线训练差异很小，但不能省略其离线成本。

| 成本范围 | 实测值及解释 |
| --- | --- |
| 五轨迹同步Step总计 | 3545.627秒；是训练attempt的子项，不与下行重复相加 |
| 五轨迹完整训练attempt | 3705.880秒，约61.76分钟；含载入/保存等，0已知丢弃或重放步骤 |
| DINO离线缓存一次性构建 | 174.457秒，其中HR提取96.913秒、其余IO/绑定等77.544秒；55209帧/1673片段；非纯GPU计时，整个study只计一次 |
| VAE目标缓存 | 复用公共FM既有clean latent，baseline也需要，公共历史构建成本视为sunk，未假定其为0 |
| 校准预飞 | preflight范围35.946秒；VAE/DINO梯度校准另有绑定3.186/2.315秒计时，覆盖关系见costs.csv，不任意相加 |
| 完整val execute | 3806.600秒，约63.44分钟；含模型载入、恢复、指标、PNG/MP4及清理，排除prepare/verify |
| 完整test execute | 750.562秒，约12.51分钟；同上scope |
| 未被上述receipt覆盖的准备/核验/分析 | 未汇总为统一GPU总时长，未知项不记作0 |

实际全程peak allocated约3.58/3.59/3.60 GiB（baseline/VAE/DINO）；reserved峰值约3.66/3.68/3.70 GiB，不等于整机显存占用。单次在线Step成本约0.344/0.346/0.348秒，教师特征均缓存，不含在线DINO大模型驻留。

## 实现和正确性证据

沿用10个业务模块：data负责固定配对/来源，vae复用原编码缩放，model提供block7学生特征，teacher组织冻结目标/时空缓存，losses计算两种对齐，train负责公平训练与完整状态恢复，infer正常生成，evaluate封存/选择/验证，metrics算五指标，checkpoints管理原权重及状态。文件用途和真实CLI见仓库README、CLI_GUIDE。辅助头仅在训练时使用，正常推理不输入HR或教师。

两组都选block7（第8/30层，已接受LR注入0/5，后续9不变）。VAE是同一次FM clean normalized HR latent，按native THW及patch内channel最快排列成64维，Linear1536→64、98368参数、元素平均SmoothL1 β1。DINO是官方冻结B/14，HR128×224等比AA bicubic至112×196，去CLS/register得到8×14局部tokens；首帧独立、后续每4帧平均成9位置后channel-L2，Linear1536→768、1180416参数、mean(1−cos)。教师eval/no_grad；时间映射是明确的名义压缩网格，不声称等于VAE/DINO真实感受野逆映射。

107项CPU/6个tinyWan共18VJP及真实27更新预飞通过。零权重三个GPU步骤逐步精确复现原baseline；两头学生-only reload与完整student/head/AdamW/四RNG恢复精确。已有constant LR无独立LR调度器、bf16无GradScaler；native噪声scheduler配置绑定。train0/1在线VAE/DINO与缓存逐元素精确相等。所有20CP的488/488学生参数变化，16个aux CP的2/2 head参数变化且AdamW二阶矩非零；五轨迹数据/σ/timestep以及4个CP的四RNG逐字相同。

关闭新增路径的推理回归：31保留窗口×2采样×33=2046张共享起点PNG与历史Base01/clean逐SHA相同，排除1替换窗口；这是0新增推理的CPU验证。HR→VAE→Decoder对两训练clip另有重建诊断（详细五指标在calibration_diagnostics.csv），不能当SR性能上界。

辅助loss首256→末256：VAE-low0.16022→0.03569、VAE-high0.12378→0.02908、DINO-low0.80350→0.52186、DINO-high0.77406→0.50375；高权重候选仍有更差完整推理，说明aux下降本身不保证SR收益。分噪声箱统计已交付；首末窗口数据/噪声不同，不据此宣称因果收敛速度。

本轮没有训练故障或resume。只修复两项不改变数值的外围兼容问题：旧Base text_cache字典先canonical化（新pipeline_v2，在0评价前替换旧等待父）；最终文本导出允许合法首字符“-”的sourceID（新exporter_v2，17CPU及单字符AST/安全复核）。旧版本、失败记录均保留，模型/数据/采样/预算未更改。

## 固定视频与视觉观察

[六份私有完整对比视频的hash清单](video_comparison_private_manifest.json)固定source排序前三片段，每个seed42/43；五列依次HR、LR上采样、baseline、VAE、DINO。每个视频已逐帧decode33帧、25FPS、172×1120。实际复核了六张0/8/16/24/32 contact sheet，以及从六份真实MP4解码的全33帧局部ROI图（人脸/婚纱/车轮），未开展盲评或主观视频评分实验；不把局部时序图当整个视频播放评分。

- 动画人物 `-mSvhSt_Q8g`：三模型相对LR恢复更清楚的轮廓/眼镜，但脸部细纹、表情与HR仍有偏差；seed42的起始脸部较糊，seed43明显不同，采样差异比两教师间肉眼差异更显著。VAE并未消除这种现象；两seed平均LPIPS反而比baseline高0.002581，DINO高0.008599。
- 婚纱运动 `0_xW8oRPKRs`：三模型都能恢复拖尾和较清晰外轮廓，HR织物/头发细节仍未可靠复现。VAE的LPIPS小幅改善0.000928，同时PSNR下降0.137201dB、SSIM下降0.006450，属于感知与保真权衡。逐帧ROI中动作大体连贯，没有足够证据判定某教师显著减少闪烁。
- 汽车 `0gI38E1VsiY`：大轮廓和运动方向较LR清楚，但轮辐细节不同于HR，逐帧局部纹理有变化；三组均保留此失败，VAE/DINO也未完整恢复真实结构。两seed平均LPIPS分别比baseline高0.000426/0.003872，不能作为方法视觉成功例子。

这些预声明片段没有换成最好看的视频；全32片段逐视频差在CSV，包含成功和失败。后验逐clip检查显示VAE最强LPIPS收益为UhPIA9ttlgU(+0.012271)，最差v-BvRlsbUiU(−0.003220)；这两个数字只作误差定位，没有用来选择权重或替换固定展示。

## 结论边界与下一轮建议

本轮是单训练seed、LoRA/LR分支的2048步续训；不能证明从预训练冷启动、全参数训练或多训练seed效果。两个教师头输出64/768维、参数量差约12倍，损失/归一化也不同，结论限定于完整监督配置。

数据按原始source划分；原ID/原文件SHA无跨集合交集，但视觉筛查另发现1val/3test跨ID近重复，事前固定替换并保留原训练/Base/历史记录。1705片段声明筛查无flag，不代表穷尽所有offset/crop/transcode重复。原test被历史模型用过，不是项目级pristine盲测；正式本轮test RGB/模型分数在val选择后开启，事前自动完整性缩略图访问例外已披露。

一个事后样本敏感性检查（不替代预声明32片段主结果、不增设门槛）：VAE在保留29原窗口的LPIPS收益均值0.000887，3个替换源为0.004942；DINO对应0.0000785/0.000956。VAE信号受片段构成影响，应以新的独立source与训练seed进一步确认。该拆分没有再作选参或显著性宣称。

建议优先确认低权重HR-VAE：先前瞻固定保真权衡与主要指标，再做多训练seed；加入同层、相近头容量的普通固定特征对照，判断辅助正则本身的贡献。VAE隐藏特征对比可排在确认结果之后。当前DINO与高权重结果不支持优先扩大其训练。EMA/Self-Flow更复杂，暂放第二轮后续方案；若采用，必须明确教师/学生输入和噪声，并保留同噪声或同损坏而无教师的控制。**本报告不启动任何第二轮任务。**

## 可复现材料

- [验证集：全部候选/中间点/成本/逐视频结果](analysis/val/report_text_public.md)；[测试集：冻结选择后的结果](analysis/test/report_text_public.md)。两目录各有7CSV、analysis.json、两图和delivery_manifest.json。
- [准确命令、模型/数据版本、完整恢复和20个checkpoint路径/SHA](reproduce_and_checkpoints.md)。已完成run不可覆盖，重现需全新目录与另行预算授权。权重/checkpoint/cache/原始媒体只保留本机。
- [训练公平性及更新证据](training_diagnostics/README.md)；[2046 PNG精确回归](reviews/shared_initial_regression.json)。
- [论文适配记录](https://github.com/HaoGuo-Cyper/VSR/blob/main/docs/fixed_teacher_20261108/method_references.md)：[SRA² / VAE-REPA v2](https://arxiv.org/html/2601.17830v2)、[REPA v4](https://arxiv.org/html/2410.06940v4)、[DINOv2](https://arxiv.org/html/2304.07193v2)。本项目是视频VAE/Wan/小头/LoRA续训适配，不直接复用论文速度结论。
- 本机raw run：`/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z`；训练raw run：`/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z`。源码/配置/统计/完整文本日志/CSV公开同步，PNG/MP4/权重/tensor不上传。

公开版本仅保留文本；完整曲线PNG和6个MP4在本机 outputs/FixedTeacher_20261108 与 outputs/video_compare_fixed_teacher_20261108。下游下载代码不自动包含数据、权重、缓存或媒体。
