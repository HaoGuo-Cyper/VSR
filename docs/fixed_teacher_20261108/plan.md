# 第一轮：固定HR-VAE与HR-DINOv2教师表示

2026-11-08最新人工指令为本轮唯一实验设计依据，取代EMA/gamma与先前HRVAE-only三轨迹草案。两者均未启动GPU，旧代码/记录保留。本轮只有一个训练seed；EMA/Self-Flow第二轮仅提出建议。

| 顺序 | 工作及完成标准 | 状态 |
| --- | --- | --- |
| F0 | 资产、5080、source隔离/重复泄漏、原论文与适配核对 | 完成：声明筛查排除1val/3test跨split近重复后1705片段复筛无flag；保留局限 |
| F1 | VAE meanSmoothL1、DINO channel-L2 cosine，同block7/同head深度，严格时空/cache绑定 | 完成：107CPU＋6tinyWan18VJP检查通过 |
| F2 | 新DINO缓存、在线一致性、HR-VAE重建诊断、真实短测/梯度校准、冻结预算与两权重 | 完成：新DINO缓存与在线精确验算、两VAE重建、27GPU诊断与独审通过 |
| F3 | 同Base01/freshAdamW/数据噪声序列，baseline1+VAE2+DINO2各2048步 | 完成10240/20CP及真实数据/RNG/head/学生更新审计 |
| F4 | val共享0+每轨迹512/1024/2048；16×32×2=1024恢复并选各method最终权重 | 完成1024及native/独立审计；选VAE.11、DINO.094，全部候选保留 |
| F5 | val选择receipt绑定后，才构建/打开test像素并评baseline/VAE/DINO三个final | 进行中：selection后已构建，唯一192矩阵，不重选 |
| F6 | 全部逐视频/总体指标、两种质量曲线、成本拆分与固定片段视频、复现说明/GitHub | 所有交付齐备才完成本轮；第二轮不自动启动 |

预算依据：原TASKS Stage4约3RE，旧pilot已用.75RE；取消的gamma和HRVAE-only草案均0更新，本轮建议1.25RE，使Stage4合计2RE，未超现有授权。有效batch1、constantlr1e-4/AdamW foreachFalse和原精度/梯度裁剪保持；checkpoint512/1024/1536/2048，中间评价512/1024/2048。真实短测确认内存/耗时后记录采用依据，不因分数加跑或扩候选。诊断optimizer更新与正式预算分别计账。

数据保留原train1641；本轮修复后val32/32不同source（31原窗口+1替换），33stride1、HR128×224、LR32×56固定双三次；已有全部sourceID和原文件SHA跨split零重叠，近重复指纹检查仅数据完整性，不做test模型评价。test32从原source-heldout test确定、与train/val隔离；历史模型已评过该test，不称pristine/blind。预选来源在前，模型设置由val确定后才解码本轮test RGB资产及计算模型分数。完整视频指每个已声明33帧片段全帧生成，不冒称原始长视频全长/全分辨率。

两种teacher使用同block7（第8/30层，已有LR注入0/5，后续9仍由flow约束），仅增加训练投影头。VAE目标复用原干净latent；DINO逐帧HR全区域等比resize并去CLS/register，首帧独立及1–4…29–32分组均值后再channel-L2。所有噪声区间/全部token监督；不EMA、不LR破坏、不额外t门槛。head初始化与teacher提取RNG与共同noise/timestep隔离。所有学生共同初始化与新AdamW策略相同；没有普通推理权重EMA。

权重不根据val选尺度：用同一个预先固定trainclip，在名义σ=.2/.5/.8对应native网格分别求未clip的FM与unit辅助损失对相同共享学生参数的全局L2梯度范数。每method取FM/aux范数比中位数，乘相对强度.1及1并round2sig，生成同数量两个候选，强绑定校准报告；head梯度/损失量级也记录。零/非有限范数是实现/数值问题，不默用fallback。

LPIPS(all33,q/255)为主，同时报告RGB PSNR/SSIM及原明确定义TecoGAN tLP100/tOF。先sampler42/43→clip/source，再总体均值与配对差；单seed10000 PCG64/seed20261102描述性95%linear-percentile区间。val最终2048 LPIPS每family最低选（精确tie较弱weight），不选最佳中间CP，不看test换参数。test只报告最终3模型质量与相互对照，不二次选择。LPIPS改善但PSNR/SSIM下降明确标保真代价；辅助loss下降不能替代推理质量。

效率分别看相同更新数、同步GPU训练秒和计入一次性teacher提取/缓存后的成本。VAE已有latent缓存共用，DINO新增缓存构建和在线验算时间单列，训练/评估也分别计时。以baseline最终质量为固定目标，只在实测CP比较最早达到，不插值虚构准确达标时刻；若共享起点已达到则不能声称训练加速。保存全部配置、日志、CP、noise-bin对齐误差、固定source排序前三片段视频。失败场景与选择偏倚、多seed未确认、旧test历史使用均披露。

本轮允许诊断修复可恢复故障和完整检查点续跑，保留中断记录，恢复student/head/AdamW/constantLR参数组/4RNG/step/time/来源。当前constantLR无独立学习率scheduler；native1000训练时间步网格由绑定scheduler配置重建，不假称存在可恢复的LR调度器。所有GPU使用同一共享锁和fresh host身份，单队列。

当前封存设计和权重见[真实短测与预算](preflight_and_budget.md)。正式test RGB已在完整val选择receipt后构建；完整性缩略图自动读取为单独披露的预选前例外，不作未曾访问test像素的声明。
