# 保留 VideoMAEv2 教师的 Wan2.1 视频超分实验——前瞻方案（等待当前轮完整交付）

2026-11-08。用户已授权：当前 Wan TRD 三组实验完整结束并交付后，整理本方案、标清必须修改的部分，再启动新实验。本文件在当前三组质量分数尚未读取时编写；**不是已启动声明**。当前轮 P6、独立源码实现、CPU正确性和GPU预检均为新训练前置条件。

## 研究问题与定位

在现有 Wan2.1-T2V-1.3B 配对 ×4 超分模型上，保留 VideoREPA 的 VideoMAEv2-B 教师、干净视频目标、三层投影和关系损失，是否改善正常推理质量或训练效率？比较原始 flow-matching 基线 B 与 VideoMAEv2-TRD 方法 R。当前轮的 Wan 冻结/EMA 教师不放入本轮，不混用历史评分。

这是一项尽量保留原方法核心的 **Wan 视频超分适配**。学生、任务、数据、训练规模和评价指标改变，因此不能称 CogVideoX 论文数值复现；本轮也不能仅凭跨轮结果认定教师身份是收益的唯一原因。

## 原版来源与保留内容

- VideoREPA 固定代码 `8d581dc301e18546a1808436c724f59d259d09f9`；论文 https://arxiv.org/html/2505.23656v1 。优先实现官方默认代码的完整时空 Gram margin-mean；论文 Eq.5 的空间/时间分别平均与代码不同，明确记录这一取舍。
- 原教师是 VideoMAEv2-B `vit_b_k710_dl_from_giant.pth`，不是普通 VideoMAE、其他尺寸或不同分类微调版本。官方 HF revision `706cc172d65ebd4dedbee3f9c0183a93df9fa125`，173574417 bytes，SHA256 `8141a6955e0700d11bf15928fe6d61e5cfe482606fed8cfdddb1b922c0fd88ec`。
- 使用最后 transformer block 输出的全部 patch tokens，位于最终 fc_norm/classifier 之前；不池化、不使用 CLS，不融合其他层。冻结全部教师参数，eval/no_grad。
- 取学生 one-based 第18层，即 Wan block17 输出、最终 norm 之前。对 CogVideoX-2B 和 Wan30层均为60%深度；与 CogVideoX-5B42层的相对深度不同。
- 学生 MLP 为 Linear(1536,2048)→SiLU→Linear(2048,2048)→SiLU→Linear(2048,768)，含bias，共8917760参数，使用独立初始化RNG。保留官方隐藏宽度/深度/输出维度，不用上一轮恒等Linear。
- 默认 margin=0.1、外层权重 λ=0.5，不按梯度或当前/新轮质量分数重标，不搜索第二个权重。

真实资源准备已完成：官方文件大小/SHA一致；CPU使用 weights_only=True 加载，显式排除且仅排除 fc_norm/head 的4个未使用键，余158个encoder键、86225664参数 strict=True 且逐tensor赋值一致。教师未前向、CUDA未初始化；这不表示新网格、实际dtype、梯度或显存预检已通过。

## 必须修改与资源选择分别列明

| 项目 | 本项目安排 | 性质与原因 |
|---|---|---|
| 学生与训练目标 | CogVideoX换成Wan；保留原flow matching及LR残差条件 | 任务和架构适配，不能把文本生成的目标直接覆盖超分训练 |
| 数据与指标 | 同区域HR/LR、固定bicubic×4；LPIPS为主，报告PSNR/SSIM/tLP100/tOF | 超分任务所需；不使用生成物理合理性分数代替像素恢复评价 |
| 通道 | MLP输入维随Wan改为1536，输出仍768 | 架构适配 |
| 教师/学生时间 | 教师取HR帧1..32，tubelet2得到16位置；学生仅辅助项去latent0，余8位置插值到16 | 保留原版去首位置及学生映射到教师方向；对33帧Wan网格做尺寸适配 |
| 空间网格 | HR128×224直接给patch16教师，得到8×14；Wan token同为8×14 | 本机小crop设置下的对应选择，不对HR监督区域新增裁剪 |
| 原空间stride3 Conv | 省略此下采样层，因为原生两端网格已经一致；原Conv会错误降为2×4 | 明确的结构适配；保留MLP不等于原投影全结构逐字复制 |
| 位置编码 | 从官方原始8×14×14一维sincos出发，先空间插值到8×14，再时间插值到16 | 消除官方目标24×10×15的硬编码，保留其插值顺序/align_corners=False；不按新token数重造sincos |
| 33帧128×224、rank8 LoRA、有效batch1、2048步、单seed | 沿用当前本机小规模实验条件 | **资源和比较范围选择，不是VideoREPA/VSR方法本身必需**；原论文49帧480×720、rank128或full-SFT并未复现 |
| 已有Base续训 | 两组同Base01 8192步模型，fresh AdamW，各额外2048步 | 隔离辅助监督的增量；披露共同历史训练，不能称从官方模型零起点训练 |
| 教师执行 | 默认每次方法更新在线执行冻结VideoMAEv2，teacher no_grad前向在student计算图之前 | 保留原版在线监督；不为方便先增加离线特征缓存，显存不可行时停报而非自动fallback |

原版教师会把480×720的HR区域缩至160×240。本轮保留128×224整块HR区域而不继续按1/3缩小；这是小crop条件下的尺度选择，须在报告中保留说明。空间变换采用对应区域，不扭曲宽高比或混用其他帧裁剪。

## 精确目标与数值策略

教师输入来自构造同一FM样本的HR区域。原版将RGB[-1,1]还原到[0,1]再做ImageNet mean/std归一化；本项目已封存 paired_rgb.pt 的 hr 本来就是FP32 [0,1]，直接归一化，不能再错误加一除二。仅辅助分支去frame0。教师Conv3d nominal tubelet依次覆盖[1,2]、[3,4]…[31,32]，最终token仍具有全局上下文。学生VAE是因果编码，不能宣称其每个latent恰好等于独立4帧平均。

学生全9×8×14特征先经MLP，再去首latent；按官方方向做trilinear时间8→16（align_corners=False），空间不变。记录首尾clamp、THW展平及token位置的实际检查。插值是官方使用的近似对应，不是逐token精确感受野相等。

两端通道L2 normalize，原调用eps=1e-12；全部1792 tokens组成完整关系矩阵：

`L = L_FM + 0.5 * mean(relu(abs(Gram(student) - Gram(teacher)) - 0.1))`

包含所有有序token对、自对角、同帧与跨帧；不改成分别均值之和、上三角平均、局部关系或随机抽对。F=16时同帧对占1/16、跨帧占15/16，原F=24的权重比例不同，这是时间规模变化的结果，须披露。

教师权重/目标与教师normalize/Gram为FP32，teacher eval/no_grad且显式关闭外层autocast。学生MLP延续bf16训练路径；在关系normalize前显式转bf16，student normalize/Gram维持bf16，差值按类型提升FP32后做abs/hinge/mean。官方在空间Conv前有bf16 cast，本轮省略Conv但保留这个数值边界。MLP返回/插值的实际dtype还受混合精度包装影响，不能宣称整条路径逐字节复现。预检记录投影、插值、normalize、两端Gram及最终loss实际dtype，并与精确参考函数核对值和梯度。不能静默继承上一轮统一FP32/eps1e-6。若原路径非有限或资源失败，保留证据后停止，明确讨论数值适配；不为出分自动切精度、改margin或lambda。

## 公平设置和预算

两组使用同Base01 SHA `28356e22e4e718d3c378588793f60732dbc93825ec5d78b8a85c8150060da63f`，同1641训练样本、相同顺序/时间步/噪声/增强，seed20261102。沿用LoRA8、LR注入0/5/9、bf16DiT/FP32VAE、固定文本、fresh AdamW(lr1e-4,wd.01,foreach=False)、constantLR、global clip1、有效batch1。教师/MLP初始化与公共4RNG分离；不引入训练或推理EMA。

| 预算项 | 固定安排 |
|---|---:|
| 科学训练 | B/R各2048，共4096更新=.5RE |
| 检查点 | 每组512/1024/1536/2048，共8个完整状态 |
| 目标梯度检查 | 固定训练清单前5片段×σ .2/.5/.8，共15梯度探针案例，0更新；每例FM/aux分别求VJP，共30次VJP调用而非15次backward；只测量不校准λ |
| GPU诊断更新 | 26：零λ B/R各3=6；R连续2与1+恢复1=4；B/R各2warmup+6timed=16 |
| 教师重复提取 | 固定train0/train1各提取两遍，共4次0-update forward；同device/dtype/batch1逐字匹配 |
| 验证 | (共享0+2组×512/1024/2048)×32clip×2sampler=448恢复 |
| 测试 | 2组最终2048×32clip×2sampler=128恢复，无分数选择 |
| 总评价 | 576恢复、19008张预测PNG，全部33帧；MP4用于观看。无额外诊断native恢复，预飞仅验证载入后的student状态 |
| 固定视觉样本 | 沿用已声明3clip×42/43，共6套HR/LR/B/R比较 |

诊断权重不复用。训练不能新增第三臂、第二权重、4096续训或多seed。预算在任何本轮质量评分前固定；不因旧轮结果修改候选。教师实际forward按角色逐条计数：4次重复提取、15例梯度探针、12次有辅助目标的诊断更新，共31次诊断/检查forward；零λ绕过teacher/head。正式方法2048次在线teacher forward。完整计算开销分别列诊断、有效同步训练Step（含在线教师预处理/前向）、额外HR读取、整段训练wall、评价、失败浪费，避免重叠计时相加。若读取位于Step计时之外，明确单列并计入完整训练wall。基线不额外加载教师以人为加大基线成本。

默认不构建新的教师特征缓存；每次方法更新从已封存的对应 paired_rgb.pt/HR [0,1] 提取目标。现有1641份RGB均存在且与selection同序，教师只读这些既有数据。FP32教师参数约329MiB；1792tokens/12heads单attention矩阵147MiB，softmax瞬间可能双份294MiB；no_grad串行层不同时保留12层attention。教师先于student graph执行后，仅保留目标5.25MiB和常驻权重。静态估计额外教师live工作集约0.7–1.0GiB，**不是16GiB可行的实测保证**。若真实资源检查不通过，停止并报告；不能自动改缓存或精度。预留30GiB额外磁盘；CPU实现/审核时间与GPU窗口分别记录，时长由短测更新。

## 启动检查与执行顺序

1. 当前Wan-TRD完成全部640/192、独立统计、2112PNG回归、6套视频实际复核、复现材料和文本同步，才关闭其P6；不把训练完成或parent awaiting_root_delivery当结案。
2. 在新隔离源码位置实施本方法，保留当前11模块及历史结果。继续10业务模块结构；共享库复用须保持旧行为。把本方案、官方/本地差异和最终预算记录进新协议，正常通过后的启动已获授权。
3. CPU验证：严格教师加载、位置插值、去首时间边界/空间/THW对应、官方full-Gram值/梯度参考、teacher无梯度、零λ公共轨迹、MLP/学生更新、完整保存恢复学生/头/AdamW/4显式RNG与全部全局RNG/游标/成本；推理仅加载student。
4. 独占GPU做原精度教师重复提取核对（2个固定样本各2次）、15例梯度探针及26诊断更新；所有15例必须有限，至少8例共享学生FM与辅助梯度均正且覆盖至少2个σ档。若不满足则停在目标可用性，不改阈值/层/权重。
5. 对全1641已有RGB与FM缓存核source/start/crop/增强/selection链，绑定teacher源码/权重/位置编码/预处理；保存固定真实样本重复提取目标hash。目标来自FM使用的同一HR区域；无新增随机增强，不提前提取val/test教师特征。
6. 从相同新鲜Base训练B/R各2048，完整审计8CP/数据和RNG后，原生推理val448→锁定两个final2048→test128→全部核验。正常通过无需重复询问；故障保存证据和完整状态，不降低协议标准。

## 评价和交付

复用已修复但历史使用的source-heldout val32/test32，披露其不是全新盲测。保持原UniPC20/CFG1/CPU noise42,43/固定text/FP32VAE/量化/LPIPSAlex.1/RGBPSNR/SSIM/TecoGANtLP100/tOF；每clip先平均采样seed，再等权32来源。全部输出完整33帧128×224裁剪片段，不宣称原始长视频全长或原分辨率。

主要差值为LPIPS_B−LPIPS_R，正值更好。10000次pairedclip bootstrap，NumPyPCG64 seed20261102、95% linear-percentile；单训练seed的条件描述区间，无p值/多seed显著性。区间全正为探索性增益信号，跨或碰0为证据不足，[0,0]为未观察差异；全负为本配置退化。所有PSNR/SSIM和时间指标代价完整报告，不依结果换主指标，也不继承旧EMA/G4的spread或1.5开销门槛。

绘制质量随额外步数和实际同步Step时间变化的曲线，完整计入在线teacher前向/预处理；HR读取、加载与保存等在完整训练wall中另报。没有新增教师缓存成本；共同历史VAE/RGB缓存属于已有成本，必须披露。以B最终验证LPIPS为共同目标，只用0/512/1024/2048实测点，不插值；若shared0已达到该值，不能称训练加速。辅助loss下降而正常推理不改善时明确写出这一结果。

交付实现/必要改动、固定配置与模型版本、执行及完整恢复命令、日志和8CP本机索引、全部逐视频/配对指标、两曲线、6套固定视频与实际视觉复核、成本和局限。无增益和证据不足同样完成本轮；下一轮建议不自动变成新增实验。代码/元数据/完整文本持续授权公开同步GitHub，权重/缓存/PNG/MP4留在本机。

## 证据文件

- `work/scratch/videorepa_videomaev2_fidelity_review_20261108.md`：原文/固定官方代码逐项核查，含原权重和位置策略。
- `work/videomaev2_assets_20261108/download_receipt.json`：真实下载大小/SHA。
- `work/videomaev2_assets_20261108/checkpoint_cpu_audit.json`：真实strict encoder加载，尚无forward。
- `work/videomaev2_assets_20261108/checkpoint_cpu_audit_environment.json`：复用当前环境、无依赖安装/变更。
- `work/scratch/videorepa_videomaev2_resource_plan_20261108_v2.md`：在线教师优先的资源路径/预算支持记录。早期缓存方案保留为已替代草案，不执行。

