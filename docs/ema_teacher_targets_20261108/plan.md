# Wan2.1 教师目标对照实验方案

日期：2026-11-08。状态：**仅规划，未实现新方法，未启动训练、GPU 预飞或评价。**

本轮保留现有 Wan 视频超分框架，比较原始基线、冻结 Base 教师和 EMA 教师，回答：在完全相同的学生训练条件下，持续更新教师能否比固定教师带来更好的超分结果？建议采用 VideoREPA 的 token 关系蒸馏思想，固定对齐方式，先研究教师目标。

建议规模为一个训练种子、三条轨迹各 2048 步，共 6144 步（0.75 RE）；验证 640 次、最终测试 192 次正常视频恢复。完成实现和正确性检查后，预计需预留约 2–3 小时 GPU 窗口。该时间是历史数据外推，执行需要后续明确指令；本方案不授权启动任何 GPU 工作。

## 研究范围与原文关系

[VideoREPA 原文](https://arxiv.org/html/2505.23656v1)使用 CogVideoX、冻结视频编码器和 token 关系蒸馏，研究视频生成。这里使用 Wan2.1-T2V-1.3B、配对视频超分和 Wan Base 教师，属于 **VideoREPA TRD 的 Wan 适配与教师更新机制对照**。它不是原论文的数值复现，也不能回答 VideoMAEv2 教师是否优于 Wan 教师。

官方代码核对版本为 `aHapBean/VideoREPA@8d581dc301e18546a1808436c724f59d259d09f9`。默认实现将全部时空 token 展平，比较完整余弦关系矩阵，并对差值应用 margin 后取平均；论文 Eq.5 的空间均值加时间均值与此不完全相同。本轮选择**官方默认代码的全 token 对均值定义**，不混用两种归约方式。具体源码路径和定位见下方参考资料。

与官方设置的适配差异事前确定：

| 项目 | 本轮选择 |
| --- | --- |
| 学生和任务 | 当前 Wan NoRef Base 的 LoRA/LR 分支续训，正常固定双三次 ×4 超分 |
| 教师 | 同一 Wan Base 的固定副本或 EMA 副本；不加载 VideoMAEv2/DINO |
| 教师输入 | 与学生相同 HR latent、LR 和噪声方向，但扩散噪声幅度约减半 |
| 对齐层 | 师生均取零起算 block 9 输出，已接受 LR 注入 0、5、9 |
| 时空网格 | 两端原生 9×8×14，同一裁剪和片段；保留首个 latent 位置，不做跨编码器插值 |
| 学生投影 | 单个 Linear(1536,1536)，权重初始化为单位矩阵、bias 为零；2,360,832 个参数 |
| 关系损失 | 官方 full-Gram margin L1，margin=0.1；共同外层权重由训练样本校准 |

官方多层投影和空间降采样服务于异构教师网格。本轮同构网格采用较小的恒等初始化投影，作为明确简化。冻结教师与 EMA 教师使用完全相同的投影结构、初始化、损失和权重。因此，主要对比能估计这套适配配置下教师持续更新的额外作用；不能单独证明 TRD 优于逐 token 对齐。后者需要另一个控制实验，不在本轮追加。

## 三组及共同训练条件

| 组别 | 学生目标 | 教师参数更新 | 本轮回答的问题 |
| --- | --- | --- | --- |
| B 原始基线 | 原始 flow matching 损失 | 无教师 | 公共训练参照 |
| F 冻结 Base | flow matching＋λ×TRD | 始终固定在共同 Base 起点 | 加入该辅助监督配置的作用 |
| E EMA Base | flow matching＋同一个 λ×TRD | 每次成功优化后更新 EMA | 相对 F，动态教师是否额外有益 |

B 不需要无效投影头；F/E 的新增参数完全相同。主比较是 E 对 F，B 对 F、B 对 E 同时报。三组都重新训练，不把旧轮 baseline 分数或权重当作本轮新结果，也不复用诊断更新后的参数。

共同起点为：

`/home/hugo/Projects/VSR_/runs/stage2_protocol33_base_seed20261102_20261106T025845.129456Z/Base/checkpoint_008192.pt`

SHA256：`28356e22e4e718d3c378588793f60732dbc93825ec5d78b8a85c8150060da63f`。

| 设置 | 固定值 |
| --- | --- |
| 学生主干 | Wan2.1-T2V-1.3B，官方 revision `0fad780a534b6463e45facd96134c9f345acfa5b` |
| 可训练部分 | rank-8 attention LoRA、现有 LR residual 分支；F/E 另有学生投影头 |
| 输入 | 33 连续帧、stride 1；HR 128×224；LR 32×56，×4 bicubic |
| 数据 | 现有 1641 训练片段；原视频隔离的已修复 val32/test32，保持清单与像素不变 |
| 随机性 | 训练 seed 20261102；公共数据顺序、增强、FM 时间步和噪声逐项配对；不新增增强 |
| 优化器 | fresh AdamW，lr=1e-4，weight_decay=0.01，foreach=False；无历史动量继承 |
| 更新 | 有效 batch=1，每组 2048 步，constant LR；全可训练参数 global clip norm=1 |
| 精度 | bf16 DiT、FP32 VAE、原 gradient checkpointing；固定文本 embedding，不加载 T5 |
| 主损失 | 原噪声减 clean latent 的 flow target，原 1000 点 scheduler 均匀 index 采样 |
| 条件输入 | 三组都使用正常双三次 LR，不增加遮挡、噪声、丢帧或 reference image |
| 推理权重 | 三组均使用保存的原始学生权重；不把 E 的教师 EMA 当作推理权重 |

共同 Base 已训练 8192 步，这里报告额外 2048 步续训，不能表述为从官方模型零起点训练。现有 Base 训练成本是共同历史成本，不重复计入三组增量耗时，也不隐藏它。

## 教师输入和 EMA 定义

令 z 为本次 FM 使用的同一份归一化干净 HR latent，ε 为公共高斯噪声，c 为同一 LR 条件。

学生输入为 `x_s=(1−σ_s)z+σ_s ε`，σ_s 使用原有 scheduler。教师目标噪声为 `0.5×σ_s`；候选只包括原 native grid 的1000个成对 `(sigmas[i], timesteps[i])`，i=0…999，不包含没有对应timestep的末尾terminal sigma。在其中选择 σ 不大于 σ_s、且距目标最近的一项，精确并列时选择较小 σ，再并列选择较小 index。使用该项对应的真实 timestep 构造教师输入，不把时间步编号直接除以二。记录每次 σ_s、σ_t、index 和实际比值；最小σ边界允许两者相等。

F/E 输入均为 `x_t=(1−σ_t)z+σ_t ε`，并共享相同 LR、文本、时间和空间位置。教师只提供训练目标，eval/no_grad，不加入学生主输入，也不参与推理。所有原始训练噪声区间保留；不按观察到的损失或分数筛选 timestep。

E 的教师与 F 在 step 0 参数逐字相同。E 仅对学生的 LoRA 和 LR 分支更新：

`θ_teacher ← 0.999 θ_teacher + 0.001 θ_student`

在成功的 student optimizer step 之后更新一次，保存更新计数；teacher forward 使用本步更新前的状态。冻结主干保持相同，不更新 VAE、文本编码器或学生投影头的 EMA。teacher shadow 使用 FP32 更新并完整保存，forward 按原 bf16 autocast 规则执行。F 始终保持起点参数不变。半衰期约 693 步，2048 次更新后初始化系数约 12.9%。本轮不同时扫描 decay、层号、噪声比或权重 warmup。

该教师视角比学生噪声更少，是待检验假设，不代表教师一定更准确；它也可能带来难以预测的 HR 信息或跨噪声表示冲突。

## 固定关系损失与共同权重

师生特征形状为 `[B,1008,1536]`。学生特征经过上述投影头，教师特征 detach；两端在通道维做 FP32 L2 normalize，`eps=1e-6`。这是本轮明确的数值约定，与官方调用的默认eps=1e-12不同，不能称逐字节原损失复现。随后得到完整 Gram 矩阵：`G_s=normalize(P(H_s)) normalize(P(H_s))ᵀ`，`G_t=normalize(H_t) normalize(H_t)ᵀ`。

`L_TRD = mean(relu(abs(G_s−G_t)−0.1))`

包含全部有序 token 对、自对角和同帧块，不用光流、top-k、局部边或随机关系子集。1008²=1,016,064 项，其中空间同帧块为112,896项（1/9），跨帧为903,168项（8/9）。自对角共1008项，归一化正常时差值接近零。空间和时间的独立均值同时作为诊断日志，不改变总损失权重。

两张 FP32 Gram 共约7.75 MiB，仅指矩阵存储；特征、差值、梯度、投影头和教师前向的显存另计，不能将该数字当作总峰值。

外层 λ 只校准一次，F/E 共用：在未来获准的预飞中，用训练清单前5个固定片段、预定 σ=.2/.5/.8 最近 native grid、独立诊断噪声流，共15例。使用共同起点和投影，计算**共享学生 LoRA/LR 参数**上的未裁剪 FM 与单位 TRD 梯度全局 L2 范数。λ 为有限正梯度案例的 `median(||g_FM||/||g_TRD||)×0.1`，保留2位有效数字；投影头梯度单独报告，不纳入这个比值。

15例全部保留并报告，包括零梯度案例；任何一例feature、loss或gradient非有限，整项立即失败，不能排除该例继续校准。允许不进入比值计算的仅为有限零梯度案例。至少8例具有有限、严格正的两种梯度，且覆盖至少2个 σ 档，否则停止在目标可用性检查，不进入三组训练。这个规则用于避免共同起点与 margin 导致辅助目标大面积失活；不得为了进入训练静默把 margin 改为0、改层号或跳过失败样本。最终数值 λ 目前尚未测得，只冻结校准公式；无需凭空指定一个常数。正常通过后 F/E 用同一固定 λ 完成2048步，不独立调参，也不根据 val/test 添加第二个权重。

## 正确性和资源检查

本轮当前仅编制方案。以下检查属于未来实现和启动前置工作：

1. CPU：对官方 full-Gram margin 定义做独立参考验证；检查归约、margin边界、THW顺序、首位置、空间/跨帧计数、有限值与教师 detach。用标记 token 检查对应索引，不能只核 shape。
2. CPU：EMA 更新公式、仅更新白名单、F 参数不变、σ与真实timestep匹配；完整 checkpoint 恢复学生/头/AdamW/teacher shadow/计数/数据游标和所有RNG。沿用原constant LR和无GradScaler设置，不虚构不存在的状态。
3. 未来 GPU：B/F/E 在 λ=0 时各3步，公共学生轨迹精确复现，共9诊断更新。teacher目标在step0一致；同噪声同层且恒等投影时原始关系矩阵一致。
4. 未来 GPU：F/E 分别比较2步连续训练与1步保存恢复后再1步，共8次诊断更新，核完整状态和后续输出一致。对齐必须产生进入主干的有效梯度，不能只更新投影头。
5. 未来 GPU：三组各2 warmup＋6 timed完整更新，共24诊断更新；总诊断预算41更新，与6144科学更新分列，诊断权重不复用。训练样本梯度校准是0 optimizer update，耗时独立计入。

记录所有 timed 值、均值/SD、allocated/reserved峰值、整卡占用、teacher次数、EMA次数、投影/主干梯度、梯度夹角、关系差值超过margin的比例，以及σ分箱TRD。资源目标为本机RTX5080 16GiB；性能报告真实teacher开销，不继承旧G3的1.5倍门槛。所有 GPU 工作仍需未来明确启动指令、fresh desktop身份检查和同一shared GPU lock；不启动第二队列或H100任务。

若新实现导致 λ=0 不匹配、teacher泄漏梯度、teacher全失活、非有限或OOM，保留诊断并停止依赖训练；不改变科学设置来自动通过。

## 数据和评价协议

复用固定教师轮已修复清单：train1641、val32、test32，来源和固定裁剪/起点不变。依次保持OpenCV顺序解码、等比缩放和中心裁剪；PyTorch bicubic、antialias=True、align_corners=False、clamp生成×4 LR。沿用配套VAE posterior.mode和逐通道latent缩放，33帧整片编码，边界重置规则不变。

这些 val/test 来源与训练按原视频隔离，并经过声明范围的近重复筛查；但它们已被前几轮实际使用，本轮是**复用已暴露来源的探索性评价，不是新的盲测**。筛查也不是穷尽所有变速、裁剪或转码重复的证明。不在本轮临时更换划分；独立新来源与多训练种子确认留给后续任务。

训练保存512/1024/1536/2048 checkpoint。验证只评价共享起点0，以及各组512/1024/2048，三组最终统一用2048，无best-middle选择、候选淘汰或测试集调参。

| 阶段 | 精确组成 | 恢复次数 |
| --- | --- | ---: |
| 验证曲线 | `(共享0＋3组×3检查点)×32clip×2采样seed` | 640 |
| 最终测试 | `3组最终2048×32clip×2采样seed` | 192 |
| 总计 | 每次完整生成33帧 | 832 |

正常推理沿用 native UniPC20、CFG1、CPU初始噪声42/43、固定文本、FP32 VAE、tile256×448/overlap64和原PNG量化。每次生成全部33帧PNG与MP4，预计27456张预测PNG与832份预测MP4；属于128×224裁剪片段，不是原始长视频全长。推理只加载student LoRA/LR状态，不加载teacher或projection。评分用无损像素，MP4只用于观看。

主要指标LPIPS Alex0.1，全33帧RGB；同时沿用原PSNR、SSIM、TecoGAN tLP100和tOF定义，禁止换Y通道、裁边或时序计算范围。每clip先平均42/43，再等权汇总32个source；显示均值与clip sample SD。原定义详见现有 `docs/fixed_teacher_20261108/data_and_metric_protocol.md`。

固定视觉样本沿用上轮的3个测试片段，每个42/43，共6套完整33帧对比，面板为HR、LR放大、B、F、E。预声明片段为 `-mSvhSt_Q8g_85_92to239`、`0_xW8oRPKRs_100_113to216`、`0gI38E1VsiY_18_125to238`；不依据本轮结果挑选漂亮片段。额外失败案例可补充，但与固定六套区分。

## 评价标准和结论边界

最终主对比为 `Δ_EF = mean(LPIPS_F − LPIPS_E)`；正值表示EMA更好。辅助比较同时报告B−F与B−E。全部逐视频差值、噪声分箱训练诊断和视频效果保留。

10000次配对clip bootstrap，NumPy PCG64 seed20261102，95% linear-percentile区间。三组共享同一重采样clip索引，推理seed先在clip内平均。只有一个训练seed，因此区间仅描述此训练seed下clip差异，不报告多seed显著性或p值，也不把clip SD写成训练seed SD。

| 观察 | 允许的结论 |
| --- | --- |
| Δ_EF描述区间下界>0 | 本配置、本训练seed出现EMA相对固定教师的LPIPS增量信号 |
| Δ_EF区间包含0 | EMA增量证据不足，不能宣称等效或无效 |
| Δ_EF区间上界<0 | 本配置下固定教师的LPIPS更好 |
| 区间退化为[0,0] | 这些输出上未观察到LPIPS差别，不能推广为方法等效 |
| E优于F，但未优于B | 尚无证据表明EMA方案改善原始基线 |
| LPIPS变好，PSNR/SSIM下降 | 报告感知收益及保真代价，不标为全面改善 |

“有增量且保真保持”的候选标签额外要求：E对F及E对B的LPIPS区间下界均>0，且E相对F、B的**未取整PSNR与SSIM均值差分别≥0**。零容差为本轮规划的保守标签，不沿用旧G4的三Base spread门槛，也不改写旧G4失败。均值非负不等于统计上的非劣证明，四个保真差的配对区间仍全部报告。

tLP100/tOF分别报告差值和区间；均值恶化必须披露。它们不临时加入或删除为上述候选标签的AND门槛；即使候选标签通过，也不能在时间指标变差时宣称所有维度改善。视频检查关注真实细节、内容失真、颜色、闪烁和运动连续性。

质量曲线分别用额外更新数与真实累计同步Step墙钟时间。teacher forward、关系矩阵、EMA更新和诊断开销都计入Step；装载/保存、预飞、失败重放、评价另列。以本轮B最终2048验证LPIPS为共同达标目标，只使用实测0/512/1024/2048点，不插值、不外推。若共享0已达标，则该目标不能证明训练加速；较少步数但更长实际时间也不能称时间加速。

本轮完成三组和全部评价即停止；失败、无增益或证据不足都是有效结果。不会因结果不理想自动增加权重、第四组、4096步或种子。后续VideoMAEv2教师、直接特征与关系监督比较、层号或噪声比消融单独立项。

## 预算和预计耗时

| 资源 | 本轮建议预算 |
| --- | --- |
| 科学训练 | 3×2048＝6144 updates＝0.75 RE，1个训练seed |
| 辅助配置 | F/E共享一个校准λ、一个margin、一个噪声比和层号，不做网格搜索 |
| 诊断 | 未来41 optimizer updates，单独计账；不作为科学训练成果 |
| 训练检查点 | 三组各4份，共12份完整恢复checkpoint |
| 完整评价 | val640＋test192＝832 restores；不复用旧行 |
| 新磁盘预留 | 建议至少10GiB，不包含已有官方权重与公共latent缓存；执行前测实际容量 |
| 训练时间估算 | 含三组装载/保存，约50–75分钟 |
| 完整推理评价估算 | 约50–65分钟，不含CPU全量hash核验、分析和人工视觉检查 |
| GPU窗口预留 | 约2–3小时，包含未来预飞余量；不是已经测得的耗时或启动承诺 |

估算锚点：上一轮baseline2048同步Step为704.994秒；在线Wan教师新方法暂估每组17–25分钟。旧EMA G3测过约1.406倍，但仅17/20计时步启用teacher，不能直接用于新全步teacher/TRD。历史val平均约3.72秒/restore，640次约40分钟，test192约12.5分钟。新TRD峰值、完整教师开销和读盘需预飞后实际测量；若预算失配，先记录实测、报告修订需求，不自动减少帧数/样本数来凑预算。

实现、CPU测试、协议审查和交付分析按独立工作时间安排，不计为纯GPU耗时。公共已有缓存成本披露为历史成本；F/E教师本轮在线提取，不为冻结教师单独预计算不同数据流的缓存。

## 实施顺序和交付

| 顺序 | 内容 | 当前状态 |
| --- | --- | --- |
| P0 | 论文/官方损失差异、方案、预算和判定定义 | 本次文档完成 |
| P1 | 在保留旧源码版本的独立开发位置实现TRD、F/E路径和完整恢复；沿用10业务模块 | 未实施 |
| P2 | 针对性CPU测试与独立审查，生成新协议草案 | 未实施 |
| P3 | 明确启动指令后才运行41步预飞与训练集梯度校准；冻结λ和source | 未启动 |
| P4 | 三条全新轨迹各2048步，日志/来源/12CP审核 | 未启动 |
| P5 | 完整val640，锁定最终2048比较；test192及逐行验证 | 未启动 |
| P6 | 配对指标、两曲线、固定6套视频、成本/失败分析与文本同步 | 未开始 |

本轮不把现有 `ft-execute` 或旧 `pilot` 命令冒充为新方法入口；新runner/CLI尚未实现，执行与恢复命令在P1/P2完成后按实际实现交付。当前只新增本方案和跟踪记录，不改11个数值模块、不复用旧运行目录、不部署actual、不恢复已删除的自动巡检。

未来交付包括具体配置与版本、执行/恢复命令、全日志、12CP本机路径、逐视频及汇总指标、两种质量曲线、6套视频、教师激活/梯度/EMA诊断和适配差异。项目文本可继续同步GitHub；权重、视频、PNG、tensor缓存和凭证留在本机。

## 复用资产和参考资料

- 当前基线源码版本：`5e1098bc12808bc6741db8d615f58d04af7b25b8`；本次文档提交不改变数值实现。
- 当前源码目录：`/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync`。
- 原canonical源码 `work/VSR-sync@a91a95f59bb46a81f0c6715df8250f090772d5fb` 与实际runtime `/home/hugo/Projects/VSR_@41407b0e62748d879d2d9e1d8c5b58fa5202d09f` 保持不变。
- clean cohort目录：`/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_clean_cohorts_20261108`。
- val plan SHA：`f6e17c582e84e2a2cb90c2dea016e08195edae3f2f8558c26378b75d25bc27bd`；val dataset SHA：`698fc703e1421305fa47a2f5250dfc175059e1acb09413b71e14fef52a2f9eae`；test plan SHA：`f3224734e43365f1cb8454b14554d362ff7c7e290322fd2a714b78d5a984931f`。
- [VideoREPA 论文](https://arxiv.org/html/2505.23656v1)：TRD与预训练视频模型微调背景。
- [VideoREPA 主TRD分支](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/lora_trainer.py#L324-L354)：full-Gram、margin和统一均值；未来实现须按此版本做独立参考测试。
- [官方启动配置](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/scripts/multigpu_VideoREPA_5B_lora.sh#L74-L85)、[总损失](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/trainer.py#L460-L468)与[原投影头](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/models/cogvideox_align.py#L27-L34)：原margin0.1、外层系数0.5和异构投影。此轮外层系数改为共同训练集梯度校准。
- [Self-Flow](https://arxiv.org/html/2603.06507v1)：较清晰教师视角的参考；本轮全片单一σ教师不属于其逐token双时间步完整实现。
- 现有固定教师轮 `docs/fixed_teacher_20261108/final_report.md`、`data_and_metric_protocol.md` 与旧EMA `docs/local_continuation_20261107/G4_report.md`：只作为既有证据与成本参照，不替代本轮结果。
