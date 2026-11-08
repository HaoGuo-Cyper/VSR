# VideoMAEv2–VideoREPA / Wan VSR：在线教师优先资源方案 v2

日期：2026-11-08。本v2在任何下一轮GPU前向或训练启动前，取代以预计算教师feature cache为默认的v1草案。选择在线teacher是为了更贴近原VideoREPA，并直接核算其完整每步成本；不是根据质量分数修改方案。本文是独立的下一轮准备记录，不是运行完成报告。最新人工已授权在当前 Wan teacher-target 实验完整评价、分析和交付后，实施并启动保留 VideoMAEv2 教师的实验。当前只能做只读研究、独立资源下载及 CPU 检查；不得修改正在运行的冻结源码、启动第二个 GPU 队列或读取当前 partial 质量分数。

## 1. 可执行的最小范围

建议两个独立且配对的训练臂：

| 臂 | 目标 | 初始化和预算 |
|---|---|---|
| baseline | 现有 flow-matching VSR 损失 | 同一个已审计 Base01，fresh AdamW，2048 更新 |
| videomaev2_trd | 相同 FM + 0.5 × 官方 full-Gram margin-mean TRD；margin=0.1 | 相同 Base01、数据/噪声/时间步，fresh AdamW，2048 更新 |

总计 **4096 科学更新 = 0.5 RE**，一个训练种子 `20261102`，每臂仅一个设置。没有 frozen-Wan/EMA-Wan 臂，不增加权重网格、训练种子或4096续训。以小 crop、LoRA 和短预算检验实际收益；这些是资源约束选择，不是原 VideoREPA 或视频超分必需的限制，不能声称复现论文数值。

在线teacher先于student graph执行，eval/no_grad，不保留其autograd；原始HR RGB缓存只是输入数据，不是教师特征缓存。两臂都从原 Base01 的8192步 checkpoint 启动，不复用本轮 Wan-TRD 或固定教师研究的续训参数。共同优化器采用现有 constant LR=1e-4、weight decay=.01、foreach=False、gradient clip=1、有效 batch=1，Wan bf16/配套 VAE FP32、rank8 attention LoRA 与现有 LR 注入分支。头初始化独立 RNG；教师提取不消耗公共数据/噪声/时间步 RNG。保留33连续帧、128×224 HR、32×56 LR、固定双三次×4、固定text。

## 2. 方法保留与适配边界

方法细节由配套 `videorepa_videomaev2_fidelity_review_20261108.md` 的固定版本源码证据约束。关键选择如下。

- 保留官方 **VideoMAEv2-B/16、K710 distilled-from-giant checkpoint**，干净 HR 输入，教师 eval/no_grad/`requires_grad_(False)`。输出是第12 block后全部 patch tokens，既不加最终norm，也不取CLS/分类head或平均池化。
- 保留原默认 one-based layer18：Wan zero-based **block17** 后、最终norm之前。CogVideoX-2B和Wan各30层时均为60%深度；不沿用旧实验block7/9。
- 保留三层MLP：`1536→2048→2048→768`、两SiLU、bias，共 **8,917,760** 参数。原官方额外stride3空间Conv不用于本机新网格；这属于明确的结构适配，不能称完整投影头逐字复制。
- 教师只取同一 HR crop 的帧1..32，ImageNet归一化；本机128×224已被16整除，不再按原480×720数据的1/3缩小。tubelet2、patch16得到 **16×8×14×768**。教师名义tubelet覆盖[1,2]至[31,32]，最终token仍有整段自注意力上下文。
- 学生完整9 latent照常参与FM；辅助项也先对全部9 latent做逐token MLP，再去掉第0位置留下latent1..8，trilinear时间8→16、align_corners=False，空间保持8×14。该对应延续原“学生插值到教师”方向，是近似网格映射，不是宣称因果VAE每latent恰为独立四帧平均。
- 教师位置编码从原 **8×14×14固定sin/cos** 出发，先bicubic空间→8×14，再trilinear时间→16，align_corners=False；不能重建新长度sin/cos，也不能照搬官方写死的时间×3/160×240。
- 保留完整ordered-pair Gram的统一mean、对角线及全部同/跨帧pairs：`mean(relu(abs(Gram(S)-Gram(T))-.1))`。通道normalize原默认eps=1e-12与教师LayerNorm eps=1e-6是不同设置；不可混用。原论文印刷公式与默认代码聚合有差异，本轮选择可复现的默认代码目标并披露。
- **lambda固定.5，不按梯度重新标定**，margin固定.1。15个train-only梯度probe只诊断finite、信号和噪声区间，不据其或质量分数选另一lambda。先测原源码实际dtype路径，记录teacher/projector/normalize/Gram/loss dtype；不能静默继承上一轮统一FP32关系计算或lambda.3。若确需数值稳定性改变，先保留失败与变更依据，不能声称原路径已通过。

## 3. 官方资源已取得，尚未运行教师

最初在 `/home/hugo/Projects`、`/home/hugo/.cache` 及 WORK 的含隐藏/忽略文件检索中，没有发现 VideoMAEv2/VideoREPA代码或该权重。此结论限定于已搜索目录。随后按人工后续准备授权，仅下载官方B权重与小份官方源码到全新目录：

`/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_assets_20261108/`

| 项目 | 实际证据 |
|---|---|
| 权重 | `vit_b_k710_dl_from_giant.pth`，**173574417 bytes** |
| HF固定revision | `706cc172d65ebd4dedbee3f9c0183a93df9fa125` |
| 官方与本机SHA256 | `8141a6955e0700d11bf15928fe6d61e5cfe482606fed8cfdddb1b922c0fd88ec` |
| 下载记录 | `download_receipt.json`，成功且官方size/hash完全匹配；本次耗时52.676808秒，网络准备成本，不是GPU成本 |
| 官方特征类 | `VideoREPA_VideoMAEv2_8d581dc.py`，VideoREPA commit `8d581dc301e18546a1808436c724f59d259d09f9`；本地SHA `a2d7cb25263bc96566c6dcfd3ce163c3fc32ebd016a4e0603443abbce48673f7` |
| 真实CPU strict审计 | `checkpoint_cpu_audit.json`，162个FP16 tensor；明确排除4项，剩余158个encoder tensors **strict=True**，missing=0、shape mismatch=0，载入后全部tensor与checkpoint转换到目标dtype逐一相同 |
| 仅排除项 | `fc_norm.weight/bias`（官方特征路径不使用）、`head.weight[710,768]`和`head.bias[710]`（分类头）。没有其他encoder漏载 |
| 实际模型 | **86,225,664**参数，12 blocks/12 heads/width768，Conv3d权重[768,3,2,16,16]，LayerNorm eps1e-6，初始非参数pos[1,1568,768] |
| 环境 | 已有 Python3.12.16、torch2.14.1+cu130、timm1.0.30、numpy2.5.2；CPU线程2；没有安装或修改依赖 |
| 执行范围 | 只构造模型、读权重、strict加载/数值逐项比对、eval/freeze检查；**没有teacher forward，CUDA未初始化，0 optimizer update** |

`checkpoint_cpu_audit_environment.json`保留一次纯CPU导入脚手架问题：`importlib`需先注册`sys.modules`供`timm.register_model`查询；修正仅限独立audit harness，官方源码未修改。现有timm版本与原requirements不同，CPU类加载通过不等于未来前向数值已验证。官方loader的strict=False没有被用于通过检查。

来源：官方 [VideoREPA teacher代码](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/models/ssl/VideoMAEv2.py)、[VideoMAEv2 model zoo](https://github.com/OpenGVLab/VideoMAEv2/blob/29eab1e8a588d1b3ec0cdec7b03a86cca491b74b/docs/MODEL_ZOO.md)、[HF固定提交](https://huggingface.co/OpenGVLab/VideoMAE2/commit/706cc172d65ebd4dedbee3f9c0183a93df9fa125)、[固定权重下载](https://huggingface.co/OpenGVLab/VideoMAE2/resolve/706cc172d65ebd4dedbee3f9c0183a93df9fa125/distill/vit_b_k710_dl_from_giant.pth)。VideoREPA代码Apache-2.0、上游VideoMAEv2代码MIT、HF权重标记Apache-2.0分别记录；本地保留`LICENSE_VideoREPA`。不需要整个18.6GB模型仓库、ViT-g、CogVideoX权重或OpenVid数据。

## 4. 数据、模型和评价资源可复用

| 内容 | 已存在位置/身份 |
|---|---|
| 官方Wan与VAE | `/home/hugo/Projects/ReprVSR-pretrained/models/Wan2.1-T2V-1.3B-Diffusers`；官方revision `0fad780a534b6463e45facd96134c9f345acfa5b` |
| 原始HQ HR | `/home/hugo/Projects/ReprVSR-pretrained/data/HQ-VSR` |
| Base01及原配置 | `/home/hugo/Projects/VSR_/runs/stage2_protocol33_base_seed20261102_20261106T025845.129456Z/config.json` 与 `Base/checkpoint_008192.pt`；CP SHA `28356e22e4e718d3c378588793f60732dbc93825ec5d78b8a85c8150060da63f` |
| 配对训练缓存 | 同Base01 run的producer cache，5024 artifact seal，1641 train+32原val记录；只读取已固定training_ids对应的1641个`paired_rgb.pt`/`fresh_latents.pt`及metadata，**不从有损hr.mp4重算教师** |
| 固定text | `/home/hugo/Projects/VSR/runs/formal_hq_v2_noref_20261102T142214.588571Z/fresh_text_embeddings.pt`，SHA `1dd219cde3005b4d16febaad67ad87e79c1403ae9bdaae69f2723ab9cbf239c2`；旧binding无bytes时按现有canonical binding规范化 |
| clean val32 | WORK下`fixed_teacher_clean_cohorts_20261108/validation_assets/dataset.json`，SHA `698fc703e1421305fa47a2f5250dfc175059e1acb09413b71e14fef52a2f9eae` |
| val计划 | WORK下`fixed_teacher_clean_cohorts_20261108/iteration_000/val_plan.json`，SHA `f6e17c582e84e2a2cb90c2dea016e08195edae3f2f8558c26378b75d25bc27bd` |
| clean test32计划 | 同iteration目录`test_plan.json`，SHA `f3224734e43365f1cb8454b14554d362ff7c7e290322fd2a714b78d5a984931f` |
| 已有test RGB | WORK下`fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/test_assets/dataset.json`，SHA `b348ea155d774c928ef01a3897494546bba18dae9040c950ee2d4eeae88ab206`（本次已重新核文件SHA） |
| 指标 | 现有`/home/hugo/Projects/VSR/weights/`的LPIPS-Alex0.1/AlexNet及仓库TecoGAN tLP100/tOF实现；正常推理仍用现有统一入口 |

WORK为 `/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work`。本轮不改训练数据、不重新划分。近重复清理此前替换1val/3test，保留31/29原窗口，screen为声明范围内0flag而非穷尽offset/crop/transcode证明。test历史模型已使用过，故只能称 **source-held-out但重复使用的探索性测试集**，不能称blind/pristine。下一轮若需独立确认，应另行冻结新来源测试集；不在本轮预算中悄悄扩集。

训练HR读取的精确模板为 `/home/hugo/Projects/VSR_/runs/stage2_protocol33_base_seed20261102_20261106T025845.129456Z/clip_{0..1640}/paired_rgb.pt` 的 `hr` 字段；本次实际CPU读clip0确认FP32、[1,3,33,128,224]、范围[0,1]，不是[-1,1]，因此不能再错误地做一次`(x+1)/2`。1641个文件均存在，metadata record_id与selection.training_ids逐一同序；合计已有RGB容器38,432,265,948B。本次只做metadata/存在性核验和clip0 payload检查，没有重复全量38GB二进制hash；未来协议仍绑定原seal并做所需严格校验。记录在独立资源目录 `reusable_input_assets_audit.json`。

val/test各32个已物化HR/clean NPY均存在。可直接复用例如val路径 `WORK/fixed_teacher_clean_cohorts_20261108/validation_assets/oHeY69nSZkk_9_0to379/{HR,clean}.npy`，test路径 `WORK/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/test_assets/8I_mCCVLves_35_43to189/{HR,clean}.npy`；实际所有路径、source indices与SHA取各自dataset.assets，不能另选帧。**不需要重新decode原视频**。旧DINO缓存只有教师features及元数据，不能作为HR RGB；旧teacher features也不能转换成VideoMAEv2 targets。如任何已绑定RGB缺失，先报告具体项，不能任意重新解码替代；若后续显式恢复必须严格复现相同source indices/resize/crop/归一化并核同字节。

缓存clip metadata已包含源SHA、source_indices、stride、FPS、原图/resize/crop和退化。例train clip0为33个连续索引921..953，HR128×224来自resize128×228后crop[0,2,128,224]；LR使用`torch bicubic`、antialias=True、align_corners=False、clamp[0,1]、所有帧同参数。这足以绑定教师到实际HR区域，仍需新在线重复提取一致性验证。

## 5. 在线教师优先及静态显存分析

默认 **每个method训练step从配对HR在线提取VideoMAEv2特征**。教师约86.2M参数、FP32参数约328.925MiB，不能因为此前DINO采用cache就假设它也必须预计算。保持教师eval、所有参数requires_grad=False、torch.no_grad；在构造学生前向计算图之前完成teacher preprocess/forward，只保留最后一层目标及必要诊断标量。不得在列表/日志里累计教师中间激活、attention或跨step目标。

每步建议顺序：读取已封存HR/LR/latent → 根据公共RNG取得样本/时间步/噪声 → 计时开始 → HR转移与教师预处理/在线前向 → 原学生FM前向+中间层MLP/网格映射+TRD → backward/clip/optimizer/finite检查 → 同步计时结束。HR输入实际已是FP32[0,1]，直接ImageNet normalize，不再做一次(x+1)/2；教师保留原FP32参数路径，原关系计算实际dtype仍须preflight捕获，不自动统一到FP32。不得把teacher forward藏到计时边界外。

静态公式按实际官方eager Attention（`q@k.T`后独立softmax，并非假定FlashAttention）计算。N=16×8×14=1792，D=768，heads12、batch1，表中激活按FP32作资源核算：

| 存活对象 | 静态存储 |
|---|---:|
| 全教师参数 | 328.925MiB |
| 一张12-head attention score/prob矩阵 | 147MiB |
| softmax分配时输入score与输出prob短暂重叠 | 294MiB |
| QKV投影合并张量 | 15.75MiB |
| 一个N×D token张量（例如最终目标） | 5.25MiB |
| 一个FFN4D中间张量 | 21MiB |
| 一份33帧HR输入FP32 | 10.828MiB |
| 学生投影头参数+grad+两个AdamW moments，均按FP32估算 | 136.074MiB |
| 两张完整TRD Gram，均按FP32 | 24.5MiB |

由于teacher处于no_grad且12blocks顺序执行，**不把12层attention同时累加**；softmax/FFN两种峰值也不总在同一时刻。加上LayerNorm/残差/位置编码/预处理的临时数组，可将独立teacher可见live工作集粗略看作 **约0.7–1.0GiB**。这不是实测保证或严格上界：CUDA/cuBLAS workspace、allocator碎片/缓存、输入dtype和算子实现都可能另占内存。

设W为Wan/VAE/当前optimizer的常驻量，S为学生step的峰值，则全step峰值概念上是 `max(W + teacher参数 + teacher临时量, S + teacher参数 + retained target + 新head/关系开销)`，不是把两个完整前向图相加。教师大attention在student graph建立前即可释放；学生阶段常驻teacher参数约.321GiB和目标.005GiB，另有head/关系梯度开销。

已有已完成Wan-TRD preflight中的baseline峰值allocated3,845,416,960B、reserved3,946,840,064B，仅说明此小crop下旧基线有显存余量，**不是新block17/3层MLP在线teacher的实测峰值**。静态分析支持优先在16GiB做真实预检，无理由现在就构建8.4GiB feature cache。但正式可行性仍取决于完整26步诊断中的实际peak allocated/reserved、NVML占用、teacher目标存活和清理检查。

如果在线预检发生OOM或不可接受的运行问题，保留记录并停止依赖训练，讨论是否明确改为feature cache或其他适配；**绝不自动fallback、改变teacher dtype/网格/层号、调margin或删除慢值**。v1计算的1641条FP32目标payload8.413330GiB只作为潜在替代成本记录，本方案不创建该cache。

硬件静态检查为RTX5080报告16303MiB、主机约29.95GiB RAM；当次磁盘可用590,071,926,784B（动态值）。在线方案新增磁盘建议预留 **30GiB**，包含8个训练CP、576组33帧PNG/MP4与记录，不包含已有输入资产；没有8.4GiB教师cache。576组裸RGB像素1,634,992,128B只是像素总量，压缩文件和CP体积以真实写入统计。数据读取保持单clip流式，不能一次装入已有38GB paired RGB。

## 6. 固定诊断预算及准确forward计数

| 诊断项 | optimizer updates | 在线teacher forwards |
|---|---:|---:|
| baseline与TRD lambda0，各3步；lambda0直接绕过teacher/head，公共状态/RNG/FM逐步匹配 | 6 | 0 |
| TRD连续2步，与1步保存/完整恢复再第2步；student/head/AdamW/LR/4RNG逐一核验 | 4 | 4 |
| baseline与TRD各2warmup+6timed；每个完整TRD Step计teacher preprocess/forward | 16 | 8 |
| 前5个固定train样本×nominal sigma .2/.5/.8的15个probe cases，每case FM/aux两次VJP、共30VJP，0更新，不跨case缓存teacher | 0 | 15 |
| 固定train0、train1各独立提取两遍；同device、dtype、batch1、同预处理逐字比较 | 0 | 4 |
| **预检合计** | **26** | **31** |

科学method轨迹另有 **2048次在线teacher forward**；baseline科学轨迹0次。上述31不含“临时加的”额外warmup/资源forward，任何额外次数如不可避免必须逐项先登记，不能混在0-update字样中隐藏成本。两次一致性检查意为两个样本的两个重复对，共4次forward，不是cache核验。

全部26诊断更新的权重最终丢弃。CPU检查覆盖原full-Gram值/梯度/归约、时空映射首尾、teacher frozen/eval、projection进入student梯度、lambda0基线精确、student-only正常推理CP兼容、头RNG隔离。原checkpoint无teacher训练状态；完整恢复绑定官方固定weight/source/pos/preprocess hashes，恢复student/head/optimizer/LR/RNG/data cursor；不把86M冻结教师复制成可训练shadow或EMA。

预飞仅做student-only state加载/一致性检查，不新增诊断native restore；本轮正常native恢复调用严格为val448+test128=576次。预检可以设置30分钟观察上限，超过则记录具体瓶颈再决定运行安排，不增加诊断更新预算。必须保留所有6个timed值，不删outlier。

lambda=.5、margin=.1固定；VJP只检查finite和信号，不重新缩放lambda、不选第二权重。真实nonfinite、teacher梯度未冻结、对齐有效梯度缺失、OOM、重复提取不一致或完整恢复不精确时，停止依赖GPU并保留证据，不暗改精度/容差/层号/预算。正常工程故障恢复按完整状态和新attempt记录执行，不丢中断成本。

## 7. 完整评价及成本归因

保存每臂512/1024/1536/2048共8个CP。评价预算保持不变：

| 阶段 | 角色 | 完整恢复数 | 全33帧PNG |
|---|---|---:|---:|
| validation | shared0 + 两臂各512/1024/2048，共7roles；32clips×sampler42/43 | **448** | 14,784 |
| test | 两臂最终2048；32clips×sampler42/43 | **128** | 4,224 |
| 合计 | 不拼历史行、不选中间最佳CP、不加权重 | **576** | **19,008** |

沿用native UniPC20/CFG1、固定text、FP32VAE、固定量化、空间tile256×448/overlap64、sampler42/43；每行生成33帧128×224完整crop及MP4，不称原长视频/原分辨率。val完整verify后锁定两臂最终CP再test，无选参、test不调设置。固定3clips×两sampler生成HR/LR/baseline/VideoMAEv2四panel对比视频，实际观看后报告细节/失真/闪烁。

主比较仍是按clip先平均两个sampler的LPIPS(baseline)−LPIPS(method)；报告均值、clip SD、每clip配对差，10000 paired-clip PCG64/seed20261102/95% linear-percentile条件描述区间。strict positive区间支持感知收益，strict negative为退化，跨/触0为证据不足，exact[0,0]单列无观察差异。PSNR/SSIM/tLP100/tOF全报；PSNR或SSIM点估计负向时明确质量权衡，不称全面提升。单trainingseed、历史使用test和小crop的限制保持披露，不套旧G4规则。

效率曲线横轴分别为更新数和 **committed synchronized full Step wall**，在线teacher preprocess/forward完整计入method step；数据磁盘读取单列，设备转移/teacher输入预处理在step边界内。记录teacher部分时间是full Step的组成，不能再重复加到总step时间。完整attempt墙钟包括模型/teacher加载、输入I/O、CP保存和清理，单独列出；诊断/评价/下载另列。

本方案 **offline教师feature cache构建成本为0，因为未创建**；不能说teacher计算是0，2048个method step都付出真实在线成本。官方权重下载52.676808秒是一次性网络准备，教师加载一次/attempt是训练attempt成本。旧公共RGB/VAE缓存是共同已有资产，双方同口径披露为sunk preprocessing，不每臂重算。吞吐和GPU效率不能据只算student的子计时宣称加速。

不猜测新VideoMAEv2吞吐。短测后使用 `2048×(baseline_full_step_mean + method_full_step_mean) + 576×restore_mean + measured_load_data_IO_verify` 做预算；均值保留所有timed行。例：**若**baseline≈.35s、包含在线teacher的method≤1s、每次恢复≤6s，则训练约46.1分钟、评价约57.6分钟，另给加载/诊断/验证/导出20–40分钟，总约2–3小时；这是排期条件假设，不是新方法测量或承诺。若真实短测更慢，更新预计时长而非删预算或将teacher移出计时。

## 8. 下一步与执行边界

1. 当前Wan-TRD完整val/test/analysis/视频交付并确认父服务结束、shared lock释放后，才进入本研究实现/预检。当前11core与bound helpers不改；旧actual41407b0、canonical和固定教师工作树保留。
2. 新独立源码工作树/dated runs内保留10业务模块，实现在线VideoMAEv2接口、固定pos网格、block17 tap、官方3层MLP和TRD；所有不可避免/资源约束差异前瞻记录。
3. 资源下载及CPU strict加载已完成；数值forward、真实dtype、显存和新方法正确性尚未测。先完整CPU审查，再唯一共享GPU队列执行26诊断更新（其中12次teacher forward）及19次零更新teacher forward，总teacher31次；通过才两臂4096科学更新、val448、test128、verify/报告/视频。不得把本文件当已通过GPU门槛。
4. 无teacher feature cache、无offline extraction job、无EMA分支；只有真实资源问题才停报讨论另一个显式方案，不能后台自动构建缓存继续。
5. 公开GitHub同步仅代码、配置、统计、完整文本日志/metadata；官方weight、CP、PNG/MP4和tensor不公开。交付放原OUTPUTS。本轮之后只提后续多seed/较大crop/投影对照建议，不自动扩展。

本轮能判断这套保持原教师/默认关系目标且在线计费的Wan小规模配置是否优于matched baseline。与旧Wan frozen/EMA横向差别同时包含teacher输入、层号、头、网格和dtype路径，不能将所有效果唯一归因于教师身份。
