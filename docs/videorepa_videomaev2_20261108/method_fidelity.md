# VideoREPA / VideoMAEv2 忠实适配核查（只读研究，2026-11-08）

本文件为下一轮方案提供源码证据；没有启动 GPU、下载权重、读取当前 Wan-TRD 部分质量分数或修改冻结源码。建议在当前 Wan-TRD 三组完整交付之后，另立 baseline / frozen VideoMAEv2 两臂实验。它应称为“保留原 VideoREPA 教师与关系目标的 Wan VSR 适配”，不能称复现原论文数值。

## 固定来源与已核实权重

- VideoREPA 官方代码固定为 `8d581dc301e18546a1808436c724f59d259d09f9`，不是滚动 main。
- 论文核对版本是 [arXiv v1](https://arxiv.org/html/2505.23656v1)。原任务为 CogVideoX 文本生成视频的物理合理性；论文并未验证配对 ×4 超分、LPIPS 改善或当前小规模 LoRA 续训。因此迁移后的恢复收益仍需本项目实验判断。
- 官方训练入口明确构造 `vit_base_patch16_224()`，读取 `vit_b_k710_dl_from_giant.pth`：[teacher 初始化](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/lora_trainer.py#L31-L40)，[README 下载位置](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/README.md#L182-L186)。不要替换为普通 VideoMAE、ViT-g、K400 微调版或随机初始化 ViT-B。
- OpenGVLab 官方 [model zoo](https://github.com/OpenGVLab/VideoMAEv2/blob/29eab1e8a588d1b3ec0cdec7b03a86cca491b74b/docs/MODEL_ZOO.md#L32-L41) 将该 checkpoint 列为 K710 上从 `vit_g_hybrid_pt_1200e_k710_ft` 蒸馏的 ViT-base。源代码版本为 `29eab1e8a588d1b3ec0cdec7b03a86cca491b74b`。
- 官方 HF revision **`706cc172d65ebd4dedbee3f9c0183a93df9fa125`** 的 [提交页](https://huggingface.co/OpenGVLab/VideoMAE2/commit/706cc172d65ebd4dedbee3f9c0183a93df9fa125) 给出精确 LFS bytes **173574417**、SHA256 **`8141a6955e0700d11bf15928fe6d61e5cfe482606fed8cfdddb1b922c0fd88ec`**。固定版本的 [下载地址](https://huggingface.co/OpenGVLab/VideoMAE2/resolve/706cc172d65ebd4dedbee3f9c0183a93df9fa125/distill/vit_b_k710_dl_from_giant.pth) 已由官方文件与提交链接确认；本次没有实际下载、hash 文件或载入 tensor，不能把远端 metadata 核对写成真实加载通过。
- [VideoREPA 代码许可](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/LICENSE) 是 Apache-2.0；[VideoMAEv2 上游代码](https://github.com/OpenGVLab/VideoMAEv2/blob/29eab1e8a588d1b3ec0cdec7b03a86cca491b74b/LICENSE) 是 MIT；[官方 HF 页面](https://huggingface.co/OpenGVLab/VideoMAE2) 标记权重为 Apache-2.0。代码与权重的标记分别记录，不能混用。

## 默认脚本及真实代码

| 项目 | 官方固定版本的真实设置 | 对照证据 |
|---|---|---|
| 教师 | VideoMAEv2-B，patch16、tubelet2、768 通道、12 blocks、12 heads、MLP ratio4、qkv bias；block LayerNorm eps=1e-6；dropout/drop-path 默认0 | [构造器](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/models/ssl/VideoMAEv2.py#L330-L394)，[factory](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/models/ssl/VideoMAEv2.py#L547-L559) |
| 教师 feature tap | 第12 block 后的全部 patch token；最终 norm / fc_norm / classifier 被注释。没有 CLS/register、没有 mean pooling、没有多个层平均 | [forward_features](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/models/ssl/VideoMAEv2.py#L471-L498) |
| 教师模式 | eval，提取在 no_grad 内；教师不进入 optimizer。源码 `param.require_grad=False` 是拼写错误，不是真正冻结参数标记；迁移应改为 `requires_grad_(False)` 并验证梯度为空 | [初始化](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/lora_trainer.py#L31-L40)，[no_grad](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/lora_trainer.py#L201-L208) |
| 教师输入 | 干净训练视频，不是 student 的 noisy latent，也不是较低 sigma 的 Wan；完整49帧480×720样本去 RGB frame0，保留48帧 | [原始输入及去首帧](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/lora_trainer.py#L162-L191) |
| 教师预处理 | 原训练 RGB[-1,1]→[0,1]→ImageNet mean(.485,.456,.406)/std(.229,.224,.225)，然后逐帧 bicubic resize 到 H//3,W//3，即160×240；该 interpolate 未传 antialias（默认False）或 align_corners（等效False） | [预处理](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/lora_trainer.py#L168-L208) |
| 教师 token 网格 | 48/2×160/16×240/16 = 24×10×15 =3600，flatten顺序 T,H,W；Conv3d 无 padding | [patch embedding](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/models/ssl/VideoMAEv2.py#L267-L302) |
| 学生对齐层 | one-based18，即执行完 zero-based block17 后，norm_final 之前；2B/5B 脚本同值 | [feature tap](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/models/cogvideox_align.py#L181-L208)，[2B script](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/scripts/multigpu_VideoREPA_2B_lora.sh#L74-L84) |
| 学生 projector | Linear(D,2048)→SiLU→Linear(2048,2048)→SiLU→Linear(2048,768)，含bias；后续还有独立的可训练 Conv2d(768,768,kernel3,stride3,bias=True) | [MLP](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/models/cogvideox_align.py#L27-L34)，[构造](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/models/cogvideox_align.py#L110-L118) |
| 学生网格适配 | 全13×30×45先经MLP，去第0 latent→12×30×45，trilinear时间×2→24×30×45，再上述stride3 Conv→24×10×15；不是把教师时间平均到学生网格 | [真实分支](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/lora_trainer.py#L292-L354) |
| 默认 loss | 完整时空 normalized Gram 差异经过 margin hinge，再对所有元素mean；margin=.1，外层lambda=.5 | [TRD](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/lora_trainer.py#L324-L354)，[总loss](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/trainer.py#L456-L468) |

原数据先按 source/声明区间抽帧，再中心裁剪到目标宽高比并 bilinear resize 至训练尺寸；此步 align_corners=True、antialias=True，与教师分支后续 bicubic 是两个独立处理步骤。[dataset utils](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/datasets/utils.py#L98-L226)，[中心裁剪与resize](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/datasets/data_transform.py#L28-L84)。不能把其49帧抽样策略描述成本项目33连续帧，也不能直接复制裁剪而破坏本项目已配对的HR/LR区域。

## 损失、论文与 dtype 的差异

记投影并完成网格映射后的学生为 S，clean HR 教师为 T，均为 B×N×768；通道L2归一化为 U,V。默认代码等价于

`L_TRD = mean(relu(abs(U @ U.transpose(-1,-2) - V @ V.transpose(-1,-2)) - 0.1))`

所有 ordered pairs、同帧/跨帧、自身对角线都留在统一分母中。若对齐时间长度F，则同帧占1/F、跨帧占(F-1)/F。原F24是1/24与23/24；Wan建议F16是1/16与15/16。只算上三角而不正确加权、排除对角线、分别给空间/时间相同权重，均会改变目标。官方还有 only_spatial/only_temporal 分支，它们不是默认TRD。

论文 Eq5 分别归一化空间项和跨帧项后相加，打印公式未写margin；Appendix A补充VideoMAEv2 margin=.1，Appendix C讨论lambda=.5及深度18。本项目若优先可复现官方默认代码，应明确选“full-Gram margin-mean”，不能称与Eq5逐字同式。[论文 Eq5及附录](https://arxiv.org/html/2505.23656v1#S3.SS3)。

dtype 必须区分可直接证明的源码事实与未来实测：教师创建后只有 `.to(device)`、没有显式 dtype 转换，默认参数为FP32；其输入由 `.float()` RGB预处理得到。教师没有交给 accelerator.prepare，提取在no_grad中。学生launch为bf16；TRD在空间Conv前显式 `align.to(torch.bfloat16)`，两端normalize调用没有指定eps（PyTorch默认1e-12），两个bmm也没有统一`.float()`。不能把这条路径写成“官方统一FP32 loss/eps1e-6”；1e-6实际是教师LayerNorm的eps。外层混精度/DeepSpeed会影响实际运算dtype，当前未运行官方脚本来测量，故应在移植preflight捕获并记录teacher、projector、归一化、Gram及loss实际dtype。先测试原路径的值、finite和梯度；如果选择FP32关系计算，需另记为数值稳定性适配，不能无记录地继承上一轮Wan-TRD设置，也不据分数选择dtype。[bf16 script](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/scripts/multigpu_VideoREPA_2B_lora.sh#L29-L39)，[dtype关键调用](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/lora_trainer.py#L329-L354)。

## 位置编码与 checkpoint 能证明到哪里

原构造器先为16帧、224×224、tubelet2/patch16的 **8×14×14** 网格，按展平token序号产生1D sin/cos。它是普通Tensor，不是可训练Parameter，也不是注册buffer，默认不在state_dict中。官方加载权重后将它reshape回T,H,W：先逐时间bicubic空间14×14→10×15，再trilinear时间8→24；二者align_corners=False。代码根据768通道硬编码目标160×240和时间×3，`align_video_resolution`并不能自动泛化到任意目标。[sin/cos](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/models/ssl/VideoMAEv2.py#L305-L376)，[插值](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/models/ssl/VideoMAEv2.py#L415-L445)。

Wan适配应保持上述原始8×14×14 sin/cos及插值顺序，只将目标显式设为16×8×14；不要按新N=1792从头重建1D sin/cos，也不要把它当作checkpoint学习到的pos。位置Tensor的CPU字节/hash应记录，在线/缓存使用同一版本。

原loader读取`module`键并可去`module.`前缀，fallback读`model`且可去`encoder.`；用strict=False仅打印missing/unexpected。[loader](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/models/ssl/VideoMAEv2.py#L500-L526)。上游分类模型有fc_norm与head，而VideoREPA特征类将它们去掉；因此合理预期存在这些非encoder键，但**仅靠网页/源码不能证明真实checkpoint恰好只有这几项差异**。[上游分类模型](https://github.com/OpenGVLab/VideoMAEv2/blob/29eab1e8a588d1b3ec0cdec7b03a86cca491b74b/models/modeling_finetune.py#L390-L402)。

后续下载完成后必须CPU核对官方bytes/SHA，列出顶层键、每个encoder键和shape，使用精确允许的classifier/fc_norm排除项，要求encoder missing=0、unexpected=0、全部参数被权重赋值；未识别的pos/decoder/teacher包装键不能静默忽略。128×224/32帧只改变token数与非参数pos，不改变patch Conv或encoder权重shape，这说明结构上可加载，但当前不构成真实strict加载通过证据。按特征类构造式推算参数量86,225,664，未来应以实际模型count核对。

## 建议的 Wan 映射及变化分类

| 选择 | 建议 | 类型与理由 |
|---|---|---|
| 核心方法 | 上述官方VideoMAEv2-B、clean HR、last-block patch tokens、full-Gram hinge、margin.1/lambda.5 | 保留原版核心；不加入EMA、低噪声Wan教师、DINO逐帧平均或mask |
| 学生 | 现有Wan2.1-T2V-1.3B VSR及原LR条件、flow loss、配对bicubic数据 | 任务适配：CogVideoX→Wan，生成→SR；原diffusion target不能覆盖现有FM训练 |
| 对齐深度 | one-based18，即Wan block17后、最终norm前 | 保留原绝对层。Cog2B30层与Wan30层均60%；Cog5B42层是42.9%，不能泛称相对深度完全一致。不同于当前Wan-TRD block9，也不同于第一轮block7 |
| HR与教师 | 对同一33帧128×224 HR crop，仅辅助分支去frame0；保留原生128×224，无额外裁剪/缩放；teacher Conv3d产生16×8×14 | 本机小crop条件下的几何适配选择：宽高均被16整除，完整区域保持。不是声称原官方160×240可原封不动使用 |
| 学生时间 | 完整9 latent经过Wan；对齐用latent1..8，先MLP，trilinear时间8→16、align_corners=False，空间8×14不变 | 保留“学生插值到教师”方向及首latent排除。仅辅助项排除首位置，原FM loss和推理仍覆盖全部33帧 |
| 学生空间 | 不施加原stride3 Conv | 网格已同为8×14，原Conv会误降到2×4；明确为删去原投影结构的空间适配，不能称完整原投影头逐字复用。无需求添加另一随机1×1 Conv |
| 学生MLP | 1536→2048→2048→768，两SiLU，默认随机Linear初始化，独立RNG | 保留官方MLP深度/隐藏维/输出维，输入维随Wan变化；不是identityLinear。参数8,917,760 |
| 教师位置编码 | 原8×14×14 sin/cos→空间8×14→时间16 | 必须消除原硬编码目标；沿用原插值策略，不重新发明位置编码 |
| 33帧、小crop、rank8 LoRA、2048更新、单seed | 可作为资源受限探索方案，但预算由父方案事前固定 | **资源约束选择，不是VSR或VideoREPA方法本身所必需**。原2B脚本是full SFT、LR2e-6；原5B是rank128/alpha64 LoRA、LR1e-4；原49×480×720，不能将本机设置称原论文训练复现 |
| 教师缓存 | 只cache训练集对应HR crop，冻结teacher源码/weight/preprocess/pos/grid/fps/source/start/crop | 工程适配；需在线与cache一致。val/test正常推理不用教师特征，不需要创建这些缓存 |

[原2B/5B配置](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/scripts/multigpu_VideoREPA_2B_lora.sh#L64-L84)、[5B LoRA](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/scripts/multigpu_VideoREPA_5B_lora.sh#L64-L85)、[官方2B层数](https://huggingface.co/zai-org/CogVideoX-2b/raw/main/transformer/config.json)、[官方5B层数](https://huggingface.co/zai-org/CogVideoX-5b/raw/main/transformer/config.json)。

教师tubelet nominal覆盖HR[1,2],[3,4],…,[31,32]，每个最终token经全局自注意力后还包含整段上下文。学生latent位置来自因果VAE，不能把它简化为互不重叠四帧平均。trilinear上采样不是repeat2，也不是逐位置精确感受野匹配；它保留的是官方采用的近似网格对应。未来应测试帧索引、首尾clamp、T/H/W flatten以及中间tensor形状，明确这种近似限制。

## 资源静态估算与实施前检查

建议网格N=1792，全Gram为3,211,264元素；两张FP32 Gram仅张量存储24.5MiB，**不是整个训练峰值**。教师12层注意力、临时差值、autograd、投影头参数/梯度/AdamW还需实测。教师特征FP32每clip为5,505,024 bytes：1641训练clip约8.413330GiB，若错误地缓存1673则约8.577393GiB（均未含元数据/文件系统）。保留FP32全特征而非只存Gram可重核通道归一化，并避免N²缓存膨胀。以上为静态乘法计算，不是吞吐/显存承诺。

进入本轮实现后、出分之前需完成：真实权重strict审计；原dtype及两端feature/Gram实测；教师eval与无grad；zero-weight公共FM/RNG/推理一致；全Gram与官方分块bmm数值及梯度对照；映射边界synthetic检查；至少真实train片段在线/cache一致；完整head/optimizer/RNG恢复；新head RNG不消耗公共训练流。先保持margin.1/lambda.5，不复用上一轮梯度校准的lambda.3，不用质量分数改margin、层号或教师输入。资源/数值不可行时应记录具体问题，再决定显式适配，不能预先宣布原路径可行。

当前研究可以支持明确的方法实现选择，但不能保证VideoMAEv2的动作/物理表示改善像素忠实度。关系监督对特征旋转/尺度的自由度较大，首位置排除及教师全局上下文也可能限制局部细节恢复。新两臂结果只能回答这套完整配置相对matched baseline是否改善质量/时间成本，不能仅凭与旧Wan-teacher研究横向比较就把差异全部归因于教师身份：层号、投影头、网格和教师输入均不同。

核查结论：官方教师、checkpoint发布路径/版本、默认训练设置与源码均有证据；实际权重下载/strict加载、原dtype真实运行及Wan适配的数值正确性仍待后续实施验证。本次没有开展这些实验。

## 后续资源核查补记：真实下载与 CPU strict load 已通过

本节更新上文研究初稿中“尚未下载/载入”的状态。根 agent 另行授权只准备公开资源后，资源 agent 已完成下载和一次 CPU 严格加载。本核查者只独立阅读其收据、环境报告与两个 harness，核本地文件大小、源码一致性及文本证据 hash，**没有重跑加载审计、重新hash大权重或执行教师前向**。

- [download_receipt.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_assets_20261108/download_receipt.json) 记录固定 HF revision、173574417 bytes 和官方 SHA256 `8141a6955e0700d11bf15928fe6d61e5cfe482606fed8cfdddb1b922c0fd88ec`。下载 harness 在写入时逐块hash，要求大小与SHA均相等后才将新partial文件改名；本次独立stat也为173574417。文件与旧资源未覆盖。
- [checkpoint_cpu_audit.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_assets_20261108/checkpoint_cpu_audit.json) 实际读取162个FP16 tensors，只排除 `fc_norm.weight/bias` 与 `head.weight[710,768]/bias[710]` 四键。其余 **158 keys / 86,225,664 parameters** 使用官方feature类 `load_state_dict(strict=True)`，missing和shape mismatch为空；随后逐一 `torch.equal` 比较载入tensor与checkpoint cast结果，全部相同。模型eval且所有参数冻结。
- [审计 harness](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_assets_20261108/checkpoint_cpu_audit.py) 的四键排除列表是精确相等断言，不是宽泛 `strict=False`；缺key、shape不符、逐tensor比较不等、CUDA已初始化都会拒绝。没有model.forward或调用model(input)。本地官方源码SHA `a2d7cb25263bc96566c6dcfd3ce163c3fc32ebd016a4e0603443abbce48673f7` 与本核查通过GitHub连接独立读取的固定commit文本一致，仅额外一个文件末尾换行，无计算代码差异。
- [环境报告](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_assets_20261108/checkpoint_cpu_audit_environment.json)：既有Python3.12.16、torch2.14.1+cu130、timm1.0.30、numpy2.5.2；CPU2threads，没有安装/修改依赖或初始化CUDA。这不是原requirements中torch2.6.0/timm0.4.12环境的完整复现。第一次导入时timm注册模块需要sys.modules登记，harness补登记后执行；原官方源码不变，早期错误发生在模型构造与权重载入之前。

独立结论：**资源身份与官方feature encoder的CPU key/shape/精确加载通过，没有阻塞项。** 该审计构造的是原始pos形状 `[1,1568,768]`；它尚未实现/验证建议的1792-token位置插值。原dtype的真实前向和梯度、32帧128×224输入、cache在线一致、训练显存/吞吐仍未测试；不能由strict load推导这些已经通过。当前实验pipeline及其冻结源码未改变。

文本证据SHA256：download receipt `1d5f6faea3fec83cb33f7bb8cbe76c94dc914016176b00a94fa03dd683a05cc6`；CPU audit `58222d45e6c88e7563b1ec9eb404add51bf4fc1de3a4541e4de580547ec71720`；环境 `d4cc974a4458d6831cd89756ca148b2ea89e208b49523240cf47e0a81484000d`；下载harness `4b42e4339f97c39e690e486bd2966b10a3633b9c9d815b7af384d0f7667dec98`；CPU审计harness `d0be85ba156b7834beb181df5bd216a7e17b845754430f42205103e451bb5dbc`。
