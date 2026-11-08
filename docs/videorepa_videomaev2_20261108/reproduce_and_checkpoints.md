# VideoREPA / VideoMAEv2 → Wan VSR：复现与 checkpoint 索引

编制快照：2026-11-08T16:27:08.486382+00:00。这是本轮实验的**操作和资产索引**，不包含质量评分或效果结论。编制仅使用完成训练的协议、summary、已有审计、评价状态/命令和源码；没有再次 hash/load 大型 checkpoint、读取评价质量分数或启动 GPU。

## 1. 本轮边界与完成状态

| 阶段 | 固定规模 | 本索引核到的状态 |
|---|---:|---|
| 独立 GPU preflight | 26 个诊断更新、31 次教师前向、15 个梯度案例/30 次 VJP；0 科学更新 | completed |
| 科学训练 | baseline / videomaev2 各 2048；共 4096 = 0.5 RE | completed；8 个完整 CP，完整训练与独立审计通过 |
| 原生验证 | (shared0 + 两组 × 512/1024/2048) × 32 clip × seed42/43 = 448 | completed；native/source/PNG/MP4 完整 verify 通过 |
| 原生测试 | 两组 final2048 × 32 clip × seed42/43 = 128 | completed128；native/source/PNG/MP4 verify exit0。本索引不读取质量分数或给出效果结论 |

这里的 1 RE 定义为 8192 次科学优化器更新；诊断另计。每组只有一个训练 seed20261102。8 个 CP 的步数是从同一 **Base01 的 8192 步状态开始额外训练**的 512/1024/1536/2048，不是从官方 Wan 零步开始的模型。

训练、已完成的 preflight/validation 目录均为不可覆盖的历史记录。下面历史命令用于说明真实执行方式；**不要复制命令后再次启动现有作业**。额外复现实验需要新目录和另行确认的计算预算；本文件本身不授权新训练或 GPU 重跑。

## 2. 源码与冻结证据

- 本轮实际源码：[VSR-sync](/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync)。
- 启动时源码 commit：`ebd8508aa78605f3bf57e1c663028732cb72d2d3`。
- 编制时本地 HEAD：`366781cda844d0cce914acdd75d26fa73040d68b`；本机 `origin/main` 公开同步记录：`366781cda844d0cce914acdd75d26fa73040d68b`。这是本机已记录的发布快照，本索引未另做远端查询。仓库： [HaoGuo-Cyper/VSR](https://github.com/HaoGuo-Cyper/VSR)。后续文档提交不能替代协议中的源码 SHA。
- 正式 study：[study_training_20261108.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_runs/study_training_20261108.json)，SHA256 `98f2e53341a2f0b3cf935f6c0e2c2e6fe3c5ad21397e5d9b3b6d8bc7a0795521`。
- 事前科学方案：[plan.md](/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync/docs/videorepa_videomaev2_20261108/plan.md)，SHA256 `609d79334e4afb295336e1f022e41c3e323bc8655eb77146e20aca020d822655`。标题中的“等待启动”属于事前记录；真实完成状态以 run/status 和完成审计为准。
- 训练协议：[protocol.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_runs/paired_training_20261108T150511Z/protocol.json)，SHA256 `5292bbd39d21303ac4f57541d80c2e897b02782291b4e73817eb01bd97676bdf`；包含全部 1641 样本/模型/教师/运行时来源绑定。
- 训练 summary：[summary.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_runs/paired_training_20261108T150511Z/summary.json)，SHA256 `a7b5a572e923ebd2c04aff4c6a4db952b4f97f938207db9b2f105d1645bad8bf`。
- 完整 CPU 审计：[training_audit.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/VideoREPA_VideoMAEv2_20261108/training_diagnostics/training_audit.json)；独立复核：[videomaev2_training_completion_independent_review_20261108.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/scratch/videomaev2_training_completion_independent_review_20261108.json)。两者覆盖 4096 行、8 CP、匹配数据/噪声序列和 RNG、学生与头更新、教师/EMA 计数及来源链。本索引复用其大型 CP 证据，不重复加载。

协议冻结的 10 个业务模块与包入口如下；编制时逐个核过当前小型源码文件 SHA 与协议一致。

| 文件 | 冻结 SHA256 |
|---|---|
| [__init__.py](/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync/vsr/__init__.py) | `7ff7f1b099772cc4a207c4e924cf34ddffd6057ba8ca7525afe9b52ec5838af1` |
| [checkpoints.py](/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync/vsr/checkpoints.py) | `69eb3cdfcf0b030c299b4574a42417fb0850af418c92f6b92ef119a8df17e574` |
| [data.py](/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync/vsr/data.py) | `9e3433b17a76fab831ba0ccca00dbff7d7643061ab7b80829bea474f782795ff` |
| [evaluate.py](/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync/vsr/evaluate.py) | `47ad3691aaf695e5fa44b222f18b14eb7b2c0c1ed638e80f7bd1a39bb424c2d9` |
| [infer.py](/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync/vsr/infer.py) | `16af3fb011a3284d25952ed1d28bdadac65f9deb8ed50e52d3216e31050df69d` |
| [losses.py](/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync/vsr/losses.py) | `bca592300cb79ad5e0b13ae6c423566e9ad96a3ec966739e328f674f835e5983` |
| [metrics.py](/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync/vsr/metrics.py) | `ea1005380e788506ccce7050650e5b0e6031f8b27ca4c26f92931cd845484ad6` |
| [model.py](/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync/vsr/model.py) | `7394e76203f29c24232dc586d7a65af69738b02c235996066c675e92e5ca4598` |
| [teacher.py](/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync/vsr/teacher.py) | `23e46cb92b59ad814752d2f59d1e25976bb153936616015f9dedc1f7e7415cac` |
| [train.py](/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync/vsr/train.py) | `943f4d087cc2ac40b76499d0ff70ad306ed21686591ab8bacfd35027ba8c3d47` |
| [vae.py](/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync/vsr/vae.py) | `e83d0b88cdc23c4fd47c72bc278089df8c2a6615bcb7a9112d9aca72541956e6` |

冻结父 controller：[videomaev2_pipeline.py](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_pipeline.py)。其 launch receipt 与 helper SHA 在 [active_videomaev2.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/active_videomaev2.json) 的 `continuation` 中；逐阶段真实 argv/退出码保存在 [phases](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_runs/pipeline_paired_training_20261108T150511Z/phases)。不要一边使用 prepared 协议，一边修改绑定的数值源码、依赖或 helper。

## 3. 官方教师与本地适配

- 原论文：[VideoREPA: Learning Physics for Video Generation through Relational Alignment with Foundation Models](https://arxiv.org/abs/2505.23656v1)。原任务为 CogVideoX 文本生成视频；本轮是 Wan 配对 ×4 视频超分适配，**不声称复现原论文数值**。
- 教师论文：[VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking](https://arxiv.org/abs/2303.16727v2)；[官方 VideoMAEv2 model zoo](https://github.com/OpenGVLab/VideoMAEv2/blob/29eab1e8a588d1b3ec0cdec7b03a86cca491b74b/docs/MODEL_ZOO.md)。
- 取用 [VideoREPA 官方教师实现](https://github.com/aHapBean/VideoREPA/blob/8d581dc301e18546a1808436c724f59d259d09f9/finetune/models/cogvideox_t2v_align/models/ssl/VideoMAEv2.py)，revision `8d581dc301e18546a1808436c724f59d259d09f9`。本地文件 [VideoREPA_VideoMAEv2_8d581dc.py](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_assets_20261108/VideoREPA_VideoMAEv2_8d581dc.py)，SHA256 `a2d7cb25263bc96566c6dcfd3ce163c3fc32ebd016a4e0603443abbce48673f7`。
- 官方权重为 **VideoMAEv2-B，vit_b_k710_dl_from_giant.pth**（K710 蒸馏 ViT-B），不是普通 VideoMAE、DINO、ViT-g 或 Wan teacher。[固定官方下载地址](https://huggingface.co/OpenGVLab/VideoMAE2/resolve/706cc172d65ebd4dedbee3f9c0183a93df9fa125/distill/vit_b_k710_dl_from_giant.pth)，HF revision `706cc172d65ebd4dedbee3f9c0183a93df9fa125`。
- 本地权重：[vit_b_k710_dl_from_giant.pth](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_assets_20261108/vit_b_k710_dl_from_giant.pth)；173574417 bytes；SHA256 `8141a6955e0700d11bf15928fe6d61e5cfe482606fed8cfdddb1b922c0fd88ec`。真实下载和 strict encoder 载入证据：[download_receipt.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_assets_20261108/download_receipt.json)、[checkpoint_cpu_audit.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_assets_20261108/checkpoint_cpu_audit.json)。encoder 仅排除 fc_norm/head 的 4 个未用键，余 158 个键 strict=True；教师 86,225,664 参数，FP32、eval/no_grad、requires_grad=False，不进优化器。

本轮固定科学设置：

| 内容 | 本轮契约 |
|---|---|
| 目标 | `L_FM + 0.5 * mean(relu(abs(Gram(student)-Gram(teacher)) - 0.1))`；full 时空 Gram 的全部有序对/自对角，不抽样、不改分母 |
| 学生 tap / 头 | Wan zero-based block17（one-based18）输出；1536→2048→2048→768 三层含 bias MLP/SiLU；6 个头张量，8,917,760 头参数 |
| 教师输入 | 同一 FM 样本的完整 HR 区域：33 帧中仅辅助项去 frame0，取 1..32；FP32 [0,1] RGB，ImageNet mean/std；不额外 resize/crop/pad |
| token 网格 | 教师 tubelet2/patch16 → 16×8×14 = 1792 tokens，最后 block、最终 norm/classifier 之前；无 CLS/pooling。学生全 9×8×14 经 MLP 后去 latent0，8→16 trilinear，THW 展平 |
| 位置编码 | 从官方 8×14×14 flatten-1D sin/cos 开始；空间 14×14→8×14 bicubic，再时间 8→16 trilinear，align_corners=False。最终 FP32 [1,1792,768] SHA `8efed9994d8f3ba40f5d4baeab45e24e316f38931ecb1f8ec98ba1bdb50edc81` |
| 精度 | 教师/目标/Gram FP32；学生投影混精度路径、关系 normalize/Gram bf16，差值提升 FP32；normalize eps=1e-12。实际 dtype 与值/梯度已在 preflight 校验 |
| 公平对照 | baseline λ0，method λ.5；同一 Base、训练 seed/样本顺序/噪声/时间步、各 fresh AdamW，LR1e-4/wd.01/foreach=False、clip1、constant LR；无 EMA |
| 资源范围 | 33 连续帧 stride1、HR128×224 / LR32×56、rank8 LoRA、有效 batch1、单训练 seed、每组2048 |

保留官方教师、MLP 隐藏宽度/深度、去首位置、学生映射方向、margin/λ 和官方代码 full-Gram mean。必要差异包括 CogVideoX→Wan/LR条件、输入通道、时间/空间网格、位置插值尺寸；两端空间网格原本相同，因此省略官方 stride3 Conv。原论文 49 帧480×720、不同 rank/full-SFT 与生成物理合理性评价没有在这里重现。full-Gram 官方代码与论文 Eq.5 分项归一化也不完全同式；采用前者的取舍已事前固定。小 crop、LoRA8、2048步和既有 Base 续训属于资源/比较范围选择。原文逐项对照见 [videorepa_videomaev2_fidelity_review_20261108.md](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/scratch/videorepa_videomaev2_fidelity_review_20261108.md)。

冻结教师每个 method 更新**在线**提取目标，2048 次；本轮不建教师特征缓存。共同历史 VAE/RGB 缓存不是本轮新增教师缓存。因果 VAE 与 tubelet 的时间适配不等于逐 token 感受野完全一致。

## 4. 私有数据、Wan、文本及指标依赖

GitHub 代码本身不足以运行。需要单独取得以下本机私有资产；换服务器必须在**新配置**中明确重映射路径，并重新 prepare 审查协议，不能改旧协议冒充同一恢复运行。

| 依赖 | 精确位置 / 绑定 |
|---|---|
| Wan 官方模型 | [Wan2.1-T2V-1.3B-Diffusers](/home/hugo/Projects/ReprVSR-pretrained/models/Wan2.1-T2V-1.3B-Diffusers)；[Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/tree/0fad780a534b6463e45facd96134c9f345acfa5b)，revision `0fad780a534b6463e45facd96134c9f345acfa5b` |
| 共同 Base01 | [checkpoint_008192.pt](/home/hugo/Projects/VSR_/runs/stage2_protocol33_base_seed20261102_20261106T025845.129456Z/Base/checkpoint_008192.pt)；SHA256 `28356e22e4e718d3c378588793f60732dbc93825ec5d78b8a85c8150060da63f`；Base/config文件 SHA `90df7f9992b967bcf686610b7e5e9a98ad7afa66563f70311f27e85ceeeb41de` |
| HQ-VSR 原数据 | [HQ-VSR](/home/hugo/Projects/ReprVSR-pretrained/data/HQ-VSR)；当前manifest为 [hq_vsr.json](/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync/manifests/hq_vsr.json)，内容canonical digest（manifest内 `manifest_sha256`）`305d71993836df42def707ff5ed2bda1fad838e298aaee7d36a83873d2d47484`；此JSON文件raw-byte SHA `196ad8f04ee75ae52900fe33d643cec1d96429242df4d9a06cf2eb6c0d6f3347`，两者不是同一hash定义 |
| 训练1641样本缓存 | [stage2_protocol33_base_seed20261102_20261106T025845.129456Z](/home/hugo/Projects/VSR_/runs/stage2_protocol33_base_seed20261102_20261106T025845.129456Z)；同序 `clip_<index>/fresh_latents.pt`、`paired_rgb.pt`、`metadata.json`；需同时保留selection/config及seal，不可拿历史9帧stride4缓存替代 |
| 训练cache seal | [cache_asset_seal.json](/home/hugo/Projects/VSR_/runs/stage2_protocol33_base_seed20261102_20261106T025845.129456Z/cache_asset_seal.json)；SHA256 `aa4fdd751016ec6b03229f1f27581667e9f87bf91b6fedcf8fd03b2b9be19c73`；已有5024文件全量封存链，本轮只用原1641 training IDs |
| clean val32 | [val_plan.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_clean_cohorts_20261108/iteration_000/val_plan.json)；SHA `f6e17c582e84e2a2cb90c2dea016e08195edae3f2f8558c26378b75d25bc27bd`；dataset [dataset.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_clean_cohorts_20261108/validation_assets/dataset.json)，SHA `698fc703e1421305fa47a2f5250dfc175059e1acb09413b71e14fef52a2f9eae` |
| clean test32 | plan [test_plan.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_clean_cohorts_20261108/iteration_000/test_plan.json)，SHA `f3224734e43365f1cb8454b14554d362ff7c7e290322fd2a714b78d5a984931f`；已由历史首轮生成的dataset [dataset.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/test_assets/dataset.json)，SHA `b348ea155d774c928ef01a3897494546bba18dae9040c950ee2d4eeae88ab206` |
| 固定文本 | [fresh_text_embeddings.pt](/home/hugo/Projects/VSR/runs/formal_hq_v2_noref_20261102T142214.588571Z/fresh_text_embeddings.pt)；4,196,043 bytes，SHA `1dd219cde3005b4d16febaad67ad87e79c1403ae9bdaae69f2723ab9cbf239c2`。训练cache中有同SHA拷贝；保留固定 caption 接口，无语义提升声称 |
| LPIPS AlexNet | [alexnet-owt-7be5be79.pth](/home/hugo/Projects/VSR/weights/alexnet-owt-7be5be79.pth)；SHA `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` |
| LPIPS v0.1 linear | [lpips_v0.1_alex.pth](/home/hugo/Projects/VSR/weights/lpips_v0.1_alex.pth)；SHA `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0` |
| 本地评价依赖 | [.deps](/home/hugo/Projects/VSR/.deps)；文件树与包版本/实现SHA由评价protocol.runtime绑定 |
| 教师额外依赖 | timm1.0.30（入口、dist-info、drop/helpers/weight_init等源码均由teacher_dependencies绑定）；上述 VideoREPA教师源码与权重 |

Wan 大文件 SHA 从已完成协议摘录（本次不重新 hash）：

| 相对 Wan 模型根目录的文件 | SHA256 |
|---|---|
| `transformer/config.json` | `0b093fa072e9ff28763febe9b964ee582f566733a6d6709deb9dfba1bde16b81` |
| `transformer/diffusion_pytorch_model-00001-of-00002.safetensors` | `6d011927dbd2cc8afe53d57abab04a8fd86f615d83324770d985fb058ece3a24` |
| `transformer/diffusion_pytorch_model-00002-of-00002.safetensors` | `b92ec2309b1f239af6f746431815a881afcc938abb26a4f08d9a2fd6c892f872` |
| `transformer/diffusion_pytorch_model.safetensors.index.json` | `dcbcf3497134a3f50557ff069dd7d2c84b5c4d8c5932472f6bdb780fb4016589` |
| `vae/config.json` | `f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527` |
| `vae/diffusion_pytorch_model.safetensors` | `d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793` |
| `scheduler/scheduler_config.json` | `3fed2abbd9bbc301a74db01947198057ec5049808910dccab320925bf27bea6e` |

数据独立性范围：source-ID 和 raw-byte SHA 跨 split 为0；原始视频各 segment 保持同一 split；已确认的视觉重复源在本轮沿用的 repaired cohort 中排除。自动 source0/mid/end 与绑定窗口0/16/32 缩略图筛查、dHash/pHash候选和 SSIM/MSE 多时间复核有覆盖局限，不能据此声称穷尽所有转码/偏移/裁切近重复。完整性 receipt：[summary.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_clean_cohorts_20261108/iteration_000/integrity/summary.json)。这些 val/test source 在历史研究轮次已使用，且 test RGB 已在历史首轮生成；本轮是 **historical source-heldout exploratory evaluation，不是新的盲测**。不把旧Base配置里的历史“test untouched”描述当作当前事实。

## 5. 原环境与原生推理契约

Python：`3.12.16 (main, Oct 26 2026, 11:37:43) [GCC 15.3.0]`；执行器 [python3.12](/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python3.12)。launch使用同环境的 `bin/python`。本轮主机是 RTX5080 16GiB；device cuda:0，原GPU UUID `GPU-4414baf2-fd94-f31a-18e6-ebe95dcd1813`，共享锁 [stage2_gpu.lock](/home/hugo/Projects/VSR_/runs/stage2_gpu.lock)，限制 17179869184 bytes。桌面进程由封存准入和每次 fresh host 检查同时核验；换服务器不能照搬本机PID/UUID/桌面receipt。

| 包 | 本轮版本 |
|---|---|
| `torch` | `2.14.1+cu130` |
| `numpy` | `2.5.2` |
| `diffusers` | `0.42.1` |
| `transformers` | `5.21.0` |
| `peft` | `0.21.2` |
| `accelerate` | `1.15.0` |
| `safetensors` | `0.8.0` |
| `lpips` | `0.1.4` |
| `scikit-image` | `0.25.2` |
| `Pillow` | `12.3.0` |
| `opencv-python-headless` | `4.14.0.94` |
| `scipy` | `1.18.1` |
| `torchvision` | `0.29.1+cu130` |
| `timm` | `1.0.30` |

版本号之外，protocol.runtime/teacher_dependencies 还绑定实际包文件、native Wan transformer/VAE/UniPC源码和本地依赖树；仅 `pip install` 同名版本不等于逐字节环境一致。当前同设备精确恢复预飞通过，不作跨GPU/驱动/包版本 bitwise 复现保证。

真实子进程环境（路径按原主机）：

```bash
export CUDA_VISIBLE_DEVICES=GPU-4414baf2-fd94-f31a-18e6-ebe95dcd1813
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 XFORMERS_DISABLED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/hugo/Projects/VSR/.deps:/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync
```

还会由源码配置 strict deterministic runtime、关闭TF32与flash/memory-efficient/CUDNN SDP并启用math SDP；不要只设置环境变量后跳过程序内检查。正式训练/评价GPU worker各自取得同一shared lock；父controller只串行调度，Restart=no，没有自动重试。

原生恢复固定：UniPC20 / CFG1、CPU采样noise seed42和43、固定text、FP32 VAE、tile256×448/overlap64、原量化/LPIPSAlex.1/RGBPSNR/SSIM/TecoGAN tLP100/tOF。全部33帧128×224固定crop，**不是源视频全长或源分辨率**。448val包含shared0和2组512/1024/2048；1536 CP用于完整状态/恢复审计但不增加质量评价角色。test仅保留两组final2048，完整validation后锁定，两者均测试，**不按分数择优**。

## 6. 八个完整 checkpoint

以下路径、大小、SHA直接取自已完成summary与已有8CP审计。文件保持私有；本索引没有再次读取其tensor或计算大文件SHA。表中每个 CP 均带相邻 arm/config 和完整训练日志来源。

| arm / 额外更新 | CP绝对路径 | bytes | SHA256 |
|---|---|---:|---|
| baseline / 512 | [baseline/Base/checkpoint_000512.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_runs/paired_training_20261108T150511Z/arm_runs/baseline/Base/checkpoint_000512.pt) | 78768497 | `59fdfbac43634d2d9a42adfdba7b3b72f164c0dab6a9befd43dc533726e4201d` |
| baseline / 1024 | [baseline/Base/checkpoint_001024.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_runs/paired_training_20261108T150511Z/arm_runs/baseline/Base/checkpoint_001024.pt) | 78768497 | `418f04186849991c2ae050a1423ca438f784284d0addd4a6498e18163ebbfea0` |
| baseline / 1536 | [baseline/Base/checkpoint_001536.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_runs/paired_training_20261108T150511Z/arm_runs/baseline/Base/checkpoint_001536.pt) | 78768497 | `6280d7de3d640dfe29e115040aa1d6fed98746c1e75a5cb48b894ff497198a7d` |
| baseline / 2048 | [baseline/Base/checkpoint_002048.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_runs/paired_training_20261108T150511Z/arm_runs/baseline/Base/checkpoint_002048.pt) | 78768497 | `d31c7e2415c61a973e76ff46d85e8f6b6c5e3105c97fa144397b1ac407866842` |
| videomaev2 / 512 | [videomaev2/Base/checkpoint_000512.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_runs/paired_training_20261108T150511Z/arm_runs/videomaev2/Base/checkpoint_000512.pt) | 185819417 | `478d6e6e79b48cf757c5e58709c710ab8e1e20591a5399ebb29210baf8de04d2` |
| videomaev2 / 1024 | [videomaev2/Base/checkpoint_001024.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_runs/paired_training_20261108T150511Z/arm_runs/videomaev2/Base/checkpoint_001024.pt) | 185819417 | `5b4647ce206380dad984ca1db475a658f92e098666787c3f05fac1a99363e958` |
| videomaev2 / 1536 | [videomaev2/Base/checkpoint_001536.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_runs/paired_training_20261108T150511Z/arm_runs/videomaev2/Base/checkpoint_001536.pt) | 185819417 | `381ffc72ea2451c3cea6e8f1ac098bc30bb26e14c8ce387c8526b359d4c414ac` |
| videomaev2 / 2048 | [videomaev2/Base/checkpoint_002048.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_runs/paired_training_20261108T150511Z/arm_runs/videomaev2/Base/checkpoint_002048.pt) | 185819417 | `83810994e1c5cd49da94053fd1ffa5634165f69d5fd1d48bb93279bac6432458` |

两组初始 student trainable digest 均为 `c9920b247f7b6dbe5b2920074f03fff21e2ff9823e9065e733f7f02ea8d7e3b6`，488张量/6,500,992参数；基线没有训练头，方法另有6个MLP头张量。最终两组都保留student-only推理入口；完整CP也包含训练专用状态，不能把只导出的student权重当成精确恢复CP。

| arm | 完整日志 | 日志SHA256 |
|---|---|---|
| baseline | [training.jsonl](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_runs/paired_training_20261108T150511Z/arm_runs/baseline/training.jsonl) | `80b7a83c5a09d38becc4f4929d31fe997d20682b17442308cc8cc2f39ae748a9` |
| videomaev2 | [training.jsonl](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_runs/paired_training_20261108T150511Z/arm_runs/videomaev2/training.jsonl) | `0760323fd02686f0bee51da13763d98c11b83a902864c19bc94864290d2b544a` |

## 7. 历史真实命令索引（勿重跑这些完成目录）

下面变量展开到真实历史路径。`vmae-prepare`只作metadata准备；`vmae-preflight` / `vmae-execute`会运行GPU。prepare返回的protocol SHA必须审核并显式传给execute，不能用随手改写的JSON续跑。

```bash
SOURCE=/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync
WORK=/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work
PY=/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python
DESIGN="$SOURCE/configs/videorepa_videomaev2_design_20261108.json"
PREFLIGHT="$WORK/videomaev2_runs/preflight_20261108T145516Z"
TRAIN_STUDY="$WORK/videomaev2_runs/study_training_20261108.json"
TRAIN="$WORK/videomaev2_runs/paired_training_20261108T150511Z"
PIPE="$WORK/videomaev2_runs/pipeline_paired_training_20261108T150511Z"
cd "$SOURCE"

# 原 metadata prepare 与独立诊断；这两个目录现已完成
"$PY" -m vsr.train vmae-prepare --config "$DESIGN" --run-dir "$PREFLIGHT"
"$PY" -m vsr.train vmae-preflight --run-dir "$PREFLIGHT" --expected-protocol-sha 3a4938607b58b566619f30c2f468cfbdc0500a15c4ceab9207ee1f5e17f20523

# 训练study在成功preflight后绑定其报告，再prepare并启动一次正式训练
"$PY" -m vsr.train vmae-prepare --config "$TRAIN_STUDY" --run-dir "$TRAIN"
"$PY" -m vsr.train vmae-execute --run-dir "$TRAIN" --expected-protocol-sha 5292bbd39d21303ac4f57541d80c2e897b02782291b4e73817eb01bd97676bdf

# 完整训练后：7角色 ×32×2 =448；这些历史阶段均已完成
"$PY" -m vsr.evaluate vmae-build-config --training-run "$TRAIN" --phase val --output "$PIPE/val_config.json" --run-root "$PIPE"
"$PY" -m vsr.evaluate vmae-prepare --config "$PIPE/val_config.json" --run-dir "$PIPE/validation"
"$PY" -m vsr.evaluate vmae-execute --run-dir "$PIPE/validation" --expected-protocol-sha 950cce2ced2b8d0adf914b9c2d7b6beaafe7b5050c599ccc4b70cdd48f72e60c
"$PY" -m vsr.evaluate vmae-verify --run-dir "$PIPE/validation"
"$PY" -m vsr.evaluate vmae-lock-finals --validation-run "$PIPE/validation" --output "$PIPE/finals_receipt.json"

# 两组 final2048 ×32×2 =128；这些历史阶段均已完成
"$PY" -m vsr.evaluate vmae-build-config --training-run "$TRAIN" --phase test --output "$PIPE/test_config.json" --run-root "$PIPE" --validation-run "$PIPE/validation" --finals-receipt "$PIPE/finals_receipt.json"
"$PY" -m vsr.evaluate vmae-prepare --config "$PIPE/test_config.json" --run-dir "$PIPE/test"
"$PY" -m vsr.evaluate vmae-execute --run-dir "$PIPE/test" --expected-protocol-sha ca366a6d6267a104025716655622cdf65dc24bdcee13a1b289b2e462793e3d5a

# 原父controller已执行完成，退出码0；本索引没有重复执行verify
"$PY" -m vsr.evaluate vmae-verify --run-dir "$PIPE/test"
```

真实 preflight launch：[launch.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_runs/preflight_20261108T145516Z/launch.json)；训练launch：[launch.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_runs/paired_training_20261108T150511Z/launch.json)。实际service分别为 `vsr-videomaev2-preflight-20261108t150235z.service`、`vsr-videomaev2-training-20261108t151157z.service`；后处理父service为 `vsr-videomaev2-continuation-20261108t151223z.service`。父service不会启动第二次训练。原阶段argv见 [phases](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_runs/pipeline_paired_training_20261108T150511Z/phases)；原validation verify已核14784预测PNG/448预测MP4及1056GT PNG/32GT MP4，未在本任务重复校验或查看视频。

## 8. 获授权后在新位置复现

先保留上述历史目录。创建全新的源码checkout/研究配置/run root；复制代码不能替代私有模型、Base01、RGB/VAE缓存、固定text、val/test NPY和指标权重。使用已有制品时按原SHA核对来源；重建资产时记录producer/帧索引/crop/退化/RNG，不能仅因形状相同便称同一数据。

新主机需要改映射的字段至少包括：`base_config`、`base_checkpoint`、`allowed_base_root`、`cache_root`、`model_directory`、`teacher_source`、`teacher_checkpoint`、`run_root`、`gpu_admission`、`gpu_lock`、`device/gpu_uuid`、评价seal/dataset/testplan、文本与指标权重、本地dependency root和运行环境。GPU准入须由新现场生成，不能复用本机的PID/UUID。**不要对旧protocol做路径替换，也不要借`--resume`绕过新主机绑定**；重新prepare生成新SHA并记录与本轮固定科学设置的差异。若新环境/精度/硬件不同，先完成其正确性与资源预检，不能预设bitwise相同。

在新路径、预算和环境确认后，使用第7节同一CLI顺序：新design(mode=preflight) → 新preflight prepare/审核SHA/执行 → 新training study(mode=training，preflight_report指向这次成功summary) → 新训练prepare/审核SHA/执行 → 完整训练审计 → 新val配置/prepare/执行/verify → 锁定两个final → 新test配置/prepare/执行/verify。`vmae-prepare`拒绝已存在的run目录；不要复用当前完成目录。新运行的SHA必然需要重新读取，不能复制上面的历史SHA给新JSON。

原论文/本轮固定目标不因移机自动更改：不增加候选/权重、改layer/λ/margin、削减评价行、切教师或改teacher在线路径。增补多seed、大尺寸或H100主实验属于另一项预算和研究计划。

## 9. 同协议完整恢复（工程中断才用）

本轮正式训练只有 `attempt_000`，`resume_requested=false`；两次评价初始attempt也均非resume/复用0行。preflight包含恢复一致性测试，但**不代表正式训练曾经中断恢复**。

训练真实恢复入口只有下式；程序从同run当前未完成arm的最后一个完整CP自动恢复，没有可随意指定其他协议checkpoint的CLI：

```bash
# 仅适用于确已中断、经诊断允许恢复的同一run；不要对completed的TRAIN运行
cd /home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python -m vsr.train vmae-execute \
  --run-dir /ABSOLUTE/PATH/TO/INTERRUPTED_SAME_PROTOCOL_RUN \
  --expected-protocol-sha ORIGINAL_64_HEX_PROTOCOL_SHA \
  --resume
```

恢复前仍验证完整原协议/当前源码和依赖/配置/Base初始化/CP来源/日志前缀/成本。恢复student trainable、方法6头张量、AdamW完整moments与param groups、4显式RNG（`noise`、`timestep`、`mask`、`corruption`；data index由cursor确定）及Python/NumPy/Torch CPU/CUDA全局RNG、步数/数据cursor、教师前向计数和累计同步Step时间；constant LR存在optimizer中，无独立scheduler状态。冻结VideoMAEv2按原source/weights绑定重新载入，教师权重不存入任务CP。完整CP路径必须保留其同arm `config.json` 和原run日志/协议链，不能改Base优化器或别轮CP充作本轮恢复。

未提交日志尾和中断attempt必须保留，rollback/浪费成本另报，不重复算入有效训练曲线。`--resume`接受原实现允许的running/failed_recoverable工程状态；completed及VideoMAEv2 OOM/nonfinite科学/资源硬停状态会拒绝，不能靠改status文件强行恢复。任何源码修复若使原protocol SHA链变化，需要另行明确恢复/新协议方案，不能静默接受。

评价的恢复单位为同protocol已完成且校验过的行：

```bash
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python -m vsr.evaluate vmae-execute \
  --run-dir /ABSOLUTE/PATH/TO/INTERRUPTED_SAME_PROTOCOL_EVALUATION \
  --expected-protocol-sha ORIGINAL_64_HEX_EVAL_PROTOCOL_SHA \
  --resume
```

既有行先核身份/来源/PNG完整性再复用；不能跨run或删掉不喜欢的行。已经completed的评价使用 `vmae-verify --run-dir ...`，不再execute或另起第二矩阵。实际恢复前先确认旧service与GPUworker已退出，仍只允许一个shared-lease GPUworker。

## 10. 普通推理只需要 student

完整checkpoint中教师绑定、训练头、优化器/RNG是训练/复现证据。`load_training_checkpoint`普通加载只恢复restoration student；推理不使用VideoMAEv2、MLP头或HR输入。仍需官方Wan transformer/VAE、该任务CP与config、固定text、LR视频。metadata preview本身不加载模型：

```bash
PY=/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python
SOURCE=/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync
CP=/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_runs/paired_training_20261108T150511Z/arm_runs/videomaev2/Base/checkpoint_002048.pt
CP_ROOT=/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_runs/paired_training_20261108T150511Z
cd "$SOURCE"
"$PY" -m vsr.infer preview --checkpoint "$CP" --allowed-checkpoint-root "$CP_ROOT"

# 仅作未来获授权新输出的语法示例；本索引没有执行此命令
"$PY" -m vsr.infer run --checkpoint "$CP" --allowed-checkpoint-root "$CP_ROOT" \
  --model-dir /home/hugo/Projects/ReprVSR-pretrained/models/Wan2.1-T2V-1.3B-Diffusers \
  --input /ABSOLUTE/PATH/TO/LR_VIDEO.mp4 --output /ABSOLUTE/PATH/TO/NEW_OUTPUT \
  --text-embedding /home/hugo/Projects/VSR/runs/formal_hq_v2_noref_20261102T142214.588571Z/fresh_text_embeddings.pt \
  --caption 'A high quality video.' --device cuda:0 \
  --gpu-lock /home/hugo/Projects/VSR_/runs/stage2_gpu.lock \
  --gpu-admission /ABSOLUTE/PATH/TO/FRESH_HOST_ADMISSION.json \
  --frames 33 --seed 42 --steps 20 --tile-height 256 --tile-width 448 --overlap 64 --execute
```

这只是单次用户文件恢复语法；正式val/test完整矩阵必须用第7节的评价入口，才能保留完整角色/clip/seed/指标和来源校验。

## 11. 发布与后续更新

可公开：源码、配置、协议和完整文本日志/实验统计、SHA及本机路径索引。**不上传任务checkpoint/官方权重、原视频、PNG/MP4、RGB/VAE或教师tensor缓存、原始RNG状态、凭证。** 上述链接是本机私有路径，GitHub文本发布不会自动携带其二进制内容；服务器需要按对应许可单独取得依赖。

本索引记录到test128执行/verify已完成的操作快照，不给质量结论、选择方法或单凭运行完成宣称研究结案。指标、视频实际复核与最终交付由根代理写入新的完成记录；保留本编制快照和全部历史run。
