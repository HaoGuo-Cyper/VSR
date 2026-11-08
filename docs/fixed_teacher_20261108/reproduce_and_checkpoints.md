# 固定教师 VSR：复现、恢复与 checkpoint 索引

元数据核对时间：`2026-11-08T06:03:16.491625+00:00`。本文件只新增文本；没有运行 GPU、训练、恢复、评价或重复测试，也没有读取 test 的观察行、部分分数或图像。

## 先分清已完成记录与新执行

- 正式训练已完成 **5×2048=10240 updates（1.25RE）**，20个完整checkpoint；protocol SHA：`7b620162ea4518f52ff7bb84a18ff955b2030ca4cdc94fd357547ece11021509`。
- 新val已完整完成并经过核心`ft-verify`：**16roles×32sources×2samplers=1024 restores**；protocol SHA：`b638fd326211724d9f3b561d2bc1d0fa87c3deb478aff4ca2d1810d2acb64c0a`。
- 索引时仅核test状态元数据：`completed`，protocol SHA：`d1572aeb22f736d1c1823aae858f8c1aaba838a748c4875610507aaa2c011869`；不据此宣称test已完成，最终结果以其完整`ft-verify`及总报告为准。
- **现有训练/val RUN 已完成且不可覆盖或再次执行。** 以下历史执行命令用于追溯，不是现在应运行的队列。新重复实验必须使用新目录/新协议绑定，额外GPU与训练预算需另行授权；该索引不构成授权。
- 正式训练只有`attempt_000`且`resume_requested=false`；val/test的首个attempt均`resume_requested=false`、`verified_rows_reused=0`。本轮正式阶段截至索引时未做中断恢复。预飞中的保存/加载/恢复一致性诊断不等于正式实验发生过resume。

设计依据：[plan.md](/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/docs/fixed_teacher_20261108/plan.md)；完成条件：[completion_checklist.md](/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/docs/fixed_teacher_20261108/completion_checklist.md)。旧清单中的计划态不能覆盖本文件引用的真实完成状态。

## 入口、源版本与环境

- GitHub：<https://github.com/HaoGuo-Cyper/VSR>；本机checkout：`/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync`。索引时Git HEAD：`b6111e10eee17b05f22c958c99b56118b2a4d452`。后续文档/日志commit可能继续变化，**实际数值源码身份以训练protocol的11个文件SHA为准**，不是只看HEAD。
- 声明Python：`/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python3.12`（Python3.12.16）；metric辅助依赖：`/home/hugo/Projects/VSR/.deps`。
- Study：[/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/configs/fixed_teacher_study_20261108.json](/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/configs/fixed_teacher_study_20261108.json)，SHA `13206686e3c6513867f25131967d48949d75cdb0f958f08e716ecdf79bb61216`。
- 训练：[/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z)；流水线：[/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z)；已审调度：[fixed_teacher_pipeline_v2.py](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_pipeline_v2.py)。旧`fixed_teacher_pipeline.py`及旧等待服务只保留历史，勿启动。
- 共享GPU租约：`/home/hugo/Projects/VSR_/runs/stage2_gpu.lock`；本轮设备`cuda:0` / `GPU-4414baf2-fd94-f31a-18e6-ebe95dcd1813`；admission文件 `/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/hr_vae_admission_20261108.json`。设备/桌面身份必须在任何新GPU执行前重新现场核验，不能跨重启/机器盲用旧PID或UUID。

| 依赖 | 训练绑定版本 |
| --- | --- |
| torch | `2.14.1+cu130` |
| diffusers | `0.42.1` |
| transformers | `5.21.0` |
| peft | `0.21.2` |
| numpy | `2.5.2` |
| safetensors | `0.8.0` |

包入口/分发记录/native源码的完整路径、大小和SHA在训练protocol的`dependencies`；校准protocol也绑定同一链。

| 数值源码 | SHA256 |
| --- | --- |
| [__init__.py](/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/vsr/__init__.py) | `7ff7f1b099772cc4a207c4e924cf34ddffd6057ba8ca7525afe9b52ec5838af1` |
| [checkpoints.py](/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/vsr/checkpoints.py) | `fee88eb3f27075e99c8219af6b7786aff29f434a4b0bf21fc370d8bf17875171` |
| [data.py](/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/vsr/data.py) | `82e475daddaa41b595bb535540e31f66d2863cfe9a82bd1743c1b4f7f9f00416` |
| [evaluate.py](/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/vsr/evaluate.py) | `5d4f0b663c68524c9375eed20ce3755ff74f8c47987ac69afaa732f0c687f026` |
| [infer.py](/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/vsr/infer.py) | `16af3fb011a3284d25952ed1d28bdadac65f9deb8ed50e52d3216e31050df69d` |
| [losses.py](/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/vsr/losses.py) | `ea0a0ca248caa39a377e1f3044008c9b99c3b2fbb02f8f8aaa627d265bc2329a` |
| [metrics.py](/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/vsr/metrics.py) | `ea1005380e788506ccce7050650e5b0e6031f8b27ca4c26f92931cd845484ad6` |
| [model.py](/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/vsr/model.py) | `7394e76203f29c24232dc586d7a65af69738b02c235996066c675e92e5ca4598` |
| [teacher.py](/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/vsr/teacher.py) | `27e573ac755a2d62a385678fa2c826946678fde88e2643de32e6db842f312a96` |
| [train.py](/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/vsr/train.py) | `089b4d26401d461e60a8ea706ce6fd9fd3dbd808ad29843b50360f9ccb5f49fe` |
| [vae.py](/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/vsr/vae.py) | `e83d0b88cdc23c4fd47c72bc278089df8c2a6615bcb7a9112d9aca72541956e6` |

实际共同环境如下；`CUDA_VISIBLE_DEVICES`只在明确授权的新GPU执行时指定。CPU校验命令应置空该变量。

```bash
FT_WORK=/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work
FT_REPO=/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync
FT_PYTHON=/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python3.12
FT_TRAIN=/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z
FT_PIPELINE=/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z
export PYTHONPATH=/home/hugo/Projects/VSR/.deps:$FT_REPO
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 XFORMERS_DISABLED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$FT_REPO"
```

## 私有数据、模型与缓存资产

GitHub仅提供代码与允许公开的文本；以下原视频、权重、checkpoint、NPY/PT特征及PNG/MP4 **均不随仓库分发**。在其他机器准备这些资产后必须重新绑定路径/硬件/依赖；不能直接拿本机protocol做跨机resume。

| 资产 | 本机路径/版本与校验入口 |
| --- | --- |
| HQ-VSR原视频 | `/home/hugo/Projects/ReprVSR-pretrained/data/HQ-VSR`；原manifest：[/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/VSR-sync/manifests/hq_vsr.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/VSR-sync/manifests/hq_vsr.json) |
| 原配对RGB+归一化VAE latent共同缓存 | `/home/hugo/Projects/VSR_/runs/stage2_protocol33_base_seed20261102_20261106T025845.129456Z`；source cache seal `aa4fdd751016ec6b03229f1f27581667e9f87bf91b6fedcf8fd03b2b9be19c73`；1641train和32个旧工程验证例，旧32例不用于本轮优化/选参 |
| 新val32 | seal：[/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_clean_cohorts_20261108/iteration_000/val_plan.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_clean_cohorts_20261108/iteration_000/val_plan.json)；dataset：[/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_clean_cohorts_20261108/validation_assets/dataset.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_clean_cohorts_20261108/validation_assets/dataset.json) |
| 新test32 | 事前plan：[/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_clean_cohorts_20261108/iteration_000/test_plan.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_clean_cohorts_20261108/iteration_000/test_plan.json)；正式RGB由完整val选择后生成于 `/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/test_assets/dataset.json` |
| 完整性修复与CPU资产绑定 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_clean_cohorts_20261108/CPU_assets_handoff.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_clean_cohorts_20261108/CPU_assets_handoff.json)；训练集/Base不改，排除1val/3test近重复源，31val旧NPY字节保留；多点screen不是穷尽重复证明 |
| 官方Wan1.3B模型/VAE/scheduler | `/home/hugo/Projects/ReprVSR-pretrained/models/Wan2.1-T2V-1.3B-Diffusers`；repo `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`；revision `0fad780a534b6463e45facd96134c9f345acfa5b`；各文件SHA见protocol.`official_model_files` |
| 共同起点Base01 | [/home/hugo/Projects/VSR_/runs/stage2_protocol33_base_seed20261102_20261106T025845.129456Z/Base/checkpoint_008192.pt](/home/hugo/Projects/VSR_/runs/stage2_protocol33_base_seed20261102_20261106T025845.129456Z/Base/checkpoint_008192.pt)；SHA `28356e22e4e718d3c378588793f60732dbc93825ec5d78b8a85c8150060da63f`；仅继承学生参数，不继承其optimizer/RNG |
| DINOv2-B/14源码 | `/home/hugo/Projects/VSR/vendor/dinov2`；官方[仓库](https://github.com/facebookresearch/dinov2)；commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`；203个tracked文件SHA在DINO manifest |
| DINOv2权重 | `/home/hugo/Projects/VSR/weights/dinov2_vitb14_pretrain.pth`；SHA `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73` |
| 新DINO目标缓存 | `/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/dinov2_cache_20261108T032522Z`；`manifest.json`/`seal.json`/`completion_receipt.json`/`extraction_receipt.json`；全部1673clip/55209帧重新提取，不能复用旧9frame/stride4缓存 |
| native缓存text | `/home/hugo/Projects/VSR/runs/formal_hq_v2_noref_20261102T142214.588571Z/fresh_text_embeddings.pt`；SHA `1dd219cde3005b4d16febaad67ad87e79c1403ae9bdaae69f2723ab9cbf239c2`；旧引用含`source_run`而无`bytes`，v2按core规范化实测path/bytes/sha |
| LPIPS alexnet | `/home/hugo/Projects/VSR/weights/alexnet-owt-7be5be79.pth`；244408911B；SHA `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` |
| LPIPS linear | `/home/hugo/Projects/VSR/weights/lpips_v0.1_alex.pth`；6009B；SHA `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0` |

缓存和采样的固定定义：33连续帧/stride1，HR128×224、LR32×56，shared bicubic/AA；VAE为冻结FP32、posterior.mode、原channel归一化，33→9latent；token9×8×14。正式指标取全部33帧量化RGB，UniPC20/CFG1、tile256×448/overlap64、sampler42/43。完整说明见[data_and_metric_protocol.md](/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/docs/fixed_teacher_20261108/data_and_metric_protocol.md)。

## 五条轨迹的20个checkpoint

每个都是同一Base01的独立fresh AdamW续训：baseline0、HR-VAE .11/1.1、HR-DINO .094/.94，各2048步。四个保存点是512/1024/1536/2048；用于val的中间点只有512/1024/2048。路径目标均为本机私有文件。SHA来自已完成summary，并已由真实CPU诊断重算20个文件；本次编索引只复核存在/大小，不重复加载权重或跑测试。

| 轨迹 | 权重 | 步数 | checkpoint绝对路径 | SHA256 |
| --- | ---: | ---: | --- | --- |
| baseline | 0.0 | 512 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/baseline/Base/checkpoint_000512.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/baseline/Base/checkpoint_000512.pt) | `d9cea2e743afedd98ced9accc80f7ffd8200a61741c231dd0ec94830823a4dfa` |
| baseline | 0.0 | 1024 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/baseline/Base/checkpoint_001024.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/baseline/Base/checkpoint_001024.pt) | `36433b3732870ea57c999de5b2c4524089af4094ce135f6e6ed278f8182da3e3` |
| baseline | 0.0 | 1536 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/baseline/Base/checkpoint_001536.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/baseline/Base/checkpoint_001536.pt) | `0e734b5a62dcaf288d9a02d1a17bc345a5debd1713416d706040d9b26af8c4c5` |
| baseline | 0.0 | 2048 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/baseline/Base/checkpoint_002048.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/baseline/Base/checkpoint_002048.pt) | `5df6e5b1e95bb18993be33647b1b62f6d1a3090671e7e21bbc450f7d546ef3be` |
| vae_low | 0.11 | 512 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/vae_low/Base/checkpoint_000512.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/vae_low/Base/checkpoint_000512.pt) | `53cd1ae5c0ba02d26c3a4f700fb986e01953dcbcbbbdd95f23bf3912f699be07` |
| vae_low | 0.11 | 1024 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/vae_low/Base/checkpoint_001024.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/vae_low/Base/checkpoint_001024.pt) | `756bcc82411dd2c8b3e400dda6bad83d7cf6d5024090b7d65b9e724e4a9a2f5d` |
| vae_low | 0.11 | 1536 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/vae_low/Base/checkpoint_001536.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/vae_low/Base/checkpoint_001536.pt) | `c5d39e861c509d2bfbe376ca251b5bc89b46b8c6abcbb48f73a53b5d1609f238` |
| vae_low | 0.11 | 2048 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/vae_low/Base/checkpoint_002048.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/vae_low/Base/checkpoint_002048.pt) | `aa16349a897f8b1b9aad4419355a42ad0221e922e3bd85a17248ded5e0d2b4be` |
| vae_high | 1.1 | 512 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/vae_high/Base/checkpoint_000512.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/vae_high/Base/checkpoint_000512.pt) | `72e9eb10dab068f932d9ea2f66e12c5de36e404bf15d9d0ca427eebbb2c8269d` |
| vae_high | 1.1 | 1024 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/vae_high/Base/checkpoint_001024.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/vae_high/Base/checkpoint_001024.pt) | `4400c145efa9c07477a5ad8d7dfad004bc5a3c423809aeef8e395e63011ed95d` |
| vae_high | 1.1 | 1536 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/vae_high/Base/checkpoint_001536.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/vae_high/Base/checkpoint_001536.pt) | `113067027d328f57ab498bc80fa04eb0ea4a3701661f5084c222104b62486940` |
| vae_high | 1.1 | 2048 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/vae_high/Base/checkpoint_002048.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/vae_high/Base/checkpoint_002048.pt) | `d9643d453be7905c6ae4d5f643145e76e2cbf6befa6073580c4d681f40faf75e` |
| dino_low | 0.094 | 512 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/dino_low/Base/checkpoint_000512.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/dino_low/Base/checkpoint_000512.pt) | `7dd58c2ceaa11e3b371ae329575111a67f3dc112b1b15b8d60ad2816a55c4da7` |
| dino_low | 0.094 | 1024 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/dino_low/Base/checkpoint_001024.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/dino_low/Base/checkpoint_001024.pt) | `c182d1e1c48cde13fc8136e03c87558d3fcd9ac6a77d682e459bb8b9fd7a7590` |
| dino_low | 0.094 | 1536 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/dino_low/Base/checkpoint_001536.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/dino_low/Base/checkpoint_001536.pt) | `0d2249a738cf5606e1f93fc55900082b133b7f39a245117e522efec15e2ed90a` |
| dino_low | 0.094 | 2048 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/dino_low/Base/checkpoint_002048.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/dino_low/Base/checkpoint_002048.pt) | `0158f85ec7ef85a48ad89aa4a6d0e16da509366cc20dc13edf13e39b59470f71` |
| dino_high | 0.94 | 512 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/dino_high/Base/checkpoint_000512.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/dino_high/Base/checkpoint_000512.pt) | `c884dacd8795955566a4fe051f62b4f17a7079108e04da980f40c7d4e84ed880` |
| dino_high | 0.94 | 1024 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/dino_high/Base/checkpoint_001024.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/dino_high/Base/checkpoint_001024.pt) | `eb648bdf35dfe84eaf8db79f77d59335c1a9f3f844ba0104f227e0378669fd87` |
| dino_high | 0.94 | 1536 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/dino_high/Base/checkpoint_001536.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/dino_high/Base/checkpoint_001536.pt) | `29bd11e368703a2d8f75c1a498e700d7d45ee10e40fbd0b0ad7e651594590c14` |
| dino_high | 0.94 | 2048 | [/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/dino_high/Base/checkpoint_002048.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/dino_high/Base/checkpoint_002048.pt) | `5448d57fa204e4d116317d1129aaf6be72cacce991cec4577b05a6a22360a270` |

所有CP还含私有`training_only`状态：student/head、AdamW参数组与完整moments、4条显式RNG、step、累计同步step时间、Base来源及protocol/config身份。普通推理只读student trainable_state，忽略训练head、optimizer/RNG；没有推理EMA。

val最终2048按每family LPIPS选出的**冻结test角色**（不表示test结论）：baseline → `baseline`（weight=0.0）；hr_vae → `vae_low`（weight=0.11）；hr_dino → `dino_low`（weight=0.094）。选择receipt：[/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/selection.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/selection.json)。

## 实际执行记录与安全的只读复核

训练实际只执行一次，以下逐字取自原training launch argv，**当前完成RUN不得重跑**。原argv的`bin/python`与pipeline解析后的`bin/python3.12`是同一解释器，均为Python3.12.16。

```bash
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python -m vsr.train ft-execute --run-dir /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z --expected-protocol-sha 7b620162ea4518f52ff7bb84a18ff955b2030ca4cdc94fd357547ece11021509
```

下面均逐项摘自已完成`pipeline/phases/*.json`，未补造参数。它们解释本轮如何得到配置、完整val及选择后test资产；本文件没有执行它们。

### `02_val_config`

```bash
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python3.12 /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_eval_configs.py val --source-root /home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync --study /home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/configs/fixed_teacher_study_20261108.json --training-run /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z --run-root /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z --dependency-root /home/hugo/Projects/VSR/.deps --output /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/val_config.json
```

### `03_val_prepare`

```bash
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python3.12 -m vsr.evaluate ft-prepare --config /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/val_config.json --run-dir /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/validation
```

### `04_val_execute`

```bash
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python3.12 -m vsr.evaluate ft-execute --run-dir /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/validation --expected-protocol-sha b638fd326211724d9f3b561d2bc1d0fa87c3deb478aff4ca2d1810d2acb64c0a
```

### `05_val_verify`

```bash
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python3.12 -m vsr.evaluate ft-verify --run-dir /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/validation
```

### `06_val_select`

```bash
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python3.12 -m vsr.evaluate ft-select --validation-run /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/validation --output /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/selection.json
```

### `07_test_materialize`

```bash
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python3.12 /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_data.py test-materialize --plan /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_clean_cohorts_20261108/iteration_000/test_plan.json --study /home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/configs/fixed_teacher_study_20261108.json --selection-receipt /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/selection.json --output /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/test_assets
```

### `08_test_config`

```bash
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python3.12 /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_eval_configs.py test --source-root /home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync --study /home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/configs/fixed_teacher_study_20261108.json --training-run /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z --run-root /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z --dependency-root /home/hugo/Projects/VSR/.deps --output /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/test_config.json --validation-run /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/validation --selection-receipt /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/selection.json --test-dataset /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/test_assets/dataset.json
```

### `09_test_prepare`

```bash
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python3.12 -m vsr.evaluate ft-prepare --config /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/test_config.json --run-dir /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/test
```

test执行命令已记录在pipeline当前状态/事件中；其scope仅为192恢复、不训练。这里只列协议/CLI，不读取test分数，也不把运行中的test写成已验证完成。

```bash
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python3.12 -m vsr.evaluate ft-execute --run-dir /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z/test --expected-protocol-sha d1572aeb22f736d1c1823aae858f8c1aaba838a748c4875610507aaa2c011869
```

已完成val的只读核心校验入口（CPU，不是再次推理；本次没有重复执行）：

```bash
CUDA_VISIBLE_DEVICES='' "$FT_PYTHON" -m vsr.evaluate ft-verify --run-dir "$FT_PIPELINE/validation"
```

真实训练诊断已通过：[/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/FixedTeacherTrainingDiagnostics_20261108/training_diagnostics.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/FixedTeacherTrainingDiagnostics_20261108/training_diagnostics.json)；完整数据/σ/timestep匹配、四RNG逐字节相同、20CP更新证据已记录。31保留窗口的shared_initial/2046PNG检查与成本/视频交付按completion_checklist单独封存；这些都不能用来代替method质量结论。

## 仅在另行授权预算后的全新复现入口

必须准备新路径/新study和新RUN目录，保留现有全部记录。下面尖括号项是待审的**新路径/新protocol SHA**，不是现有RUN的可执行命令。`ft-prepare`只绑定metadata；`ft-execute`才会训练。若换设备、路径、依赖或设计，原校准不能假装仍然有效，需重新封存/验收。

```bash
"$FT_PYTHON" -m vsr.train ft-calibration-prepare --config '<NEW_DESIGN_JSON>' --run-dir '<NEW_CALIBRATION_RUN>'
"$FT_PYTHON" -m vsr.train ft-calibrate --run-dir '<NEW_CALIBRATION_RUN>' --expected-protocol-sha '<NEW_CALIBRATION_PROTOCOL_SHA>'
# 仅将该校准实际生成的两种权重写入新的fixed_teacher_study_v1，审查后再prepare：
"$FT_PYTHON" -m vsr.train ft-prepare --config '<NEW_FINAL_STUDY_JSON>' --run-dir '<NEW_TRAIN_RUN>'
"$FT_PYTHON" -m vsr.train ft-execute --run-dir '<NEW_TRAIN_RUN>' --expected-protocol-sha '<NEW_TRAIN_PROTOCOL_SHA>'
# completed/exit0后，使用新输出目录串行完整val→selection→test→analysis：
"$FT_PYTHON" "$FT_WORK/fixed_teacher_pipeline_v2.py" --source-root "$FT_REPO" --study '<NEW_FINAL_STUDY_JSON>' --training-run '<NEW_TRAIN_RUN>' --run-root '<NEW_STUDY_RUN_ROOT>' --python "$FT_PYTHON" --output-dir '<NEW_CHILD_OF_WORKSPACE_OUTPUTS>' --dependency-root /home/hugo/Projects/VSR/.deps --after-training --execute
```

新缓存的真实CLI入口如下，同样会产生额外准备成本，不能对现有目录覆盖；准备缓存不等于获准新增训练。

```bash
"$FT_PYTHON" -m vsr.data prepare-cache --config '<BOUND_BASE_CONFIG_JSON>' --output '<NEW_COMMON_CACHE>' --project-root "$FT_REPO" --device cuda:0 --gpu-lock '<SHARED_GPU_LOCK>' --gpu-admission '<FRESH_REVIEWED_ADMISSION_JSON>'
"$FT_PYTHON" -m vsr.data prepare-dino-cache --training-cache-root '<BOUND_COMMON_CACHE>' --teacher-source /home/hugo/Projects/VSR/vendor/dinov2 --teacher-weights /home/hugo/Projects/VSR/weights/dinov2_vitb14_pretrain.pth --frame-batch-size 8 --output '<NEW_DINO_CACHE>' --device cuda:0 --gpu-lock '<SHARED_GPU_LOCK>' --gpu-admission '<FRESH_REVIEWED_ADMISSION_JSON>' --execute
```

## 中断恢复与普通学生推理：不同入口

**当前已完成训练/val不可resume。** 仅对原同协议、尚未完成的工程中断，保留attempt/失败/日志并确认无活跃executor后，才可调用真实恢复CLI：

```bash
"$FT_PYTHON" -m vsr.train ft-execute --run-dir '<INTERRUPTED_SAME_PROTOCOL_TRAIN_RUN>' --expected-protocol-sha '<ORIGINAL_UNCHANGED_TRAIN_PROTOCOL_SHA>' --resume
"$FT_PYTHON" -m vsr.evaluate ft-execute --run-dir '<INTERRUPTED_SAME_PROTOCOL_VAL_OR_TEST_RUN>' --expected-protocol-sha '<ORIGINAL_UNCHANGED_EVAL_PROTOCOL_SHA>' --resume
```

训练CLI没有任意`--checkpoint`参数：它在原arm目录内选择最新完整CP，严格恢复student/head/AdamW/4RNG/step/time，并验证完整日志前缀；checkpoint之后的中断尾部先保留为`interrupted_log_attempt_*`/`rollback_attempt_*`，废弃计算另计成本。CP缺失/损坏、source/config/data/protocol不符或没有完整CP时不能伪造resume。constantLR没有独立LR scheduler；学习率在AdamW参数组中恢复，native1000训练时间步网格由绑定UniPC配置重建。

评价resume只复用同协议完整且经SHA/marker校验的行，保留attempt/failure；不能改角色、减少行、换数据或用旧行拼接另一protocol。既有pipeline/交付目录不可用控制器自动重试；任何手工恢复应由维护者核对后单独记录。

普通恢复视频只用学生参数，不恢复训练优化器。下面`preview`为元数据入口，不加载模型；`run`需要新LR视频/新MP4路径和显式GPU授权，不是封存F32矩阵的逐字节替代。压缩LR视频解码会与原NPY像素不同，正式表格仍须使用`vsr.evaluate ft-prepare/ft-execute/ft-verify`的sealed NPY协议。

```bash
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python3.12 -m vsr.infer preview --checkpoint /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/vae_low/Base/checkpoint_002048.pt --allowed-checkpoint-root /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python3.12 -m vsr.infer run --checkpoint /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z/arm_runs/vae_low/Base/checkpoint_002048.pt --allowed-checkpoint-root /home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/fixed_teacher_runs/paired_training_20261108T033800Z --model-dir /home/hugo/Projects/ReprVSR-pretrained/models/Wan2.1-T2V-1.3B-Diffusers --input '<NEW_LRV_33FRAME_VIDEO>' --output '<NEW_OUTPUT_MP4>' --text-embedding /home/hugo/Projects/VSR/runs/formal_hq_v2_noref_20261102T142214.588571Z/fresh_text_embeddings.pt --device cuda:0 --gpu-lock /home/hugo/Projects/VSR_/runs/stage2_gpu.lock --gpu-admission '<FRESH_REVIEWED_ADMISSION_JSON>' --frames 33 --seed 42 --steps 20 --tile-height 256 --tile-width 448 --overlap 64 --execute
```

以上CLI均按冻结源码`train.py`/`evaluate.py`/`infer.py`/`data.py`的parser和实际phase argv核对；没有新增不存在的参数。完整2048步跨RUN/跨机器bitwise重复实验尚未另跑，不把短预飞/匹配RNG证明冒充该结论。

## 原始论文与本轮明确适配

依据已核对的[method_references.md](/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync/docs/fixed_teacher_20261108/method_references.md)和[适配记录](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/scratch/fixed_teacher_paper_adaptation_20261108.md)，本轮是方法适配，不是原论文逐项复现：

- [SRA² / VAE-REPA，2601.17830v2](https://arxiv.org/html/2601.17830v2)：原文SD-VAE紧凑latent/SiT/多层MLP、SmoothL1 β=.05；本轮为冻结Wan视频VAE的normalized posterior.mode、33→9、elementmean β=1、单Linear1536→64。
- [REPA，2410.06940v4](https://arxiv.org/html/2410.06940v4)；[作者loss](https://github.com/sihyun-yu/REPA/blob/main/loss.py)和[SiT投影头](https://github.com/sihyun-yu/REPA/blob/main/models/sit.py)：原方法干净视觉patch表征对齐噪声hidden，作者实现channel-L2/negative cosine与三Linear/两SiLU；本轮单Linear1536→768、mean(1−cos)，与负cos相差常数但梯度相同。
- [DINOv2，2304.07193v2](https://arxiv.org/html/2304.07193v2)和[官方仓库](https://github.com/facebookresearch/dinov2)：本轮B/14无register，取final x_norm_patchtokens、排除CLS，HR全区域128×224→112×196，patch14→8×14；33帧各自编码，frame0独立、1–4…29–32均值后channel-L2映射到9个latent时刻。**名义时间分组不是VAE卷积完整感受野的精确逆映射。**
- 学生统一block7（0-based，第8/30层）；Wan主干冻结，只训练同一LoRA/LR分支及训练期head。各arm从同一已训练Base01续训，freshAdamW foreachFalse；不是ImageNet从头全模型训练。VAE/DINO head分别98,368/1,180,416参数，尺寸差异必须计入成本解释。
- 权重由train-only σ=.2/.5/.8梯度校准冻结为两档，不照搬论文λ；本轮无EMA/无LR条件破坏，旧EMA/G4失败记录不被重解释为本轮通过。EMA仅第二轮候选，不自动启动。
- 目标是clean×4视频SR，主LPIPS并报PSNR/SSIM及固定TecoGAN时间指标；不是FID/ImageNet生成结论。仅一个训练seed、配对clip描述CI，不能宣称多seed显著性或照搬论文加速/GFLOPs。已有VAE公共缓存为共同沉没成本，新DINO一次性缓存、校准、训练step wall、评价/IO与废弃尝试分别计账；辅助loss下降不等于SR质量改善。
- 新test仅在本轮val选参后正式开启，仍属于历史上被旧模型评估过的split；自动缩略图完整性筛查是事前例外，不能称pristine/从未访问test像素。全部输出是完整33帧128×224 crop，不冒称原始长视频全长或720p原分辨率。

