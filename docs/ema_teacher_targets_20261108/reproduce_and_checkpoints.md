# Wan TRD：复现、恢复和检查点索引

记录时间：2026-11-08T13:44:42.866230+00:00。本文件是当前实验的复现材料；没有从本文件启动额外训练或推理。

## 当前完成范围

三条训练轨迹已完成：baseline、冻结 Base 教师、EMA Base 教师各 2048 步，共 6144 个科学更新；41 个正确性／资源／梯度校准更新独立计数。全部 12 个训练检查点保留。训练审计通过，5 张 CSV 共 47 行已独立核对，原审计文件未修改。

父流程已正常退出0/MainPID0，完整640验证及192测试原生verify通过；全部三组最终2048已固定并评价，没有按分数淘汰教师、选择中间checkpoint或更换权重。75个配对区间独立审计、2112PNG共享起点精确回归、六套固定视频及root实际时序ROI检查均完成。质量结论及证据限制见同目录final_report.md；不由实验完成推出方法有效。

- 训练目录：`/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/paired_training_20261108T124055Z`
- 父流程目录：`/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/pipeline_paired_training_20261108T124055Z`
- 训练协议 SHA256：`6d5221ce42f2264e10a78c51c04b4d833a3137e146aa7aa54d2c176151b4f072`
- 完成训练 summary SHA256：`8f6bb62a1d02bff66922629196cea9ca5ded9d2244c3ae17cb368d4603720d1d`
- 实际校准协议 SHA256：`a74e1a44f362a569d934696ff171f71cb26a5351beab35b0b567d361ceb6c650`
- 最终统一权重配置：`/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/study_common_lambda_20261108.json`，SHA256 `ff9db1b4d317d3dfac338089a16c7b79dcb4ed0a2efa341ed57096e7d4fcc8e6`
- 训练诊断：[training_audit.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/training_diagnostics/training_audit.json)；独立文本复核：[training_diagnostics_independent_review.json](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/outputs/WanTRD_20261108/training_diagnostics_independent_review.json)，SHA256 `f29ab5b52659a668c8dbf98f41fddcea7d2d161a81c256624471e81d5c35ede7`。

独立文本复核重算完整训练日志、CSV 汇总、源文件和小型元数据的绑定关系，并核对 12 个检查点的声明路径／文件大小。它没有重跑原训练审计，没有再次加载或哈希检查点。实际检查点张量、四条 RNG 原始字节、AdamW 状态及 FP32 教师检查由先前已经完成的 `training_audit.json` 负责；大型官方权重的完整绑定另由原生协议检查负责。

## 实验定义和已有资源

这是保留 Wan2.1 VSR 框架的 VideoREPA 启发式关系监督适配，不是 CogVideoX／VideoMAEv2 原论文数值复现。三组共用同一 Base 学生初始化、训练数据顺序、噪声和时间步；两种教师的输入噪声序列相同，教师权重更新策略是两组间的主要区别。

- Wan2.1-T2V-1.3B Diffusers，官方 revision `0fad780a534b6463e45facd96134c9f345acfa5b`；本地模型目录 `/home/hugo/Projects/ReprVSR-pretrained/models/Wan2.1-T2V-1.3B-Diffusers`。
- 共同起点：`/home/hugo/Projects/VSR_/runs/stage2_protocol33_base_seed20261102_20261106T025845.129456Z/Base/checkpoint_008192.pt`，SHA256 `28356e22e4e718d3c378588793f60732dbc93825ec5d78b8a85c8150060da63f`。这是已完成的 8192 步 NoRef Base；每条新轨迹只载入学生任务权重，重新创建 AdamW 和随机流。
- 数据：HQ-VSR，同一 1641 个训练片段；33 连续帧、stride 1、HR 128×224、固定 ×4 双三次 LR 32×56、原 FPS。固定 VAE 编码和既有完整缓存，LR 注入 block 0／5／9，训练 rank-8 注意力 LoRA 和 LR 分支。
- 学生／教师均取 block 9，1008 个 token；全 token 关系监督，margin .1、归一化 epsilon 1e-6；F/E 均用 identity 初始化的 Linear 1536→1536 头（2,360,832 参数）。共同辅助权重 λ=.3，仅由训练校准梯度决定，不看验证／测试分数。
- 教师与学生共享 LR 和基础噪声 epsilon；教师使用最接近 `0.5×学生 sigma` 的原生时间网格位置。冻结教师始终为原始 Base；EMA 仅对 FP32 可训练参数 shadow 在成功更新后做 decay .999 的更新。
- batch 1；AdamW lr 1e-4、weight decay .01、`foreach=False`、clip norm 1；恒定学习率。DiT bf16、VAE FP32、gradient checkpointing；使用缓存的固定文本，不加载 T5。
- 推理只使用各组最终／规定中间检查点的学生；不载入投影头，不使用教师或 EMA 权重。

| 组 | λ | 科学更新 | 已提交同步训练 step 时间（秒） | EMA 更新数 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.0 | 2048 | 696.574673 | 0 |
| frozen | 0.3 | 2048 | 988.026295 | 0 |
| ema | 0.3 | 2048 | 1054.971584 | 2048 |

总已提交同步 step 时间 `2739.572551` 秒；实际完整训练 attempt 时间 `2828.383745` 秒；丢弃的已完成 step 时间为 0。前者用于曲线，不是纯 GPU kernel 时间，也不含教师校准、评估、分析和完整初始化 I/O。不要把两种 scope 相加。

验证／测试采用预封存的各 32 个源视频窗口，两个推理种子 42／43，原生 UniPC 20 步、CFG1、FP32 VAE、tile 256×448／overlap64、固定 PNG 量化和现有 LPIPS-Alex0.1／PSNR／SSIM／TecoGAN tLP100／tOF。每条输出完整 33 帧 crop 和 MP4；这不是原长视频全长／原分辨率推理。复用的测试来源曾用于历史轮次，属于探索性测试，不能描述为盲测或未接触测试集。本轮没有按测试选参。

## 环境与完整执行命令

以下是实际 CLI 的复现用法。**当前训练已经完成、当前验证由唯一父服务执行中，不要对现有目录再次运行这些命令，也不要另启 GPU 队列。** 重新复现应使用新日期目录和单独预算；跨机器需要重新审查 GPU UUID、桌面 admission、资源和资产路径，不能直接移植旧 admission。

```bash
SOURCE='/home/hugo/.codex/worktrees/vsr-wan-trd/VSR-sync'
WORK='/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work'
PY='/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python'
TRAIN='/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/paired_training_20261108T124055Z'
PIPELINE='/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/pipeline_paired_training_20261108T124055Z'
PREFLIGHT='/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/preflight_20261108T123515Z'
export PYTHONPATH='/home/hugo/Projects/VSR/.deps:'"$SOURCE"
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false XFORMERS_DISABLED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=GPU-4414baf2-fd94-f31a-18e6-ebe95dcd1813
cd "$SOURCE"
```

实际 41-update preflight 由设计配置准备，完成后冻结共同 λ=.3 的训练配置。下列命令说明该次执行链；现有目录的重复执行会被 guard 拒绝。

```bash
"$PY" -m vsr.train trd-prepare   --config "$SOURCE/configs/wan_trd_design_20261108.json" --run-dir "$PREFLIGHT"
"$PY" -m vsr.train trd-preflight --run-dir "$PREFLIGHT"   --expected-protocol-sha 'a74e1a44f362a569d934696ff171f71cb26a5351beab35b0b567d361ceb6c650'

"$PY" -m vsr.train trd-prepare   --config "$WORK/wan_trd_runs/study_common_lambda_20261108.json" --run-dir "$TRAIN"
"$PY" -m vsr.train trd-execute --run-dir "$TRAIN"   --expected-protocol-sha '6d5221ce42f2264e10a78c51c04b4d833a3137e146aa7aa54d2c176151b4f072'
```

若只在同一冻结环境复现最终科学比较，可复用已通过的 preflight 收据和最终配置，以新的训练目录执行 `trd-prepare`，审阅该新协议，然后将其实际 SHA256 传入 `trd-execute`。所有 3×2048 更新必须从同一 Base 重新开始，不能把 smoke 或校准状态当共同起点。源码、依赖或绑定资产有变化时，旧 preflight 收据不能证明新版本正确。

正常验证／测试由父流程串行执行。其训练诊断命令已经完成，不要再次写入原 `training_diagnostics/`。下面列出父流程核心推理链，便于在全新复现目录中逐步复现；`VAL_PROTOCOL_SHA`／`TEST_PROTOCOL_SHA` 是相应 metadata prepare 返回的 SHA，不是训练协议 SHA。

```bash
# 当前正在执行的验证：共享初始学生 + 3组×(512,1024,2048)，共640次。
"$PY" -m vsr.evaluate trd-build-config --training-run "$TRAIN" --phase val   --output "$PIPELINE/val_config.json" --run-root "$PIPELINE"
"$PY" -m vsr.evaluate trd-prepare --config "$PIPELINE/val_config.json"   --run-dir "$PIPELINE/validation"
"$PY" -m vsr.evaluate trd-execute --run-dir "$PIPELINE/validation"   --expected-protocol-sha 'c8dbde5e11d5e151e0673e152829e161221db1164644e766a22835573376aa05'
"$PY" -m vsr.evaluate trd-verify --run-dir "$PIPELINE/validation"

# 待完整验证和verify通过：锁定全部三组final2048，不做分数选参。
"$PY" -m vsr.evaluate trd-lock-finals --validation-run "$PIPELINE/validation"   --output "$PIPELINE/finals_receipt.json"
"$PY" -m vsr.evaluate trd-build-config --training-run "$TRAIN" --phase test   --output "$PIPELINE/test_config.json" --run-root "$PIPELINE"   --validation-run "$PIPELINE/validation" --finals-receipt "$PIPELINE/finals_receipt.json"
"$PY" -m vsr.evaluate trd-prepare --config "$PIPELINE/test_config.json"   --run-dir "$PIPELINE/test"
# 把新prepare返回的SHA赋给TEST_PROTOCOL_SHA，审阅后才能执行。
"$PY" -m vsr.evaluate trd-execute --run-dir "$PIPELINE/test"   --expected-protocol-sha "$TEST_PROTOCOL_SHA"
"$PY" -m vsr.evaluate trd-verify --run-dir "$PIPELINE/test"
```

实际父流程命令／返回码／stdout／stderr 保存于 `$PIPELINE/phases/`；训练和 preflight 的完整启动环境分别保存于 `$TRAIN/launch.json`、`$PREFLIGHT/launch.json`。父服务 `vsr-wan-trd-continuation-20261108t125401z.service` 只等待并评价已有训练，不重新训练。所有 GPU worker 强制复核共享 `/home/hugo/Projects/VSR_/runs/stage2_gpu.lock` 和新鲜 GPU／系统桌面身份；无并行第二矩阵、无自动协议修改。

## 中断恢复

本轮实际训练无恢复、无丢弃更新。以下只用于**已诊断为可恢复的工程中断**，不是对已完成轨迹加训的命令，也不能绕过非有限、OOM、科学门槛或数据／协议变化。

```bash
# 同一未完成训练目录、同一协议SHA；服务完全退出并确认GPU锁释放后：
"$PY" -m vsr.train trd-execute --run-dir "$INTERRUPTED_TRAIN"   --expected-protocol-sha "$UNCHANGED_TRAIN_PROTOCOL_SHA" --resume

# 同一未完成评价目录、同一协议SHA；验证已完成行后继续：
"$PY" -m vsr.evaluate trd-execute --run-dir "$INTERRUPTED_EVAL"   --expected-protocol-sha "$UNCHANGED_EVAL_PROTOCOL_SHA" --resume
```

训练恢复自动选择该 arm 最后一个完整检查点；已完成 arm 先核验原完成记录再跳过。检查点保存学生、F/E 投影头、AdamW 全状态和命名分组、四条配对 RNG、Python／NumPy／Torch CPU／CUDA 全局 RNG、step／data cursor、FP32 教师状态及更新数、共同起点和协议来源、累计已提交 step 时间。学习率恒定，由绑定配置确定。恢复先完整验证再修改；失败可回滚到加载前状态。比检查点更晚的中断日志会另存，回退后的浪费时间独立记录，不能当有效训练进度。推理加载只恢复学生权重，训练头和教师不进入推理。

## 全部 12 个检查点

完整根目录为 `/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/paired_training_20261108T124055Z`，各文件的完整路径可点击。SHA 和大小来自已完成 summary，原训练审计已经逐个实际哈希和加载；此次独立复核未重复读取大文件内容。

| arm | 更新数 | 完整路径 | 字节数 | SHA256 |
| --- | ---: | --- | ---: | --- |
| baseline | 512 | [checkpoint_000512.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/paired_training_20261108T124055Z/arm_runs/baseline/Base/checkpoint_000512.pt) | 78761969 | `d05e580606f32bffcd68a2a682f4d0170c60c5c7eca974ce56ab762ef96b4314` |
| baseline | 1024 | [checkpoint_001024.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/paired_training_20261108T124055Z/arm_runs/baseline/Base/checkpoint_001024.pt) | 78761969 | `373bd97c2551dc6e18a6f2ced3c42c03dc4495269d54b96e00f58c09b0ea6a6a` |
| baseline | 1536 | [checkpoint_001536.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/paired_training_20261108T124055Z/arm_runs/baseline/Base/checkpoint_001536.pt) | 78761969 | `22ce217b5c78fbe137e7ad2290d41e0e5ea4f518274c2bca31935e2edb1c85dd` |
| baseline | 2048 | [checkpoint_002048.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/paired_training_20261108T124055Z/arm_runs/baseline/Base/checkpoint_002048.pt) | 78761969 | `5964a0646c67657fe63f3882793c5b6b2296fb03dfdafaba644231d299f3e897` |
| frozen | 512 | [checkpoint_000512.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/paired_training_20261108T124055Z/arm_runs/frozen/Base/checkpoint_000512.pt) | 133265409 | `e72ad4f8a17e571ff737b866c9e6394cc5e0a7de46d3911123b3ef8bda7396bf` |
| frozen | 1024 | [checkpoint_001024.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/paired_training_20261108T124055Z/arm_runs/frozen/Base/checkpoint_001024.pt) | 133265409 | `533ac84816cec508c2e33e32b5a455d3237ba51ab6e232d0bfb3ea62676b5a33` |
| frozen | 1536 | [checkpoint_001536.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/paired_training_20261108T124055Z/arm_runs/frozen/Base/checkpoint_001536.pt) | 133265409 | `c613ffa6f69f4a9232e4b74db9efbb1b72403629e1b6b2db8ecaee6f85456ae4` |
| frozen | 2048 | [checkpoint_002048.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/paired_training_20261108T124055Z/arm_runs/frozen/Base/checkpoint_002048.pt) | 133265409 | `161ccbb7fcda1b9afa318ba405890ac87571cc7daa70b8e50e149f4bf4678424` |
| ema | 512 | [checkpoint_000512.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/paired_training_20261108T124055Z/arm_runs/ema/Base/checkpoint_000512.pt) | 133265409 | `4070015a5ba851efdecfe04b6894d7d933a69d054c63dd81a0e9a67706057dec` |
| ema | 1024 | [checkpoint_001024.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/paired_training_20261108T124055Z/arm_runs/ema/Base/checkpoint_001024.pt) | 133265409 | `722af2fe0331fe4ea4652ce507527502dfca94cb38f9bcad864eaf4d4534ad19` |
| ema | 1536 | [checkpoint_001536.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/paired_training_20261108T124055Z/arm_runs/ema/Base/checkpoint_001536.pt) | 133265409 | `8bca66334f8f7009996c653def0ae0a5f2b68e7e8363b06d3aefbcf2408de0e7` |
| ema | 2048 | [checkpoint_002048.pt](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/wan_trd_runs/paired_training_20261108T124055Z/arm_runs/ema/Base/checkpoint_002048.pt) | 133265409 | `1d5c29cba84190d9076ba21eb6dc05f99887b2b51f25eee73bd0d4545aa926eb` |

## 冻结的数值源码

以下 11 个文件从校准到训练及本轮验证／测试结束保持不变；无需为复现增加业务 Python 模块。历史源码、实验和实际 checkout 均保留。

| 文件 | SHA256 |
| --- | --- |
| `__init__.py` | `7ff7f1b099772cc4a207c4e924cf34ddffd6057ba8ca7525afe9b52ec5838af1` |
| `checkpoints.py` | `3bafc745a1ad35a269a970c4e0e0e7545b880e1370381e9785b62d07a96cb9d5` |
| `data.py` | `82e475daddaa41b595bb535540e31f66d2863cfe9a82bd1743c1b4f7f9f00416` |
| `evaluate.py` | `eddb93a0b32f4c3ab5120fea061e20a50cc24dc46a14c0b27fa43b0a2d5a1f67` |
| `infer.py` | `16af3fb011a3284d25952ed1d28bdadac65f9deb8ed50e52d3216e31050df69d` |
| `losses.py` | `001aee24f56c9d4750299aca97a274e324b0521cecafc2a2a131ed5278d4d66f` |
| `metrics.py` | `ea1005380e788506ccce7050650e5b0e6031f8b27ca4c26f92931cd845484ad6` |
| `model.py` | `7394e76203f29c24232dc586d7a65af69738b02c235996066c675e92e5ca4598` |
| `teacher.py` | `162dc28427d27c5443985d37f23871c403cc593f13f8c33e52c3776a9a6bfe38` |
| `train.py` | `83746d7c668f7202f781542b9403b9686669b596029a17f06fedbd0719040c18` |
| `vae.py` | `e83d0b88cdc23c4fd47c72bc278089df8c2a6615bcb7a9112d9aca72541956e6` |

权重、检查点、原始视频、PNG／MP4 和 tensor 缓存均保留本机，不进入公开 GitHub；本文件、配置、日志和统计文本可以同步。
