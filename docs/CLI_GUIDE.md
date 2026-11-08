# 当前主线入口说明

2026-11-07。下列是已实现接口；当前本地5080任务状态及门槛见[本地继续计划](local_continuation_20261107/plan.md)，H100迁移延期。示例中的`/private/...`、SHA与UUID需按实际机器和新协议填写，不能直接复制为启动命令，也不能重复启动已有任务。

## 1. 开发时先看帮助和CPU测试

```bash
python -m vsr.data --help
python -m vsr.train --help
python -m vsr.infer --help
python -m vsr.evaluate --help
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

CLI help/import不加载GPU模型，不下载权重。普通CPU测试无需私有数据；有本机官方指标权重及已有Base资产时执行额外CPU兼容检查，否则明确skip。

## 2. 数据：data.py

- `manifest --raw-root ... --output ... [--seed ...]`：原视频清单与来源隔离，CPU工作。
- `prepare-cache --config ... --output ... [--project-root ...] [--device cpu]`：固定文本缓存、流式LR/HR准备与VAE编码，新目录、新seal；不构建T5/DINO/任务模型。已有已审计数据优先复用。
- `check-cache --cache-root ... [--seal-sha256 ...] [--verify-all]`：检查旧protocol33缓存。verify-all重算全部资产，普通reader只核对实际使用的条目。

CUDA预处理额外要求`--gpu-lock`与`--gpu-admission`，使用和训练/推理/评价相同的租约。CPU默认不查询GPU。若固定数据/窗口改变，应另立科学协议，不覆盖已有缓存。

## 3. 训练：train.py

`prepare`只绑定元数据/文件hash，不加载GPU模型。`execute`需要已准备的新run及明确SHA。

```text
python -m vsr.train prepare
  --mode base|smoke|resource|pilot
  --config /private/matched-Base-source-config.json
  --cache-root /private/sealed-training-cache
  --model-directory /private/official-Wan-directory
  --allowed-checkpoint-root /private/runs
  --device cuda:0
  --gpu-lock /private/runs/stage2_gpu.lock
  --gpu-admission /private/host-admission.json
  --memory-cap-gib <事前声明的上限>
  --run-dir /private/runs/NEW-RUN
  --run-root /private/runs
```

- `base`：fresh official Base8192入口；现有三个Base已完成，本次及下一轮无需重训。
- `smoke`：另加`--base-checkpoint`与`--expected-base-sha`，仅用Base01和首3训练clip；固定3A2诊断updates+保存/学生重载+1次UniPC20恢复，0正式RE。无需G2/G3，不能用于声称科学门槛通过；本次已完成，不自动重跑。
- `resource`：另加`--base-checkpoint`、`--expected-base-sha`及`--g2-run`；必须重算完整G2通过，执行70诊断updates。
- `pilot`：resource的参数外再加`--g3-run`和`--g4-definitions`；同Base01、fresh optimizer、flow-only/C1/A2各2048步。G4原定义在任何方法训练/评分之前绑定。

实际执行命令形态为`python -m vsr.train execute --run-dir ... --run-root ... --expected-protocol-sha ...`；按前置门槛启用新run：完整Base评价G2真实通过后才进入resource，G3通过后才进入pilot。重构前的old run不能交给新入口resume。传入matched Base的原config，模型目录迁移使用独立`--model-directory`，不改原checkpoint配置。

## 4. 推理：infer.py

- `preview --checkpoint ... --allowed-checkpoint-root ... [--model-dir ...]`：只预览来源配置。
- `run`额外需要`--input`、`--output`、`--text-embedding`和显式`--execute`；device默认cpu。CUDA要求`--gpu-lock`与`--gpu-admission`。
- 采样选项：`--seed`、`--steps`、`--tile-height`、`--tile-width`、`--overlap`；默认使用完整输入帧与原FPS。只有显式`--frames`才截取前缀。
- 固定文本缓存直接提供，不加载T5。主线NoRef，不接受HRI参考图像。仅学生恢复权重参与推理。

## 5. 评价：evaluate.py

1. 以[配置模板](../configs/evaluation_bases.template.json)填入明确绝对路径、3个已审计Base SHA、固定text及LPIPS权重、host准入。
2. `python -m vsr.evaluate prepare --config ... --run-dir ...`创建新协议。
3. 审核新协议及前置检查后用`execute --run-dir ... --expected-protocol-sha ...`执行；既定本地计划正常通过门槛后继续，无需重复许可，已有run不重复启动。
4. `verify --run-dir ...`重查全1152行/预测文件/完成marker/来源hash并重算门槛；部分run不能解锁后续阶段。

`kind=bases`要求3训练seed×32clips×6条件×2采样seed；`kind=pilot`要求flow-only/C1/A2的3arm矩阵，另需`base_evaluation_run`、`pilot_training_run`和`G4_definitions`，核对真实6144步训练及对应最终2048 checkpoint。

旧val32 seal和224资产内容不改。`path_map`只决定到哪里寻找同一批字节；旧源码绑定应指向`legacy/pre_refactor_20261106/`的相同字节。不同host上的FFmpeg/CPU依赖差异必须显式验收并事前记录，不关闭校验来求通过。H100迁移当前延期；未来启用时仍需现场环境与GPU验证。

## 6. 设备身份与公开同步

所有CUDA执行共用checkpoints.gpu_lease。admission声明`expected_gpu_uuid`，可选`desktop_processes`完整pin；空清单要求目标GPU无compute进程，不能仅按进程名字放行。CPU检查使用mock inventory；本机5080已验证真实桌面准入/共享锁，H100准入尚未现场验证。

GitHub只同步代码、配置、协议、统计和文本日志；模型、checkpoint、原视频、tensor缓存及凭证通过私有通道提供。实验期间固定协议绑定的数值源码与依赖，不git pull覆盖运行源码；文档和文本日志可单独同步。


## HR VAE latent：第一轮独立成对实验

先阅读 `docs/hr_vae_latent_20261108/plan.md`，使用本机声明Python与固定依赖；机器路径在`configs/hr_vae_latent_20261108.json`显式列出。权重和缓存不在Git仓库中。

```bash
python -m vsr.train hr-prepare --config configs/hr_vae_latent_20261108.json --run-dir /absolute/new/study_run
python -m vsr.train hr-probe --run-dir /absolute/new/study_run --expected-protocol-sha PREPARED_SHA
python -m vsr.train hr-execute --run-dir /absolute/new/study_run --expected-protocol-sha PREPARED_SHA
# 同本轮协议的故障恢复：保留中断记录并完整恢复head/AdamW/RNG，不是重用旧实验权重
python -m vsr.train hr-execute --resume --run-dir /absolute/new/study_run --expected-protocol-sha PREPARED_SHA
python -m vsr.evaluate hr-prepare --config /absolute/evaluation_config.json --run-dir /absolute/new/evaluation_run
python -m vsr.evaluate hr-execute --run-dir /absolute/new/evaluation_run --expected-protocol-sha EVALUATION_SHA
python -m vsr.evaluate hr-verify --run-dir /absolute/new/evaluation_run
```

真实GPU小测使用27诊断更新，不复用其参数；正式基线/0.01/0.1各2048步。评价10个role×32片段×2samplers=640次，所有选定33帧都经标准`vsr.infer.restore_video`生成、保存及计分；不把完整裁剪片段称为原始长视频全长。指标从PNG等价q/255像素计算，MP4只作视觉展示。

结果分析工具在任务工作区的`work/analyze_hr_vae_study.py`，待完整验证后产生LPIPS主指标、PSNR/SSIM/时间指标、两种训练曲线与单seed描述性区间。不会用辅助loss下降替代真实推理效果。


# Fixed-teacher 评价配置与核心 CLI

`work/fixed_teacher_eval_configs.py` 只生成新的、不可覆盖的 JSON 配置，不创建 protocol/run，不调用 GPU。必须等五条训练轨迹全部完成且同版本源码未变；它检查真实完成链、10240行日志、20个训练checkpoint与每个评价角色的绑定/时间。最终 `ft-prepare` 再完成原生依赖、真实资产和选择链验证。

以下变量中的 study/training/val run/config 路径需由根代理按真实完成产物填写；本文件没有执行这些操作。

```bash
FT_PYTHON=/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python
FT_WORK=/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work
FT_REPO=/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync
FT_DEPS=/home/hugo/Projects/VSR/.deps
FT_STUDY=/absolute/path/to/final_fixed_teacher_study.json
FT_TRAINING=/absolute/path/to/completed_training_run
FT_RUN_ROOT=/absolute/path/to/evaluation_run_root
FT_VAL_CONFIG=/absolute/path/to/new_validation_config.json
FT_VAL_RUN=/absolute/path/to/new_validation_run
FT_SELECTION=/absolute/path/to/new_selection_receipt.json
FT_TEST_DATASET=/absolute/path/to/selected_after_test_assets/dataset.json
FT_TEST_CONFIG=/absolute/path/to/new_test_config.json
FT_TEST_RUN=/absolute/path/to/new_test_run
```

验证集配置：16角色＝共享step0＋五条轨迹各512/1024/2048，完整1024次恢复。输入只取 final study 的 repaired validation seal/dataset 绑定，不使用旧val名单。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$FT_DEPS:$FT_REPO" "$FT_PYTHON" "$FT_WORK/fixed_teacher_eval_configs.py" val --source-root "$FT_REPO" --study "$FT_STUDY" --training-run "$FT_TRAINING" --run-root "$FT_RUN_ROOT" --dependency-root "$FT_DEPS" --output "$FT_VAL_CONFIG"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$FT_DEPS:$FT_REPO" "$FT_PYTHON" -m vsr.evaluate ft-prepare --config "$FT_VAL_CONFIG" --run-dir "$FT_VAL_RUN"
```

`ft-prepare` 输出的 protocol SHA 必须由根代理审查后才用于唯一GPU执行：`ft-execute --run-dir <run> --expected-protocol-sha <SHA>`。本轮故障允许同一protocol显式`--resume`；已完成行须完整hash验证，未提交行保留后重做。不能以新参数覆盖旧protocol。

全val完成后，先检查服务正常退出和GPU释放，再完整验证、生成不可修改选择receipt：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$FT_DEPS:$FT_REPO" "$FT_PYTHON" -m vsr.evaluate ft-verify --run-dir "$FT_VAL_RUN"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$FT_DEPS:$FT_REPO" "$FT_PYTHON" -m vsr.evaluate ft-select --validation-run "$FT_VAL_RUN" --output "$FT_SELECTION"
```

每family只依据最终2048的all33 LPIPS选候选；精确并列选弱weight。数据helper验证并全文重算receipt后才生成正式test RGB资产；预先自动完整性缩略图是单独绑定的`integrity_only`例外，不用于模型质量选择。

已有选后test dataset后，生成三个最终角色（baseline/VAE/DINO）的192次恢复配置：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$FT_DEPS:$FT_REPO" "$FT_PYTHON" "$FT_WORK/fixed_teacher_eval_configs.py" test --source-root "$FT_REPO" --study "$FT_STUDY" --training-run "$FT_TRAINING" --run-root "$FT_RUN_ROOT" --dependency-root "$FT_DEPS" --validation-run "$FT_VAL_RUN" --selection-receipt "$FT_SELECTION" --test-dataset "$FT_TEST_DATASET" --output "$FT_TEST_CONFIG"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$FT_DEPS:$FT_REPO" "$FT_PYTHON" -m vsr.evaluate ft-prepare --config "$FT_TEST_CONFIG" --run-dir "$FT_TEST_RUN"
```

builder只读receipt中的已冻结选择和验证文件绑定，不重新读取或解释质量分数；`ft-prepare` 独立完整重算val选择，拒绝手工选checkpoint。test阶段不重新调参。每一行完整输出33帧PNG和MP4；它们是固定128×224评价crop，不是源视频全长或原分辨率。

CPU验收记录：[fixed_teacher_repaired_val_metadata_and_config_builder_CPU_review_20261108.json](fixed_teacher_repaired_val_metadata_and_config_builder_CPU_review_20261108.json)。最终repaired cohort无flag仅指声明的自动多帧筛查，不能称穷尽证明；原test历史暴露仍需披露。
