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
