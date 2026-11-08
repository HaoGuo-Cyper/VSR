# 完整 Wan TRD 评价文本导出

这是独立的 CPU 后处理工具，不增加项目业务模块，不修改训练、评价核心或现有 exporter。只有当前轮完整 val640 或 test192 **已经 completed，且原生 `trd-verify` 收据已保存**后才能运行。编写和测试期间只用了合成数据，没有读取真实未完成矩阵的 observations／metrics，没有实际导出当前矩阵。

工具：[export_wan_trd_completed_evaluation.py](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/export_wan_trd_completed_evaluation.py)。SHA256：`131cc207f523571728c31fd179fb4aff3e5a4bf492fcbb28a742d265491ff3c8`。

发布前元数据兼容修正：封存test清单包含合法首字符为`-`的clip ID。新版本仅为clip引入允许该首字符的独立安全规则；role／目标目录规则、路径逃逸拒绝和数值协议均不变。旧版`cbfa89...`尚未执行真实导出；原合成验证记录保留。

先读 completed 状态、protocol、完整 native verify 收据和 summary 哈希链，随后才读取质量文本。检查全部 role×32clip×2seed、规范顺序、每行 metrics SHA／complete marker、33帧／28时序对、推理输入声明和学生载入／状态审计。缺行、重复、路径逃逸、符号链接或绑定改变均拒绝；从不打开权重、checkpoint、tensor、PNG 或 MP4，也不重复 hash PNG。

导出白名单包括：完整 observations、每行 metrics／complete、每 role 的 load／state audits、GT 文本 complete markers、根 protocol／status／summary／runtime、可选 progress／执行时间／console，以及全部规范 attempt／admission／failure JSON。未知文件不打开、不复制。11 个冻结源码仅核 SHA，不重复导出。

所有来源文本保留原始 UTF-8 字节及 CRLF。超过1MB的文本按最多800000字节的有效 UTF-8 分块保存，manifest 包含原始完整 SHA／字节数和逐块顺序／SHA；原始文本可完全恢复。这个上限为 GitHub 工具的2.6MB JSON payload留出转义余量。

## 完成后的实际用法

下面是未来调用说明，不是当前阶段的启动指令。父流程已管理唯一评价矩阵，禁止另启 worker 或导出未完成结果。目标必须是本源码树 `docs/ema_teacher_targets_20261108/experiments/` 下全新的、对应 `val_` 或 `test_` 前缀子目录；默认名称包含 phase、run名和协议摘要。已有目标永不覆盖。

```bash
WORK='/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work'
SOURCE='/home/hugo/.codex/worktrees/vsr-wan-trd/VSR-sync'
PIPELINE="$WORK/wan_trd_runs/pipeline_paired_training_20261108T124055Z"
PY='/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python'

# 仅在完整val640及05_val_verify成功后：
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$PY" \
  "$WORK/export_wan_trd_completed_evaluation.py" \
  --run-dir "$PIPELINE/validation" \
  --verification-json "$PIPELINE/phases/05_val_verify.result.json" \
  --source-root "$SOURCE"

# 仅在完整test192及10_test_verify成功后：
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$PY" \
  "$WORK/export_wan_trd_completed_evaluation.py" \
  --run-dir "$PIPELINE/test" \
  --verification-json "$PIPELINE/phases/10_test_verify.result.json" \
  --source-root "$SOURCE"
```

运行返回 `complete_verified_text_export` 和 manifest SHA 后，才能把该新目录作为完整文本交付同步。原文件中的本机路径及媒体／模型 hash 可以公开，但不代表二进制已被包含。

## 无损恢复分块文本

将导出目录设为 `EXPORT_DIR`，将全新的恢复目录设为 `RESTORED_DIR` 后，可运行下列纯 CPU 代码。它恢复的是原始文本文件字节，不能据此重建二进制权重或视频。

```python
import hashlib, json, os
from pathlib import Path
src, dst = Path(os.environ['EXPORT_DIR']), Path(os.environ['RESTORED_DIR'])
assert not dst.exists()
dst.mkdir(parents=True)
manifest = json.loads((src / 'manifest.json').read_text())
for entry in manifest['files']:
    if entry['storage'] == 'copy':
        raw = (src / entry['path']).read_bytes()
    else:
        parts = []
        for part in entry['parts']:
            data = (src / part['path']).read_bytes()
            assert len(data) == part['bytes']
            assert hashlib.sha256(data).hexdigest() == part['sha256']
            parts.append(data)
        raw = b''.join(parts)
    assert len(raw) == entry['bytes']
    assert hashlib.sha256(raw).hexdigest() == entry['sha256']
    out = dst / entry['path']
    assert '..' not in Path(entry['path']).parts and not Path(entry['path']).is_absolute()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
```

## 合成 CPU 验证

9项测试方法（含17个子场景）通过，0失败／0错误，用时3.033秒。覆盖完整 val640／test192逐字节保留、running提前拒绝、缺／重复行、逃逸／符号链接、SHA／marker变化、未知二进制排除，以及大文本／多字节UTF-8／CRLF无损分块。新增完整test192合法首`-`clip，以及空值、`..`、`../escape`、斜杠和反斜杠拒绝。完整记录：[CPU validation v2 JSON](/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/scratch/wan_trd_completed_evaluation_exporter_cpu_validation_v2_20261108.json)。

该工具只验证和导出已保存文本及收据链，不能代替真正的原生 `trd-verify` 对媒体、模型、数据和完整数值来源的检查；它也不作质量收益判断。
