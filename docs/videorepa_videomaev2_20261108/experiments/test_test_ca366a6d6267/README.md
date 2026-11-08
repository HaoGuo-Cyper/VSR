# 完整 Wan VideoMAEv2 评价文本

只有 completed448-val 或128-test及保存的原生vmae-verify收据通过后才导出。包含完整observations、每行metrics/complete、每role载入/状态审计、GT文本marker和根/attempt白名单文本。所有来源文件按原字节保存（含CRLF）；未知文件、媒体、权重和tensor不读取也不复制。媒体已由原生verify检查，本工具不重复hash PNG，不产生推理或质量结论。

manifest.json给出原始相对路径、完整SHA和字节数。若file_entries_storage=ordered_manifest_parts，先按manifest_parts顺序读取各JSON数组并连接得到files列表，每部分先核bytes/SHA。storage=copy直接复制；storage=utf8_byte_chunks按parts顺序以二进制拼接，逐part核SHA，再核完整bytes/SHA即可恢复原文件字节。所有文本分块和清单部分最多400000字节，保持有效UTF8。文本中的模型/媒体路径和hash仅为引用。
