# 完整 Wan TRD 评价文本

只有 completed640-val 或192-test及保存的原生trd-verify收据通过后才导出。包含完整observations、每行metrics/complete、每role载入/状态审计、GT文本marker和根/attempt白名单文本。所有来源文件按原字节保存（含CRLF）；未知文件、媒体、权重和tensor不读取也不复制。媒体已由原生verify检查，本工具不重复hash PNG。

manifest.json给出每个原始相对路径、完整SHA和字节数。storage=copy直接复制；storage=utf8_byte_chunks按parts列表顺序以二进制方式拼接，逐part核SHA，再核完整bytes/SHA即可恢复原文件字节。分块保持有效UTF8，每块最多800000字节，适合受限GitHub上传payload。文本中的模型/媒体路径和hash是引用，不代表包含对应二进制。
