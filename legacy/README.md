# 旧实现归档

本目录保存源码整理前的完整业务代码、阶段脚本和测试，来源提交为 `60e2b4b178210806d32c00bb562eb9e8a4f3197e`。

- `pre_refactor_20261106/vsr/`：原53个业务/历史Python模块。
- `pre_refactor_20261106/scripts/`：原31个Python脚本及其已受Git管理的支持文本。
- `pre_refactor_20261106/tests/`：原54个Python测试。
- [source_manifest.json](pre_refactor_20261106/source_manifest.json)：144个文件的原路径、归档路径、字节数与SHA256。

文件字节未修改。原始配置、checkpoint、实验结果及停止记录保留原处。新主线不导入这里的实现。

本目录供查阅和回归参考，不从此目录直接启动历史GPU任务。若需要按旧命令复现，应检出原提交并使用对应的冻结环境/资产/协议；归档本身不授权重启实验。旧Python包中的相对ROOT和旧阶段脚本路径依赖原仓库布局。

当前GPU实验和自动任务仍暂停。主线入口见[README](../README.md)。
