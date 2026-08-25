# BPE 实现来源说明

当前 `architecture_lab/tokenization/bpe.py` 是基于以下公开仓库的设计和实现思路改写：

```text
https://github.com/lxx3f/assignment1-basics
```

参考模块：

- `cs336_basics/tokenizer.py`；
- `cs336_basics/train_bpe.py`。

当前项目没有将该仓库作为运行时依赖，也没有复制整个仓库。改写内容包括：

- 使用当前项目的 `BPETokenizer` 接口；
- 使用 JSON tokenizer artifact；
- 增加 artifact SHA-256 完整性校验；
- 增加类型和边界条件检查；
- 适配 `architecture_lab` 包结构；
- 增加当前项目的单元测试。

原仓库 `LICENSE` 声明为 MIT License，版权声明为 Stanford University 2025。以下实现仍需保留该来源和许可证信息。
