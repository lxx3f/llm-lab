# BPE tokenizer 子阶段审查

## 完成范围

- 将参考仓库中的 BPE 设计改写到 `architecture_lab/tokenization/bpe.py`；
- 提供 `BPETokenizer`、`train_bpe` 和 `train_bpe_from_file`；
- 支持 GPT-2 风格 regex pre-tokenization；
- 支持 special tokens；
- 支持 JSON tokenizer artifact；
- 增加 artifact SHA-256 完整性校验；
- 增加来源和 MIT License 说明。

## 未完成范围

- 尚未接入 DenseTransformer；
- 尚未实现 tokenizer 训练 CLI；
- 尚未编码真实训练/验证数据；
- 尚未实现 Dense 训练 loop；
- 尚未实现 checkpoint metadata；
- 尚未完成 P0-01 Dense 训练闭环。

## 测试结果

```text
BPE tokenizer tests: 5 passed
Dense + MoE tests: 13 passed
Stage 0 schema tests: 6 passed
```

另外通过 Python 编译检查和 `git diff --check`。

## 实验结果

本子阶段没有产生模型训练或性能结论。测试仅验证 tokenizer 训练、编码/解码、artifact 保存/加载和篡改检测。

## 新发现问题

- tokenizer 测试目前需要单独运行，尚未纳入统一的架构测试命令；
- BPE 训练当前接收完整字符串，后续需要增加文件输入、数据 hash 和 artifact metadata；
- tokenizer artifact 版本与模型配置、训练数据缓存的绑定尚未完成。

这些问题继续记录在 `docs/plans/open-issues.md` 的 P0-01 中，不阻塞本 BPE 子阶段提交，但阻塞 Dense 训练闭环完成。

## 计划调整

后续不复制参考仓库的完整训练脚本，而是：

```text
当前 BPE module
→ tokenizer 训练/保存脚本
→ tokenizer artifact metadata
→ 文本数据编码缓存
→ Dense training loop
```

## 是否允许进入下一阶段

有条件允许。BPE 子模块可以提交，P0-01 整体仍未完成，下一步继续处理 tokenizer artifact 与 Dense 训练数据接口。

## Git 提交

本子阶段审查通过后创建一个阶段 commit。未执行 `git push`。
