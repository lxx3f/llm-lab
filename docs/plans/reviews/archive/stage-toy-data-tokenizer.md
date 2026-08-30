# toy 数据与 tokenizer artifact 子阶段审查

## 完成范围

- 固定第一版人工构造 toy corpus；
- 固定数据版本 `D0` 和许可证 `CC0-1.0`；
- 按文档/文本块预先分离 train 和 validation；
- 固定 special token 方案：仅使用 `<|endoftext|>`；
- 生成 `artifacts/tokenizers/toy-bpe/v0.1.0/`；
- 生成并验证 `tokenizer.json` 与 `metadata.json`；
- metadata 记录 source path、license、data version、source hash、artifact hash 和 config hash；
- 增加 toy 数据说明文档。

## 未完成范围

- 尚未实现编码后的 train/validation dataset cache；
- 尚未接入 DenseTransformer；
- 尚未实现 Dense training loop；
- 尚未实现 checkpoint 和 generation；
- toy corpus 尚未作为正式能力评测数据。

## 固定决策

```text
数据：人工构造 toy corpus
数据版本：D0
许可证：CC0-1.0
special token：<|endoftext|>
BOS/EOS/PAD：暂不增加
tokenizer：byte-level BPE
vocab size：512
```

## 验证结果

- tokenizer artifact 可成功加载；
- 文本编码/解码 round-trip 通过；
- special token ID 为 256；
- train source hash 与 metadata 一致；
- BPE、CLI、Dense/MoE 和 schema 测试全部通过；
- 总测试数：26 passed。

## 新发现问题

1. 当前 toy corpus 很小，只能用于训练链路 smoke 验证；
2. tokenizer artifact 目前只由训练集生成，validation 文本不会参与 BPE 训练；
3. 训练数据和 tokenizer artifact 尚未建立统一 registry；
4. 尚未验证 BPE token stream 的 batch 采样和 cache hash 机制。

## 是否允许进入下一阶段

有条件允许。数据和 tokenizer 决策已固定，但 P0-01 整体尚未完成。下一阶段处理编码缓存和 Dense 训练数据接口。

## Git 提交

审查通过后创建阶段 commit，未执行 `git push`。
