# tokenizer CLI 与 artifact 子阶段审查

## 完成范围

- 新增 `scripts/train_bpe_tokenizer.py`；
- 支持从 UTF-8 文本训练 BPE；
- 支持 `--name`、`--version`、`--vocab-size` 和重复 `--special-token`；
- 固定 artifact 布局：`artifacts/tokenizers/<name>/<version>/`；
- 固定 artifact 文件：`tokenizer.json` 和 `metadata.json`；
- metadata 记录 tokenizer 配置、source SHA-256、文件大小、artifact SHA-256、config SHA-256 和创建时间；
- 非空 artifact 默认拒绝覆盖，使用 `--force` 才能覆盖；
- 新增协议文档 `docs/protocols/tokenizer-artifact.md`。

## 未完成范围

- 尚未接入 DenseTransformer；
- 尚未实现训练数据编码缓存；
- 尚未固定训练数据集、许可证和 split；
- 尚未实现 Dense 训练 loop；
- 尚未实现 checkpoint metadata；
- 尚未完成 P0-01 Dense 训练闭环。

## 测试结果

```text
BPE tokenizer tests: 5 passed
Tokenizer CLI tests: 2 passed
Stage 0 schema tests: 6 passed
Dense + MoE tests: 13 passed
总计: 26 passed
```

另外通过：

- `py_compile`；
- `git diff --check`。

## 实验结果

本子阶段没有产生模型训练或性能结论。CLI 测试验证了：

- artifact 目录布局；
- metadata 字段；
- source 和 artifact hash；
- 非空目录覆盖保护。

## 新发现问题

1. CLI 当前读取完整文本到内存，适用于小型实验；大语料需要后续评估内存和训练速度；
2. metadata 已记录 source hash，但还没有独立的数据集版本 registry；
3. artifact 还没有接入 Dense 训练配置和 checkpoint；
4. 当前 `artifacts/` 目录尚未生成真实项目 tokenizer，避免提交未经确认的数据产物。

## 计划调整

P0-01 后续顺序固定为：

```text
确定 special token 方案
→ 确定第一版训练数据和许可证
→ 训练并提交/登记 tokenizer artifact
→ 编码 train/validation 数据并缓存
→ Dense training loop
→ checkpoint 和 generation
```

## 是否允许进入下一阶段

有条件允许。tokenizer CLI 与 artifact 协议子阶段通过；P0-01 整体仍未完成。

## Git 提交

审查通过后创建一个阶段 commit，未执行 `git push`。
