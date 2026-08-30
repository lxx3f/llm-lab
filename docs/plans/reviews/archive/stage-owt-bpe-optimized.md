# OWT 增量 BPE merge 与 tokenizer artifact 审查

## 完成范围

- 将 BPE merge 从全表重复扫描改为增量 pair count；
- 增加 pair 到 pre-tokenized word 的倒排索引；
- 使用 lazy heap 选择最高频 pair；
- 保持旧实现的 pair frequency 和 tie-break 语义；
- 与旧实现进行 vocab/merges 等价性验证；
- 64 MiB OWT 前缀、4096 vocab benchmark 通过；
- 生成本地 OWT tokenizer artifact：

```text
artifacts/tokenizers/owt-bpe/v0.2.0/
```

配置：

```text
source: data/raw/owt-sample/owt_train.txt
training prefix: 67104212 bytes（newline-aligned）
vocab size: 8192
merge count: 7935
special token: <|endoftext|> -> 256
data version: OWT-SAMPLE-v1
```

## 验证结果

```text
旧实现与新实现 vocab/merges 等价：PASS
OWT artifact encode/decode round-trip：PASS
BPE tokenizer tests：8 passed
Tokenizer CLI tests：8 passed
Dense + MoE tests：13 passed
Stage 0 schema examples：5 passed
```

64 MiB、4096 vocab benchmark：

```text
约 87.5 秒
```

## 当前边界

- `owt-bpe/v0.2.0` 使用的是 64 MiB 前缀，不是完整 OWT train；
- 词表为 8192，不是最终计划中的 32k；
- artifact 位于被 `.gitignore` 排除的 `artifacts/`，不提交模型/大产物；
- 当前 artifact 只用于推进 OWT token cache 和 Dense 训练接口，不用于正式模型效果结论；
- 尚未实现 token cache；
- 尚未实现 Dense training loop、validation、checkpoint 和 generation。

## 下一阶段

```text
加载 owt-bpe/v0.2.0
→ 编码 OWT train/valid
→ token cache metadata 绑定 source/tokenizer hash
→ Dense training loop
```

## 是否允许进入下一阶段

通过。BPE merge 性能优化和 OWT tokenizer artifact 子阶段完成；允许进入 OWT token cache 实现。

## Git 提交

审查通过后创建阶段 commit，未执行 `git push`。
