# OWT 大语料 BPE 分块读取阶段审查

## 完成范围

- 新增 `train_bpe_iterable`；
- `train_bpe_from_file` 改为 newline-aligned 分块读取；
- special token 以文档边界优先处理，避免跨 chunk merge；
- CLI 新增：
  - `--chunk-size-bytes`；
  - `--max-training-bytes`；
  - `--source-kind`；
- metadata 新增实际训练范围：
  - `training_bytes`；
  - `training_byte_limit`；
  - `newline_aligned_prefix`；
- source metadata 支持 OWT 的 external dataset、upstream terms 和数据版本；
- 16 MiB OWT train 前缀试跑成功；
- toy 数据没有恢复。

## 验证结果

```text
BPE tokenizer tests: 8 passed
Tokenizer CLI tests: 8 passed
Dense + MoE tests: 13 passed
Stage 0 schema examples: 5 passed
```

16 MiB 试跑 metadata：

```text
requested max_training_bytes: 16777216
actual training_bytes: 16777109
newline_aligned_prefix: true
vocab_size: 512
merge_count: 255
```

本地试跑 artifact 位于被忽略的 `artifacts/`，不提交到 Git。

## 未完成范围

- 尚未生成正式的 1 GiB、32k vocab OWT tokenizer artifact；
- BPE pair frequency 更新仍是全表重复扫描，训练大范围 OWT 可能较慢；
- 尚未实现高效增量 pair count、优先队列或持久化统计；
- 尚未实现 OWT token cache；
- 尚未实现 Dense training loop。

## 风险说明

当前“分块读取”解决了完整文本一次性读入内存的问题，但没有完全解决 BPE 训练的时间和 pair frequency 内存复杂度。不能把 16 MiB smoke artifact 当作正式 OWT tokenizer，也不能据此报告正式训练结果。

## 是否允许进入下一阶段

有条件通过。读取边界和 metadata 协议已验证；下一阶段先优化 BPE merge 性能，再生成正式 OWT artifact。

## Git 提交

审查通过后创建阶段 commit，未执行 `git push`。
