# Dense training MVP 阶段审查

审查模型：minimax-cn/MiniMax-M3
审查 agent：reviewer（subagent dispatch 名称）
实际 provider/model：PI_PROVIDER=minimax-cn, PI_MODEL=MiniMax-M3
可选运行标识：PI_AGENT_NAME（本次 harness 未提供）

## 完成范围

- 新增 `architecture_lab/data/batching.py`，实现连续 token stream 的随机窗口和确定性 epoch 遍历；
- 新增 `load_token_cache()`，训练前复用 cache validator 并支持 memory-mapped uint16 cache；
- 新增 `architecture_lab/training/dense_training.py`；
- tokenizer artifact 动态决定 Dense `vocab_size`；
- 新增显式 next-token causal loss；
- 新增 AdamW、gradient clipping、validation loss；
- 新增原子 checkpoint、resume 及 tokenizer/cache metadata hash 校验；
- 新增 tokenizer generation；
- 新增 `scripts/train_dense.py` 和训练配置；
- 新增 batch/training 协议文档和 Dense MVP 实验记录；
- 增加 batch sampler、causal loss、动态 vocab mismatch、checkpoint resume 和绑定拒绝测试；
- `test_checkpoint_rejects_cache_metadata_change` 与 `test_model_vocab_size_mismatch_is_rejected` 为独立测试；
- validation 使用 `try/finally` 恢复训练模式，即使 validation 失败也不会遗留 eval 状态；
- epoch 计数只在完整 epoch 后递增；
- checkpoint pickle 仅限受信本地产物，并记录后续 `weights_only`/safetensors 计划；

## 验证结果

```text
Project tests: 25 passed
Dense/MoE tests: 13 passed
BPE tests: 8 passed
Stage 0 schema examples: 5 passed
```

训练 smoke：

```text
step 50 validation_loss: 55.168423
step 100 validation_loss: 22.880113
step 100 last_train_loss: 20.823641
checkpoint: artifacts/checkpoints/dense-owt-mvp.pt
```

## 结果边界

- 当前只使用 1 MiB train/validation newline-aligned cache；
- 当前模型为 `vocab_size=8192`、`d_model=64`、2 layers 的 MVP；
- generation 输出存在明显重复，不能作为语言质量结论；
- 尚未生成正式 512 MiB/64 MiB cache；
- 尚未实现 scheduler、AMP、梯度累积或分布式采样；
- 尚未进行 Dense/MoE 公平训练对比。

## 审查状态

审查模型：minimax-cn/MiniMax-M3
审查 agent：reviewer（subagent dispatch 名称）
实际 provider/model：PI_PROVIDER=minimax-cn, PI_MODEL=MiniMax-M3
可选运行标识：PI_AGENT_NAME（本次 harness 未提供）

本阶段已由 `reviewer` 使用 `minimax-cn/MiniMax-M3` 完成只读审查。Critical：无；Warnings：已处理 validation 异常模式恢复和测试拆分问题；审查结论：通过。
