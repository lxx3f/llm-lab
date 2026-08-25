# Dense 优化与正式结果 schema 阶段审查

审查模型：minimax-cn/MiniMax-M3
审查 agent：reviewer（subagent dispatch 名称）
实际 provider/model：PI_PROVIDER=minimax-cn, PI_MODEL=MiniMax-M3
可选运行标识：PI_AGENT_NAME（本次 harness 未提供）

## 完成范围

- 新增 warmup + cosine learning-rate scheduler；
- scheduler 按 optimizer update 推进；
- 新增 `gradient_accumulation_steps`，区分 micro-batch、optimizer step 和 effective batch size；
- 新增 PyTorch 2.10 `torch.autocast` AMP；
- CUDA fp16 使用 `torch.amp.GradScaler`，bf16/CPU 不使用 scaler；
- checkpoint 保存/恢复 optimizer、scheduler 和 scaler state；
- 新增 `schemas/dense_training_result.schema.json`；
- 新增正式训练结果 builder 和 CLI schema 校验；
- 结果记录模型、数据 hash、优化器、scheduler、AMP、loss、checkpoint 和 generation；
- 1 MiB smoke 与正式 512 MiB/64 MiB cache 均完成 scheduler 版本的 100-step 运行。

## 验证结果

```text
Project tests: 29 passed
Dense/MoE tests: 13 passed
BPE tests: 8 passed
Stage 0 schema examples: 5 passed
```

正式 cache scheduler smoke：

```text
step 50 validation_loss: 57.813842
step 100 validation_loss: 51.347665
step 100 last_train_loss: 49.561340
step 100 final learning rate: 0.000030
```

结果 JSON 已通过 `dense_training_result.schema.json` 校验，并记录 train/validation metadata SHA-256。

## 结果边界

- AMP 默认配置为关闭；CPU 测试覆盖 bfloat16 autocast 元数据路径，CUDA fp16 scaler 尚未进行独立性能结论；
- 当前 `max_steps` 表示 optimizer updates；
- 当前尚未实现分布式 rank-aware sampler；
- 当前结果仍是小模型、100-step smoke，不代表完整 OWT 训练质量；
- checkpoint 使用受信本地 PyTorch pickle payload。

## 审查状态

本阶段已由 `reviewer` 使用 `minimax-cn/MiniMax-M3` 完成只读审查。Critical：无；Warnings：非阻塞；审查结论：通过。
