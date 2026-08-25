# N1 MoE Top-1 训练闭环阶段审查

审查模型：minimax-cn/MiniMax-M3
审查 agent：reviewer
实际 provider/model：PI_PROVIDER=minimax-cn, PI_MODEL=MiniMax-M3
可选运行标识：PI_AGENT_NAME（本次 harness 未提供）

## 阶段目标

完成 MoE Top-1 训练闭环：

- MoE `collect_stats` 开关，默认关闭；
- 训练、validation、auxiliary loss；
- 原子 checkpoint/resume 与模型、MoE、tokenizer/cache hash 绑定；
- Top-1 active parameter 统计；
- 独立 MoE training result schema；
- 1MiB 与正式 512MiB/64MiB cache 的 100 optimizer-step smoke；
- 训练后 generation，显式记录 prefill `capacity_factor=1.0`、decode `capacity_factor=2.0`。

## 实现范围

```text
architecture_lab/models/moe_transformer.py
architecture_lab/training/moe_training.py
architecture_lab/training/moe_results.py
scripts/train_moe.py
schemas/moe_training_result.schema.json
configs/moe_training.example.yaml
configs/moe_training.owt-formal.example.yaml
tests/test_moe_training.py
```

## 验证证据

```text
MoE model tests: 7 passed
MoE training tests: 10 passed
Project full suite: 61 tests passed
Stage 0 schema examples: 5 passed
```

结果文件（本地 Git ignored artifacts）：

```text
artifacts/moe-owt-mvp-result.json
artifacts/moe-owt-formal-cache-result.json
```

两者均通过：

```text
schemas/moe_training_result.schema.json
```

formal cache smoke 指标：

```text
step 50 validation lm_loss: 60.627878
step 100 validation lm_loss: 58.920591
step 100 validation aux_loss: 1.042519
step 100 validation total_loss: 58.931017
```

参数统计：

```text
total parameters: 656,192
top1 active parameters: 582,464
```

## 阶段边界

以下不属于 N1：

- Dense/MoE 公平 benchmark；
- routing stats 分析脚本和 latency 统计；
- aux loss/capacity sweep；
- 多 seed 和统一 benchmark 元数据；
- 分布式训练；
- 严格 prefill/decode 数值等价。

## 审查状态

审查结论：通过（无 Critical；Warnings 为非阻塞一致性修复，已在 commit 前处理）。
