# N8 Dense n_heads 消融协议

> 状态：N8 阶段交付。本文档描述 medium 2.10M × n_heads ∈ {2, 4, 8} 的消融训练协议与产物。

## 目标

观察 n_heads（head 数）对 medium 2.10M 模型 5000 步 val_loss 曲线的影响：

- n_heads=2（head_dim=64）
- n_heads=4（head_dim=32，N5 medium 默认）
- n_heads=8（head_dim=16）

观察内容：
- val_min 是否随 n_heads 变化（attention head 数量 vs 单 head 维度）；
- head_dim 越小（head 越多）是否带来 val_loss 改善；
- 5000 步内最佳 val_loss。

**不是**：n_heads 推荐值；multi-head attention 理论分析；长序列评测。

## 共享训练协议（与 N5 medium 完全一致）

| 维度 | 固定值 |
|---|---|
| 架构 | d_model=128, n_layers=4, d_ff=512 |
| 参数数 | 2,098,304 |
| 数据 | OWT 正式 cache train 512 MiB / validation 64 MiB（sha256=`e7ece4c755cd54be74eefeb50a4cbbc26af99bba60b0313349dcabc53021d1bd`）|
| Tokenizer | owt-bpe v0.2.0（vocab 8192） |
| Max seq length | 64 |
| Batch size | 8 |
| Optimizer | AdamW (lr=3e-4, weight_decay=0.01, grad_clip=1.0) |
| Scheduler | warmup_cosine (warmup=100, min_lr_ratio=0.1) |
| AMP | bf16 |
| Max steps | 5000 |
| Log interval | 50 |
| Validation interval | 200 |
| Validation batches | 8 |
| Seed | 42 |

> 仅 `model.n_heads` 字段变化；d_model=128 必须能被 n_heads 整除且 head_dim 必须为偶数（d_model // n_heads % 2 == 0）。

## 训练命令

```bash
# n_heads=2
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-medium-heads-2.example.yaml \
    --output artifacts/dense-owt-formal-curve-medium-heads-2-result.json

# n_heads=4（control；commit 时 regenerate 对齐 HEAD）
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-medium.example.yaml \
    --output artifacts/dense-owt-formal-curve-medium-result.json

# n_heads=8
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-medium-heads-8.example.yaml \
    --output artifacts/dense-owt-formal-curve-medium-heads-8-result.json
```

## 训练产物（3 个 result JSON）

- `artifacts/dense-owt-formal-curve-medium-heads-2-result.json`
- `artifacts/dense-owt-formal-curve-medium-result.json`（n_heads=4 control）
- `artifacts/dense-owt-formal-curve-medium-heads-8-result.json`

均按 `schemas/dense_training_result.schema.json` v1.1 校验。

## 3-curve overlay PNG

```bash
.venv/python.exe scripts/plot_dense_curve.py \
    --input artifacts/dense-owt-formal-curve-medium-heads-2-result.json \
    --input artifacts/dense-owt-formal-curve-medium-result.json \
    --input artifacts/dense-owt-formal-curve-medium-heads-8-result.json \
    --output artifacts/dense-owt-formal-curve-medium-heads-sweep.png \
    --overlay
```

调色板：索引 0 → tab:blue / tab:orange（n_heads=2），索引 1 → tab:green / tab:olive（n_heads=4 control），索引 2 → tab:red / tab:brown（n_heads=8）。

PNG 尺寸严格 800×500 dpi=100。

## 与 N5/N6/N7 关系

- N5 sweep 变量 = 架构（d_model/n_layers/d_ff）
- N6 sweep 变量 = dropout
- N7 sweep 变量 = rope_base
- N8 sweep 变量 = n_heads（attention 头数 / head_dim）
- N8 不改变 N5/N6/N7 schema/dense_training.py/results.py/plot_dense_curve.py/tests。

## 不构成正式结论

N8 n_heads 消融的观察严格限定于：
- 固定架构（medium 2.10M, d_model=128, n_layers=4, d_ff=512）；
- 单 seed（42）；
- OWT 正式 cache；
- 5000 步训练上限；
- n_heads ∈ {2, 4, 8}（d_model=128 整除约束下）；
- max_seq_len=64；
- Dense Transformer 单架构。

任何超出上述范围的论断（外推到更大 d_model / 长序列 / 不同 seed）均**不构成 N8 协议内的有效结论**。

## 退出条件

- 3 个 result JSON schema v1.1 valid + metadata 对齐 HEAD；
- 1 个 3-curve overlay PNG 生成（共享 2 axes，800×500）；
- 协议/实验/审查三类文档落盘；
- `scripts/run_tests.py full` 仍 100 tests passing；
- 一次 stage review 通过 isolated auditor。