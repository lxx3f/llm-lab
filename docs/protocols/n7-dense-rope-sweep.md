# N7 Dense RoPE base 消融协议

> 状态：N7 阶段交付。本文档描述 medium 2.10M × rope_base ∈ {10000, 50000, 100000} 的消融训练协议与产物。

## 目标

观察 RoPE base（θ）对 medium 2.10M 模型 5000 步 val_loss 曲线的影响：

- rope_base=10000（N5 medium 默认）
- rope_base=50000
- rope_base=100000

观察内容：
- val_min 是否随 rope_base 变化（RoPE base 越大，高频位置编码越精确）；
- 5000 步内最佳 val_loss。

**不是**：RoPE base 推荐值；位置编码理论分析；长上下文评测。

## 共享训练协议（与 N5 medium 完全一致）

| 维度 | 固定值 |
|---|---|
| 架构 | d_model=128, n_heads=4, n_layers=4, d_ff=512 |
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

> 仅 `model.rope_base` 字段变化；其它全部固定。

## 训练命令

```bash
# rope_base=10000（control，N5 medium 等同；commit 时 regenerate 对齐 HEAD）
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-medium.example.yaml \
    --output artifacts/dense-owt-formal-curve-medium-result.json

# rope_base=50000
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-medium-ropebase-50k.example.yaml \
    --output artifacts/dense-owt-formal-curve-medium-ropebase-50k-result.json

# rope_base=100000
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-medium-ropebase-100k.example.yaml \
    --output artifacts/dense-owt-formal-curve-medium-ropebase-100k-result.json
```

## 训练产物（3 个 result JSON）

- `artifacts/dense-owt-formal-curve-medium-result.json`（rope_base=10000 control）
- `artifacts/dense-owt-formal-curve-medium-ropebase-50k-result.json`
- `artifacts/dense-owt-formal-curve-medium-ropebase-100k-result.json`

均按 `schemas/dense_training_result.schema.json` v1.1 校验。

## 3-curve overlay PNG

```bash
.venv/python.exe scripts/plot_dense_curve.py \
    --input artifacts/dense-owt-formal-curve-medium-result.json \
    --input artifacts/dense-owt-formal-curve-medium-ropebase-50k-result.json \
    --input artifacts/dense-owt-formal-curve-medium-ropebase-100k-result.json \
    --output artifacts/dense-owt-formal-curve-medium-ropebase-sweep.png \
    --overlay
```

调色板：索引 0 → tab:blue / tab:orange（rope=10k），索引 1 → tab:green / tab:olive（rope=50k），索引 2 → tab:red / tab:brown（rope=100k）。

PNG 尺寸严格 800×500 dpi=100。

## 与 N5/N6 关系

- N5 sweep 变量 = 架构（d_model/n_layers/d_ff）
- N6 sweep 变量 = dropout
- N7 sweep 变量 = rope_base（位置编码）
- N7 不改变 N5/N6 schema/dense_training.py/results.py/plot_dense_curve.py/tests。

## 不构成正式结论

N7 RoPE base 消融的观察严格限定于：
- 固定架构（medium 2.10M）；
- 单 seed（42）；
- OWT 正式 cache（train 512 MiB / validation 64 MiB）；
- 5000 步训练上限；
- rope_base ∈ {10000, 50000, 100000}；
- max_seq_len=64（短序列，RoPE 优势在更长序列才显著）；
- Dense Transformer 单架构。

任何超出上述范围的论断（外推到更长序列 / 不同架构 / 不同 seed）均**不构成 N7 协议内的有效结论**。

## 退出条件

- 3 个 result JSON schema v1.1 valid + metadata 对齐 HEAD；
- 1 个 3-curve overlay PNG 生成（共享 2 axes，800×500）；
- 协议/实验/审查三类文档落盘；
- `scripts/run_tests.py full` 仍 100 tests passing；
- 一次 stage review 通过 isolated auditor。