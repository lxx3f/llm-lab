# N6 Dense Dropout 消融协议

> 状态：N6 阶段交付。本文档描述 medium 2.10M × dropout ∈ {0.0, 0.1, 0.2} 的消融训练协议与产物。

## 目标

在 N5 已交付的 medium 2.10M 模型基础上，观察 dropout 对 val_loss 曲线的影响：

- dropout=0.0（control，N5 medium 已生成）
- dropout=0.1（新增）
- dropout=0.2（新增）

观察内容：
- val_loss 末段是否反弹（dropout=0.0 的 Δ_val_loss=0.0 是否在更高 dropout 下仍出现）；
- train_loss / val_loss gap 是否随 dropout 增大（过拟合信号）；
- 5000 步内最佳 val_loss 是否随 dropout 改善。

**不是**：dropout 推荐值；模型质量排名；过拟合 vs 欠拟合理论分析。

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

> 仅 `model.dropout` 字段变化；其它全部固定。

3 个 result JSON 共享同一 `metadata.dataset_hash`。

## Dropout 实现

`architecture_lab/models/dense_transformer.py`：

- `TransformerConfig.dropout: float = 0.0`（默认 0.0）
- attention forward：`dropout_p=self.dropout if self.training else 0.0`
- eval 模式（`model.eval()`）下 dropout 自动关闭，不影响 validation_loss

## 训练命令

```bash
# dropout=0.0（复用 N5 medium artifact；commit 时 regenerate 对齐 HEAD）
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-medium.example.yaml \
    --output artifacts/dense-owt-formal-curve-medium-result.json

# dropout=0.1（新增）
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-medium-dropout01.example.yaml \
    --output artifacts/dense-owt-formal-curve-medium-dropout01-result.json

# dropout=0.2（新增）
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-medium-dropout02.example.yaml \
    --output artifacts/dense-owt-formal-curve-medium-dropout02-result.json
```

预计耗时：2 × ~65s ≈ **2.5 min**（dropout=0.0 复用，不重跑）。

## 训练产物（3 个 result JSON）

- `artifacts/dense-owt-formal-curve-medium-result.json`（dropout=0.0，N5 已生成）
- `artifacts/dense-owt-formal-curve-medium-dropout01-result.json`（新增）
- `artifacts/dense-owt-formal-curve-medium-dropout02-result.json`（新增）

均按 `schemas/dense_training_result.schema.json` v1.1 校验。

每个 result JSON 必须含：
- `metadata.git_commit` == `git rev-parse HEAD`（commit 时 regenerate 3 个 artifact）；
- `metadata.dataset_hash` == `e7ece4c755cd54be74eefeb50a4cbbc26af99bba60b0313349dcabc53021d1bd`；
- `metadata.gpu_compute_capability` == `"12.0"`（dotted，无 sm_ 前缀）。

## 3-curve overlay PNG（dropout sensitivity）

```bash
.venv/python.exe scripts/plot_dense_curve.py \
    --input artifacts/dense-owt-formal-curve-medium-result.json \
    --input artifacts/dense-owt-formal-curve-medium-dropout01-result.json \
    --input artifacts/dense-owt-formal-curve-medium-dropout02-result.json \
    --output artifacts/dense-owt-formal-curve-medium-dropout-sweep.png \
    --overlay
```

输出 PNG：
- 800×500，dpi=120；
- 双轴（ax.twinx()）：左轴 train_loss + 右轴 val_loss，共享同一对 axes；
- 3 条曲线叠加，每个 run 用不同颜色（dropout=0.0 蓝 / 0.1 绿 / 0.2 红）；
- 标题 `Overlay: medium dropout=0.0 vs dropout=0.1 vs dropout=0.2`。

## 与 N5 规模 sweep 的关系

- N5 sweep 变量是 `d_model / n_layers / d_ff`（架构变化）；
- N6 消融变量是 `dropout`（正则化，固定架构）；
- N5 / N6 互补，不互证：N6 在同一架构上观察正则化效果，N5 在不同架构上观察规模效果。

## 与 P1-03 多 seed 协议的关系

N6 仍使用单 seed（42），不构成 P1-03 的多 seed sweep。N6 观察不能外推到不同 seed 下。

## 不构成正式结论

N6 dropout 消融的观察严格限定于：
- 固定架构（medium 2.10M）；
- 单 seed（42）；
- OWT 正式 cache（train 512 MiB / validation 64 MiB）；
- 5000 步训练上限；
- dropout ∈ {0.0, 0.1, 0.2}；
- Dense Transformer（不涉及 MoE / GQA / MLA）。

任何超出上述范围的论断（外推到更大 dropout / weight_decay / label_smoothing / stochastic depth / 不同 seed）均**不构成 N6 协议内的有效结论**。

## 退出条件

- 3 个 result JSON schema v1.1 valid + metadata 对齐 HEAD；
- 1 个 3-curve overlay PNG 生成（共享 2 axes）；
- 协议/实验/审查三类文档落盘；
- `scripts/run_tests.py full` 仍 97 tests passing；
- 一次 stage review 通过 isolated auditor。