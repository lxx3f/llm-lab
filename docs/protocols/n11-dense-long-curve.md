# N11 Dense 长训练曲线协议（50000 步）

> 状态：N11 阶段交付。本文档描述 baseline 0.66M + medium 2.10M 各 50000 步的长训练协议与产物。

## 目标

观察 N4/N5/N6 的 5000 步训练在 **10× 步数（50000 步）** 下的行为：

- 5000 步时 val_min 都在 step 5000（曲线未收尾）——50000 步是否能看到 val 触底/反弹；
- train_loss 是否持续下降或饱和；
- 长训练的 val_min 相比 5000 步改善多少。

**不是**：正式模型质量结论；多 seed 统计；超参调优。

## 规模点

| 名称 | d_model | n_layers | d_ff | 参数 | 5000 步 val_min | 预计 50000 步耗时 |
|---|---|---|---|---|---|---|
| baseline | 64 | 2 | 256 | 655,680 | 7.23 @ 5000 | ~17 min |
| medium | 128 | 4 | 512 | 2,098,304 | 7.06 @ 5000 | ~25 min |

## 共享训练协议（与 N4/N5/N6 一致）

| 维度 | 固定值 |
|---|---|
| 数据 | OWT 正式 cache train 512 MiB / validation 64 MiB |
| Tokenizer | owt-bpe v0.2.0（vocab 8192） |
| Max seq length | 64 |
| Batch size | 8 |
| Optimizer | AdamW (lr=3e-4, weight_decay=0.01, grad_clip=1.0) |
| Scheduler | warmup_cosine (warmup=100, min_lr_ratio=0.1) |
| AMP | bf16 |
| **Max steps** | **50000**（5000 的 10×）|
| Log interval | 500（100 采样点）|
| Validation interval | 2000（25 val 点）|
| Validation batches | 8 |
| Seed | 42 |

## 训练命令

```bash
# baseline 50000 步
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-long-baseline.example.yaml \
    --output artifacts/dense-owt-formal-curve-long-baseline-result.json

# medium 50000 步
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-long-medium.example.yaml \
    --output artifacts/dense-owt-formal-curve-long-medium-result.json
```

## 训练产物（2 个 result JSON）

- `artifacts/dense-owt-formal-curve-long-baseline-result.json`
- `artifacts/dense-owt-formal-curve-long-medium-result.json`

均按 `schemas/dense_training_result.schema.json` v1.1 校验（含 train_losses + curve_summary，采样间隔 log_interval=500）。

## 2-curve overlay PNG

```bash
.venv/python.exe scripts/plot_dense_curve.py \
    --input artifacts/dense-owt-formal-curve-long-baseline-result.json \
    --input artifacts/dense-owt-formal-curve-long-medium-result.json \
    --output artifacts/dense-owt-formal-curve-long-sweep.png \
    --overlay
```

调色板：索引 0 → tab:blue / tab:orange（baseline），索引 1 → tab:green / tab:olive（medium）。

PNG 尺寸严格 800×500 dpi=100。

## 与 5000 步曲线对比口径

N11 的 baseline + medium 与 N4/N5 的 5000 步曲线共享：
- 同一架构（d_model/n_layers/d_ff）；
- 同一 cache / tokenizer / seed / batch / seq / lr / warmup / AMP；
- 唯一差异是 max_steps（5000 vs 50000）与采样间隔（log_interval 50 vs 500）。

对比维度：
- val_min 绝对值和出现 step；
- val_min 改善（50000 vs 5000）；
- 5000 步时 val 是否已触底。

## 不构成正式结论

N11 长训练观察严格限定于：
- 单 seed（42）；
- baseline + medium 两个规模点；
- OWT 正式 cache；
- 50000 步训练上限；
- 5000 步 → 50000 步对比仅限同 seed 同架构。

任何超出上述范围的论断均**不构成 N11 协议内的有效结论**。

## 退出条件

- 2 个 result JSON schema v1.1 valid + metadata 对齐 HEAD；
- 1 个 2-curve overlay PNG 800×500；
- 协议/实验/审查三类文档落盘；
- `scripts/run_tests.py full` 仍 100 tests passing；
- 一次 stage review 通过 isolated auditor。