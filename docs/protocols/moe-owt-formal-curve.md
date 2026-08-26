# MoE 5000 步训练曲线协议

> 状态：阶段交付。本文档描述 MoE Top-1 在 OWT 正式 cache 上的 5000 步训练曲线协议，补 N1/N2 的训练曲线缺口（N1 仅 100 步 smoke、N2 仅 128 token budget benchmark）。

## 目标

在 N5/N6/N7/N8/N9 Dense 训练曲线体系之外，补齐 **MoE Top-1 的 5000 步训练曲线**，与 Dense medium (2.10M total) 直接可比：

- total params 对齐：Dense medium 2.10M vs MoE 2.10M；
- active params 对比：Dense 2.10M (100%) vs MoE 1.51M (72%)；
- val_loss 曲线对比（MoE 是否在 5000 步内达到 Dense 相近的 val_min）。

**不是**：MoE 质量排名；capacity 消融；routing stats 分析（那是 N2 专门脚本的范围）。

## MoE 配置

| 维度 | 值 |
|---|---|
| 架构 | MoETransformer（d_model=128, n_heads=4, n_layers=4, d_ff=128）|
| MoE | num_experts=4, capacity_factor=1.0, aux_loss_weight=0.01 |
| 参数 | total=2,100,352 / top1 active=1,510,528 |
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

## MoE 结果 schema v1.1

MoE schema 从 v1.0 升级到 **v1.1**，与 Dense v1.1 对齐：

- `metrics.train_losses`（list[{step, loss, lr}]，采样间隔 = log_interval）
- `metrics.curve_summary`（train_loss_first/last、val_loss_min/min_step/last、delta_train_loss、delta_val_loss、train_loss_sample_count、val_loss_count）
- `metrics.validation_metrics`（保留 MoE 特有的 lm/aux/total）
- `curve_summary.val_loss_*` 使用 validation lm_loss 计算

## 训练命令

```bash
.venv/python.exe scripts/train_moe.py \
    --config configs/moe_training.owt-formal-curve.example.yaml \
    --output artifacts/moe-owt-formal-curve-result.json
```

## 训练产物

- `artifacts/moe-owt-formal-curve-result.json`（MoE 5000 步曲线）
- 单 PNG：`artifacts/moe-owt-formal-curve.png`（800×500 dpi=100）
- overlay PNG（与 Dense medium 对比）：`artifacts/moe-vs-dense-medium-curve.png`（800×500 dpi=100）

## 与 Dense 对比口径

MoE vs Dense medium 对比：

| 维度 | Dense medium | MoE |
|---|---|---|
| total params | 2,098,304 | 2,100,352 |
| active params | 2,098,304 (100%) | 1,510,528 (72%) |
| train protocol | 同 cache / tokenizer / seed / batch / seq / lr / warmup / AMP / 5000 步 | 同 |
| 额外 | — | aux_loss_weight=0.01, capacity_factor=1.0 |

> MoE 与 Dense 的对比是 N2 公平对比协议 A/B 之外的**补充观察**，不替代 N2。N2 协议 A（same total params）的 MoE d_ff 对齐是 d_ff=128（4 experts × 32），这里也使用 d_ff=128，与协议 A 一致。

## 不构成正式结论

MoE 5000 步曲线观察严格限定于：
- 单 seed（42）；
- 单配置（num_experts=4, capacity_factor=1.0, aux_loss_weight=0.01）；
- OWT 正式 cache；
- 5000 步训练上限；
- 与 Dense medium 的对比仅限 total/active params 口径，不替代 N2 完整协议。

任何超出上述范围的论断（外推到更大 num_experts / capacity / seed sweep）均**不构成协议内的有效结论**。

## 退出条件

- 1 个 MoE result JSON schema v1.1 valid + metadata 对齐 HEAD；
- 2 个 PNG（单曲线 + Dense 对比 overlay）800×500；
- 协议/实验/审查三类文档落盘；
- `scripts/run_tests.py full` 仍 100 tests passing；
- 一次 stage review 通过 isolated auditor。