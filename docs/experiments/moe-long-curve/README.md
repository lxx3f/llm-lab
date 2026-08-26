# MoE 长训练曲线实验记录（50000 步）

> 状态：阶段交付。
>
> 本实验把 MoE Top-1（total 2.10M / active 1.51M）从 5000 步扩展到 **50000 步**，与 N11 Dense medium 50000 步对齐，观察长训练下 MoE vs Dense 的性能差距。

## 共享训练协议

| 维度 | 固定值 |
|---|---|
| 数据 | OWT 正式 cache（与 N11 相同）|
| Tokenizer | owt-bpe v0.2.0（vocab 8192）|
| 架构 | MoE Top-1：d_model=128, n_heads=4, n_layers=4, d_ff=128, num_experts=4, capacity=1.0, aux=0.01 |
| Max seq len / batch | 64 / 8 |
| Optimizer | AdamW (lr=3e-4, wd=0.01, clip=1.0) |
| Scheduler | warmup_cosine (warmup=100, min_lr=0.1) |
| AMP | bf16 |
| Max steps | 50000 |
| Log / validation interval | 500 / 2000 |
| Seed | 42 |

## 实验结果

### MoE 50000 步曲线摘要

| 模型 | params_total | params_active | val_min @ step | val_last | Δ_val_loss |
|---|---|---|---|---|---|
| **MoE 5000 步** | 2.10M | 1.51M | 7.2200 @ 4600 | 7.23 | +0.01 |
| **MoE 50000 步** | 2.10M | 1.51M | **6.0434** @ 50000 | 6.0434 | 0.0 |
| **Dense medium 50000 步** | 2.10M | 2.10M | **5.5453** @ 50000 | 5.5453 | 0.0 |

### 关键观察

1. **MoE 50000 步相比 5000 步改善 1.18 nats**（7.22 → 6.04），与 Dense medium 的改善幅度（7.06 → 5.55 = 1.51 nats）相比**更小**。
2. **长训练下 Dense 优势扩大**：
   - 5000 步：Dense medium 7.06 vs MoE 7.22（差距 0.16 nats）；
   - 50000 步：Dense medium 5.55 vs MoE 6.04（差距 **0.50 nats**）；
   - MoE 用 72% active params 在 5000 步达到 97.8% 性能，但在 50000 步降到 91.7%——**长训练下稀疏模型的相对劣势更明显**。
3. **MoE 50000 步仍未触底**（val_min_step == 50000），与 Dense 一致。
4. **训练耗时**：MoE 50000 步 ~40 min（比 Dense medium 的 ~22 min 慢约 1.8×，因为 MoE 路由开销）。

### 5000 vs 50000 步对比（MoE）

| 维度 | MoE 5000 | MoE 50000 | 改善 |
|---|---|---|---|
| val_min | 7.22 | **6.04** | 1.18 nats |
| train_last | 7.0（5000 步末） | 5.9（50000 步末） | ~1.1 nats |
| 触底 | step 4600（反弹 0.01）| step 50000（未触底）| — |

> MoE 5000 步的 val_min @ 4600 反弹在 50000 步视角下只是训练中段的普通波动；延长训练后曲线继续下降。

## overlay PNG

- `artifacts/moe-vs-dense-medium-long-curve.png`（800×500）——MoE 50000 步 vs Dense medium 50000 步共享双轴 overlay。

## 不构成正式结论

- 单 seed（42）单配置；
- MoE 配置固定（num_experts=4, capacity=1.0, aux=0.01），不 sweep；
- "Dense 优势扩大"是单 seed 观察，需要多 seed 验证（MoE 多 seed 是后续阶段）；
- 参数对比以 params_total=2.10M 对齐（MoE active 只有 1.51M）。

## 文件索引

- 协议：`docs/protocols/moe-owt-formal-curve.md`（沿用 5000 步协议，长训练参数已扩展）
- 实验记录：`docs/experiments/moe-owt-formal-curve/README.md`（5000 步版）
- 长训练记录：本文件
- Config：`configs/moe_training.owt-formal-curve-long.example.yaml`
- Artifacts（gitignored）：`artifacts/moe-owt-formal-curve-long-result.json` + `moe-vs-dense-medium-long-curve.png`