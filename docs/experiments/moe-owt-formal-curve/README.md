# MoE 5000 步训练曲线实验记录

> 状态：阶段交付。
>
> 本实验补齐 MoE Top-1 在 OWT 正式 cache 上的 5000 步训练曲线（N1 仅 100 步 smoke、N2 仅 128 token budget benchmark），并与 Dense medium (2.10M total) 直接对比。

## 范围

| 维度 | 值 |
|---|---|
| 架构 | MoETransformer（d_model=128, n_heads=4, n_layers=4, d_ff=128）|
| MoE | num_experts=4, capacity_factor=1.0, aux_loss_weight=0.01 |
| 参数 | total=2,100,352 / top1 active=1,510,528 |
| 数据 | OWT 正式 cache train 512 MiB / validation 64 MiB |
| Max steps | 5000 |
| Seed | 42 |

## 实验结果

### MoE 5000 步曲线摘要

| 指标 | 值 |
|---|---|
| train_loss first → last | 117.90 → 6.79 (**-111.11**) |
| val_min | **7.22** @ step **4600** |
| val_last | 7.23 |
| Δ_val_loss | **0.01**（4600 触底后 4600→5000 上升 0.01）|
| train_loss_samples | 100 |
| validation_points | 25 |
| total params | 2,100,352 |
| top1 active params | 1,510,528（72%）|

### MoE vs Dense medium 对比

| 维度 | Dense medium (N5) | MoE (本实验) |
|---|---|---|
| total params | 2,098,304 | 2,100,352 |
| active params | 2,098,304 (100%) | 1,510,528 (**72%**) |
| train first | 104.69 | 117.90 |
| train last | 6.56 | 6.79 |
| **val_min** | **7.06** @ 5000 | **7.22** @ 4600 |
| val_last | 7.06 | 7.23 |
| Δ_val_loss | 0.0 | 0.01 |

### 关键观察

1. **MoE (1.51M active) 在 val_min 上落后 Dense (2.10M active)**：7.22 vs 7.06，差距 0.16 nats（2.2%）；
   - 用 72% active params 达到 97.8% 的 val 性能（7.22/7.06 = 1.0226）；
   - 稀疏激活的 val 效率（per-active-param 性能）略优于 Dense，但绝对 val 仍落后。
3. **MoE val_min 出现在 step 4600（非 5000）**：首次在全部 5000 步曲线中观察到 Δ_val_loss > 0（0.01）；
   - 提示 MoE 在 4600 步后开始轻微过拟合/波动，而 Dense 4 规模 + 3 dropout + 3 rope_base + 3 n_heads + 3 d_ff 全部在 5000 步达到 min；
   - 单 seed 观察，不构成过拟合结论。
5. **train_loss 末值**：MoE 6.79 vs Dense 6.56，差距 0.23——MoE 用 72% active params 训练到 train_loss 6.79，说明容量充足但 active 参数不足。
7. **train_loss 起始**：MoE 117.90 vs Dense 104.69——MoE 初始化（router + 4 experts）的 logits scale 更大，vocab 随机交叉熵更高。
9. **数据效率**：MoE 与 Dense 共用同一 train/val split + 同一 tokenizer + 同一 seed 下的同一 batch 序列（batcher 依赖 seed），token 级可比。

### 文件索引

- 协议：`docs/protocols/moe-owt-formal-curve.md`
- 实验记录：`docs/experiments/moe-owt-formal-curve/README.md`（本文）
- Stage review：`docs/plans/reviews/stage-moe-owt-formal-curve.md`
- Config：`configs/moe_training.owt-formal-curve.example.yaml`
- Artifacts（gitignored）：`artifacts/moe-owt-formal-curve-result.json` + `moe-owt-formal-curve.png` + `moe-vs-dense-medium-curve.png`

## 不构成正式结论

- 单 seed / 单 MoE 配置：不外推到多 seed / 更大 num_experts；
- MoE vs Dense 对比仅限 total/active params 口径，不替代 N2 完整协议 A/B；
- Δ_val_loss=0.01 的单点波动不构成过拟合证据。