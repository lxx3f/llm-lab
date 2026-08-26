# N11 Dense 长训练曲线实验记录（50000 步）

> 状态：N11 阶段交付。
>
> 本实验把 N4/N5 的 baseline 0.66M + medium 2.10M 从 5000 步扩展到 **50000 步**（10×），观察 val 曲线是否触底/反弹。

## 范围

| 配置 | 参数 | 5000 步 val_min | 50000 步 val_min |
|---|---|---|---|
| baseline | 655,680 | 7.23 @ 5000 | 6.08 @ 50000 |
| medium | 2,098,304 | 7.06 @ 5000 | 5.55 @ 50000 |

## 共享训练协议

| 维度 | 固定值 |
|---|---|
| 数据 | OWT 正式 cache train 512 MiB / validation 64 MiB |
| Tokenizer | owt-bpe v0.2.0（vocab 8192） |
| Max seq length | 64 |
| Batch size | 8 |
| Optimizer | AdamW (lr=3e-4, weight_decay=0.01, grad_clip=1.0) |
| Scheduler | warmup_cosine (warmup=100, min_lr_ratio=0.1) |
| AMP | bf16 |
| Max steps | 50000 |
| Log interval | 500 |
| Validation interval | 2000 |
| Seed | 42 |

## 实验结果

### 50000 步曲线摘要

| 规模 | 参数 | train first → last (Δ) | val min @ step | val last | Δ_val_loss | samples | val_count |
|---|---|---|---|---|---|---|---|
| **baseline** | 0.66M | 61.57 → 6.00 (-55.57) | **6.08** @ 50000 | 6.08 | 0.0 | 100 | 25 |
| **medium** | 2.10M | 104.69 → 5.54 (-99.15) | **5.55** @ 50000 | 5.55 | 0.0 | 100 | 25 |

> 注意：baseline 的 train_first 61.57 是采样起点（step 500 处），因为 log_interval=500 第一次采样在 step 500。

### 2-curve overlay PNG

`artifacts/dense-owt-formal-curve-long-sweep.png`（800×500 dpi=100）——2 曲线共享双轴，调色板：baseline tab:blue/orange、medium tab:green/olive。

### 长训练观察

1. **50000 步 val_min 相比 5000 步大幅改善**：
   - baseline：7.23 → 6.08（**改善 1.15 nats = 15.9%**）；
   - medium：7.06 → 5.55（**改善 1.51 nats = 21.4%**）；
   - medium 改善幅度更大（规模优势在长训练下更明显）。
3. **val 仍未在 50000 步内触底**（两个点 val_min_step == 50000）：
   - 50000 步对 0.66M/2.10M 模型仍不是充分训练；
   - 与 5000 步观察一致（此前所有 val_min_step == 5000），步数上限始终是瓶颈。
5. **train_loss 末值**：
   - baseline 6.00 / medium 5.54——medium 在 50000 步末值仍低于 baseline 0.46 nats；
   - 与 5000 步相比（6.81 vs 6.56），两者都继续下降但差距从 0.25 → 0.46 扩大，规模优势随训练拉长更显著。
7. **token 消耗**：50000 步 × 8 batch × 64 seq = 25.6M tokens/规模，约 OWT train cache 的 17.8%（143.9M tokens）。
9. **训练耗时**（RTX 5070 Ti bf16）：
   - baseline：~17 min（50000 步 × ~8.2ms + 25 次 validation + checkpoint）；
   - medium：~22 min（50000 步 × ~13ms + 25 次 validation + checkpoint）；
   - 总 ~40 min。

### 5000 vs 50000 步对比

| 维度 | baseline 5000 | baseline 50000 | medium 5000 | medium 50000 |
|---|---|---|---|---|
| val_min | 7.23 | **6.08** | 7.06 | **5.55** |
| train_last | 6.81 | **6.00** | 6.56 | **5.54** |
| 改善 | — | **1.15 nats** | — | **1.51 nats** |

> 5000 步曲线严重低估模型能力：延长 10× 训练带来 1.2-1.5 nats val 改善。这与 N4/N5 观察"5000 步 val_min 都在 step 5000（未收尾）"一致。

## 不构成正式结论

N11 观察严格限定于：
- 单 seed（42）；
- baseline + medium 两个规模点；
- OWT 正式 cache；
- 50000 步训练上限；
- 5000 → 50000 对比仅限同 seed 同架构。

任何超出上述范围的论断：
- ❌ 不外推到多 seed；
- ❌ 不声明"最优训练步数"（50000 步仍未触底）；
- ❌ 不外推到更大规模。

## 退出条件

- ✅ 2 个 result JSON schema v1.1 valid + metadata 对齐 HEAD；
- ✅ 1 个 2-curve overlay PNG 800×500；
- ✅ 协议/实验/审查三类文档落盘；
- ✅ `scripts/run_tests.py full` 仍 100 tests passing。

完成 N11。

## 文件索引

- 协议：`docs/protocols/n11-dense-long-curve.md`
- 实验记录：`docs/experiments/n11-dense-long-curve/README.md`（本文）
- Stage review：`docs/plans/reviews/stage-n11-dense-long-curve.md`
- Configs：`configs/dense_training.owt-formal-curve-long-{baseline,medium}.example.yaml`
- Artifacts（gitignored）：`artifacts/dense-owt-formal-curve-long-{baseline,medium}-result.json` + `…-long-sweep.png`