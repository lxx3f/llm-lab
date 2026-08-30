# N7 Dense RoPE base 消融实验记录

> 状态：N7 阶段交付。
>
> 本实验在 N5/N6 medium 2.10M（rope_base=10000）的基础上，扫描 rope_base ∈ {10000, 50000, 100000}，观察位置编码 θ 对 val_loss 曲线的影响。

## 范围

| 配置 | rope_base | 来源 |
|---|---|---|
| medium rope_base=10000 | 10000.0 | N5 medium（control）|
| medium rope_base=50000 | 50000.0 | N7 新增 |
| medium rope_base=100000 | 100000.0 | N7 新增 |

固定架构：d_model=128, n_layers=4, d_ff=512, n_heads=4, **2,098,304 params**。

## 共享训练协议

| 维度 | 固定值 |
|---|---|
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

## 实验结果

### 3 个 rope_base 值的曲线摘要

| rope_base | train first → last (Δ) | val min @ step | val last | Δ_val_loss |
|---|---|---|---|---|
| **10000** | 104.69 → 6.56 (**-98.13**) | **7.06** @ 5000 | 7.06 | 0.0 |
| **50000** | 105.05 → 6.49 (**-98.56**) | **7.03** @ 5000 | 7.03 | 0.0 |
| **100000** | 104.69 → 6.46 (**-98.23**) | **7.02** @ 5000 | 7.02 | 0.0 |

> 3 个点 `curve_summary.val_loss_min_step == 5000` 且 `delta_val_loss == 0.0`。

### 3-curve overlay PNG

`artifacts/dense-owt-formal-curve-medium-ropebase-sweep.png`（800×500 dpi=100）——3 曲线共享同一对双轴（train_loss 左轴 + val_loss 右轴），调色板：

- rope_base=10000：train tab:blue / val tab:orange
- rope_base=50000：train tab:green / val tab:olive
- rope_base=100000：train tab:red / val tab:brown

### RoPE base sensitivity 观察

1. **单 seed（42）val_min 随 rope_base 单调下降**（7.06 → 7.03 → 7.02）：
   - 10× 增大 rope_base 带来 0.04 nats val_loss 改善（约 0.6%）；
   - ⚠️ **多 seed 复核（P1-03，3 seeds）不稳健**：3-seed mean = 10k/50k/100k = 7.078/7.088/7.078，无单调趋势，差异 < 0.01 nats（噪声级）。单 seed 观察不构成结论。
   - 在 max_seq_len=64 下 RoPE 优势有限（高频位置编码需要更长序列才显著）；
   - 在短序列上 RoPE base 影响是亚主导级（远小于 N5 规模 sweep 0.30 nats）。
3. **train_loss 末值也单调下降**（6.56 → 6.49 → 6.46）：
   - train 与 val 一致，提示 100000 不是过拟合（5000 步）；
   - 但 50000 → 100000 改善幅度（0.03）小于 10000 → 50000（0.07），边际效益递减。
5. **Δ_train_loss 不严格随 rope_base 增长**（-98.13 → -98.56 → -98.23）：
   - 受起始 train_loss 差异影响（105.05 略高于其它）；非架构结论。
7. **局部上升次数**（99 个相邻间隔）：rope_base=10000/50000/100000 = 44/40/43，rope_base 影响有限。

### 共同点（3 个 rope_base）

- train_loss 曲线都伴随震荡（不是单调下降）；
- val_loss 5000 步时 3 个点均达到 `val_loss_min_step: 5000`；
- warmup 100 步后 loss 进入快速下降区间；
- 起始 train_loss 几乎相同（104.69–105.05）。

### 不同点（跨 rope_base 差异）

| 观察 | 10000 | 50000 | 100000 |
|---|---|---|---|
| val_min | 7.06 | 7.03 | **7.02** |
| train_last | 6.56 | 6.49 | **6.46** |
| Δ_train_loss | -98.13 | **-98.56** | -98.23 |
| 局部上升次数 | 44/99 | 40/99 | 43/99 |

> val_min 的差异（7.06–7.02 = 0.04）远小于 N5 规模 sweep 的差异（0.30）与 N6 dropout sweep 的差异（0.10）。

## 与 N5/N6 的关系

| 阶段 | 变量 | val_min 区间 |
|---|---|---|
| N5 规模 sweep | d_model/n_layers/d_ff | 0.30 nats |
| N6 dropout sweep | dropout | 0.10 nats |
| **N7 RoPE base sweep** | rope_base | **0.04 nats** |

> 三个 sweep 在 val_min 上的影响：规模 > dropout > rope_base。N7 提示在 short-context（max_seq_len=64）下 RoPE base 影响亚主导级。

## 不构成正式结论

N7 观察严格限定于：
- 固定架构（medium 2.10M）；
- 单 seed（42）；
- OWT 正式 cache；
- 5000 步训练上限；
- rope_base ∈ {10000, 50000, 100000}；
- max_seq_len=64（短序列）；
- Dense Transformer 单架构。

任何超出上述范围的论断：
- ❌ 不外推到长序列（≥1024 token，RoPE 优势在长序列才显著）；
- ❌ 不外推到多 seed；
- ❌ 不外推到其它位置编码（如 ALiBi）；
- ❌ 不声明"最佳 rope_base"（单 seed 观察；P1-03 3-seed 复核显示无显著影响）。

## 退出条件

- ✅ 3 个 result JSON schema v1.1 valid + metadata 对齐 HEAD；
- ✅ 1 个 3-curve overlay PNG 生成（800×500，共享 2 axes）；
- ✅ 协议/实验/审查三类文档落盘；
- ✅ `scripts/run_tests.py full` 仍 100 tests passing。

完成 N7。

## 文件索引

- 协议：`docs/protocols/n7-dense-rope-sweep.md`
- 实验记录：`docs/experiments/n7-dense-rope-sweep/README.md`（本文）
- Stage review：`docs/plans/reviews/stage-n7-dense-rope-sweep.md`
- Configs：`configs/dense_training.owt-formal-curve-medium-ropebase-{10k,50k,100k}.example.yaml`
- Artifacts（gitignored）：`artifacts/dense-owt-formal-curve-medium-ropebase-{10k,50k,100k}-result.json` + `…-ropebase-sweep.png`