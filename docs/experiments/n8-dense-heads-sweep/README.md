# N8 Dense n_heads 消融实验记录

> 状态：N8 阶段交付。
>
> 本实验在 N5/N6/N7 medium 2.10M（n_heads=4）的基础上，扫描 n_heads ∈ {2, 4, 8}，观察 attention 头数对 val_loss 曲线的影响。

## 范围

| 配置 | n_heads | head_dim | 来源 |
|---|---|---|---|
| medium heads=2 | 2 | 64 | N8 新增 |
| medium heads=4 | 4 | 32 | N5 medium（control）|
| medium heads=8 | 8 | 16 | N8 新增 |

固定架构：d_model=128, n_layers=4, d_ff=512, **2,098,304 params**。

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

### 3 个 n_heads 值的曲线摘要

| n_heads | head_dim | train first → last (Δ) | val min @ step | val last | Δ_val_loss |
|---|---|---|---|---|---|
| **2** | 64 | 106.10 → 6.65 (**-99.45**) | **7.06** @ 5000 | 7.06 | 0.0 |
| **4** | 32 | 104.69 → 6.56 (**-98.13**) | **7.06** @ 5000 | 7.06 | 0.0 |
| **8** | 16 | 106.20 → 6.50 (**-99.70**) | **7.03** @ 5000 | 7.03 | 0.0 |

> 3 个点 `curve_summary.val_loss_min_step == 5000` 且 `delta_val_loss == 0.0`。

### 3-curve overlay PNG

`artifacts/dense-owt-formal-curve-medium-heads-sweep.png`（800×500 dpi=100）——3 曲线共享同一对双轴，调色板：

- n_heads=2：train tab:blue / val tab:orange
- n_heads=4：train tab:green / val tab:olive
- n_heads=8：train tab:red / val tab:brown

### n_heads sensitivity 观察

1. **val_min 随 head 数增加而改善**（n_heads=2 → 4 → 8 = 7.06 → 7.06 → 7.03）：
   - n_heads=8 比 2/4 改善约 0.03 nats（0.4%）；
   - 在 d_model=128 下，单 head 维度降到 16 后仍能维持或改善 val_loss；
   - 提示 16 维 head 在短序列（max_seq_len=64）下仍能表达足够 attention pattern。
3. **train_loss 末值随 head 数增加而单调下降**（6.65 → 6.56 → 6.50）：
   - 与 val_loss 趋势一致，提示 head 越细（head_dim 越小）模型表达力越强；
   - Δ_train_loss 同样单调（-99.45 → -98.13 → -99.70），N8 中 n_heads=8 最佳。
5. **起始 train_loss 略高**（106.10 / 104.69 / 106.20）：
   - 多 head 初始化在 logits scale 上略有差异；
   - 不作为架构结论。
7. **局部上升次数**（99 个相邻间隔）：n_heads=2/4/8 = 42/44/43，n_heads 对振荡频率影响有限。

### 共同点（3 个 n_heads）

- train_loss 曲线都伴随震荡（不是单调下降）；
- val_loss 5000 步时 3 个点均达到 `val_loss_min_step: 5000`；
- warmup 100 步后 loss 进入快速下降区间；
- 参数数严格相同（2,098,304）—— n_heads 变化不影响参数总数。

### 不同点（跨 n_heads 差异）

| 观察 | heads=2 | heads=4 | heads=8 |
|---|---|---|---|
| val_min | 7.06 | 7.06 | **7.03** |
| train_last | 6.65 | 6.56 | **6.50** |
| Δ_train_loss | -99.45 | -98.13 | **-99.70** |
| 局部上升次数 | 42/99 | 44/99 | 43/99 |

> val_min 差异（7.06 → 7.03 = 0.03）小于 N5 规模 sweep（0.30）与 N6 dropout sweep（0.10）；与 N7 rope_base sweep（0.04）相当。

## 与 N5/N6/N7 的关系

| 阶段 | 变量 | val_min 区间 |
|---|---|---|
| N5 规模 sweep | d_model/n_layers/d_ff | 0.30 nats |
| N6 dropout sweep | dropout | 0.10 nats |
| N7 RoPE base sweep | rope_base | 0.04 nats |
| **N8 n_heads sweep** | n_heads | **0.03 nats** |

> N8 影响亚主导级，与 N7 RoPE base sweep 相当。在 d_model=128 + max_seq_len=64 下，n_heads 变化对 val_loss 影响很小。

## 不构成正式结论

N8 观察严格限定于：
- 固定架构（medium 2.10M, d_model=128）；
- 单 seed（42）；
- OWT 正式 cache；
- 5000 步训练上限；
- n_heads ∈ {2, 4, 8}（d_model=128 整除约束下）；
- max_seq_len=64；
- Dense Transformer 单架构。

任何超出上述范围的论断：
- ❌ 不外推到更大 d_model（GQA 类架构下不同）；
- ❌ 不外推到长序列；
- ❌ 不外推到多 seed；
- ❌ 不声明"最佳 n_heads"（仅观察值）。

## 退出条件

- ✅ 3 个 result JSON schema v1.1 valid + metadata 对齐 HEAD；
- ✅ 1 个 3-curve overlay PNG 生成（800×500，共享 2 axes）；
- ✅ 协议/实验/审查三类文档落盘；
- ✅ `scripts/run_tests.py full` 仍 100 tests passing。

完成 N8。

## 文件索引

- 协议：`docs/protocols/n8-dense-heads-sweep.md`
- 实验记录：`docs/experiments/n8-dense-heads-sweep/README.md`（本文）
- Stage review：`docs/plans/reviews/stage-n8-dense-heads-sweep.md`
- Configs：`configs/dense_training.owt-formal-curve-medium-heads-{2,8}.example.yaml`
- Artifacts（gitignored）：`artifacts/dense-owt-formal-curve-medium-heads-{2,8}-result.json` + `…-heads-sweep.png`