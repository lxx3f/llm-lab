# N9 Dense d_ff 消融实验记录

> 状态：N9 阶段交付。
>
> 本实验在 N5 medium（d_model=128, d_ff=512）基础上，扫描 d_ff ∈ {256, 512, 1024}，观察 FFN hidden size 对 val_loss 曲线的影响。

## 范围

| 配置 | d_ff | 参数 | d_ff / d_model |
|---|---|---|---|
| medium d_ff=256 | 256 | 1,705,088 | 2× |
| medium d_ff=512 | 512 | 2,098,304 | 4× |
| medium d_ff=1024 | 1024 | 2,884,736 | 8× |

固定架构：d_model=128, n_heads=4, n_layers=4。**d_ff 变化同时改变参数总数**。

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
| Seed | 42 |

## 实验结果

### 3 个 d_ff 值的曲线摘要

| d_ff | params | train first → last (Δ) | val min @ step | val last | Δ_val_loss |
|---|---|---|---|---|---|
| **256** | 1.71M | 95.18 → 7.65 (**-87.53**) | **7.56** @ 5000 | 7.56 | 0.0 |
| **512** | 2.10M | 104.69 → 6.56 (**-98.13**) | **7.06** @ 5000 | 7.06 | 0.0 |
| **1024** | 2.88M | 113.00 → 6.27 (**-106.73**) | **6.97** @ 5000 | 6.97 | 0.0 |

> 3 个点 `curve_summary.val_loss_min_step == 5000` 且 `delta_val_loss == 0.0`。

### 3-curve overlay PNG

`artifacts/dense-owt-formal-curve-medium-dff-sweep.png`（800×500 dpi=100）——3 曲线共享同一对双轴，调色板：

- d_ff=256：train tab:blue / val tab:orange
- d_ff=512：train tab:green / val tab:olive
- d_ff=1024：train tab:red / val tab:brown

### d_ff sensitivity 观察

1. **val_min 随 d_ff 增大而单调下降**（7.56 → 7.06 → 6.97）：
   - 2× → 4× d_ff 改善 0.50 nats（6.6%）；
   - 4× → 8× d_ff 改善 0.09 nats（1.3%），边际效益递减；
   - 提示在 d_model=128 下，FFN 容量是 val_loss 的主要瓶颈。
3. **train_loss 末值同样单调下降**（7.65 → 6.56 → 6.27）：
   - 起始 train_loss 随 d_ff 增大而增大（95.18 → 104.69 → 113.00，vocab 随机交叉熵）；
   - 但末值下降更多（-1.09），FFN 容量越大收敛越好。
5. **参数数 vs val_min 改善**：
   - 1.71M → 2.10M（+23% params）→ val_min 改善 0.50 nats；
   - 2.10M → 2.88M（+37% params）→ val_min 改善 0.09 nats；
   - 边际效益快速递减，符合深度学习参数 scaling law 经验。
7. **局部上升次数**（99 个相邻间隔）：d_ff=256/512/1024 = 41/44/42，d_ff 对振荡频率影响有限。

### 共同点（3 个 d_ff）

- train_loss 曲线都伴随震荡；
- val_loss 5000 步时 3 个点均达到 `val_loss_min_step: 5000`；
- warmup 100 步后 loss 进入快速下降区间。

### 不同点（跨 d_ff 差异）

| 观察 | d_ff=256 | d_ff=512 | d_ff=1024 |
|---|---|---|---|
| params | 1.71M | 2.10M | 2.88M |
| val_min | 7.56 | 7.06 | **6.97** |
| train_last | 7.65 | 6.56 | **6.27** |
| Δ_train_loss | -87.53 | -98.13 | **-106.73** |

> val_min 差异（7.56 → 6.97 = 0.59）远大于 N5 规模 sweep（0.30）、N6 dropout sweep（0.10）、N7 rope_base sweep（0.04）、N8 n_heads sweep（0.03）。N9 是 val_min 最大变化的 sweep。

## 与 N5/N6/N7/N8 的关系

| 阶段 | 变量 | val_min 区间 | 是否参数变化 |
|---|---|---|---|
| N5 规模 sweep | d_model/n_layers/d_ff | 0.30 nats | 是 |
| N6 dropout sweep | dropout | 0.10 nats | 否 |
| N7 RoPE base sweep | rope_base | 0.04 nats | 否 |
| N8 n_heads sweep | n_heads | 0.03 nats | 否 |
| **N9 d_ff sweep** | d_ff | **0.59 nats** | **是** |

> N9 是 N5 的局部细化，但 d_ff 独立 sweep 比 N5 整体规模 sweep 影响更大（0.59 vs 0.30）。  
> 在 d_model=128 下，FFN 容量比 attention 头数、RoPE base、dropout 对 val_min 影响大一个数量级。

## 不构成正式结论

N9 观察严格限定于：
- 固定架构（d_model=128, n_heads=4, n_layers=4）；
- 单 seed（42）；
- OWT 正式 cache；
- 5000 步训练上限；
- d_ff ∈ {256, 512, 1024}；
- max_seq_len=64；
- Dense Transformer 单架构；
- d_ff 变化同时改变参数总数。

任何超出上述范围的论断：
- ❌ 不外推到更大 d_model；
- ❌ 不外推到多 seed；
- ❌ 不声明"最佳 d_ff / d_model 比例"（仅观察值）。

## 退出条件

- ✅ 3 个 result JSON schema v1.1 valid + metadata 对齐 HEAD；
- ✅ 1 个 3-curve overlay PNG 生成（800×500，共享 2 axes）；
- ✅ 协议/实验/审查三类文档落盘；
- ✅ `scripts/run_tests.py full` 仍 100 tests passing。

完成 N9。

## 文件索引

- 协议：`docs/protocols/n9-dense-dff-sweep.md`
- 实验记录：`docs/experiments/n9-dense-dff-sweep/README.md`（本文）
- Stage review：`docs/plans/reviews/stage-n9-dense-dff-sweep.md`
- Configs：`configs/dense_training.owt-formal-curve-medium-dff-{256,1024}.example.yaml`
- Artifacts（gitignored）：`artifacts/dense-owt-formal-curve-medium-dff-{256,1024}-result.json` + `…-dff-sweep.png`