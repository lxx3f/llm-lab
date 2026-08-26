# N6 Dense Dropout 消融实验记录

> 状态：N6 阶段交付。
>
> 本实验在 N5 medium 2.10M（dropout=0.0）的基础上，新增 **dropout=0.1** 与 **dropout=0.2** 两个对照，得到 medium 2.10M 在 OWT 正式 cache 上的 dropout sensitivity 曲线。

## 范围

| 配置 | dropout | 来源 |
|---|---|---|
| medium dropout=0.0 | 0.0 | N5 medium（control）|
| medium dropout=0.1 | 0.1 | N6 新增 |
| medium dropout=0.2 | 0.2 | N6 新增 |

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

> 仅 `model.dropout` 字段变化；其它全部固定。

## 实验结果

### 3 个 dropout 值的曲线摘要

| dropout | train first → last (Δ) | val min @ step | val last | Δ_val_loss |
|---|---|---|---|---|
| **0.0** | 104.69 → 6.56 (**-98.13**) | **7.06** @ 5000 | 7.06 | 0.0 |
| **0.1** | 104.83 → 6.51 (**-98.32**) | **6.99** @ 5000 | 6.99 | 0.0 |
| **0.2** | 104.79 → 6.59 (**-98.20**) | **7.09** @ 5000 | 7.09 | 0.0 |

> 3 个点 `curve_summary.val_loss_min_step == 5000` 且 `delta_val_loss == 0.0`；停步点即最小值点。

### 3-curve overlay PNG

`artifacts/dense-owt-formal-curve-medium-dropout-sweep.png`（92KB，800×500 dpi=120）——3 曲线共享同一对双轴（train_loss 左轴 + val_loss 右轴），每个 run 用不同颜色（dropout=0.0 蓝 / 0.1 绿 / 0.2 红）。

### Dropout sensitivity 观察

1. **val_min 在 dropout=0.1 时最佳**（6.99 vs dropout=0.0 的 7.06 vs dropout=0.2 的 7.09）：
   - 单次 5000 步训练下，dropout=0.1 改善 1.0%（相对 0.0）；
   - dropout=0.2 反而劣于 0.0（7.09 vs 7.06，约 0.4% 退化）；
   - 提示 medium 2.10M 在 5000 步 + OWT cache 上的"最佳" dropout 在 0.0–0.1 区间，而非更高。
3. **train_loss 末值接近**（6.56 / 6.51 / 6.59，区间 0.08）：
   - dropout 在 5000 步内对 train_loss 末值影响不显著；
   - 与 N5 不同规模的 train_last 区间（6.41–6.81）相比，dropout 对 train_last 的影响比规模影响小一个数量级。
5. **Δ_train_loss 几乎相同**（-98.13 / -98.32 / -98.20，区间 0.19）：
   - 三者起始 train_loss 几乎相同（104.69 / 104.83 / 104.79，标准差 0.07）；
   - 5000 步内的"信息量"基本一致。
7. **train_loss 末段均震荡**（与 N5 一致）：99 个相邻间隔中局部上升次数 dropout=0.0/0.1/0.2 = 44/42/43，dropout 几乎不影响振荡频率。
9. **val_last == val_min**（三者 delta_val_loss=0.0）：
   - 5000 步仍不是充分训练步数；继续训练才能判断 val 是否反弹。
   - 与 N5 4 个规模观察一致（`val_loss_min_step == 5000` 在所有 4 个规模点都成立）。

### 共同点（3 个 dropout）

- train_loss 末段都伴随震荡（不是单调下降）；
- val_loss 5000 步时 3 个点均达到 `val_loss_min_step: 5000`，停步点即最小值点；
- warmup 100 步后 loss 进入快速下降区间；
- 起始 train_loss 几乎相同（vocab 8192 随机 batch 的交叉熵差异由随机初始化主导）。

### 不同点（跨 dropout 差异）

| 观察 | 0.0 | 0.1 | 0.2 |
|---|---|---|---|
| val_min | 7.06 | **6.99** | 7.09 |
| train_last | 6.56 | 6.51 | 6.59 |
| Δ_train_loss | -98.13 | -98.32 | -98.20 |
| 局部上升次数 | 44/99 | 42/99 | 43/99 |

> val_min 的差异（6.99–7.09 = 0.10）远小于 N5 规模 sweep 的差异（7.23–6.93 = 0.30）。

## 与 N5 规模 sweep 的关系

- N5 改变架构（d_model/n_layers/d_ff），观察 val_min 从 7.23 → 6.93（7.8× 规模 → 0.30 nats 改善）；
- N6 改变 dropout（架构固定），观察 val_min 从 7.06 → 7.09 → 6.99（dropout sweep → 0.10 nats 区间）；
- 规模对 val_min 的影响是 dropout 的 3×；
- 两者互补，不互证。

## 不构成正式结论

N6 观察严格限定于：
- 固定架构（medium 2.10M）；
- 单 seed（42）；
- OWT 正式 cache（train 512 MiB / validation 64 MiB）；
- 5000 步训练上限；
- dropout ∈ {0.0, 0.1, 0.2}；
- Dense Transformer 单架构。

任何超出上述范围的论断：
- ❌ 不外推到其它架构规模（仅 medium）；
- ❌ 不外推到多 seed（单 seed）；
- ❌ 不外推到其它正则化（weight_decay / label_smoothing 不在 N6）；
- ❌ 不外推到 MoE / GQA / MLA；
- ❌ 不声明"最佳 dropout"（仅观察值；正式推荐需 P1-03 多 seed sweep 后再做）。

## 退出条件

- ✅ 3 个 result JSON schema v1.1 valid + metadata 对齐 HEAD；
- ✅ 1 个 3-curve overlay PNG 生成（共享 2 axes）；
- ✅ 协议/实验/审查三类文档落盘；
- ✅ `scripts/run_tests.py full` 仍 97 tests passing。

完成 N6。

## 文件索引

- 协议：`docs/protocols/n6-dense-dropout-sweep.md`
- 实验记录：`docs/experiments/n6-dense-dropout-sweep/README.md`（本文）
- Stage review：`docs/plans/reviews/stage-n6-dense-dropout-sweep.md`
- Configs：`configs/dense_training.owt-formal-curve-medium{,-dropout01,-dropout02}.example.yaml`
- Artifacts（gitignored）：`artifacts/dense-owt-formal-curve-medium{,-dropout01,-dropout02}-result.json` + `…-medium-dropout-sweep.png`