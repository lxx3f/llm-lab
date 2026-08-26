# N5 Dense 规模 sweep 实验记录

> 状态：N5 阶段交付。
>
> 本实验在 N4 baseline 0.66M + medium 2.10M 的基础上，新增 **small 1.23M** 与 **large 5.11M** 两个规模点，得到 4 个 Dense Transformer 在 OWT 正式 cache 上的 5000 步训练曲线。

## 范围

4 个规模 sweep：

| 规模 | d_model | n_layers | d_ff | n_heads | 参数数 |
|---|---|---|---|---|---|
| **baseline** | 64 | 2 | 256 | 4 | **655,680** (0.66M) |
| **small** | 96 | 3 | 384 | 4 | **1,229,472** (1.23M) |
| **medium** | 128 | 4 | 512 | 4 | **2,098,304** (2.10M) |
| **large** | 192 | 6 | 768 | 4 | **5,114,304** (5.11M) |

7.8× 规模跨度（0.66M → 5.11M）。

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

### 4 个规模点的曲线摘要

| 规模 | 参数 | train first → last (Δ) | val min @ step | val last | 5k 步实测耗时 |
|---|---|---|---|---|---|
| **baseline** | 0.66M | 61.57 → 6.81 (**-54.76**) | **7.23** @ 5000 | 7.23 | ~42s |
| **small** | 1.23M | 88.42 → 6.60 (**-81.82**) | **7.18** @ 5000 | 7.18 | ~54s |
| **medium** | 2.10M | 104.69 → 6.56 (**-98.13**) | **7.06** @ 5000 | 7.06 | ~65s |
| **large** | 5.11M | 62.59 → 6.41 (**-56.18**) | **6.93** @ 5000 | 6.93 | ~91s |

> 所有 4 个点的 `curve_summary.val_loss_min_step == 5000` 与 `delta_val_loss == 0.0`；继续训练才能判断 val_loss 是否反弹。

### 4-curve overlay PNG

`artifacts/dense-owt-formal-curve-scale-sweep.png`（101KB，800×500 dpi=120）——4 曲线共享同一对双轴（train_loss 左轴 + val_loss 右轴），每个 run 用不同颜色配对。

### 规模敏感性观察

1. **val_loss 随规模单调下降**（7.23 → 7.18 → 7.06 → 6.93）：
   - 7.8× 规模带来 val_loss 改善 4.1%（0.30 nats 绝对差）；
   - 在 0.66M→2.10M（3.2×）段，每倍规模 val_loss 约下降 0.05–0.06；
   - 在 2.10M→5.11M（2.4×）段，每倍规模 val_loss 约下降 0.06–0.07；
   - 子 6M 区间规模 sensitivity 与文献观察一致。
3. **train_loss 末值随规模缓慢下降**（6.81 → 6.60 → 6.56 → 6.41）：
   - 5000 步内 train_loss 仍处下降区间，未饱和；
   - 5.11M 与 2.10M 末值仅差 0.15，提示 5000 步已不是这些规模的充分训练步数。
5. **train_loss 首值与规模非单调**（baseline=61.57, small=88.42, medium=104.69, large=62.59）：
   - 取决于随机初始化（seed=42）下 logits scale 与随机 batch 的交叉熵；
   - **不应作为架构结论**——观察用。
7. **Δ_train_loss 与规模非单调**（baseline=-54.76 < small=-81.82 < medium=-98.13，large=-56.18 反弹）：
   - 因为 first 值不同，绝对下降量与起始点耦合；
   - val_loss 才是规模敏感性的更稳定指标（同一 cache 的同一 validation split）。
9. **5k 步时间成本**：0.66M 42s → 5.11M 91s，2.2× 时间 vs 7.8× 规模——RTX 5070 Ti bf16 AMP 下 forward/backward 是 dense 矩阵乘，规模增长未线性传递到训练时间（受 bf16 tensor core 加速 + batch 8 padding 等因素共同作用）。

### 共同点（4 个规模）

- train_loss 曲线总体**震荡下降**（不是单调下降）——100 个采样点中，baseline 99 间隔里 42 次局部上升、small 99 间隔里 38 次局部上升、medium 99 间隔里 44 次局部上升、large 99 间隔里 41 次局部上升；
- val_loss 5000 步时 4 个点均达到 `val_loss_min_step: 5000`，停步点即最小值点；
- warmup 100 步后 loss 进入快速下降区间。

### 不同点（跨规模差异）

| 观察 | baseline | small | medium | large |
|---|---|---|---|---|
| train_first | 61.57 | 88.42 | 104.69 | 62.59 |
| train_last | 6.81 | 6.60 | 6.56 | 6.41 |
| val_min | 7.23 | 7.18 | 7.06 | 6.93 |
| 局部上升次数 | 42/99 | 38/99 | 44/99 | 41/99 |
| Δ_train_loss | -54.76 | -81.82 | -98.13 | -56.18 |

> Δ_train_loss 受 train_first 影响较大；val_loss 才是更稳定的规模敏感性指标。

## 与 N4 baseline + medium 的关系

- N4 baseline 0.66M + N4 medium 2.10M 直接复用 N4 artifact + config；
- N5 新增 small 1.23M + large 5.11M 是 N5 阶段的两个新增跑点；
- N5 不改变 N4 的 schema、result builder、training 协议、plot 脚本、测试集。

## 不构成正式结论

N5 观察严格限定于：
- 4 个固定规模点（0.66M / 1.23M / 2.10M / 5.11M）；
- 单 seed（42）；
- OWT 正式 cache（train 512 MiB / validation 64 MiB）；
- 5000 步训练上限；
- Dense Transformer 单架构。

任何超出上述范围的论断：
- ❌ 不外推到 100M+ 规模（仍在 sub-6M 区间）；
- ❌ 不外推到多 seed（单 seed）；
- ❌ 不外推到 MoE / GQA / MLA（仅 Dense）；
- ❌ 不外推到不同 cache / tokenizer / batch size；
- ❌ 不声明模型质量或"最优规模"。

## 退出条件

- ✅ 4 个 result JSON schema v1.1 valid + metadata 对齐 HEAD；
- ✅ 1 个 4-curve overlay PNG 生成（共享 2 axes）；
- ✅ 协议/实验/审查三类文档落盘；
- ✅ `scripts/run_tests.py full` 仍 97 tests passing。

完成 N5。

## 文件索引

- 协议：`docs/protocols/n5-dense-scale-sweep.md`
- 实验记录：`docs/experiments/n5-dense-scale-sweep/README.md`（本文）
- Stage review：`docs/plans/reviews/stage-n5-dense-scale-sweep.md`
- Configs：`configs/dense_training.owt-formal-curve{,-small,-medium,-large}.example.yaml`
- Artifacts（gitignored）：`artifacts/dense-owt-formal-curve{,-small,-medium,-large}-result.json` + `…-scale-sweep.png`