# N4 Dense 正式训练曲线实验记录

## 范围

按 [`docs/protocols/n4-dense-curve.md`](../../protocols/n4-dense-curve.md) 跑两个规模的 Dense Transformer 训练曲线：

| 实验 | config | 模型 | 参数量 | 步数 |
|---|---|---|---|---|
| **baseline** | `configs/dense_training.owt-formal-curve.example.yaml` | d_model=64, n_layers=2, d_ff=256 | 655,680 | 5000 |
| **medium** | `configs/dense_training.owt-formal-curve-medium.example.yaml` | d_model=128, n_layers=4, d_ff=512 | 2,098,304 | 5000 |

## 数据

- Tokenizer：`artifacts/tokenizers/owt-bpe/v0.2.0/`（vocab=8192）
- Train cache：`data/processed/owt-sample/train.tokens.uint16`（143,918,122 tokens ≈ 287 MiB）
- Validation cache：`data/processed/owt-sample/validation.tokens.uint16`（17,999,093 tokens ≈ 35 MiB）
- 1 epoch ≈ 1024 steps（512 tokens/batch × 64 seq_len × 1024 ≈ 33.5M tokens）；5000 步 ≈ 4.9 epoch

## 训练超参（两份共享）

| 参数 | 值 |
|---|---|
| optimizer | AdamW (lr=3e-4, weight_decay=0.01) |
| scheduler | warmup_cosine (warmup=100, min_lr_ratio=0.1) |
| batch_size × seq_len | 8 × 64 = 512 tokens/step |
| gradient_clip_norm | 1.0 |
| AMP | bfloat16 |
| log_interval / val_interval | 50 / 200（→ 100 train + 25 val 采样）|
| val_batches | 8 (4096 tokens / 17M = ~0.024%) |
| seed | 42 |
| device | cuda (RTX 5070 Ti Laptop) |

## 关键数字

| 实验 | 参数量 | train first → last (Δ) | val min @ step | val last | 实际耗时 |
|---|---|---|---|---|---|
| **baseline** | 0.66M | 61.57 → 6.81 (**-54.76**) | **7.23** @ 5000 | 7.23 | ~42s |
| **medium** | 2.10M | 104.69 → 6.56 (**-98.13**) | **7.06** @ 5000 | 7.06 | ~65s |

**曲线可视化**：双轴 PNG（train_loss 左轴 + val_loss 右轴），overlay 模式在共享 axes 上叠加两曲线对比。

## 观察记录

### 共同点

- 两条曲线都看到明显的 train_loss 单调下降（Δ_train_loss < 0，符合协议 sanity check）
- val_loss 5000 步时两实验均达到 `val_loss_min_step: 5000` 与 `delta_val_loss: 0.0`（末点就是最小值，因采样到 step 5000 就停步；继续训练才能判断是否反弹）
- warmup 100 步后 loss 进入快速下降区间
- medium 起始 train_loss 比 baseline 高约 1.7×（vocab 随机采样+大模型的 logits scale）
- medium 末段 val_loss（7.06）**低于** baseline（7.23）约 0.17——规模 3.2× 带来 val_loss 改善 2.4%

### Baseline（0.66M）

- train_loss 5000 步时仍在 6.8 附近震荡（lr 已到 min_lr_ratio=0.1 的下限 ~3e-5）
- val_loss 4200 步开始稳定在 7.23 附近；5000 步略降（7.2323 < 7.2491 @ 4800）；`val_loss_min_step: 5000`，与 last 一致

### Medium（2.10M）

- train_loss 末段 5 步：6.56-7.26 区间震荡；lr 已到下限
- val_loss 末段 5 步：7.06-7.10 区间收敛
- Δ_val_loss = 0.0（min = last = step 5000），曲线尾端在 step 5000 达到最小值后停步；更长时间训练才能观察是否反弹（5000 步为协议上限）

### 与 N3 smoke（100 步）对比

| 实验 | 步数 | train last | val min | 模型 |
|---|---|---|---|---|
| N3 smoke | 100 | ~14.6 | 14.5 | 0.66M |
| N4 baseline | 5000 | **6.81** | **7.23** | 0.66M |
| N4 medium | 5000 | **6.56** | **7.06** | 2.10M |

N4 vs N3：相同模型、50× 步数 → train_loss 从 14.6 → 6.81（**减半**），val_loss 从 14.5 → 7.23（同样减半）。曲线协议生效。

## 产物

| 文件 | 大小 | 说明 |
|---|---|---|
| `artifacts/dense-owt-formal-curve-result.json` | schema v1.1 | baseline result，含 train_losses / curve_summary |
| `artifacts/dense-owt-formal-curve.png` | 43 KB | baseline 曲线图 |
| `artifacts/dense-owt-formal-curve-medium-result.json` | schema v1.1 | medium result |
| `artifacts/dense-owt-formal-curve-medium.png` | 40 KB | medium 曲线图 |
| `artifacts/dense-owt-formal-curve-overlay.png` | 74 KB | baseline vs medium 对比 |

PNG 与 JSON artifact 都在 `artifacts/`，gitignored。

## 运行方式

```bash
# Baseline
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve.example.yaml \
    --output artifacts/dense-owt-formal-curve-result.json

# Medium
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-medium.example.yaml \
    --output artifacts/dense-owt-formal-curve-medium-result.json

# 绘图
.venv/python.exe scripts/plot_dense_curve.py \
    --input artifacts/dense-owt-formal-curve-result.json \
    --output artifacts/dense-owt-formal-curve.png

.venv/python.exe scripts/plot_dense_curve.py \
    --input artifacts/dense-owt-formal-curve-medium-result.json \
    --output artifacts/dense-owt-formal-curve-medium.png

.venv/python.exe scripts/plot_dense_curve.py \
    --input artifacts/dense-owt-formal-curve-result.json \
    --input artifacts/dense-owt-formal-curve-medium-result.json \
    --output artifacts/dense-owt-formal-curve-overlay.png \
    --overlay
```

## 限制

- 单 seed（seed=42）——多 seed sweep 在 P1-03 阶段处理
- 无 mean/std/CI 区间——同上
- d_model ≤ 256、n_layers ≤ 8（小规模 sanity）——更大规模属 N5+
- 无 perplexity 计算——独立协议
- 无 checkpoint 加载推理——N4 仅产出曲线与 result JSON，checkpoint 在 artifacts/checkpoints/（gitignored）

## 不在 N4 范围

- 多 seed / 多超参 sweep → P1-03
- 真实模型质量评估 → 后续评测协议
- checkpoint 转 safetensors → 后续 checkpoint 安全化阶段
- 与 MoE 的 fairness 对比 → 已由 N2 完成（独立协议）