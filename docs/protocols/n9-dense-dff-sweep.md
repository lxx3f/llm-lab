# N9 Dense d_ff 消融协议

> 状态：N9 阶段交付。本文档描述 medium 架构（d_model=128, n_heads=4, n_layers=4）× d_ff ∈ {256, 512, 1024} 的消融训练协议与产物。

## 目标

观察 d_ff（FFN hidden size）对 val_loss 曲线的影响：

- d_ff=256（1.71M params）
- d_ff=512（2.10M params，control）
- d_ff=1024（2.88M params）

观察内容：
- val_min 是否随 d_ff 增大（FFN 容量增加）；
- 参数总量随 d_ff 增大（每个 transformer block 增加 4 × d_model × d_ff 参数）；
- 5000 步内最佳 val_loss。

**注意**：d_ff 变化同时改变参数总数（不像 N6/N7/N8 是严格的"参数相同"消融）。N9 同时观察 val_min vs 参数总数曲线。

## 共享训练协议（与 N5 medium 一致）

| 维度 | 固定值 |
|---|---|
| 架构 | d_model=128, n_heads=4, n_layers=4 |
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

> 仅 `model.d_ff` 字段变化；其它全部固定。

## 训练命令

```bash
# d_ff=256
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-medium-dff-256.example.yaml \
    --output artifacts/dense-owt-formal-curve-medium-dff-256-result.json

# d_ff=512（control；commit 时 regenerate 对齐 HEAD）
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-medium.example.yaml \
    --output artifacts/dense-owt-formal-curve-medium-result.json

# d_ff=1024
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-medium-dff-1024.example.yaml \
    --output artifacts/dense-owt-formal-curve-medium-dff-1024-result.json
```

## 训练产物（3 个 result JSON）

- `artifacts/dense-owt-formal-curve-medium-dff-256-result.json`
- `artifacts/dense-owt-formal-curve-medium-result.json`（d_ff=512 control）
- `artifacts/dense-owt-formal-curve-medium-dff-1024-result.json`

均按 `schemas/dense_training_result.schema.json` v1.1 校验。

## 3-curve overlay PNG

```bash
.venv/python.exe scripts/plot_dense_curve.py \
    --input artifacts/dense-owt-formal-curve-medium-dff-256-result.json \
    --input artifacts/dense-owt-formal-curve-medium-result.json \
    --input artifacts/dense-owt-formal-curve-medium-dff-1024-result.json \
    --output artifacts/dense-owt-formal-curve-medium-dff-sweep.png \
    --overlay
```

调色板：索引 0 → tab:blue / tab:orange（d_ff=256），索引 1 → tab:green / tab:olive（d_ff=512 control），索引 2 → tab:red / tab:brown（d_ff=1024）。

PNG 尺寸严格 800×500 dpi=100。

## 与 N5/N6/N7/N8 关系

- N5 sweep 变量 = 架构（d_model/n_layers/d_ff 一起扫）
- N6 sweep 变量 = dropout
- N7 sweep 变量 = rope_base
- N8 sweep 变量 = n_heads
- N9 sweep 变量 = d_ff（仅 FFN hidden size，与 N5 的部分重叠）

N9 是 N5 的局部细化：N5 4 个规模点全部用 d_ff=4×d_model；N9 在 d_model=128 上扫描 d_ff ∈ {256, 512, 1024}（即 {2×, 4×, 8× d_model}）。

## 不构成正式结论

N9 d_ff 消融的观察严格限定于：
- 固定架构（d_model=128, n_heads=4, n_layers=4）；
- 单 seed（42）；
- OWT 正式 cache；
- 5000 步训练上限；
- d_ff ∈ {256, 512, 1024}；
- max_seq_len=64；
- Dense Transformer 单架构；
- d_ff 变化伴随参数总数变化（不像 N6/N7/N8 严格参数相同）。

任何超出上述范围的论断均**不构成 N9 协议内的有效结论**。

## 退出条件

- 3 个 result JSON schema v1.1 valid + metadata 对齐 HEAD；
- 1 个 3-curve overlay PNG 生成（共享 2 axes，800×500）；
- 协议/实验/审查三类文档落盘；
- `scripts/run_tests.py full` 仍 100 tests passing；
- 一次 stage review 通过 isolated auditor。