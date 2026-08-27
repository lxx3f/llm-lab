# N11 Dense 长训练曲线协议（50000 步）

> 状态：N11 阶段交付（含多 seed 扩展）。本文档描述 baseline 0.66M + medium 2.10M + large 5.11M 各 50000 步的长训练协议与产物。主体实验为单 seed=42；多 seed 扩展（3 seeds = {42, 123, 7}）见 `docs/experiments/n11-long-multi-seed/README.md`，统计口径见 `docs/protocols/p1-03-multi-seed.md`。

## 目标

观察 N4/N5/N6 的 5000 步训练在 **10× 步数（50000 步）** 下的行为：

- 5000 步时 val_min 都在 step 5000（曲线未收尾）——50000 步是否能看到 val 触底/反弹；
- train_loss 是否持续下降或饱和；
- 长训练的 val_min 相比 5000 步改善多少。

**不是**：正式模型质量结论；超参调优。多 seed 统计属 N11 扩展（见 `docs/experiments/n11-long-multi-seed/README.md`），N=3 不构成正式统计显著性。

## 规模点

| 名称 | d_model | n_layers | d_ff | 参数 | 5000 步 val_min | 预计 50000 步耗时 |
|---|---|---|---|---|---|---|
| baseline | 64 | 2 | 256 | 655,680 | 7.23 @ 5000 | ~17 min |
| medium | 128 | 4 | 512 | 2,098,304 | 7.06 @ 5000 | ~25 min |
| large | 192 | 6 | 768 | 5,113,856 | 6.93 @ 5000 | ~70 min |

## 共享训练协议（与 N4/N5/N6 一致）

| 维度 | 固定值 |
|---|---|
| 数据 | OWT 正式 cache train 512 MiB / validation 64 MiB |
| Tokenizer | owt-bpe v0.2.0（vocab 8192） |
| Max seq length | 64 |
| Batch size | 8 |
| Optimizer | AdamW (lr=3e-4, weight_decay=0.01, grad_clip=1.0) |
| Scheduler | warmup_cosine (warmup=100, min_lr_ratio=0.1) |
| AMP | bf16 |
| **Max steps** | **50000**（5000 的 10×）|
| Log interval | 500（100 采样点）|
| Validation interval | 2000（25 val 点）|
| Validation batches | 8 |
| Seed | 42（主体）；3-seed 扩展 {42, 123, 7}（见 P1-03）|

## 训练命令

```bash
# baseline / medium / large 各 50000 步（单 seed=42）
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-long-baseline.example.yaml \
    --output artifacts/dense-owt-formal-curve-long-baseline-result.json

.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-long-medium.example.yaml \
    --output artifacts/dense-owt-formal-curve-long-medium-result.json

.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-long-large.example.yaml \
    --output artifacts/dense-owt-formal-curve-long-large-result.json
```

## 多 seed 扩展（3 seeds = {42, 123, 7}）

```bash
.venv/python.exe -u scripts/run_multi_seed.py --only long
# 输出：artifacts/dense-owt-formal-curve-long-{baseline,medium,large}-seed{42,123,7}-result.json（9 个）
#       artifacts/multi-seed-overview.json
```

## 训练产物（单 seed 3 个 + 多 seed 9 个 result JSON）

单 seed（seed=42）：
- `artifacts/dense-owt-formal-curve-long-baseline-result.json`
- `artifacts/dense-owt-formal-curve-long-medium-result.json`
- `artifacts/dense-owt-formal-curve-long-large-result.json`

多 seed（seeds = {42, 123, 7}）：`artifacts/dense-owt-formal-curve-long-{baseline,medium,large}-seed{42,123,7}-result.json`（9 个）。

均按 `schemas/dense_training_result.schema.json` v1.1 校验（含 train_losses + curve_summary，采样间隔 log_interval=500）。

## 3-curve scale-sweep overlay PNG

```bash
.venv/python.exe scripts/plot_dense_curve.py \
    --input artifacts/dense-owt-formal-curve-long-baseline-result.json \
    --input artifacts/dense-owt-formal-curve-long-medium-result.json \
    --input artifacts/dense-owt-formal-curve-long-large-result.json \
    --output artifacts/dense-owt-formal-curve-long-scale-sweep.png \
    --overlay
```

调色板：索引 0 → tab:blue / tab:orange（baseline），索引 1 → tab:green / tab:olive（medium），索引 2 → tab:red / tab:brown（large）。

PNG 尺寸严格 800×500 dpi=100。

## 与 5000 步曲线对比口径

N11 的 baseline + medium + large 与 N4/N5 的 5000 步曲线共享：
- 同一架构（d_model/n_layers/d_ff）；
- 同一 cache / tokenizer / seed / batch / seq / lr / warmup / AMP；
- 唯一差异是 max_steps（5000 vs 50000）与采样间隔（log_interval 50 vs 500）。

对比维度：
- val_min 绝对值和出现 step；
- val_min 改善（50000 vs 5000）；
- 5000 步时 val 是否已触底。

## 不构成正式结论

N11 长训练观察严格限定于：
- 三规模 baseline + medium + large × 3 seeds {42, 123, 7}（主体文档记录单 seed=42 曲线；3-seed 复核 mean/std 见 `docs/experiments/n11-long-multi-seed/README.md`）；
- OWT 正式 cache；
- 50000 步训练上限；
- 5000 步 → 50000 步对比仅限同 seed 同架构。

任何超出上述范围的论断均**不构成 N11 协议内的有效结论**（N=3 无显著性检验；50000 步未触底；不外推到 3-seed 之外的 seed 数、更大规模或更长序列）。

## 退出条件

- 单 seed 3 个 + 多 seed 9 个 result JSON schema v1.1 valid，`metadata.git_commit` 指向 `tests/test_artifact_provenance.py::KNOWN_NIGHT_RUN_COMMITS` 中的已知 commit（artifact 可指向其训练启动时的 commit，不要求等于审查时 HEAD）；
- 1 个 3-curve scale-sweep overlay PNG 800×500（+ 多 seed overview JSON）；
- 协议/实验/审查三类文档落盘；
- `scripts/run_tests.py full` 通过；
- 一次 stage review 通过 isolated auditor。