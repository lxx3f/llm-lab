# N5 Dense 规模 sweep 协议

> 状态：N5 阶段交付。本文档描述 4 个规模 sweep 的训练协议与产物；4 规模 sweep 是 N4 baseline + medium 的扩展。

## 目标

把 N4 的 2 点曲线（baseline 0.66M + medium 2.10M）扩展到 4 点，**新增 small (1.23M) 与 large (5.11M)** 两个规模点，得到一条完整的 Dense Transformer 在 OWT 正式 cache 上的**规模 sweep 曲线**。

观察内容：
- 4 个规模下 train_loss 的下降曲线（局部振荡 + 总体趋势下降）；
- 4 个规模下 val_loss 的最小值与最小值出现的 step；
- 4 个规模下 val_loss 收敛速度（5000 步内是否反弹或持续下降）；
- 规模 vs val_loss 的趋势敏感性。

**不是**：模型质量排名、perplexity 报告、跨任务评测。这是规模敏感性 sweep。

## 4 个规模点

| 名称 | d_model | n_layers | d_ff | n_heads | 参数数 | 5k 步实测耗时 |
|---|---|---|---|---|---|---|
| baseline | 64 | 2 | 256 | 4 | 655,680 | ~42s |
| small | 96 | 3 | 384 | 4 | 1,229,472 | ~54s |
| medium | 128 | 4 | 512 | 4 | 2,098,304 | ~65s |
| large | 192 | 6 | 768 | 4 | 5,114,304 | ~91s |

`n_heads` 固定为 4（d_model 必须能被 4 整除且 head_dim 为偶数）。

## 共享训练协议

4 个点使用完全相同的训练设置（仅 `d_model` / `n_layers` / `d_ff` 不同）：

| 维度 | 固定值 |
|---|---|
| 数据 | `data/processed/owt-sample/` 正式 cache（train 512 MiB / validation 64 MiB） |
| Tokenizer | `artifacts/tokenizers/owt-bpe/v0.2.0/` |
| Vocab size | 8192（由 tokenizer artifact 动态绑定） |
| Max seq length | 64 |
| Batch size | 8 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Gradient clip | 1.0 |
| Gradient accumulation | 1 |
| Scheduler | warmup_cosine（warmup=100, min_lr_ratio=0.1）|
| AMP | bf16 |
| Max steps | 5000 |
| Log interval | 50（每 50 步采样一次 train_loss）|
| Validation interval | 200（每 200 步跑一次 validation_loss）|
| Validation batches | 8 |
| Seed | 42 |

> 重要：所有 4 个点共享同一 `metadata.dataset_hash`（= sha256(train.metadata.json bytes) = `e7ece4c755cd54be74eefeb50a4cbbc26af99bba60b0313349dcabc53021d1bd`），保证训练数据可比。

## 训练命令

每个规模点用 `scripts/train_dense.py` 跑：

```bash
# baseline（复用 N4）
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve.example.yaml \
    --output artifacts/dense-owt-formal-curve-result.json

# small（新增）
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-small.example.yaml \
    --output artifacts/dense-owt-formal-curve-small-result.json

# medium（复用 N4）
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-medium.example.yaml \
    --output artifacts/dense-owt-formal-curve-medium-result.json

# large（新增）
.venv/python.exe scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-large.example.yaml \
    --output artifacts/dense-owt-formal-curve-large-result.json
```

## 训练产物（4 个 result JSON）

- `artifacts/dense-owt-formal-curve-result.json`（baseline，N4 已生成）
- `artifacts/dense-owt-formal-curve-small-result.json`（N5 新增）
- `artifacts/dense-owt-formal-curve-medium-result.json`（medium，N4 已生成）
- `artifacts/dense-owt-formal-curve-large-result.json`（N5 新增）

均按 `schemas/dense_training_result.schema.json` v1.1 校验（必须含 `metrics.train_losses` + `metrics.curve_summary`）。

每个 result JSON 必须含：
- `metadata.git_commit` == `git rev-parse HEAD`（commit 时 regenerate 4 个 artifact）；
- `metadata.dataset_hash` == `e7ece4c755cd54be74eefeb50a4cbbc26af99bba60b0313349dcabc53021d1bd`；
- `metadata.gpu_compute_capability` == `"12.0"`（dotted，无 sm_ 前缀）。

## 4-curve overlay PNG

```bash
.venv/python.exe scripts/plot_dense_curve.py \
    --input artifacts/dense-owt-formal-curve-result.json \
    --input artifacts/dense-owt-formal-curve-small-result.json \
    --input artifacts/dense-owt-formal-curve-medium-result.json \
    --input artifacts/dense-owt-formal-curve-large-result.json \
    --output artifacts/dense-owt-formal-curve-scale-sweep.png \
    --overlay
```

输出 PNG：
- 800×500，dpi=120；
- 双轴（ax.twinx()）：左轴 train_loss + 右轴 val_loss，共享同一对 axes；
- 4 条曲线叠加，每个 run 用不同颜色（train_loss 与 val_loss 颜色一致配对）；
- 标题 `Overlay: dense-owt-formal-curve vs dense-owt-formal-curve-small vs dense-owt-formal-curve-medium vs dense-owt-formal-curve-large`；
- 不使用 subplot 分面；所有曲线在同一图上叠加。

## 单曲线 PNG（每规模独立）

每个规模也生成单独的 PNG（`dense-owt-formal-curve[-small|-medium|-large].png`），便于单独观察每个 run 的细节（已在 N4 阶段交付，N5 复用）。

## 与 N4 baseline + medium 的关系

- N4 baseline 0.66M + N4 medium 2.10M：直接复用 N4 artifact + config；N5 不重跑。
- N5 新增 small 1.23M + N5 新增 large 5.11M：跑 5000 步并生成 artifact。

N5 不改变 N4 的 schema、result builder、training 协议、plot 脚本、测试集。

## 与 N2 公平对比协议的关系

N2 公平对比协议 A/B（fixed total params / fixed active params）是 Dense vs MoE 的对比，**不适用于规模 sweep**。N5 仅做 Dense 单架构规模变化；Dense vs MoE 仍是 N2 的范围。

## 与 P1-03 多 seed 协议的关系

N5 仍使用单 seed（42），不构成 P1-03 的多 seed sweep。N5 的 4 个点观察不能外推到不同 seed 下。

## 不构成正式结论

N5 规模 sweep 的观察严格限定于：
- 4 个固定规模点；
- 单 seed（42）；
- OWT 正式 cache（train 512 MiB / validation 64 MiB）；
- 5000 步训练上限；
- Dense Transformer（不涉及 MoE / GQA / MLA）。

任何超出上述范围的论断（外推到更大规模、外推到不同 seed、外推到不同 tokenizer、外推到不同 cache）均**不构成 N5 协议内的有效结论**。

## 退出条件

- 4 个 result JSON schema v1.1 valid + metadata 对齐 HEAD；
- 1 个 4-curve overlay PNG 生成（共享 2 axes）；
- 协议/实验/审查三类文档落盘；
- `scripts/run_tests.py full` 仍 97 tests passing；
- 一次 stage review 通过 isolated auditor。