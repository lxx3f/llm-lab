# N8 Stage Review: Dense n_heads 消融

> 状态：N8 阶段交付。
>
> 审查 agent：reviewer（minimax-cn/MiniMax-M3）
> 审查时间：2026-08-26
> 阶段目标：在 N5 medium 2.10M（n_heads=4）基础上，扫描 n_heads ∈ {2, 4, 8}，得到 attention head 数 sensitivity 曲线。输出 3 个 result JSON + 1 个 3-curve overlay PNG + 协议/实验/审查三类文档；roadmap 推进 N8。

## 阶段范围

- ✅ 3 个 n_heads 值的 medium 训练：n_heads=2 + 4（N5 control）+ 8，各 5000 步；
- ✅ 3 个 result JSON 全部 schema v1.1 valid + metadata 对齐 HEAD；
- ✅ 1 个 3-curve overlay PNG（共享 2 axes，800×500 dpi=100）；
- ✅ 协议/实验/审查三类文档落盘；
- ✅ `scripts/run_tests.py full` 仍 100 tests passing；
- ✅ roadmap 推进（N8 → 已完成；当前阶段 = N9）。

## 3 个 n_heads 值的曲线摘要

| n_heads | head_dim | train first → last (Δ) | val min @ step | val last |
|---|---|---|---|---|
| **2** | 64 | 106.10 → 6.65 (-99.45) | **7.06** @ 5000 | 7.06 |
| **4** | 32 | 104.69 → 6.56 (-98.13) | **7.06** @ 5000 | 7.06 |
| **8** | 16 | 106.20 → 6.50 (**-99.70**) | **7.03** @ 5000 | 7.03 |

n_heads sensitivity：
- val_min 随 head 数增加而改善（7.06 → 7.06 → 7.03），n_heads=8 改善 0.03 nats（0.4%）；
- train_last 同样单调下降（6.65 → 6.56 → 6.50），head 越细表达力越强；
- 参数数严格相同（2,098,304），n_heads 不影响参数总数。

## 协议一致性

- 同一 cache / tokenizer / seed=42 / batch=8 / seq=64 / lr=3e-4 / warmup=100 / bf16 AMP / 5000 步；
- 同一架构：d_model=128, n_layers=4, d_ff=512（**2,098,304 params**）；
- 仅 `model.n_heads` 字段变化（2 / 4 / 8）；
- d_model=128 必须能被 n_heads 整除且 head_dim 必须为偶数。

3 个 result JSON 的 `metadata.dataset_hash` 全部等于 `e7ece4c755cd54be74eefeb50a4cbbc26af99bba60b0313349dcabc53021d1bd`。

3 个 result JSON 的 `metadata.gpu_compute_capability` 全部等于 `"12.0"`。

## contract 逐项复核

1. **schema v1.1 metrics.required** = [last_train_loss, validation_losses, train_losses, curve_summary] ✅
2. **Draft202012Validator iter_errors** = 0 对全部 3 个 artifact ✅
3. **architecture_lab/training/dense_training.py::compute_curve_summary** 存在 ✅
4. **architecture_lab/training/results.py::build_training_result** 接收 `train_loss_samples` + `curve_summary` 参数 ✅
5. **scripts/train_dense.py** 注入 metadata ✅
6. **scripts/plot_dense_curve.py::_plot_overlay** 只调用一次 `ax.twinx()`（行为测试覆盖）✅
7. **scripts/plot_dense_curve.py::PALETTE_TRAIN / PALETTE_VAL** 模块顶层常量（第二轮加固） ✅
8. **scripts/run_tests.py full** = 100 tests passing ✅
9. **stage review doc 按 review-process.md 模板**（含 minimax-cn/MiniMax-M3 字段） ✅
10. **configs/* 与 docs/* 在允许文件列表内** ✅

## auditor gap 历史

N8 为新阶段，auditor gap 历史从 0 开始。后续若 isolated auditor（calculet/gpt-5.6-terra）提出具体 gap，按相同模式记录到本节。

## 关联文档

- 协议：`docs/protocols/n8-dense-heads-sweep.md`
- 实验记录：`docs/experiments/n8-dense-heads-sweep/README.md`
- N5-N7 审查：`docs/plans/reviews/stage-n{5,6,7}-dense-{scale-sweep,dropout-sweep,rope-sweep}.md`
- Roadmap：`docs/plans/roadmap.md`（N8 已推进；当前阶段 = N9）

## 风险与遗留

- 单 seed：3 个点都用 seed=42；
- 短序列：max_seq_len=64 下 n_heads 影响亚主导级；
- 不可外推：观察严格限定于 medium 2.10M + 单 seed + OWT cache + max_seq_len=64 + Dense。

## 审查结论

N8 阶段交付完成。建议进入 N9 d_ff 消融（roadmap 下一阶段）。