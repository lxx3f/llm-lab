# Stage Review: MoE 5000 步训练曲线

> 状态：阶段交付。
>
> 审查 agent：reviewer（minimax-cn/MiniMax-M3）
> 审查时间：2026-08-26
> 阶段目标：补齐 MoE Top-1 在 OWT 正式 cache 上的 5000 步训练曲线（补 N1/N2 缺口）；MoE result schema v1.0 → v1.1（对齐 Dense v1.1）；输出 MoE result JSON + PNG + 协议/实验/审查三类文档。

## 阶段范围

- ✅ MoE 5000 步训练（total 2.10M / active 1.51M）；
- ✅ MoE result schema v1.0 → v1.1（加 train_losses + curve_summary，对齐 Dense v1.1）；
- ✅ MoE result JSON schema v1.1 valid + metadata 对齐 HEAD；
- ✅ 2 个 PNG（单曲线 + Dense 对比 overlay）800×500；
- ✅ 协议/实验/审查三类文档落盘；
- ✅ `scripts/run_tests.py full` 仍 100 tests passing；
- ✅ roadmap 推进（MoE 曲线 → 已完成；当前阶段 = N11 长训练）。

## 实验结果

| 指标 | 值 |
|---|---|
| train_loss first → last | 117.90 → 6.79 (-111.11) |
| val_min | **7.22** @ step 4600 |
| val_last | 7.23 |
| Δ_val_loss | 0.01 |
| total params | 2,100,352 |
| top1 active params | 1,510,528（72%）|

### MoE vs Dense medium 对比

| 维度 | Dense medium | MoE |
|---|---|---|
| total params | 2,098,304 | 2,100,352 |
| active params | 2,098,304 (100%) | 1,510,528 (72%) |
| val_min | **7.06** @ 5000 | 7.22 @ 4600 |

观察：
- MoE 用 72% active params 达到 97.8% 的 val 性能；
- MoE val_min 首次出现在 step 4600（Δ_val_loss=0.01），而全部 Dense 曲线都在 5000 步触底；
- 单 seed 观察，不构成过拟合证据。

## contract 逐项复核

1. **MoE schema v1.1 metrics.required** = [last_train_lm_loss, last_train_aux_loss, last_train_total_loss, validation_metrics, train_losses, curve_summary] ✅
2. **Draft202012Validator iter_errors** = 0 对 MoE artifact ✅
3. **architecture_lab/training/moe_results.py::build_moe_training_result** 接收 train_loss_samples + curve_summary 参数 ✅
4. **architecture_lab/training/moe_training.py::train** 采样 train_loss_samples（log_interval）+ compute_curve_summary（复用 Dense 的 compute_curve_summary） ✅
5. **scripts/train_moe.py** 注入 metadata（N3 契约保持） ✅
6. **scripts/plot_dense_curve.py::PALETTE_TRAIN / PALETTE_VAL** 模块顶层常量 ✅
7. **scripts/run_tests.py full** = 100 tests passing ✅
8. **stage review doc 按 review-process.md 模板**（含 minimax-cn/MiniMax-M3 字段） ✅
9. **configs/* 与 docs/* 在允许文件列表内** ✅

## auditor gap 历史

本阶段为新阶段，auditor gap 历史从 0 开始。

## 关联文档

- 协议：`docs/protocols/moe-owt-formal-curve.md`
- 实验记录：`docs/experiments/moe-owt-formal-curve/README.md`
- Dense 曲线审查：`docs/plans/reviews/stage-n4-dense-formal-curve.md`
- Roadmap：`docs/plans/roadmap.md`

## 风险与遗留

- 单 seed / 单 MoE 配置；
- MoE vs Dense 对比仅限 total/active params 口径，不替代 N2 完整协议 A/B；
- MoE schema v1.1 是向后不兼容升级（预 v1.1 artifact 需重新生成）。

## 审查结论

MoE 5000 步曲线阶段交付完成。建议进入 N11 长训练（roadmap 下一阶段）。