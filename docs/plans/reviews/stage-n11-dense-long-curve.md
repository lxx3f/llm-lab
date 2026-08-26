# N11 Stage Review: Dense 长训练曲线（50000 步）

> 状态：N11 阶段交付。
>
> 审查 agent：reviewer（minimax-cn/MiniMax-M3）
> 审查时间：2026-08-26
> 阶段目标：把 baseline 0.66M + medium 2.10M 从 5000 步扩展到 50000 步，观察 val 曲线是否触底/反弹。输出 2 个 result JSON + 1 个 2-curve overlay PNG + 协议/实验/审查三类文档。

## 阶段范围

- ✅ baseline + medium 各 50000 步训练（log_interval=500, validation_interval=2000）；
- ✅ 2 个 result JSON schema v1.1 valid + metadata 对齐 HEAD（train_losses 100 点 + curve_summary）；
- ✅ 1 个 2-curve overlay PNG 800×500；
- ✅ 协议/实验/审查三类文档落盘；
- ✅ `scripts/run_tests.py full` 仍 100 tests passing；
- ✅ roadmap 推进（N11 → 已完成）。

## 实验结果

| 规模 | 参数 | 5000 步 val_min | **50000 步 val_min** | 改善 |
|---|---|---|---|---|
| baseline | 0.66M | 7.23 @ 5000 | **6.08** @ 50000 | 1.15 nats |
| medium | 2.10M | 7.06 @ 5000 | **5.55** @ 50000 | 1.51 nats |

观察：
- 延长 10× 训练带来 1.2-1.5 nats val 改善；
- medium 改善幅度大于 baseline（规模优势随训练拉长更显著）；
- 50000 步仍未触底（val_min_step == 50000），与 5000 步观察一致。

## contract 逐项复核

1. **schema v1.1 metrics.required** = [last_train_loss, validation_losses, train_losses, curve_summary] ✅
2. **Draft202012Validator iter_errors** = 0 对 2 个 artifact ✅
3. **architecture_lab/training/dense_training.py::compute_curve_summary** 存在 ✅
4. **scripts/train_dense.py** 注入 metadata ✅
5. **scripts/plot_dense_curve.py::_plot_overlay** 只调用一次 `ax.twinx()`（行为测试覆盖）✅
6. **scripts/plot_dense_curve.py::PALETTE_TRAIN / PALETTE_VAL** 模块顶层常量 ✅
7. **scripts/run_tests.py full** = 100 tests passing ✅
8. **stage review doc 按 review-process.md 模板**（含 minimax-cn/MiniMax-M3 字段） ✅
9. **configs/* 与 docs/* 在允许文件列表内** ✅

## auditor gap 历史

N11 为新阶段，auditor gap 历史从 0 开始。

## 关联文档

- 协议：`docs/protocols/n11-dense-long-curve.md`
- 实验记录：`docs/experiments/n11-dense-long-curve/README.md`
- N4 5000 步审查：`docs/plans/reviews/stage-n4-dense-formal-curve.md`
- Roadmap：`docs/plans/roadmap.md`（N11 已推进）

## 风险与遗留

- 单 seed：2 个点都用 seed=42；
- 未触底：50000 步仍未观察到 val 反弹，无法判断过拟合；
- 不可外推：观察严格限定于 baseline + medium + 单 seed + OWT cache + Dense。

## 审查结论

N11 阶段交付完成。建议进入多 seed 实跑（roadmap 下一阶段）。