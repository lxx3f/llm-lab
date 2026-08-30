# N9 Stage Review: Dense d_ff 消融

> 状态：N9 阶段交付。
>
> 审查 agent：reviewer（minimax-cn/MiniMax-M3）
> 审查时间：2026-08-26
> 阶段目标：在 N5 medium（d_ff=512）基础上，扫描 d_ff ∈ {256, 512, 1024}，得到 FFN hidden size sensitivity 曲线。输出 3 个 result JSON + 1 个 3-curve overlay PNG + 协议/实验/审查三类文档；roadmap 推进 N9。

## 阶段范围

- ✅ 3 个 d_ff 值的训练：d_ff=256 + 512（N5 control）+ 1024，各 5000 步；
- ✅ 3 个 result JSON 全部 schema v1.1 valid + metadata 对齐 HEAD；
- ✅ 1 个 3-curve overlay PNG（共享 2 axes，800×500 dpi=100）；
- ✅ 协议/实验/审查三类文档落盘；
- ✅ `scripts/run_tests.py full` 仍 100 tests passing；
- ✅ roadmap 推进（N9 → 已完成；当前阶段 = MoE 5000 步曲线）。

## 3 个 d_ff 值的曲线摘要

| d_ff | params | train first → last (Δ) | val min @ step |
|---|---|---|---|
| **256** | 1.71M | 95.18 → 7.65 (-87.53) | **7.5573** @ **3200**（Δ_val_loss=+0.0348）|
| **512** | 2.10M | 104.69 → 6.56 (-98.13) | **7.0582** @ 5000 |
| **1024** | 2.88M | 113.00 → 6.27 (**-106.73**) | **6.9668** @ 5000 |

d_ff sensitivity：
- val_min 单调下降（7.5573 → 7.0582 → 6.9668），变化区间 0.59 nats（7.8%），是所有 sweep 中最大；
- 边际效益递减：2× → 4× 改善 0.50，4× → 8× 改善 0.09；
- FFN 容量是 val_min 的主要瓶颈；
- **d_ff=256 在 step 3200 触底后反弹（Δ_val_loss=+0.0348）**，指示 FFN 容量不足以支持 5000 步继续下降；d_ff=512/1024 容量足够，5000 步仍未触底。

## 协议一致性

- 同一 cache / tokenizer / seed=42 / batch=8 / seq=64 / lr=3e-4 / warmup=100 / bf16 AMP / 5000 步；
- 同一架构：d_model=128, n_heads=4, n_layers=4；
- 仅 `model.d_ff` 字段变化（256 / 512 / 1024）；
- 参数总数随之变化（1.71M / 2.10M / 2.88M）。

3 个 result JSON 的 `metadata.dataset_hash` 全部等于 `e7ece4c755cd54be74eefeb50a4cbbc26af99bba60b0313349dcabc53021d1bd`。

## contract 逐项复核

1. **schema v1.1 metrics.required** = [last_train_loss, validation_losses, train_losses, curve_summary] ✅
2. **Draft202012Validator iter_errors** = 0 对全部 3 个 artifact ✅
3. **architecture_lab/training/dense_training.py::compute_curve_summary** 存在 ✅
4. **architecture_lab/training/results.py::build_training_result** 接收 `train_loss_samples` + `curve_summary` 参数 ✅
5. **scripts/train_dense.py** 注入 metadata ✅
6. **scripts/plot_dense_curve.py::_plot_overlay** 只调用一次 `ax.twinx()`（行为测试覆盖）✅
7. **scripts/plot_dense_curve.py::PALETTE_TRAIN / PALETTE_VAL** 模块顶层常量 ✅
8. **scripts/run_tests.py full** = 100 tests passing ✅
9. **stage review doc 按 review-process.md 模板**（含 minimax-cn/MiniMax-M3 字段） ✅
10. **configs/* 与 docs/* 在允许文件列表内** ✅

## auditor gap 历史

N9 为新阶段，auditor gap 历史从 0 开始。

## 关联文档

- 协议：`docs/protocols/n9-dense-dff-sweep.md`
- 实验记录：`docs/experiments/n9-dense-dff-sweep/README.md`
- N5-N8 审查：`docs/plans/reviews/stage-n{5,6,7,8}-dense-*.md`
- Roadmap：`docs/plans/roadmap.md`（N9 已推进）

## control artifact provenance

d_ff=512 control 复用 N5 medium artifact（`dense-owt-formal-curve-medium-result.json`），是 N5/N7/N8/N9 共享 control，在夜间跑最终 commit 重生成（`metadata.git_commit == HEAD`，val_min=7.0582）。同一 config + 同一 seed → 相同 val_min，跨阶段复用不改变曲线结论。

## 风险与遗留

- 单 seed：3 个点都用 seed=42；
- 参数变化：d_ff sweep 不是"严格参数相同"消融；
- 不可外推：观察严格限定于 medium + 单 seed + OWT cache + max_seq_len=64 + Dense。

## 审查结论

N9 阶段交付完成。建议进入 MoE 5000 步训练曲线（roadmap 下一阶段，补 N1/N2 缺口）。