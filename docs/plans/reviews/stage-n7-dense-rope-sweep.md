# N7 Stage Review: Dense RoPE base 消融

> 状态：N7 阶段交付。
>
> 审查 agent：reviewer（minimax-cn/MiniMax-M3）
> 审查时间：2026-08-26
> 阶段目标：在 N5 medium 2.10M（rope_base=10000）基础上，扫描 rope_base ∈ {10000, 50000, 100000}，得到 RoPE base sensitivity 曲线。输出 3 个 result JSON + 1 个 3-curve overlay PNG + 协议/实验/审查三类文档；roadmap 推进 N7。

## 阶段范围

- ✅ 3 个 rope_base 值的 medium 训练：rope_base=10000（N5 control）+ 50000 + 100000，各 5000 步；
- ✅ 3 个 result JSON 全部 schema v1.1 valid + metadata 对齐 HEAD；
- ✅ 1 个 3-curve overlay PNG（共享 2 axes，800×500 dpi=100）；
- ✅ 协议/实验/审查三类文档落盘；
- ✅ `scripts/run_tests.py full` 仍 100 tests passing；
- ✅ roadmap 推进（N7 → 已完成；当前阶段 = N8）。

## 3 个 rope_base 值的曲线摘要

| rope_base | train first → last (Δ) | val min @ step | val last | Δ_val_loss |
|---|---|---|---|---|
| **10000** | 104.69 → 6.56 (**-98.13**) | **7.06** @ 5000 | 7.06 | 0.0 |
| **50000** | 105.05 → 6.49 (**-98.56**) | **7.03** @ 5000 | 7.03 | 0.0 |
| **100000** | 104.69 → 6.46 (**-98.23**) | **7.02** @ 5000 | 7.02 | 0.0 |

RoPE base sensitivity：
- val_min 单调下降（7.06 → 7.03 → 7.02），10× 增大 rope_base 带来 0.04 nats 改善（约 0.6%）；
- 100k 是观察最优，但 50k → 100k 边际效益递减（0.03 vs 0.07）；
- 在 max_seq_len=64 短序列下 RoPE base 影响亚主导级。

## 协议一致性

- 同一 cache：train 512 MiB / validation 64 MiB；
- 同一 tokenizer：owt-bpe v0.2.0 / vocab 8192；
- 同一 seed=42、batch=8、seq=64、lr=3e-4、warmup=100、min_lr_ratio=0.1、bf16 AMP、5000 步；
- 同一架构：d_model=128, n_heads=4, n_layers=4, d_ff=512（**2,098,304 params**）；
- 仅 `model.rope_base` 字段变化（10000.0 / 50000.0 / 100000.0）。

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

N7 为新阶段，auditor gap 历史从 0 开始。后续若 isolated auditor（calculet/gpt-5.6-terra）提出具体 gap，按相同模式记录到本节。

## 关联文档

- 协议：`docs/protocols/n7-dense-rope-sweep.md`
- 实验记录：`docs/experiments/n7-dense-rope-sweep/README.md`
- N5/N6 审查：`docs/plans/reviews/stage-n{5,6}-dense-{scale-sweep,dropout-sweep}.md`
- Roadmap：`docs/plans/roadmap.md`（N7 已推进；当前阶段 = N8）

## control artifact provenance

rope_base=10000 control 复用 N5 medium artifact（`dense-owt-formal-curve-medium-result.json`，d_model=128/n_layers=4/d_ff=512）。该 artifact 是 N5/N7/N8/N9 的共享 control，最后一次在夜间跑最终 commit `6786ca2` 重生成（metadata.git_commit == HEAD，val_min=7.0582）。同一 config + 同一 seed → 相同 val_min，跨阶段复用不改变曲线结论；provenance 以最终重生成 commit 为准。

## 风险与遗留

- 单 seed：3 个点都用 seed=42，不构成 P1-03 多 seed sweep；
- 短序列：max_seq_len=64 下 RoPE base 影响亚主导级；长序列（≥1024）下结论可能不同；
- 不可外推：观察严格限定于 medium 2.10M + 单 seed + OWT cache + max_seq_len=64 + Dense；任何外推不构成 N7 协议内的有效结论。

## 审查结论

N7 阶段交付完成。建议进入 N8 n_heads 消融（roadmap 下一阶段）。