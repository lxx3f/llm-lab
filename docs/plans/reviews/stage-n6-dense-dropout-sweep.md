# N6 Stage Review: Dense Dropout 消融

> 状态：N6 阶段交付。
>
> 审查 agent：reviewer（minimax-cn/MiniMax-M3）
> 审查时间：2026-08-26
> 阶段目标：在 N5 medium 2.10M（dropout=0.0）基础上，新增 dropout=0.1 与 dropout=0.2 对照，得到 medium dropout sensitivity 曲线。输出 3 个 result JSON + 1 个 3-curve overlay PNG + 协议/实验/审查三类文档；roadmap 推进 N6。

## 阶段范围

- ✅ 3 个 dropout 值的 medium 训练：dropout=0.0（N5 control）+ 0.1（N6 新增）+ 0.2（N6 新增），各 5000 步；
- ✅ 3 个 result JSON 全部 schema v1.1 valid + metadata 对齐 HEAD；
- ✅ 1 个 3-curve overlay PNG（共享 2 axes，行为测试覆盖）；
- ✅ 协议/实验/审查三类文档落盘；
- ✅ `scripts/run_tests.py full` 仍 97 tests passing；
- ✅ roadmap 推进（N6 → 已完成；当前阶段 = P1 工具执行器）。

## 3 个 dropout 值的曲线摘要

| dropout | train first → last (Δ) | val min @ step | val last | Δ_val_loss |
|---|---|---|---|---|
| **0.0** | 104.69 → 6.56 (**-98.13**) | **7.06** @ 5000 | 7.06 | 0.0 |
| **0.1** | 104.83 → 6.51 (**-98.32**) | **6.99** @ 5000 | 6.99 | 0.0 |
| **0.2** | 104.79 → 6.59 (**-98.20**) | **7.09** @ 5000 | 7.09 | 0.0 |

Dropout sensitivity：
- val_min 在 dropout=0.1 时最佳（6.99），相比 dropout=0.0 改善 1.0%（0.07 nats 绝对差）；
- dropout=0.2 反而劣于 0.0（7.09 vs 7.06，约 0.4% 退化）；
- train_loss 末值接近（6.56 / 6.51 / 6.59，区间 0.08）——dropout 对 train_last 影响远小于规模影响。

## 协议一致性

- 同一 cache：train 512 MiB / validation 64 MiB；
- 同一 tokenizer：owt-bpe v0.2.0 / vocab 8192；
- 同一 seed=42、batch=8、seq=64、lr=3e-4、warmup=100、min_lr_ratio=0.1、bf16 AMP、5000 步；
- 同一架构：d_model=128, n_heads=4, n_layers=4, d_ff=512（**2,098,304 params**）；
- 仅 `model.dropout` 字段变化（0.0 / 0.1 / 0.2）。

3 个 result JSON 的 `metadata.dataset_hash` 全部等于 `e7ece4c755cd54be74eefeb50a4cbbc26af99bba60b0313349dcabc53021d1bd`（train cache metadata 文件字节 SHA256）。

3 个 result JSON 的 `metadata.gpu_compute_capability` 全部等于 `"12.0"`（dotted，无 sm_ 前缀）。

## contract 逐项复核

1. **schema v1.1 metrics.required** = [last_train_loss, validation_losses, train_losses, curve_summary] ✅
2. **Draft202012Validator iter_errors** = 0 对全部 3 个 artifact ✅
3. **architecture_lab/training/dense_training.py::compute_curve_summary** 存在 ✅
4. **architecture_lab/training/results.py::build_training_result** 接收 `train_loss_samples` + `curve_summary` 参数 ✅
5. **scripts/train_dense.py** 注入 metadata ✅
6. **scripts/plot_dense_curve.py::_plot_overlay** 只调用一次 `ax.twinx()`（行为测试覆盖）✅
7. **scripts/run_tests.py full** = 97 tests passing ✅
8. **stage review doc 按 review-process.md 模板**（含 minimax-cn/MiniMax-M3 字段） ✅
9. **configs/* 与 docs/* 在允许文件列表内** ✅
10. **不重写历史阶段审查** ✅

## auditor gap 历史

N6 第一轮 isolated auditor（calculet/gpt-5.6-terra）提出 2 项 blocker，已全部闭合：

1. **overlay PNG 实际尺寸与文档不一致（800×500 vs 实际 1400×587）**：原 plot 使用 `figsize=(8, 5), dpi=120, bbox_inches="tight"`，后两者使实际尺寸超标；测试仅校验 PNG 大于 5KB 而不校验尺寸 → 修复：去掉 `bbox_inches="tight"`；`dpi=120` 调为 `dpi=100` 使 figsize × dpi = 800×500；新增 `test_png_dimensions_match_protocol_800x500` 与 `test_overlay_png_dimensions_match_protocol_800x500` 两个回归测试，使用 PIL `Image.open()` 读取实际尺寸并断言 == (800, 500)。重生成 7 个 PNG 全部验证 800×500 ✅
2. **overlay 调色板与协议不符（purple/olive vs 文档记录的 red）**：原代码 `palette_train[2] == "tab:purple"` 与 `palette_val[2] == "tab:olive"`，协议文档说 dropout=0.2 应为红 / 深红 → 修复：协议描述的调色板需与代码一致；调整 `palette_train = ["tab:blue", "tab:green", "tab:red", ...]` 与 `palette_val = ["tab:orange", "tab:olive", "tab:brown", ...]`，3-input 情况下索引 0 / 1 / 2 映射为 blue/orange、green/olive、red/brown，与 N6 协议中 dropout=0.0/0.1/0.2 的描述一致；新增 `test_overlay_palette_matches_protocol_blue_green_red` 测试，运行时调用 `_plot_train` / `_plot_val`，检查实际 Line2D 的 `get_color()` 返回值含 `tab:blue` / `tab:green` / `tab:red` 与 `tab:orange` / `tab:olive` / `tab:brown` ✅

测试统计：97 → **100 tests passed**（+3 PNG 尺寸与调色板回归测试）；7 个 PNG 重生成（全部 800×500）；N6 协议与 README 同步文档描述。

## 关联文档

- 协议：`docs/protocols/n6-dense-dropout-sweep.md`
- 实验记录：`docs/experiments/n6-dense-dropout-sweep/README.md`
- N5 审查：`docs/plans/reviews/stage-n5-dense-scale-sweep.md`
- Roadmap：`docs/plans/roadmap.md`（N6 已推进；当前阶段 = P1 工具执行器）

## 风险与遗留

- 单 seed：3 个点都用 seed=42，不构成 P1-03 多 seed sweep；
- 短训练：5000 步对 2.10M 模型不充分，3 个点的 val_loss_min_step == 5000 都成立；
- 不可外推：观察严格限定于 medium 2.10M + 单 seed + OWT cache + Dense；任何外推不构成 N6 协议内的有效结论。

## 审查结论

N6 阶段交付完成。建议进入 P1 工具执行器（roadmap 下一阶段）。