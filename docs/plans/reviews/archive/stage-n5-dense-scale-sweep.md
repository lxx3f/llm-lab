# N5 Stage Review: Dense 规模 sweep（4 点曲线）

> 状态：N5 阶段交付。
>
> 审查 agent：reviewer（minimax-cn/MiniMax-M3）
> 审查时间：2026-08-26
> 阶段目标：把 N4 baseline + medium 扩展到 4 规模 sweep；输出 4 个 result JSON + 1 个 4-curve overlay PNG + 协议/实验/审查三类文档；roadmap 推进 N5。

## 阶段范围

- ✅ 4 规模 sweep：baseline 0.66M + small 1.23M + medium 2.10M + large 5.11M，各 5000 步；
- ✅ 4 个 result JSON 全部 schema v1.1 valid + metadata 对齐 HEAD；
- ✅ 1 个 4-curve overlay PNG（共享 2 axes，行为测试覆盖）；
- ✅ 协议/实验/审查三类文档落盘；
- ✅ `scripts/run_tests.py full` 仍 97 tests passing；
- ✅ roadmap 推进（N5 → 已完成；当前阶段 = P1 工具执行器）。

## 4 个规模点的曲线摘要

| 规模 | 参数 | train first → last (Δ) | val min @ step | val last | 5k 步实测耗时 |
|---|---|---|---|---|---|
| **baseline** | 0.66M | 61.57 → 6.81 (**-54.76**) | **7.23** @ 5000 | 7.23 | ~42s |
| **small** | 1.23M | 88.42 → 6.60 (**-81.82**) | **7.18** @ 5000 | 7.18 | ~54s |
| **medium** | 2.10M | 104.69 → 6.56 (**-98.13**) | **7.06** @ 5000 | 7.06 | ~65s |
| **large** | 5.11M | 62.59 → 6.41 (**-56.18**) | **6.93** @ 5000 | 6.93 | ~91s |

规模敏感性：
- val_loss 随规模单调下降（7.23 → 7.18 → 7.06 → 6.93），7.8× 规模带来 4.1% val_loss 改善；
- train_loss 末值随规模缓慢下降（6.81 → 6.60 → 6.56 → 6.41），5000 步内仍未饱和；
- Δ_train_loss 不严格随规模增长（受 train_first 影响，非架构结论）。

## 协议一致性（与 N4 baseline/medium 共享）

- 同一 cache：train 512 MiB / validation 64 MiB；
- 同一 tokenizer：owt-bpe v0.2.0 / vocab 8192；
- 同一 seed=42、batch=8、seq=64、lr=3e-4、warmup=100、min_lr_ratio=0.1、bf16 AMP、5000 步；
- 仅 `d_model` / `n_layers` / `d_ff` / `n_heads` 固定为 4 共 3 个变量。

4 个 result JSON 的 `metadata.dataset_hash` 全部等于 `e7ece4c755cd54be74eefeb50a4cbbc26af99bba60b0313349dcabc53021d1bd`（train cache metadata 文件字节 SHA256）。

4 个 result JSON 的 `metadata.gpu_compute_capability` 全部等于 `"12.0"`（dotted，无 sm_ 前缀）。

## contract 逐项复核

1. **schema v1.1 metrics.required** = [last_train_loss, validation_losses, train_losses, curve_summary] ✅
2. **Draft202012Validator iter_errors** = 0 对全部 4 个 artifact ✅
3. **architecture_lab/training/dense_training.py::compute_curve_summary** 存在 ✅
4. **architecture_lab/training/results.py::build_training_result** 接收 `train_loss_samples` + `curve_summary` 参数 ✅
5. **scripts/train_dense.py** 注入 metadata ✅
6. **scripts/plot_dense_curve.py::_plot_overlay** 只调用一次 `ax.twinx()`（行为测试覆盖）✅
7. **scripts/run_tests.py full** = 97 tests passing ✅
8. **stage review doc 按 review-process.md 模板**（含 minimax-cn/MiniMax-M3 字段） ✅
9. **configs/* 与 docs/* 在允许文件列表内** ✅
10. **不重写历史阶段审查** ✅

## auditor gap 历史

N5 为新阶段，auditor gap 历史从 0 开始。后续若 isolated auditor（calculet/gpt-5.6-terra）提出具体 gap，按相同模式记录到本节。

## 关联文档

- 协议：`docs/protocols/n5-dense-scale-sweep.md`
- 实验记录：`docs/experiments/n5-dense-scale-sweep/README.md`
- N4 审查：`docs/plans/reviews/stage-n4-dense-formal-curve.md`
- Roadmap：`docs/plans/roadmap.md`（N5 已推进；当前阶段 = P1 工具执行器）

## 风险与遗留

- 单 seed：4 个点都用 seed=42，不构成 P1-03 多 seed sweep；
- 短训练：5000 步对 5.11M 模型不充分，train_loss 仍处下降区间；
- 不可外推：观察严格限定于 4 个固定规模点 + 单 seed + Dense + OWT cache；任何外推不构成 N5 协议内的有效结论。

## 审查结论

N5 阶段交付完成。建议进入 P1 工具执行器（roadmap 下一阶段）。