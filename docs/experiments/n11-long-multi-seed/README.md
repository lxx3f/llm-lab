# N11 长训练多 seed 复核（baseline+medium+large 50000 步 × {42, 123, 7}）

> 状态：阶段交付。
>
> 本实验把 N11 的 baseline + medium + large 50000 步长训练扩展到 **3 seeds（42, 123, 7）**，验证单 seed（42）长训练结论的多 seed 稳健性。共 9 个 50000 步训练（每个 ~17-70 min）。

## 实验结果（val_loss_min mean ± std over 3 seeds）

| 规模 | seed42 | seed123 | seed7 | mean | std | 触底步 |
|---|---|---|---|---|---|---|
| **baseline 50000 步** | 6.080 | 6.083 | 6.070 | **6.0776** | 0.0057 | 全部 50000 |
| **medium 50000 步** | 5.545 | 5.552 | 5.534 | **5.5436** | 0.0073 | 全部 50000 |
| **large 50000 步** | 5.264 | 5.264 | 5.245 | **5.2578** | 0.0090 | 全部 50000 |

> std 为总体标准差（除以 N，P1-03 协议定义）。seed42 与单 seed N11 结果完全一致（baseline 6.08 / medium 5.55 / large 5.26），确认单 seed 复现性。

## 与 5000 步多 seed 对比

| 规模 | 5000 步 mean（多 seed）| 50000 步 mean（多 seed）| 改善 |
|---|---|---|---|
| baseline | 7.263 ± 0.024 | 6.078 ± 0.006 | 1.185 nats |
| medium | 7.078 ± 0.023 | 5.544 ± 0.007 | 1.534 nats |
| large | 7.027 ± 0.093 | 5.258 ± 0.009 | **1.769 nats** |

## 关键观察

1. **N11 单 seed 结论完全稳健**：std < 0.01 nats（远小于 5000 步 sweep 的 0.02-0.17），长训练下初始化噪声显著降低。
2. **三个规模 3-seed 全部在 step 50000 触底**（未触底），与单 seed 一致——50000 步对 0.66M-5.11M 仍不是充分训练。
3. **规模优势随训练拉长扩大（多 seed 确认）**：
   - 5000 步：baseline 7.263 vs medium 7.078 vs large 7.027（区间 0.236 nats）；
   - 50000 步：baseline 6.078 vs medium 5.544 vs large 5.258（区间 **0.820 nats**）；
   - 三规模单调下降（多 seed mean），区间 0.82 nats，且 std 全部 < 0.01 → 按 P1-03 判定标准（mean 差 > std 和）相邻配置差异满足判定；趋势本身作为观察记录，不做严格显著性结论（N=3 无显著性检验）。
4. **long 训练的 std 远小于 5000 步**：baseline 0.0057 vs 0.024；medium 0.0073 vs 0.023；large 0.0090 vs 0.093——训练时间越长，seed 影响越小。
5. **large 50000 步 std 略高于 baseline/medium**（0.0090 vs 0.0057/0.0073），但仍在 0.01 量级（远小于 5000 步的 0.093）。

## 不构成正式结论

- 50000 步上限不是充分训练（全部未触底）；
- 无更小/longer 步数对比（100000 步不在本阶段，见 N12）；
- 单 seed 结论更新：N11 large 原为单 seed 观察（5.2639），现以 3-seed mean 5.2578 ± 0.0090 复核确认（区间 0.82 nats 与单 seed 完全一致）。

## 文件索引

- 脚本：`scripts/run_multi_seed.py`（RUNS 含 long configs）
- 实验记录：`docs/experiments/multi-seed-sweep/README.md`（5000 步版）+ 本文件
- Artifacts（gitignored）：`artifacts/dense-owt-formal-curve-long-{baseline,medium,large}-seed{42,123,7}-result.json`（9 个）