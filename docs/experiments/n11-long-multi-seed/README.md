# N11 长训练多 seed 复核（baseline+medium 50000 步 × {42, 123, 7}）

> 状态：阶段交付。
>
> 本实验把 N11 的 baseline + medium 50000 步长训练扩展到 **3 seeds（42, 123, 7）**，验证单 seed（42）长训练结论的多 seed 稳健性。共 6 个 50000 步训练（每个 ~22-70 min）。

## 实验结果（val_loss_min mean ± std over 3 seeds）

| 规模 | seed42 | seed123 | seed7 | mean | std | 触底步 |
|---|---|---|---|---|---|---|
| **baseline 50000 步** | 6.080 | 6.083 | 6.070 | **6.0776** | 0.0057 | 全部 50000 |
| **medium 50000 步** | 5.545 | 5.552 | 5.534 | **5.5436** | 0.0073 | 全部 50000 |

> seed42 与单 seed N11 结果完全一致（baseline 6.08 / medium 5.55），确认单 seed 复现性。

## 与 5000 步多 seed 对比

| 规模 | 5000 步 mean（多 seed）| 50000 步 mean（多 seed）| 改善 |
|---|---|---|---|
| baseline | 7.263 ± 0.024 | 6.078 ± 0.006 | 1.185 nats |
| medium | 7.078 ± 0.023 | 5.544 ± 0.007 | 1.534 nats |

## 关键观察

1. **N11 单 seed 结论完全稳健**：std < 0.01 nats（远小于 5000 步 sweep 的 0.02-0.17），长训练下初始化噪声显著降低。
2. **两个规模 3-seed 全部在 step 50000 触底**（未触底），与单 seed 一致——50000 步对 0.66M/2.10M 仍不是充分训练。
3. **规模优势随训练拉长扩大（多 seed 确认）**：
   - 5000 步：baseline 7.263 vs medium 7.078（差距 0.185 nats）；
   - 50000 步：baseline 6.078 vs medium 5.544（差距 **0.534 nats**）；
   - 与单 seed 观察一致（0.25 → 0.46 nats），多 seed 下差距更大且 std 极小。
4. **long 训练的 std 远小于 5000 步**：baseline 0.0057 vs 0.024；medium 0.0073 vs 0.023——训练时间越长，seed 影响越小。

## 不构成正式结论

- 仍只有 2 个规模点；
- 50000 步上限不是充分训练（全部未触底）；
- 无更小/longer 步数对比（100000 步不在本阶段）。

## 文件索引

- 脚本：`scripts/run_multi_seed.py`（RUNS 含 long configs）
- 实验记录：`docs/experiments/multi-seed-sweep/README.md`（5000 步版）+ 本文件
- Artifacts（gitignored）：`artifacts/dense-owt-formal-curve-long-{baseline,medium}-seed{42,123,7}-result.json`（6 个）