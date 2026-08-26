# 多 seed 实跑实验记录（N5 + N6 × {42, 123, 7}）

> 状态：阶段交付。
>
> 本实验把 N5（4 规模点）+ N6（3 dropout 点，dropout=0.0 = medium）从单 seed（42）扩展到 **3 seeds（42, 123, 7）**，共 18 个 5000 步训练。

## 范围

| config | 参数 | seeds |
|---|---|---|
| baseline (d=64/L=2/ff=256) | 0.66M | 42 / 123 / 7 |
| small (d=96/L=3/ff=384) | 1.23M | 42 / 123 / 7 |
| medium (d=128/L=4/ff=512) | 2.10M | 42 / 123 / 7 |
| large (d=192/L=6/ff=768) | 5.11M | 42 / 123 / 7 |
| medium dropout=0.1 | 2.10M | 42 / 123 / 7 |
| medium dropout=0.2 | 2.10M | 42 / 123 / 7 |

## 实验结果（val_loss_min mean ± std over 3 seeds）

| config | seed42 | seed123 | seed7 | mean | std |
|---|---|---|---|---|---|
| baseline | 7.2323 | 7.2913 | 7.2651 | **7.2629** | 0.0241 |
| small | 7.1777 | 7.1962 | 7.2172 | **7.1970** | 0.0161 |
| medium | 7.0582 | 7.1099 | 7.0653 | **7.0778** | 0.0229 |
| large | 6.9292 | 7.1520 | 6.9993 | **7.0268** | 0.0930 |
| dropout=0.1 | 6.9909 | 7.1943 | 7.0769 | **7.0874** | 0.0834 |
| dropout=0.2 | 7.0872 | 7.1837 | 7.0747 | **7.1152** | 0.0487 |

## 关键观察

1. **N5 规模 sweep 结论在多 seed 下仍成立**（val_min mean 单调下降）：
   - baseline 7.26 → small 7.20 → medium 7.08 → large 7.03；
   - 规模越大 val_min 越低，3-seed mean 与单 seed 趋势一致。
3. **large 的 std 最大（0.093）**：单 seed42 的 6.93 明显低于 3-seed mean 7.03——**单 seed 低估了 large 的真实水平**；large 对初始化更敏感（5.11M 参数）。
5. **N6 "dropout=0.1 最佳"结论在多 seed 下不成立**：
   - 单 seed42：dropout=0.1 (6.99) < dropout=0.0 (7.06) < dropout=0.2 (7.09)；
   - 3-seed mean：dropout=0.0 (7.08) < dropout=0.1 (7.09) < dropout=0.2 (7.12)；
   - **dropout=0.1 的单 seed 优势（0.07 nats）是噪声**；多 seed 下 dropout 影响退化为单调递增（0.0 < 0.1 < 0.2），且差距小（0.04 nats）。
   - 教训：**N6 的结论必须用多 seed 修正**——本实验正是 P1-03 的实跑预演。
7. **规模 > dropout 仍是稳健结论**：val_min mean 的规模区间（7.26 → 7.03 = 0.23 nats）远大于 dropout 区间（7.08 → 7.12 = 0.04 nats）。

## 与 P1-03 的关系

本实验是 **P1-03 多 seed 协议的实跑预演**：
- 固定 3 seeds（42/123/7）；
- 只报告 val_loss_min 的 mean/std（不做 p50/p95 / CI）；
- 验证了单 seed 结论在哪些场景稳健（规模 sweep）哪些不稳健（dropout 消融）；
- P1-03 正式协议会把 seed 数、统计口径（mean/std/CI）、warmup-measured 规则、torch.compile/AMP/CUDA Graph 默认状态写成正式协议文档。

## 不构成正式结论

- 3 seeds 是 3 个样本，mean/std 不构成统计显著性检验（无 CI / p-value）；
- 不声明任何"最佳配置"（dropout 差异 < 0.05 nats，需更大 seed 数）；
- 观察严格限定于 OWT cache + 5000 步 + Dense 单架构。

## 文件索引

- 实验记录：`docs/experiments/multi-seed-sweep/README.md`（本文）
- 脚本：`scripts/run_multi_seed.py`
- Artifacts（gitignored）：`artifacts/dense-owt-formal-curve-{...}-seed{42,123,7}-result.json`（18 个）+ `artifacts/multi-seed-overview.json`