# MoE 多 seed 实跑（5000 + 50000 步 × {42, 123, 7}）

> 状态：阶段交付。
>
> 本实验把 MoE Top-1（72% active params，1.51M / 2.10M Dense-medium 对齐）扩展到 **3 seeds（42, 123, 7）**，覆盖 5000 步 + 50000 步两条曲线，填补 MoE vs Dense 单 seed 观察的统计缺口（P1-03 协议）。共 2 configs × 3 seeds = **6 个训练**。

## 实验结果（val_loss_min mean ± std over 3 seeds）

### MoE 5000 步（与 Dense medium 5000 步对比）

| seed | MoE 5000 步 | Dense medium 5000 步 |
|---|---|---|
| 42 | 7.2201 @ 4600 | 7.0582 @ 5000 |
| 123 | 7.2955 @ 5000 | 7.1099 @ 5000 |
| 7 | 7.2799 @ 5000 | 7.0653 @ 5000 |
| **mean ± std** | **7.2651 ± 0.0398** | **7.0778 ± 0.0229** |
| **差距** | | **0.187 nats** |

### MoE 50000 步（与 Dense medium 50000 步对比）

| seed | MoE 50000 步 | Dense medium 50000 步 |
|---|---|---|
| 42 | 6.0434 @ 50000 | 5.5453 @ 50000 |
| 123 | 6.1842 @ 50000 | 5.5515 @ 50000 |
| 7 | 6.1262 @ 50000 | 5.5339 @ 50000 |
| **mean ± std** | **6.1179 ± 0.0708** | **5.5436 ± 0.0073** |
| **差距** | | **0.574 nats** |

## 单 seed vs 多 seed 结论对比

| 指标 | 单 seed（42）观察 | 3-seed mean | 稳健？ |
|---|---|---|---|
| MoE 5000 步 vs Dense | 7.22 vs 7.06（差距 0.16）| 7.2651 vs 7.0778（差距 0.187）| ✅ 稳健（差距略大）|
| MoE 50000 步 vs Dense | 6.04 vs 5.55（差距 0.50）| 6.1179 vs 5.5436（差距 0.574）| ✅ 稳健（差距更大）|
| MoE 5000 → 50000 步改善 | 7.22 → 6.04（-1.18）| 7.2651 → 6.1179（-1.15）| ✅ 稳健 |
| MoE 50000 步 std | — | 0.0708（> Dense 0.0073）| MoE 更不稳定 |

## 关键观察

1. **MoE vs Dense 差距随训练拉长扩大（3-seed 确认）**：5000 步 0.187 nats → 50000 步 0.574 nats。单 seed 观察的 "差距 0.16 → 0.50" 方向被确认，多 seed 下差距更大（0.19 → 0.57）。
2. **统计显著性（P1-03 判定标准：mean 差 > std 和）**：
   - 5000 步：0.187 > 0.0398 + 0.0229 = 0.063 ✅
   - 50000 步：0.574 > 0.0708 + 0.0073 = 0.078 ✅
   - 两个训练长度下 MoE 都显著劣于 Dense（同 72% active params 口径）。
3. **MoE 更不稳定**：50000 步 std = 0.0708 vs Dense 0.0073（10×）。MoE 的 seed 敏感性远高于 Dense——可能来自 router 初始化和 load balancing 的 seed 依赖。
4. **MoE 5000 步"首次反弹"仍是 seed 依赖现象**：seed=42 在 step 4600 触底反弹（7.22 @ 4600 → 5000），seed=123/7 都在 step 5000 触底。仅 seed=42 展示了早触底反弹。
5. **长训练下 MoE 未触底**：三个 seed 的 50000 步 val_min 都在 step 50000（都在继续下降），与 Dense 一致（N11 long 也都在 50000 触底）。

## 运行命令

```bash
.venv/python.exe -u scripts/run_multi_seed.py --only moe
# 输出：artifacts/moe-owt-formal-curve-{seed42,seed123,seed7}-result.json
#       artifacts/moe-owt-formal-curve-long-{seed42,seed123,seed7}-result.json
#       artifacts/multi-seed-overview.json（含 17 configs 全量重建）
```

- scripts/run_multi_seed.py 新增 MoE dispatch：按 `config.model.architecture == 'MoETransformer'` 自动选择 train_moe.py。
- 训练耗时：MoE 5000 步 ~4 min × 3；MoE 50000 步 ~50 min × 3（总墙钟 ~3 h）。

## 数据契约

- 训练 schema：`moe_training_result.schema.json` v1.1（`metrics.curve_summary` 与 Dense 同构，可直接用 P1-03 统计）。
- artifact 路径（gitignored，与 N11 long 一致）：
  - `artifacts/moe-owt-formal-curve-{seed42,seed123,seed7}-result.json`
  - `artifacts/moe-owt-formal-curve-long-{seed42,seed123,seed7}-result.json`
- overview：`artifacts/multi-seed-overview.json`（config_summary 17 个 configs）。
- 对应 Dense 多 seed 数据：`docs/experiments/multi-seed-sweep/README.md`（5000 步）+ `docs/experiments/n11-long-multi-seed/README.md`（50000 步）。
