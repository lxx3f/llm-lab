# P1-03 多 seed / 统计区间协议

> 状态：已解决（2026-08-26）。本文档固定多 seed sweep 的协议定义，并记录 N5+N6 3-seed 实跑结果。

## 背景

此前所有实验（N1-N11）均为**单 seed（42）**，结果不能区分"真实差异"与"初始化噪声"。P1-03 是本项目的多 seed / 统计区间协议。

> 更新（2026-08-27）：N11 长训练已按本协议扩展为 3 seeds（{42, 123, 7}）——见 `docs/experiments/n11-long-multi-seed/README.md`（baseline/medium/large 各 50000 步）；MoE 5000+50000 步亦已 3-seed 实跑。

**实跑预演**：`docs/experiments/multi-seed-sweep/README.md`（N5 + N6 × {42, 123, 7} = 18 个 5000 步训练）已证明：
- 规模 sweep 结论多 seed 稳健；
- dropout=0.1 的单 seed 优势是噪声（多 seed 下不成立）；
- large 的 std 最大（0.093），单 seed 低估其真实水平。

## 协议定义

### 1. Seed 集合

- **默认**：3 seeds = {42, 123, 7}（seed 42 保留向后兼容 N1-N11）；
- **扩展**：如需统计显著性（CI/p-value），5 seeds = {42, 123, 7, 2024, 999}；
- **规则**：seed 必须覆盖"常见值 + 随机值"，避免全部用 round number。

### 2. 统计口径

| 指标 | 默认口径 |
|---|---|
| val_loss_min | 每 seed 的 curve_summary.val_loss_min |
| mean | 算术平均 |
| std | 总体标准差（除以 N）|
| p50 | 中位数 |
| p95 | 95 分位（排序后位置 0.95*(N-1) 线性插值）|
| CI | 不做默认要求（N=3 无意义）；N≥5 时报告 95% CI = mean ± 1.96*std/sqrt(N) |

### 3. 采样规则

- 每个 seed 使用完全相同的训练协议（cache / tokenizer / batch / seq / lr / warmup / AMP / max_steps）；
- 每个 seed 独立 checkpoint 路径（`{stem}-seed{seed}.pt`）；
- 每个 seed 独立 result JSON（`{stem}-seed{seed}-result.json`）。

### 4. 判定标准（多 seed 下）

- **配置 A 优于 B**：仅当 mean_A < mean_B **且** `(mean_B - mean_A) > std_B + std_A`（即差异超过两倍 std 之和）；
- **趋势**：跨配置的 mean 单调趋势可作为观察记录，但不能作为严格结论（无显著性检验）；
- **报告格式**：`val_min mean ± std (seed values)`。

### 5. 环境默认状态（记录字段）

- `torch.compile`：**默认关闭**（避免编译时间污染小规模实验；记录 `compile_enabled: false`）；
- AMP：**bf16 默认开**（与 N4-N11 一致；记录 `amp.dtype`）；
- CUDA Graph：**默认关闭**（小模型 graph 捕获收益有限；记录 `cuda_graph_enabled: false`）；
- warmup / measured 规则：训练类实验无 warmup/measured 区分（那是 inference latency benchmark 的规则）；本协议只适用训练曲线。

## 应用范围

- **必须多 seed**：配置对比（如 dropout 消融、超参扫描），因为单 seed 差异易被噪声淹没；
- **可单 seed**：链路验证（smoke）、单元测试、单一配置观察（如 N4 baseline 曲线本身）；
- **不适用**：inference latency benchmark（那是 P1-02/N2 的 warmup-measured 规则）。

## 已应用实验

| 实验 | seeds | 结论 |
|---|---|---|
| N5 规模 sweep（multi-seed 版） | 3 | val_min mean 7.26 → 7.03，规模结论稳健 |
| N6 dropout sweep（multi-seed 版） | 3 | dropout=0.1 优势消失（7.09 > 7.08），dropout 影响退化 |
| N7/N8/N9 单 seed | 1 | 保留单 seed 观察，不构成多 seed 结论 |

## 遗留

- CI / p-value 统计检验：N≥5 seed 时可选（当前 N=3 无意义）；
- 多 seed 结果不自动回溯改写 N1-N11 单 seed 文档（只在关键结论处标注"单 seed，未多 seed 验证"）；
- N5/N6 的多 seed 版已落 `artifacts/multi-seed-overview.json`。