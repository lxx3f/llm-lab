# SFT 工具调用训练 MVP（D1 + D1.1）

> 状态：管线交付完成，**模型结果为诚实负结果**（2026-08-27）。
>
> 本实验实现 SFT 工具调用训练闭环（数据构造 → 增强 → 训练 → 生成 → 八级分类器评测），并如实记录：**4M Dense 模型在 2000 步内未能学会稳定的工具调用 JSON 生成**。管线与评测基础设施可用，模型能力未达标。

## 交付内容

### 训练管线（完整可用）

| 模块 | 文件 | 说明 |
|---|---|---|
| SFT 数据构造 | `architecture_lab/training/sft_training.py` | tool_calling_sample → 掩码序列（user 不参与 loss，assistant 参与）；模板 `### User / ### Assistant / ### Result` |
| 确定性增强 | `architecture_lab/training/sft_augment.py` | 城市/数字/查询词替换 + mock 重算 expected_result（252 → 804 样例，0 结果不匹配）|
| 训练 CLI | `scripts/train_sft.py` | 复用 DenseTransformer + AdamW + warmup_cosine；`--init-checkpoint` 支持预训练初始化 |
| 生成 CLI | `scripts/generate_sft_tool.py` | checkpoint → assistant 续写 |
| 评测 CLI | `scripts/eval_sft_tool.py` | 模型生成 → 提取 tool_calls → P1-05 八级分类器 → 失败分布 |
| 单测 | `architecture_lab/training/tests/test_sft_training.py` | 7 个（掩码覆盖/模板/批处理/masked loss/system 消息处理）|

### 训练配置与结果

- 数据：D1（126）+ D1.1（126）train 集 → 增强 k=3 → **804 样例**；
- 初始化：`dense-owt-formal-curve-medium.pt`（OWT 5000 步预训练，架构完全匹配）；
- 2000 步，lr 5e-5 warmup_cosine，batch 8，seq 256；
- **曲线**：train_loss 8.85 → 0.48；val_loss 8.91 → **1.76**（val_min @ 2000，未触底）；
- checkpoint：`artifacts/checkpoints/sft-tool-v1.pt`（gitignored）。

## 模型结果（诚实负结果）

**模型未能学会工具调用 JSON 生成**：

- 对 D1.1 训练集内样例（北京天气）生成：`'仅��结州�，可�用工具hesh...'`——中文碎片 + 重复噪声，无 JSON 结构；
- 对 D1 dev（held-out）13 样例评测：**13/13 parse_success=False**（八级分类器第一层即失败）；
- 对 D1.1 全 126 样例：20/20（抽样）parse_success=False；
- 唯一学到的模式：偶尔输出 `{"call_id"` / `"name"` / `"arguments"` 片段，但整体 JSON 不合法。

**失败分析（4M 模型 + 2000 步）**：

1. **欠拟合确认**：训练集内样例（模型背过 2000 步）也无法生成正确输出——非泛化问题，是容量/训练量不足；
2. **任务难度**："中文自然语言请求 → 结构化 JSON 工具调用"要求模型同时掌握中文理解 + JSON 语法 + 工具 schema 映射，对 d_model=128（4M）模型过难；
3. **模板尝试**：尝试 4 种配置（从零/预训练 init、无增强/增强、`[` 数组/换行分隔）均退化；起始 token 理论（`[` 罕见）修复后无改善；
4. **val loss 信号**：val 停在 1.76（train 0.48），过拟合与欠拟合并存——小数据 + 小模型的天花板。

## 评测闭环（基础设施可用）

`scripts/eval_sft_tool.py` 完整跑通：模型生成 → tool_calls 提取（数组 + 换行对象双格式）→ P1-05 八级分类 → 失败分布 JSON：

```json
{"total": 13, "first_failure_distribution": {"parse_success": 13}, "parse_success_rate": 0.0}
```

该结果如实反映模型能力（parse 层全失败），证明评测链路能消费任意模型输出并定位失败层。

## 不构成正式结论

- ❌ 不声称"SFT 提升了工具调用能力"——模型未学会；
- ❌ 不把 train loss 0.48 表述为质量指标——生成验证失败；
- 本阶段结论：**管线可用，模型需要更大容量（≥ 20M）或更多真实数据（≥ 5000 样例）或更长训练（≥ 2 万步）**。

## 复现

```bash
# 训练（需 OWT 预训练 checkpoint 存在）
.venv/python.exe scripts/train_sft.py --config configs/sft.example.yaml \
    --init-checkpoint artifacts/checkpoints/dense-owt-formal-curve-medium.pt

# 生成测试
.venv/python.exe scripts/generate_sft_tool.py \
    --checkpoint artifacts/checkpoints/sft-tool-v1.pt --user "请帮我查一下北京天气"

# 评测
.venv/python.exe scripts/eval_sft_tool.py \
    --checkpoint artifacts/checkpoints/sft-tool-v1.pt \
    --samples-dir datasets/tool-calling-d1/dev
```

## 下一步（真实改进路径）

1. **更大模型**：d_model 256+（≥ 20M params）——当前 4M 是明确瓶颈；
2. **更多数据**：D1.1 生成器再产出数千条（真实 LLM 采样，成本可控）或模板级数据增强；
3. **更长训练 + early stopping**：val 未触底（1.76 @ 2000），2 万步可能收敛；
4. **评测闭环复用**：eval_sft_tool.py + P1-05 分类器可直接用于任何新 checkpoint。
