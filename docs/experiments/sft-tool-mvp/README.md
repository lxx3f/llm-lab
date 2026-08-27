# SFT 工具调用训练 MVP（D1 + D1.1）

> 状态：管线交付完成，**模型结果为诚实负结果**（2026-08-27）。
>
> 本实验实现 SFT 工具调用训练闭环（数据构造 → 增强 → 训练 → 生成 → 八级分类器评测），并如实记录：**medium (2.10M) 与 large (5.11M) Dense 模型在 2000 步内均未能学会可靠的工具调用 JSON 生成**。管线与评测基础设施可用，模型能力未达标。

## 交付内容

### 训练管线（完整可用）

| 模块 | 文件 | 说明 |
|---|---|---|
| SFT 数据构造 | `architecture_lab/training/sft_training.py` | tool_calling_sample → 掩码序列（user 不参与 loss，assistant 参与）；模板 `### User / ### Assistant / ### Result`；换行分隔 JSON 对象（无数组包裹，匹配 tokenizer 合并的 `{"` token）；`_find_user_turn` 正确处理 system 消息；checkpoint 中保存 `model_config` 供后续脚本自动读取 |
| 确定性增强 | `architecture_lab/training/sft_augment.py` | 城市/数字/查询词替换 + mock 重算 expected_result（252 → 804 样例，0 结果不匹配）|
| 训练 CLI | `scripts/train_sft.py` | 复用 DenseTransformer + AdamW + warmup_cosine；`--init-checkpoint` 支持预训练初始化 |
| 生成 CLI | `scripts/generate_sft_tool.py` | checkpoint → assistant 续写（自动从 ckpt 读 model_config）|
| 评测 CLI | `scripts/eval_sft_tool.py` | 模型生成 → 提取 tool_calls → P1-05 八级分类器 → 失败分布 |
| 单测 | `architecture_lab/training/tests/test_sft_training.py` | 7 个（掩码覆盖/模板/批处理/masked loss/system 消息/增强一致性）|

### 训练结果对比（两次实跑）

| 模型 | 参数量 | 预训练 init | 2000 步后 val_min | 生成质量 |
|---|---|---|---|---|
| medium | 2.10M | dense-owt-formal-curve-medium.pt (5000 步) | **1.76** | 乱码+控制字符，无结构 |
| **large** | **5.11M** | **dense-owt-formal-curve-large.pt (50000 步)** | **1.18** | **学到模板结构（### Result + JSON 片段），但默认 no_tool 路径** |

- **large 改进点**：val_loss 降低 0.58 nats；1/13 评测样例通过 parse 层进入 `argument_value_correct` 层（medium 0/13）
- **large 残留问题**：模式坍缩到 `### Result（无工具调用）直接回答...`；偶发片段输出有效 JSON 但参数错误

## 模型结果（诚实负结果）

**5.11M 模型仍未可靠学会工具调用 JSON 生成**：

- D1 dev（13 样例，在训练集内）评测：
  - **12/13 parse_success=False**
  - 1/13 通过到 `argument_value_correct`（生成了有效 JSON `{"call_id": "call-d1-007-b", "name": "d1_web_search", "arguments": {"query": "北京"}}`，但 query 错误：GRPO 强化学习 → 北京）
  - 0/13 完整通过 8 级分类
- D1.1 train（126 样例）评测：126/126 parse_success=False（样本更长更复杂，模式坍缩更严重）

**失败分析（5.11M 模型 + 2000 步 + 804 样例）**：

1. **欠拟合确认**：训练集内样例（背过 2000 步）也无法生成正确输出 → 非泛化问题，是容量/训练量/数据量不足；
2. **模式坍缩**：模型学到模板骨架但默认走 `no_tool` 路径——未学到"用户请求类型 → 工具触发"的映射；
3. **任务难度**："中文自然语言请求 → 工具选择 + 参数提取 + 结构化 JSON 输出"对 5.11M 模型仍过难；
4. **已尝试修复**：medium→large、换模板（数组→换行对象）、增强（252→804）、system 消息处理——任一变量单独修复均无法突破。

## 评测闭环（基础设施可用）

`scripts/eval_sft_tool.py` 完整跑通：模型生成 → tool_calls 提取（数组 + 换行对象双格式）→ P1-05 八级分类 → 失败分布 JSON：

```json
{"total": 13, "first_failure_distribution": {"parse_success": 12, "argument_value_correct": 1}, "no_failure": 0, "parse_success_rate": 0.077}
```

8 级分类器可定位失败层（parse_success / argument_value_correct / ...），即使模型能力不足也能精确报告问题层级。

## 不构成正式结论

- ❌ 不声称"SFT 提升了工具调用能力"——5.11M 模型同样未学会；
- ❌ 不把 train loss 0.24 / val loss 1.18 表述为质量指标——生成验证失败；
- 本阶段结论：**管线完整可用，5.11M 是当前 OWT 预训练 checkpoint 中最大规模；进一步突破需要更多真实数据（≥ 5000 样例）或更长训练（≥ 2 万步）或更大模型（d_model 256+ 需重新训 OWT 预训练）**。

## 复现

```bash
# medium 版本（2.10M）
.venv/python.exe scripts/train_sft.py --config configs/sft.example.yaml \
    --init-checkpoint artifacts/checkpoints/dense-owt-formal-curve-medium.pt

# large 版本（5.11M，推荐起点）
.venv/python.exe scripts/train_sft.py --config configs/sft-large.example.yaml \
    --init-checkpoint artifacts/checkpoints/dense-owt-formal-curve-large.pt

# 生成测试
.venv/python.exe scripts/generate_sft_tool.py \
    --checkpoint artifacts/checkpoints/sft-tool-large-v1.pt --user "请帮我查一下北京天气"

# 评测
.venv/python.exe scripts/eval_sft_tool.py \
    --checkpoint artifacts/checkpoints/sft-tool-large-v1.pt \
    --samples-dir datasets/tool-calling-d1/dev
```

## 下一步（真实改进路径）

1. **更多数据**（最高 ROI）：D1.1 生成器再产出数千条（真实 LLM 采样，成本可控）；
2. **更长 SFT 训练**：val 未触底（1.18 @ 2000），2 万步可能收敛；总耗时 ~2h（large 22 ms/step × 20000 / 60）；
3. **d_model=256+ 模型**：12M+ params，~15 min SFT 但需先 OWT 预训练 ~50 min，总成本 ~65 min；
4. **评测闭环复用**：eval_sft_tool.py + P1-05 分类器可直接用于任何新 checkpoint。