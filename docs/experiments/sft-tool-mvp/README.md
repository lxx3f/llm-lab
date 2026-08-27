# SFT 工具调用训练 MVP（D1 + D1.1）

> 状态：管线交付完成，**模型结果为诚实负结果**（2026-08-28）。
>
> 本实验实现 SFT 工具调用训练闭环（数据构造 → 增强 → 训练 → 生成 → 八级分类器评测），并如实记录：**medium (2.10M) / large (5.11M) / large + 5288 样本 + 20000 步三次实跑均未能学会可靠的工具调用 JSON 生成**。管线与评测基础设施可用，模型能力未达标。

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
| 夜间编排 | `scripts/night_run_sft.py` | 自动 fallback 扩展数据集 → 长训练，串行执行 |

### 三次实跑结果对比

| 版本 | 模型 | 数据 | 步数 | val_min | D1 dev 评测 | 状态 |
|---|---|---|---|---|---|---|
| **MVP v1** | medium 2.10M | 804 (aug k=3) | 2000 | 1.76 | 0/13 parse | 乱码/控制字符，无结构 |
| **MVP v2** | large 5.11M | 804 (aug k=3) | 2000 | 1.18 | 1/13 → argument_value | 学到模板骨架，模式坍缩到 no_tool |
| **night run** | large 5.11M | **5288** (aug k=3, 12× 数据) | **20000** | **0.33** | **0/13 parse** | val_loss 显著下降但生成更退化（严重过拟合 + 模式坍缩）|

## 模型结果（诚实负结果）

**三次实跑均未达到 SFT 目标**：

- **MVP v1**：val 1.76 → 生成乱码（控制字符 `\x18` 循环）
- **MVP v2**：val 1.18 → 学到模板骨架但默认走 no_tool 路径；1/13 进入 argument_value 层（参数错误）
- **night run**：val **0.33**（训练样本记忆）→ 生成反而更退化，**D1 dev 0/13 + D1.1 train 30/30 全 parse_fail**；输出"（无工具调用）当前可用工具列表中..."模板的无限重复

**关键诊断**：
1. **过拟合确认（night run）**：val_loss 0.33 但生成完全崩溃，说明 masked loss 在 assistant span 上极低只是模型记住了训练数据，并未真正学会"用户请求 → 工具调用"的映射
2. **模式坍缩加深**：步数从 2000 → 20000 + 数据 12× → 模型对训练集里"工具不可用"模板的背诵更深，导致推理时永远走 no_tool 路径
3. **任务难度**：5.11M 模型 + 5288 增强样本的组合在架构层面可能就不足以泛化

## 评测闭环（基础设施可用）

`scripts/eval_sft_tool.py` 完整跑通：模型生成 → tool_calls 提取（数组 + 换行对象双格式）→ P1-05 八级分类 → 失败分布 JSON。

D1 dev night run 评测 JSON（13 样例）：
```json
{"total": 13, "first_failure_distribution": {"parse_success": 13}, "no_failure": 0, "parse_success_rate": 0.0}
```

8 级分类器精确报告失败层级，**即使模型能力不足也能量化问题**。

## 不构成正式结论

- ❌ 不声称"SFT 提升了工具调用能力"——三次实跑均未达可用状态
- ❌ 不把 val_loss 0.33 表述为质量指标——对应生成完全崩溃
- 本阶段结论：**管线完整可用；5.11M 模型在 2000-20000 步范围内 + 252-5288 增强样本范围内均无法可靠学会该任务**

## 复现

```bash
# 单次训练（medium）
.venv/python.exe scripts/train_sft.py --config configs/sft.example.yaml \
    --init-checkpoint artifacts/checkpoints/dense-owt-formal-curve-medium.pt

# 单次训练（large，推荐起点）
.venv/python.exe scripts/train_sft.py --config configs/sft-large.example.yaml \
    --init-checkpoint artifacts/checkpoints/dense-owt-formal-curve-large.pt

# 夜间完整计划（数据扩展 + 长训练）
D1_LLM_API_KEY=<key> .venv/python.exe scripts/night_run_sft.py \
    --target 1500 --train-steps 20000

# 生成测试
.venv/python.exe scripts/generate_sft_tool.py \
    --checkpoint artifacts/checkpoints/sft-tool-large-v1.pt \
    --user "请帮我查一下北京天气"

# 评测
.venv/python.exe scripts/eval_sft_tool.py \
    --checkpoint artifacts/checkpoints/sft-tool-large-v1.pt \
    --samples-dir datasets/tool-calling-d1/dev
```

## 下一步（真实改进路径）

1. **架构层面突破**：d_model=256+ 模型 + 全新 OWT 预训练（12M+ params，~65 min 总成本）；当前 5.11M 可能在架构层就不足以胜任该任务
2. **数据质量**：放弃确定性增强（导致重复模板过拟合），改用 D1.1 生成器产出更多**自然变化**的真实 LLM 样本（已有 minimax/deepseek provider 自动 fallback）
3. **训练策略**：更短训练 + early stopping（night run 20000 步过拟合，5000 步可能更稳定）；或尝试 MoE 架构分散容量
4. **评测闭环复用**：eval_sft_tool.py + P1-05 分类器可精确量化任意新 checkpoint 的失败层级，无需人工检查生成文本