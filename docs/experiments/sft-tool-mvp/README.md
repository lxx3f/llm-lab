# SFT 工具调用训练 MVP（D1 + D1.1）

> 状态：管线交付完成，**模型结果为诚实负结果**（2026-08-28）。
>
> 本实验实现 SFT 工具调用训练闭环（数据构造 → 增强 → 训练 → 生成 → 八级分类器评测），并如实记录：**五次不同架构/数据/步数的实跑均未能学会可靠的工具调用 JSON 生成**。管线与评测基础设施可用，模型能力未达标。

## 交付内容

### 训练管线（完整可用）

| 模块 | 文件 | 说明 |
|---|---|---|
| Dense SFT 数据构造 | `architecture_lab/training/sft_training.py` | tool_calling_sample → 掩码序列；模板 `### User / ### Assistant / ### Result`；换行分隔 JSON 对象（匹配 tokenizer 合并的 `{"` token）；`_find_user_turn` 正确处理 system 消息 |
| MoE SFT 数据构造 | `architecture_lab/training/sft_moe_training.py` | 复用 Dense 数据构造 + 用 MoETransformer；aux_loss 加权 0.01 |
| 确定性增强 | `architecture_lab/training/sft_augment.py` | 城市/数字/查询词替换 + mock 重算 expected_result |
| Dense 训练 CLI | `scripts/train_sft.py` | 复用 DenseTransformer + AdamW + warmup_cosine |
| MoE 训练 CLI | `scripts/train_sft_moe.py` | 复用 MoETransformer + aux_loss |
| 生成 CLI | `scripts/generate_sft_tool.py` | 自动检测 Dense/MoE 架构（从 ckpt 读 model_config + moe_config）|
| 评测 CLI | `scripts/eval_sft_tool.py` | 自动检测 Dense/MoE 架构 + 提取 tool_calls（数组 + 换行对象双格式）→ P1-05 八级分类器 |
| 单测 | `architecture_lab/training/tests/test_sft_training.py` + `test_sft_moe_training.py` | 10 个（掩码/模板/批处理/masked loss/MoE forward/aux loss/system 消息/混合类型鲁棒性）|
| 夜间编排 | `scripts/night_run_sft.py` | 自动 fallback 扩展数据集 → 长训练 |
| OWT 训练 | `scripts/train_dense.py` + `scripts/train_moe.py` | Dense / MoE 预训练 |

### 五次实跑结果对比

| # | 架构 | 参数量 | OWT 预训练 init | 数据 | 步数 | val_min | D1 dev (13样例) |
|---|---|---|---|---|---|---|---|
| 1 | Dense medium | 2.10M | 5000 步 | 804 (aug k=3) | 2000 | 1.76 | 0/13 parse |
| 2 | Dense large | 5.11M | 50000 步 | 804 (aug k=3) | 2000 | 1.18 | **1/13 → argument_value_correct** |
| 3 | Dense large | 5.11M | 50000 步 | 5288 (aug k=3) | 20000 | **0.33** | 0/13 parse（严重过拟合 + 模式坍缩）|
| 4 | Dense **d256** | 12.0M | 5000 步 | 1500 | 5000 × **3 seeds** | **0.6824 ± 0.0027** | 0/13 parse × 3 |
| 5 | MoE (4 experts) | ~2M active | long-seed42 50000 步 | 1500 | 2000 | 1.94 | 0/13 parse |

**五次实跑全 D1 dev 0/13 通过 8 级分类**（除第2次 1/13 突破到 argument_value_correct 层）。

## 模型结果（诚实负结果）

**五次实跑均未达到 SFT 目标**：

- **MVP v1 (medium)**：val 1.76 → 生成乱码（控制字符 `\x18` 循环）
- **MVP v2 (large)**：val 1.18 → 学到模板骨架但默认走 no_tool 路径；1/13 进入 argument_value_correct（参数错误）
- **night run (large+5288)**：val **0.33**（训练集记忆）→ 生成完全崩溃，输出"（无工具调用）"无限重复
- **d256 (12M, 3 seeds)**：val **0.6824 ± 0.0027**（pop std）→ 0/13 × 3 seeds；生成能列出工具名但仍走 no_tool 路径
- **MoE (4 experts, OWT long init)**：val 1.94 → 0/13 parse；输出 `\n` 控制字符循环

**关键诊断**：
1. **容量不是瓶颈**：5M → 12M 改进有限（val 0.33 vs 0.69），且都 0/13 parse
2. **步数不是瓶颈**：2000 → 5000 → 20000 均失败（增加步数反而加剧过拟合）
3. **架构不是瓶颈**：Dense 大模型 + MoE 多专家均失败
4. **数据量不是唯一瓶颈**：252 → 5288 → 1500 均失败
5. **预训练不是瓶颈**：5000 vs 50000 步 OWT 预训练 + MoE OWT init 都失败

**可能根因**：2M-12M 参数量级 + 252-5288 样例的组合**对"中文自然语言→结构化 JSON 工具调用"任务而言不够**。要突破需要：
- ≥20M 真实 LLM 参数（开源模型 scale）
- ≥5 万真实 SFT 样例（含多轮对话 + 错误恢复）
- 或采用 instruction-tuned 预训练模型作为起点

## 评测闭环（基础设施可用）

`scripts/eval_sft_tool.py` 完整跑通：Dense/MoE 自动检测 → tool_calls 提取（数组 + 换行对象双格式）→ P1-05 八级分类 → 失败分布 JSON。

8 级分类器精确报告失败层级，**即使模型能力不足也能量化问题**。

## Artifact Policy（本项目）

`artifacts/*-result.json`（训练曲线）、`artifacts/*-eval*.json`（评测 JSON）、`artifacts/checkpoints/*.pt`（模型权重）均为本地产物（gitignored by project policy，见 .gitignore）。它们可由 `scripts/train_sft.py` / `scripts/train_sft_moe.py` / `scripts/eval_sft_tool.py` 从提交的 config + dataset + commit 复现，不作为 committed artifact。

夜间计划完整状态摘要（providers_tried / 最终样本数 / 训练步数）存档于 `docs/experiments/sft-tool-mvp/night-summary.json`（tracked），原 `.pi-glla/scratch/night-summary.json` 为同名本地副本。

## 不构成正式结论

- ❌ 不声称"SFT 提升了工具调用能力"——五次实跑均未达可用状态
- ❌ 不把任何 val_loss 数值表述为质量指标——生成验证全部失败
- 本阶段结论：**本项目自研模型 2M-12M 规模在当前数据集/任务上未能学会可靠的工具调用**

## 复现

```bash
# OWT 预训练（d256）
.venv/python.exe scripts/train_dense.py --config configs/dense_training.owt-formal-curve-d256.example.yaml

# Dense SFT（d256 + 3 seeds）
.venv/python.exe scripts/train_sft.py --config configs/sft-d256.example.yaml \
    --init-checkpoint artifacts/checkpoints/dense-owt-formal-curve-d256.pt

# MoE SFT（OWT long init）
.venv/python.exe scripts/train_sft_moe.py --config configs/sft-moe.example.yaml \
    --init-checkpoint artifacts/checkpoints/moe-owt-formal-curve-long-seed42.pt

# 生成测试（自动检测 Dense/MoE）
.venv/python.exe scripts/generate_sft_tool.py \
    --checkpoint artifacts/checkpoints/sft-tool-d256-seed42.pt \
    --user "请帮我查一下北京天气"

# 评测
.venv/python.exe scripts/eval_sft_tool.py \
    --checkpoint artifacts/checkpoints/sft-tool-d256-seed42.pt \
    --samples-dir datasets/tool-calling-d1/dev
```

## 下一步（真实改进路径）

1. **架构层面突破**：≥20M 真实 LLM 参数（开源模型如 Qwen2.5-0.5B 或 LLaMA-3.2-1B）——自研 12M 已到架构上限
2. **数据质量**：放弃确定性增强（导致重复模板），改用纯 D1.1 生成器产出≥5000 真实变化样例
3. **预训练起点**：从 instruction-tuned 开源模型继续 SFT，而非自研 OWT base
4. **推理后端**：接入 Transformers/vLLM 标准开源模型做统一评测
5. **评测闭环复用**：eval_sft_tool.py + P1-05 八级分类器可精确量化任意新 checkpoint 的失败层级