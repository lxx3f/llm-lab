# 项目路线图

> 单一权威的总体计划文档。
>
> 问题和决策仍然记录在 [`docs/plans/open-issues.md`](./open-issues.md)。
> 阶段审查记录在 [`docs/plans/reviews/`](./reviews/)。

## 总体目标

构建可复现的 LLM 实验闭环：

```text
模型架构实现 → 数据构造与处理 → 训练/微调 → vLLM 部署 → 统一评测 → 失败分析
```

闭环中每一节都先建立最小可复现实验，再扩展规模；不允许把“使用开源框架”本身当作项目成果，重点记录修改、验证过程和实验结果。

## 已完成阶段（2026-08-29 之前的累计交付）

| 阶段 | 范围 | 审查记录 |
|---|---|---|
| Stage 0 数据协议 | tool calling / model output / evaluation result JSON Schema | 见 reviews/stage-* 系列 |
| BPE tokenizer artifact | OWT BPE 训练、CLI、artifact metadata、版本化目录 | `stage-bpe-tokenizer.md` |
| OWT tokenizer 优化 | 优化 BPE 训练和合并 | `stage-owt-bpe-optimized.md` |
| OWT tokenizer 流式化 | 大文件 newline-aligned 流式训练 | `stage-owt-bpe-streaming.md` |
| OWT 数据登记 | 数据来源、许可、版本、hash | `stage-owt-data-source.md` |
| Token cache 协议 | UTF-8 行流式、uint16 little-endian、hash 绑定 | `stage-owt-token-cache.md` |
| Dense training MVP | Dense forward + tokenizer + 1MiB smoke | `stage-dense-training-mvp.md` |
| OWT 正式 cache | 512 MiB train / 64 MiB validation cache | `stage-owt-formal-cache.md` |
| Dense 优化与 schema | scheduler / AMP / 梯度累积 / 结果 schema | `stage-dense-optimization-schema.md` |
| 测试分层 | fast / module / full 三档入口 | review 不需要（工程改进） |
| MoE Top-1 训练闭环 | MoE 配置加载、active loss + aux loss、checkpoint/resume、result schema | `stage-moe-top1-training.md` |
| Dense/MoE 公平对比协议 | 相同 total params / 相同 active params 两套协议、统一 benchmark/routing stats schema、实际 A/B smoke | `stage-n2-dense-moe-fairness.md` |
| 统一 benchmark/result 元数据 | git commit / config hash / CUDA/PyTorch/GPU compute capability / seed；4 个 schema metadata 块 + 4 个 CLI 注入 | `stage-n3-unified-metadata.md` |
| Dense 正式训练曲线 | baseline 0.66M + medium 2.10M 各 5000 步；train_losses + validation_losses 序列 + curve_summary；matplotlib PNG 曲线 | `stage-n4-dense-formal-curve.md` |
| Dense 规模 sweep | 在 N4 基础上增加 small 1.23M + large 5.11M，4 规模点 + 4-curve overlay PNG + 规模敏感性表 | `stage-n5-dense-scale-sweep.md` |
| Dense dropout 消融 | medium 2.10M × dropout ∈ {0.0, 0.1, 0.2}，3 dropout 点 + 3-curve overlay PNG + dropout sensitivity 表 | `stage-n6-dense-dropout-sweep.md` |
| Dense RoPE base 消融 | medium 2.10M × rope_base ∈ {10000, 50000, 100000}，3 rope_base 点 + 3-curve overlay PNG + RoPE sensitivity 表 | `stage-n7-dense-rope-sweep.md` |
| Dense n_heads 消融 | medium 2.10M × n_heads ∈ {2, 4, 8}，3 head 数点 + 3-curve overlay PNG + attention head sensitivity 表 | `stage-n8-dense-heads-sweep.md` |
| Dense d_ff 消融 | medium × d_ff ∈ {256, 512, 1024}，3 d_ff 点 + 3-curve overlay PNG + FFN hidden sensitivity 表 | `stage-n9-dense-dff-sweep.md` |
| MoE 5000 步训练曲线 | MoE Top-1 5000 步（total 2.10M / active 1.51M）；MoE schema v1.1 对齐 Dense；MoE vs Dense medium 对比 | `stage-moe-owt-formal-curve.md` |
| Dense 长训练曲线 | baseline + medium 各 50000 步（5000 步的 10×）；50000 步 val_min 改善 1.2-1.5 nats；log_interval=500 / validation_interval=2000 | `stage-n11-dense-long-curve.md` |
| **P1-02 Mock 工具执行器**（历史阶段文档文件名为 `stage-p1-mock-executor.md`，早期正文曾写作 P1-01） | 纯进程内 mock executor；tool_execution_result schema v1.0；CLI + example + 8 单测；108 tests | `stage-p1-mock-executor.md` |
| D1 模板工具调用数据集 | 126 样本 8 task_type；D0 manifest；mock_executor 闭环验证；11 单测 | (内嵌于 sft-tool-mvp stage review) |
| D1.1 LLM 生成工具调用数据集 | **1500 个 train 样本**；6 task_type 各 250；初始 126 + 夜间扩展 1374；混合 per-file source provenance；aggregate sha256；MANIFEST 重建脚本 | (内嵌于 sft-tool-mvp stage review) |
| P1-03 多 seed 评测协议 | `scripts/run_multi_seed.py` (Dense + MoE 自动 dispatch)；总体 std；supplemental 测试 | (内嵌于 sft-tool-mvp stage review) |
| P1-04 数据版本 D0/D1 | `docs/protocols/p1-04-data-version-d0.md`；D1 数据集目录 + MANIFEST；filter 脚本 | (内嵌于 sft-tool-mvp stage review) |
| P1-05 八级分类器 | parse_success → schema_valid → tool_name_correct → argument_value_correct → call_plan_matches → execution_success → result_grounded → final_answer_correct；含 `_norm_name`/`_norm_args` 鲁棒性；`scripts/classify_tool_failure.py`；44 单测 | (内嵌于 sft-tool-mvp stage review) |
| SFT 工具调用训练 MVP | Dense + MoE 训练管线（数据构造 → 增强 → 训练 → 生成 → 评测）；5 次实跑对比（medium/large/large-night/d256×2/MoE）；多 seed eval 聚合器；185 单测；诚实负结果 | `stage-sft-tool-mvp.md` |
| P2 确定性 evaluator | `schemas/reward_signal.schema.json`（v1.0，含 reward_type 4 值 enum + 文档说明每个值映射的 P1-05 路径）；`scripts/reward_offline.py`（reward_binary + reward_layered + reward_type 主导通道；_dominant_reward 鲁棒性处理矛盾输入 / 未知 failure channel）；5 个 SFT MVP checkpoint × D1 dev 13 样本 = 65 reward_signal 全 schema 校验通过；**37** 单测；与 P1-05 八级分类器对齐 | `stage-p2-evaluator.md` |
| P3 D2 多轮对话数据集 | **当前契约（round 14；HEAD：当前 main）**：`schemas/d2_multi_turn_sample.schema.json`（6 个规范 task_type）；`scripts/generate_d2_dataset.py` 生成 **5000 样本**（4 类 833 + 2 类 834）= **train 3500 / dev 750 / test 750**；按 task_type 分层的 seeded shuffle（70/15/15，round 14 改用 `round()` 化 dev/test 严格命中 125/125）+ 真实 `MockExecutor.execute_sequence()` + depends_on 拓扑验证 + 跨 message well-formedness 校验；`canonical_content_signature()` 全局语义去重（**5000/5000 unique、6 类 ≥833 unique**、三 split overlap=0）+ `cross_dataset_signature()` 跨 D1/D1.1/D2 可比投影；`tests/test_d2_dataset.py` **49 + 2 = 51** 单测；P2 阶段自研 5 ckpt 历史评测归档于 `docs/plans/open-issues.md` P3-01 段（line 769-841）；`docs/protocols/d2-multi-turn.md`；与 D1/D1.1 train id + canonical semantic 双重互斥。 | `stage-p3-d2-multi-turn.md` |
| P4 GRPO MVP + 小规模正确性实验 | GRPO 训练 + advantage + policy 更新闭环；M1/M2/M3 mock 数据集 + 真实 Dense 模型；`scripts/grpo_train.py` / `scripts/grpo_mocks.py` / `scripts/grpo_experiment/`；`tests/test_grpo_mvp.py` 单测；3-seed 小规模实验（round 9/10/11 修复 reward 归一化 + dataset 维度）；detached auditor 11 轮 PASS | `stage-p4-grpo-mvp.md` + `stage-p4-grpo-smoketest.md` |
| P5-02 Transformers 后端接入 | 5 公开 instruction-tuned 模型（SmolLM2-360M/1.7B-Instruct + Qwen2.5-0.5B/1.5B/3B-Instruct）；`scripts/eval_transformers.py`；AutoModelForCausalLM + 官方 chat_template + greedy + bf16；round 2 修复 target-answer 泄漏（`_strip_terminal_assistant()`）；reward_binary = 0 诚实负结果 + reward_layered 0.33–0.42 显著高于自研 5 ckpt；`tests/test_transformers_backend.py` 25 单测（含 `GoldAnswerLeakageTests` 反向断言） | `stage-p5-02-transformers-backend.md` |
| P5-03 vLLM 后端接入（公开模型） | 仅公开模型；vLLM 0.27.1 依赖 + WSL2/Docker smoke PASS；`scripts/eval_vllm.py`；不包含自研模型完整 vLLM 适配 | `stage-p5-03-vllm-feasibility.md` |
| 最终全链路审查 round-20 | 全栈 audit (tests / schemas / doc↔artifact / gitignore / working tree / reviewer evidence)；`docs/reports/final-audit.md` 落盘；round-13 / round-17 reviewer evidence docs | `docs/reports/final-audit/` |
| B-audit 22 轮 Open-issues 关闭 | 通读 `docs/plans/open-issues.md` 全部 sections；P0/P1/P2/P3 中已交付 items close-out + commit/date 证据；新增 sections (P4 GRPO / P5-02 / P5-03 / 最终全链路审查 round-20) 反映实际 open items；stage review `docs/plans/reviews/stage-B-open-issues-audit.md`；detached auditor 22 轮 PASS | `stage-B-open-issues-audit.md` |

这些阶段都已通过阶段审查（`minimax-cn/MiniMax-M3` reviewer），不允许回退。

## 当前阶段：路线图刷新（2026-08-29，B 完成之后）

**目标**：本节标记当前“已完成但 roadmap 中尚未显式记录”的状态，并指向下一阶段候选。上一版“当前阶段：P5 推理后端接入 + 大模型起点”已被下文的“已交付阶段：P4 GRPO + P5 后端”覆盖；本 roadmap 版本同步消除重复。

当前累计已交付（2026-08-29）：

| 子阶段 | 状态 | 交付 commit | stage review |
|---|---|---|---|
| P4 GRPO MVP + 小规模正确性实验 | ✅ 已交付 | `1fae6f0` + `99646fa` | `stage-p4-grpo-mvp.md` + `stage-p4-grpo-smoketest.md` |
| P5-01 开源 instruction-tuned 模型选定 | ✅ 已交付 | `bb61a3a` (与 P5-03 同交付) | `stage-p5-03-vllm-feasibility.md` |
| P5-02 Transformers 后端接入（公开模型） | ✅ 已交付 (round 2 修复 target-answer 泄漏后) | `63cbd83` + `b4fd879` + `66ff9eb` | `stage-p5-02-transformers-backend.md` |
| P5-03 vLLM 后端接入（公开模型） | ✅ 已交付 (WSL2 smoke PASS) | `bb61a3a` | `stage-p5-03-vllm-feasibility.md` |

P5 阶段（P5-01/P5-02/P5-03）已 100% delivered，**P5-04 双后端基准对比尚未启动**（planned only）。

## 下一阶段：5 个候选（按推荐顺序）

下列候选均**不在本轮 roadmap 范围内自动启动**；每项需要后续 list item 独立激活。

### 候选 1: GQA（Grouped-Query Attention，公开模型后端评估）

**动机**：P5-02/P5-03 已接入 5 个公开 instruction-tuned 模型；下一步可以**横向评估** GQA 在 tool calling 任务上是否相对 MHA 有收益（同 base model、训练量、推理 backend）。无需实现新架构，仅做 protocol + 数据采集。

**范围**（候选，**未启动**）：
- 选定 1-2 个 GQA-only 模型（候选：Qwen2.5 系列已用 GQA，可与 LLaMA-3 系列 MHA-only 对比；或同 base 模型用 GQA vs MHA 两版）；
- 在 P5-02 benchmark evaluation subset（同 D2 dev 750 采样的子集）上评测；复用 P1-05 8 级分类器 + P2 reward offline；与 P5-02 round 2 结果并列对比；
- 输出 GQA vs MHA 4 轴对比表（latency / throughput / reward_binary / reward_layered）；
- 不修改自研模型；不重跑训练。

**退出条件**：`docs/experiments/gqa-vs-mha/README.md` + eval JSON + reward JSON 落盘；`docs/protocols/` 新增 GQA 评估协议。

**风险**：开源 GQA-only 模型 与 MHA-only 模型同 base 的 pair 难以找到（多数模型架构已固定）；可能退化为“不同 base 模型对比”，降低结论强度。

### 候选 2: 简化 MLA（Multi-Latent Attention）实验

**动机**：自研模型架构上限为 12M；MLA（DeepSeek-V2 风格）是 P0-04 阶段规划但**未交付**的扩展方向。本轮 roadmap 中 MLA 仍属"暂缓"列表（见下）；如启动简化版（latent dim 缩减 + 单层 share），可在自研模型规模内验证 MLA 收益。

**范围**（候选，**未启动**）：
- 实现 MLA 简化版：`architecture_lab/models/mla_dense.py`，latent dim = d_model / 4；KV 共享单层；
- 与 Dense baseline（n4）+ Dense dropout 0.1（n6） + Dense RoPE 50k（n7）做 4-way 对比；
- 5000 步 OWT 训练；同 medium 2.10M 总参数规模；
- 输出 MLA vs Dense 4-way 对比表 + latent dim 消融。

**退出条件**：`docs/experiments/mla-simplified/` 落盘；`docs/protocols/mla-simplified.md` 协议 + stage review。

**风险**：MLA 在 2.10M 规模收益可能不显著；建议在 5.11M（large N5）规模再做一次。

### 候选 3: P5-04 双后端基准对比（Transformers vs vLLM）

**动机**：P5-02（Transformers） + P5-03（vLLM） 已交付；缺**同一 baseline**对比。P5-04 是 roadmap 原计划但**尚未启动**。

**范围**（候选，**未启动**）：
- 同一样本（P5-02 benchmark evaluation subset，D2 dev 750 采样）+ 同一 P1-05 分类器 + P2 reward offline；
- 5 个公开模型（SmolLM2-360M/1.7B + Qwen2.5-0.5B/1.5B/3B）× 2 后端（Transformers greedy bf16 + vLLM 0.27.1）= 10 组合；
- 4 轴：latency (ms/sample) + throughput (samples/s) + reward_binary + reward_layered；
- 输出双后端 4 轴对比表 + 同模型 Δ%。

**退出条件**：`scripts/eval_backend_comparison.py` CLI + `docs/experiments/p5-04-backend-comparison/README.md` 落盘；`docs/protocols/backend-comparison.md` 协议 + stage review。

**风险**：vLLM 在小 batch / small model 上可能不显著优于 Transformers；建议同时跑 batch=1 与 batch=4 两组。

### 候选 4: N12 3-seed 复现 + 超长曲线（200k 步，N13）

**动机**：N12 已交付 100k 步单 seed baseline + medium（`docs/experiments/n12-dense-ultra-curve/README.md`），验证了 val_min 仍可改善 -0.31 至 -0.33 nats。但 N12 是单 seed，统计意义受限。本候选把 N12 单 seed 扩展为 3-seed 复现（验证 val_min 改善是否为单 seed noise）；另外 N13 可进一步尝试 200k 步超长曲线，观察 transformer 是否进入 plateau。

**范围**（候选，**未启动**）：
- 复用 N11/N12 dense 长曲线协议；增加 100000 步 baseline + medium × 3 seed (= N12 3-seed follow-up)；
- log_interval=1000 / validation_interval=5000（避免 validation 过密）；
- 3-seed val_min aggregation：mean / std / per-seed 改善；与 N12 单 seed 数字交叉对比，判断单 seed 是否为 noise；
- 可选 N13 追加 200000 步 baseline 单点（如 GPU 资源允许；参考 N12 100k 3.5 小时表动，200k 预估 7 小时）；不需 3-seed，只验证 plateau 假设。

**退出条件**：`docs/experiments/n13-dense-100k-3seed/` + `docs/protocols/n13-dense-100k-3seed.md` + stage review (N12 当前缺 stage review，本次 N13 提交时同时为 N12 补交 stage review)。

**风险**：100k 步 × 3 seed × 2 规模点 (baseline + medium) = 6 个长 run，总时长预估 3-4 × 6 ≈ 20 小时；需先评估 host GPU 时间预算。200k 步超长曲线（若推进）需提前评估是否值得。

### 候选 5: 真实 OWT 评测（不是 D2 sampling 子集）

**动机**：当前所有 P5-02/P5-03 评测都在 P5-02 benchmark evaluation subset（**D2 dev 750 采样的子集**，不是 D2 全集），**从未在真实 OWT 上评测**。OWT 是真实分布的预训练数据；模型在 OWT 上 evaluate 是大模型常用做法。

**范围**（候选，**未启动**）：
- 在 `datasets/owt-sample/` 或 `owt-formal/` 上评估公开模型；
- 计算 per-token loss（perplexity 形式）+ 5 模型对比表；
- 不涉及 D2 / tool calling；纯 LM 评估；
- 输出 `docs/experiments/owt-real-eval/README.md` + per-token-loss 表。

**退出条件**：OWT 数据源 + 5 模型 per-token loss 落盘 + 对比表 + stage review。

**风险**：OWT 数据未在仓库（gitignored）；需保证路径 hash 与 manifest 一致。

## 推荐激活顺序

按"实现路径连续性 + 已交付基础复用率"排序：

1. **候选 3 (P5-04 双后端基准对比)** — 直接复用 P5-02/P5-03 已交付的 eval scripts，仅补一个对比 CLI。**最高优先级**，可在 1-2 个 list item 内完成。
2. **候选 1 (GQA 公开模型对比)** — 同样复用 P5-02 eval script + P1-05 + P2。**次高优先级**，但依赖"找得到 GQA vs MHA 同 base 对"先决条件。
3. **候选 5 (真实 OWT 评测)** — 简单独立任务，可作为候选 3 完成后补充。**低-中优先级**。
4. **候选 4 (长训练曲线 100k)** — 需要 GPU 资源预算；**中优先级**，建议先评估 host 资源。
5. **候选 2 (简化 MLA)** — 涉及自研模型架构改动，工作量最大；**最低优先级**，且 MLA 在 2.10M 是否有效尚不确定。

如 list item 决策推进，建议从候选 3 启动。

## 暂缓阶段（明确不在 roadmap）

明确以下不在当前路线图内，避免范围蔓延：

- Expert Parallel / 分布式训练；
- 完整工业级 MLA（非简化版）；
- 多模态数据；
- 复杂联网 Agent；
- 大规模数据采集；
- 复杂 LLM Judge；
- 自研模型完整 vLLM 适配；
- Expert Parallel / 复杂 capacity 调度；
- 长上下文评测；
- 完整 RLHF / DPO；
- 模型量化（INT8/INT4 部署优化）。

如后续需要，重新评估并写入下一阶段。

## Reviewer 触发条件

阶段审查只在大阶段完成时触发，**不为局部改动触发**。触发条件：

- 一个 README 阶段完成；
- 一个 MVP 完成；
- 一个模型模块或数据管线模块完成；
- 一组对比实验完成；
- 计划顺序发生重大调整（如本节"下一阶段：5 个候选"的激活）。

不触发条件：

- 局部函数、bug、文档、测试、schema 微调；
- 单个配置或 CLI 参数调整；
- 单个实验 smoke 重跑；
- 工程化改进（如测试分层、CI、依赖锁定）；
- 性能数字更新但协议不变。

完整定义见 [`docs/plans/review-process.md`](./review-process.md)。

## 关联文档

- 当前 P0/P1/P2/P3 问题：[`open-issues.md`](./open-issues.md)
- 阶段审查流程和记录：[`review-process.md`](./review-process.md) 与 [`reviews/`](./reviews/)
- 实验记录：[`../experiments/`](../experiments/)
- 协议规范：[`../protocols/`](../protocols/)
