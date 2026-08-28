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

## 已完成阶段

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
| P2 确定性 evaluator | `schemas/reward_signal.schema.json`（v1.0，含 reward_type 4 值 enum + 文档说明每个值映射的 P1-05 路径）；`scripts/reward_offline.py`（reward_binary + reward_layered + reward_type 主导通道；_dominant_reward 鲁棒性处理矛盾输入 / 未知 failure channel）；5 个 SFT MVP checkpoint × D1 dev 13 样本 = 65 reward_signal 全 schema 校验通过；**37** 单测（8 层各 ≥ 3 例 = 25 + classifier 一致性 4 + reward_type 映射 7 + CLI 聚合 1）；与 P1-05 八级分类器对齐 | `stage-p2-evaluator.md` |
| P3 D2 多轮对话数据集 | **当前契约（round 14，HEAD `fdfc519`）**：`schemas/d2_multi_turn_sample.schema.json`（6 个规范 task_type）；`scripts/generate_d2_dataset.py` 生成 **5000 样本**（4 类 833 + 2 类 834）= **train 3500 / dev 750 / test 750**；按 task_type 分层的 seeded shuffle（70/15/15，round 14 改用 `round()` 化 dev/test 严格命中 125/125）+ 真实 `MockExecutor.execute_sequence()` + depends_on 拓扑验证 + 跨 message well-formedness 校验（tool_call_id / name / arguments / ordering / expected_answer == final assistant content）；`canonical_content_signature()` 全局语义去重（**5000/5000 unique、6 类 ≥833 unique**、三 split overlap=0）+ `cross_dataset_signature()` 跨 D1/D1.1/D2 可比投影；`tests/test_d2_dataset.py` **49 + 2 = 51** 单测覆盖 schema / depends_on / canonical uniqueness / cross-dataset disjointness / transcript well-formedness / IID split distribution / 时间戳 / expected_answer 契约 / 5000/3500/750/750 显式断言 + `D2GeneratorDefaultContractTests`；`examples/d2_multi_turn/` 正负样例；P2 阶段自研 5 ckpt 历史评测归档于 `docs/plans/open-issues.md` P3-01 段（line 769-841）；`docs/protocols/d2-multi-turn.md`；与 D1/D1.1 train id + canonical semantic 双重互斥。 | `stage-p3-d2-multi-turn.md` |

这些阶段都已通过阶段审查（`minimax-cn/MiniMax-M3` reviewer），不允许回退。

## 当前阶段：P5 推理后端接入 + 大模型起点

目标：在 SFT MVP 上诚实记录模型能力未达标后，从自研 12M 模型架构上限跳出，转向**开源 instruction-tuned 模型作为 SFT 起点** + **Transformers / vLLM 推理后端接入**，以获得可对比的标准评测能力。

| 子阶段 | 范围 | 退出条件 |
|---|---|---|
| P5-01 开源模型选定 + 下载 | Qwen2.5-0.5B-Instruct 或 LLaMA-3.2-1B-Instruct；LICENSE + 模型卡记录；`.gitignore` 覆盖 | metadata.json 落盘 + 下载脚本 |
| P5-02 Transformers 后端接入（公开模型） | 仅针对公开 instruction-tuned 模型；Transformers backend；`scripts/eval_transformers.py` CLI；与自研模型同一 P1-05 8 级分类器 + P2 reward offline 对比；**扩样后的 D2 dev (750 样本) 子集 90 × 5 模型**（SmolLM2-360M/1.7B-Instruct + Qwen2.5-0.5B/1.5B/3B-Instruct）= 450 reward_signal；**2026-08-28 下午修复 target-answer 泄漏**：`b4fd879` `_strip_terminal_assistant()` 去除 terminal assistant content message 后重跑；5 模型 reward_binary 全部 = 0（诚实负结果，原版 SmolLM2-360M 0.0111 为伪信号），reward_layered 0.33–0.42 显著高于自研 5 ckpt；`tests/test_transformers_backend.py` **25** 单测（含 `GoldAnswerLeakageTests` 反向断言）；`docs/protocols/transformers-backend.md`；`docs/experiments/p2-evaluator/README.md` §7 横向对比 | eval JSON + reward JSON 落盘；stage review 已通过（HEAD `b4fd879`） |
| P5-03 vLLM 后端接入（公开模型） | 仅针对公开 instruction-tuned 模型；vLLM 依赖安装；`scripts/eval_vllm.py` CLI；WSL/Linux/Docker 环境；throughput 对比表；不包含自研模型完整 vLLM 适配 | vLLM 推理成功 + 吞吐数字 |
| P5-04 双后端基准对比 | 同一样本 + 同一 P1-05 分类器 + P2 reward offline；latency + throughput + reward 二元 + reward 分层 四轴 | 对比 README |

任一子阶段失败必须先更新本路线图和 `open-issues.md`。

## 下一阶段：P4 GRPO + P5 后端

| 阶段 | 范围 | 触发条件 |
|---|---|---|
| P4 GRPO | 基于 P1-05 reward signal + P2 offline reward 校验 + P3 多轮数据集；advantage 计算 + policy 更新 | P3 + P5-02 完成 |
| P5-01 开源 instruction-tuned 模型选定 | SmolLM2-360M/1.7B-Instruct + Qwen2.5-0.5B/1.5B/3B-Instruct；5 模型均在 HF mirror 下载并 cache 到 `artifacts/huggingface/`；LICENSE + 模型卡记录入 `docs/protocols/transformers-backend.md` §7/§8 | — |
| P5-02 Transformers 后端接入（公开模型） | 仅公开模型 + 当前扩样版 D2 dev 750 子集 90 上 5 模型评测；自研 5 ckpt 在同 split 同子集上的历史评测见 `docs/experiments/p2-evaluator/README.md` §6 | P3 完成（5000 样本交付 + stage review HEAD `fdfc519`）|
| D2 数据集 | 多轮对话 + 错误恢复；≥5000 样 | P5-02 基础就位 |
| D2 数据集扩样到 5000+ | 6 类各 ≥833 unique variants；5000 samples（4 类 833 + 2 类 834）；3500/750/750 split 契约命中；cross-split canonical disjoint；D2 vs D1/D1.1 disjoint；MANIFEST 新增 `build_count` 字段；tests 参数化从 on-disk MANIFEST 派生并显式断言 3500/750/750 契约；`datasets/tool-calling-d2/` 加入 .gitignore 并 `git rm --cached` 移除 Git 追踪 | round 14 完成 |
| D2 数据集扩样到 5000+（round 1） | 5004 样本版本（误合同命中：3498/750/756） | detached auditor round 14 否决（偏离 3500/750/750 契约） |

## 暂缓阶段

明确以下不在当前路线图内，避免范围蔓延：

- Expert Parallel / 分布式训练；
- 完整工业级 MLA；
- 多模态数据；
- 复杂联网 Agent；
- 大规模数据采集；
- 复杂 LLM Judge；
- 自研模型完整 vLLM 适配；
- Expert Parallel / 复杂 capacity 调度；
- 长上下文评测。

如后续需要，重新评估并写入下一阶段。

## Reviewer 触发条件

阶段审查只在大阶段完成时触发，**不为局部改动触发**。触发条件：

- 一个 README 阶段完成；
- 一个 MVP 完成；
- 一个模型模块或数据管线模块完成；
- 一组对比实验完成；
- 计划顺序发生重大调整。

不触发条件：

- 局部函数、bug、文档、测试、schema 微调；
- 单个配置或 CLI 参数调整；
- 单个实验 smoke 重跑；
- 工程化改进（如测试分层、CI、依赖锁定）；
- 性能数字更新但协议不变。

完整定义见 [`docs/plans/review-process.md`](./review-process.md)。

## 关联文档

- 当前 P0/P1/P2 问题：[`open-issues.md`](./open-issues.md)
- 阶段审查流程和记录：[`review-process.md`](./review-process.md) 与 [`reviews/`](./reviews/)
- 实验记录：[`../experiments/`](../experiments/)
- 协议规范：[`../protocols/`](../protocols/)