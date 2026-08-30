# LLM Lab — 文档导航

> 本目录按"面向访客"和"内部参考"两类组织。简历用户应该从 [`SHOWCASE.md`](SHOWCASE.md) 开始。

## 核心入口（简历用户先看）

| 文档 | 用途 |
|---|---|
| [`SHOWCASE.md`](SHOWCASE.md) | 简历项目摘要：4 条 bullet + 5 张关键 plot + 同 base 判定 + 6 问面试 talking points |
| [`../README.md`](../README.md) | 项目入口：数字一览 + 项目结构 + 当前状态 + 快速命令 |
| [`environment.md`](environment.md) | 运行环境说明（PyTorch 2.13 / RTX 5070 Ti / sm_120） |

## 实验详情（`experiments/`）

按主题分组，27 个实验，每个实验一般含 `README.md` + `protocol.md` + `result.md`。

### 架构实验（自研小模型）

| 实验 | 内容 |
|---|---|
| [`n2-dense-moe-fairness/`](experiments/n2-dense-moe-fairness/) | Dense vs MoE 公平对比协议（同 total params / 同 active params） |
| [`n3-unified-metadata/`](experiments/n3-unified-metadata/) | 统一 benchmark/result metadata 规范 |
| [`n4-dense-formal-curve/`](experiments/n4-dense-formal-curve/) | Dense 0.66M / 2.10M 训练曲线（5000 步） |
| [`n5-dense-scale-sweep/`](experiments/n5-dense-scale-sweep/) | Dense scale sweep |
| [`n6-dense-dropout-sweep/`](experiments/n6-dense-dropout-sweep/) | Dense dropout sweep |
| [`n7-dense-rope-sweep/`](experiments/n7-dense-rope-sweep/) | Dense RoPE base sweep |
| [`n8-dense-heads-sweep/`](experiments/n8-dense-heads-sweep/) | Dense n_heads sweep |
| [`n9-dense-dff-sweep/`](experiments/n9-dense-dff-sweep/) | Dense d_ff sweep |
| [`n11-dense-long-curve/`](experiments/n11-dense-long-curve/) | Dense 长训练曲线（50000 步） |
| [`n12-dense-ultra-curve/`](experiments/n12-dense-ultra-curve/) | Dense ultra-curve 200k 步 |
| [`moe-top1/`](experiments/moe-top1/) | MoE Top-1 forward MVP |
| [`moe-owt-formal-curve/`](experiments/moe-owt-formal-curve/) | MoE 5000 步 OWT 训练 |
| [`moe-multi-seed/`](experiments/moe-multi-seed/) | MoE 多 seed 实验 |
| [`moe-long-curve/`](experiments/moe-long-curve/) | MoE 长训练曲线 |
| [`multi-seed-sweep/`](experiments/multi-seed-sweep/) | 多 seed 扫描协议 |
| [`n13-gqa-vs-mla-vs-mha/`](experiments/n13-gqa-vs-mla-vs-mha/) | **N13 GQA + 简化 MLA 三方对比（最新交付）** |

### 数据与训练（HF 公开模型）

| 实验 | 内容 |
|---|---|
| [`d1-llm/`](experiments/d1-llm/) | D1 LLM 生成模板版工具调用数据 |
| [`p1-mock-executor/`](experiments/p1-mock-executor/) | Mock executor（工具执行验证） |
| [`p2-evaluator/`](experiments/p2-evaluator/) | 确定性 evaluator + reward offline |
| [`p4-grpo-smoketest/`](experiments/p4-grpo-smoketest/) | GRPO MVP smoke test |
| [`sft-tool-mvp/`](experiments/sft-tool-mvp/) | SFT 工具调用训练 MVP（Dense + MoE） |
| [`dense-baseline/`](experiments/dense-baseline/) | Dense 早期 baseline |
| [`dense-training-mvp/`](experiments/dense-training-mvp/) | Dense training MVP |

### 评测与后端

| 实验 | 内容 |
|---|---|
| [`owt-real-eval/`](experiments/owt-real-eval/) | **真实 OWT 评测（5 模型 × 277MB held-out）** |
| [`p5-03-vllm-feasibility/`](experiments/p5-03-vllm-feasibility/) | vLLM 后端接入 WSL2 smoke |
| [`p5-04-backend-comparison/`](experiments/p5-04-backend-comparison/) | **P5-04 双后端对比（5 模型 × Transformers/vLLM × 2 batch × 90 样本 4 轴）** |

### 实验间 cross-reference

`docs/experiments/<name>/README.md` 都会交叉引用相关 stage review（`docs/plans/reviews/stage-<name>.md`）和协议（`docs/protocols/<name>.md`）。

## 阶段审查（`plans/reviews/`）

38 个 `stage-<name>.md` 文件，每个对应一个交付阶段的 self-review + reviewer evidence。

### 简历相关（核心交付阶段）

- `stage-n13-gqa-vs-mla-vs-mha.md` — N13 三方对比 stage review
- `stage-owt-real-eval.md` — 真实 OWT 评测 stage review
- `stage-p5-04-backend-comparison.md` — 双后端对比 stage review
- `stage-moe-owt-formal-curve.md` — MoE 5000 步训练 stage review
- `stage-n2-dense-moe-fairness.md` — Dense vs MoE 公平对比 stage review
- `stage-n4-dense-formal-curve.md` — Dense 训练曲线 stage review
- `stage-p2-evaluator.md` — 确定性 evaluator stage review
- `stage-p3-d2-multi-turn.md` / `stage-p3-d2-expansion.md` — D2 多轮数据 stage review
- `stage-sft-tool-mvp.md` — SFT 工具调用 stage review
- `stage-p4-grpo-mvp.md` / `stage-p4-grpo-smoketest.md` — GRPO stage reviews

### 早期内部阶段（可参考）

- `stage-bpe-tokenizer.md` / `stage-bpe-streaming.md` / `stage-bpe-optimized.md`
- `stage-tokenizer-cli.md` / `stage-toy-data-tokenizer.md`
- `stage-owt-data-source.md` / `stage-owt-formal-cache.md`
- `stage-dense-optimization-schema.md` / `stage-subagent-model-config.md`
- `stage-roadmap-update.md` / `stage-remove-toy-data.md`
- `stage-B-open-issues-audit.md` — open-issues 通读 audit

### Archive（已关闭的 stage）

- `archive/stage-gqa-vs-mha-no-go.md` — F 项目的 stage review（已 cancelled，**不**作为简历素材）

## 协议（`protocols/`）

22 个 `*.md` 文件，详细说明每个实验 / 阶段的具体协议。

| 协议 | 内容 |
|---|---|
| `dense-training.md` / `dense-batching.md` | Dense 训练 / batching 协议 |
| `moe-owt-formal-curve.md` / `n2-benchmark.md` | MoE 协议 |
| `n3-metadata.md` | 统一 benchmark/result metadata 协议 |
| `n4-dense-curve.md` 至 `n9-dense-dff-sweep.md` | 5 个 ablation sweep 协议 |
| `n11-dense-long-curve.md` | 长曲线协议 |
| `tokenizer-artifact.md` / `owt-token-cache.md` | tokenizer + token cache 协议 |
| `d2-multi-turn.md` | D2 多轮数据协议 |
| `grpo.md` | GRPO 协议 |
| `p1-03-multi-seed.md` / `p1-04-data-version-d0.md` / `p1-05-failure-classification.md` | 多 seed / 数据版本 / 失败分类协议 |
| `p1-mock-executor.md` / `p2-evaluator.md` | mock executor / evaluator 协议 |
| `testing.md` | 测试策略 |
| `stage0-protocol.md` | Stage 0 协议 |
| `transformers-backend.md` / `backend-comparison.md` | 后端协议 |

## 计划与决策（`plans/`）

| 文档 | 用途 |
|---|---|
| `roadmap.md` | 路线图：已完成 / 进行中 / 下一阶段 / 暂缓阶段 |
| `open-issues.md` | P0-P5 问题追踪（33 条目）+ 决策记录 |
| `review-process.md` | 阶段审查流程（reviewer / auditor 角色定义） |

## 数据契约（`data/`）

- `owt-sample.md` — OpenWebText 数据契约（SHA256 / size / encoding）
- `toy-dataset.md` — 玩具数据集（已废弃）
- `d2-expansion.md` — D2 多轮数据集扩展契约

## Reports（`reports/`）

- `final-audit.md` — 最终审计报告
- `night-run-summary.md` — 夜间运行总结
- `final-audit/round-13-reviewer-evidence.md` / `round-17-reviewer-evidence.md` — 审计 reviewer evidence

## Schemas（`../schemas/`）

11 个 JSON Schema：

- `dense_training_result.schema.json` — Dense / GQA / MLA 训练结果
- `moe_training_result.schema.json` — MoE 训练结果
- `n2_benchmark_result.schema.json` / `n2_routing_stats.schema.json` — N2 benchmark + routing
- `evaluation_result.schema.json` — 通用评测结果
- `model_output.schema.json` — 模型输出记录
- `tool_calling_sample.schema.json` / `d2_multi_turn_sample.schema.json` — 工具调用数据
- `tool_execution_result.schema.json` — 工具执行结果
- `reward_signal.schema.json` — reward 信号
- `grpo_step_result.schema.json` — GRPO step 结果

## 内部参考（`internal/`）

- `README-chinese-detail.md` — 原 README 中文规划文档（模块 A-D / MVP 范围 / 预期成果）

## Archive（`archive/`）

- `gqa-vs-mha-no-go/` — F 项目的 4 个文件（feasibility / protocol / README / reviewer-evidence）
  - **不**作为简历素材：F 是 20 轮 audit 后关闭的 incomplete feasibility review

## Tests（`../tests/`）

29 个 `test_*.py` 文件，421 tests。关键测试集：

- `test_gqa_mla_models.py` — N13 GQA + MLA 单元测试（13 tests）
- `test_gqa_vs_mha_no_go.py` + `test_gqa_vs_mha_audit_consistency.py` — F 项目 audit-consistency 测试（25 tests，保留作 archive）
- `test_dense_training.py` — Dense 训练测试
- `test_dense_result_schema.py` — Dense result schema 测试
- `test_token_cache.py` / `test_token_cache_cli.py` — token cache 测试
- `test_aggregate_d256_eval.py` — D256 评测聚合测试
- `test_d1_failure.py` / `test_d1_llm.py` / `test_d2_dataset.py` — 数据集测试
- `test_grpo_mvp.py` — GRPO MVP 测试
- `test_mock_executor.py` — mock executor 测试
- `test_moe_training.py` / `test_sft_training.py` / `test_sft_moe_training.py` — SFT/MoE 训练测试
- `test_transformers_backend.py` — Transformers backend 测试
- `test_eval_backend_comparison.py` — P5-04 后端对比测试
- `test_stage0_schemas.py` — Stage 0 schemas 测试
- `test_sweep_doc_consistency.py` — sweep 文档一致性测试
- `test_plot_dense_curve.py` — plot 脚本测试
- `test_artifact_provenance.py` — artifact provenance 测试
- `test_audit_reconciliation_extraction.py` — audit reconciliation 提取测试
- `test_experiment_metadata.py` — experiment metadata 测试
- `test_batching.py` — batching 测试
- `test_reward_offline.py` — reward offline 测试
- `test_n2_benchmark.py` / `test_n2_result_schema.py` — N2 benchmark 测试
- `test_d0_manifest.py` — D0 manifest 测试
- `test_tokenizer_artifact_cli.py` — tokenizer artifact CLI 测试
