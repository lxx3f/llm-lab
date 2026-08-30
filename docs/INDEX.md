# LLM Lab — 文档导航

> 本目录按"项目文档"、"实验记录"和"协议参考"三类组织。

## 主入口

| 文档 | 用途 |
|---|---|
| [`../README.md`](../README.md) | 项目入口：项目概览 + 数字一览 + 项目结构 + 当前状态 + 快速命令 |
| [`environment.md`](environment.md) | 运行环境说明（PyTorch 2.13 / RTX 5070 Ti / sm_120） |

## 实验记录（`experiments/`）

15 个实验，每个实验一般含 `README.md` + 可选 `protocol.md` + `result.md`。

### 架构实验（自研小模型）

| 实验 | 内容 |
|---|---|
| [`dense-training-mvp/`](experiments/dense-training-mvp/) | Dense forward + tokenizer + 1MiB smoke |
| [`moe-top1/`](experiments/moe-top1/) | MoE Top-1 训练闭环 |
| [`moe-owt-formal-curve/`](experiments/moe-owt-formal-curve/) | MoE 5000 步 OWT 训练 |
| [`moe-multi-seed/`](experiments/moe-multi-seed/) | MoE 多 seed 实验 |
| [`moe-long-curve/`](experiments/moe-long-curve/) | MoE 长训练曲线 |
| [`n2-dense-moe-fairness/`](experiments/n2-dense-moe-fairness/) | Dense vs MoE 公平对比协议（同 total params / 同 active params） |
| [`n3-unified-metadata/`](experiments/n3-unified-metadata/) | 统一 benchmark/result metadata 规范 |
| [`n4-dense-formal-curve/`](experiments/n4-dense-formal-curve/) | Dense 0.66M / 2.10M 训练曲线（5000 步） |
| [`n13-gqa-vs-mla-vs-mha/`](experiments/n13-gqa-vs-mla-vs-mha/) | **N13 GQA + 简化 MLA 三方对比** |

### 数据与训练（HF 公开模型）

| 实验 | 内容 |
|---|---|
| [`p1-mock-executor/`](experiments/p1-mock-executor/) | Mock executor（工具执行验证） |
| [`p2-evaluator/`](experiments/p2-evaluator/) | 确定性 evaluator + reward offline |
| [`p4-grpo-smoketest/`](experiments/p4-grpo-smoketest/) | GRPO MVP smoke test |
| [`sft-tool-mvp/`](experiments/sft-tool-mvp/) | SFT 工具调用训练 MVP（Dense + MoE） |

### 评测与后端

| 实验 | 内容 |
|---|---|
| [`owt-real-eval/`](experiments/owt-real-eval/) | **真实 OWT 评测（5 模型 × 277MB held-out）** |
| [`p5-04-backend-comparison/`](experiments/p5-04-backend-comparison/) | **P5-04 双后端对比（5 模型 × Transformers/vLLM × 2 batch × 90 样本 4 轴）** |

### 实验间 cross-reference

`docs/experiments/<name>/README.md` 都会交叉引用相关 stage review（`docs/plans/reviews/stage-<name>.md`）和协议（`docs/protocols/<name>.md`）。

## 阶段审查（`plans/reviews/`）

13 个 `stage-<name>.md` 文件，每个对应一个交付阶段的 self-review + reviewer evidence。

| 重点交付 | 说明 |
|---|---|
| `stage-n13-gqa-vs-mla-vs-mha.md` | N13 三方架构对比 |
| `stage-owt-real-eval.md` | 真实 OWT 评测 |
| `stage-p5-04-backend-comparison.md` | 双后端对比 |
| `stage-moe-owt-formal-curve.md` | MoE 5000 步训练 |
| `stage-n2-dense-moe-fairness.md` | Dense vs MoE 公平对比 |
| `stage-n4-dense-formal-curve.md` | Dense 训练曲线 |
| `stage-p2-evaluator.md` | 确定性 evaluator |
| `stage-p3-d2-multi-turn.md` | D2 多轮数据 |
| `stage-sft-tool-mvp.md` | SFT 工具调用 |
| `stage-p4-grpo-mvp.md` / `stage-p4-grpo-smoketest.md` | GRPO |
| `stage-moe-top1-training.md` / `stage-dense-training-mvp.md` | 早期训练 MVP |
| `stage-n3-unified-metadata.md` / `stage-n11-dense-long-curve.md` | 元数据 / 长曲线 |

### Archive（已关闭的 stage）

- [`plans/reviews/archive/`](plans/reviews/archive/) — 已归档的 stage reviews（早期 BPE、tokenizer 数据登记、ablation sweep、P1-P3 中间阶段等）

## 协议（`protocols/`）

跨实验的协议文档（24 个），包括 Dense/MoE 训练协议、N3 metadata、tokenizer/token cache、GRPO、failure classification、testing 等。

## 计划与决策（`plans/`）

| 文档 | 用途 |
|---|---|
| `roadmap.md` | 项目路线图：总体目标 / 已完成阶段 / 当前状态 / 后续可选方向 |
| `open-issues.md` | 已解决问题归并与最终全链路审查总结 |

## 数据契约（`data/`）

- `owt-sample.md` — OpenWebText 数据契约（SHA256 / size / encoding）
- `d2-expansion.md` — D2 多轮数据集扩展契约

## Schemas（`../schemas/`）

13 个 JSON Schema：

- `dense_training_result.schema.json` — Dense / GQA / MLA 训练结果
- `moe_training_result.schema.json` — MoE 训练结果
- `n2_benchmark_result.schema.json` / `n2_routing_stats.schema.json` — N2 benchmark + routing
- `evaluation_result.schema.json` — 通用评测结果
- `model_output.schema.json` — 模型输出记录
- `tool_calling_sample.schema.json` / `d2_multi_turn_sample.schema.json` — 工具调用数据
- `tool_execution_result.schema.json` — 工具执行结果
- `reward_signal.schema.json` — reward 信号
- `grpo_step_result.schema.json` — GRPO step 结果

## Archive（`archive/`）

- [`archive/gqa-vs-mha-no-go/`](archive/gqa-vs-mha-no-go/) — 公开模型 GQA vs MHA 可行性审查（no-go + incomplete feasibility）
- [`archive/sweep-experiments/`](archive/sweep-experiments/) — Dense ablation sweeps（N5-N9 / N11 / N12）
- [`archive/plans/`](archive/plans/) — 原 README 中文规划 + 阶段审查流程定义
- [`archive/reports/`](archive/reports/) — 最终全链路审查 + reviewer evidence
- [`archive/personal-backup/`](archive/personal-backup/) — 个人项目过程资料备份（不作为 GitHub 项目入口）

## Tests（`../tests/`）

项目级 pytest 文件。关键测试集：

- `test_gqa_mla_models.py` — N13 GQA + MLA 单元测试（13 tests）
- `test_dense_training.py` — Dense 训练测试
- `test_dense_result_schema.py` — Dense result schema 测试
- `test_token_cache.py` / `test_token_cache_cli.py` — token cache 测试
- `test_d1_failure.py` / `test_d2_dataset.py` — 数据集测试
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
- `test_gqa_vs_mha_no_go.py` / `test_gqa_vs_mha_audit_consistency.py` — F 项目 audit-consistency 测试（保留作 archive）
