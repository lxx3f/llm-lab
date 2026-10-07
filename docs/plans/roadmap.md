# 项目路线图

> 单一权威的总体计划文档。
>
> 已解决问题的决策摘要见 `docs/plans/open-issues.md`；阶段审查记录在 `docs/plans/reviews/`。

## 总体目标

构建可复现的 LLM 实验闭环：

```text
模型架构实现 → 数据构造与处理 → 训练/微调 → vLLM 部署 → 统一评测 → 失败分析
```

闭环中每一节都先建立最小可复现实验，再扩展规模。不允许把“使用开源框架”本身当作项目成果，重点记录修改、验证过程和实验结果。

## 已完成阶段

| 阶段 | 范围 | 交付物 / stage review |
|---|---|---|
| Stage 0 数据协议 | tool calling / model output / evaluation result JSON Schema | `schemas/` 13 个 schema |
| BPE tokenizer artifact | OWT BPE 训练、CLI、artifact metadata、版本化目录 | `architecture_lab/tokenization/` |
| OWT 数据登记 | 数据来源、许可、版本、hash | `docs/data/owt-sample.md` |
| Token cache 协议 | UTF-8 行流式、uint16 little-endian、hash 绑定 | `architecture_lab/data/token_cache.py` |
| Dense training MVP | Dense forward + tokenizer + 1MiB smoke | `stage-dense-training-mvp.md` |
| OWT 正式 cache | 512 MiB train / 64 MiB validation cache | `scripts/encode_token_cache.py` |
| Dense 优化与 schema | scheduler / AMP / 梯度累积 / 结果 schema | `stage-dense-optimization-schema.md` |
| MoE Top-1 训练闭环 | active loss + aux loss + checkpoint + result schema | `stage-moe-top1-training.md` |
| Dense/MoE 公平对比 | 相同 total params / 相同 active params 双协议 | `stage-n2-dense-moe-fairness.md` |
| 统一 metadata | git commit / config hash / CUDA/PyTorch/GPU compute capability / seed | `stage-n3-unified-metadata.md` |
| Dense 正式训练曲线 | baseline 0.66M + medium 2.10M 各 5000 步 | `stage-n4-dense-formal-curve.md` |
| MoE 5000 步训练曲线 | MoE Top-1 5000 步（total 2.10M / active 1.51M） | `stage-moe-owt-formal-curve.md` |
| Dense 长训练曲线 | baseline + medium 各 50000 步 | `stage-n11-dense-long-curve.md` |
| P1-02 Mock 工具执行器 | 进程内 mock executor；8 单测 + tool_execution_result schema | `stage-p1-mock-executor.md` |
| D1.1 工具调用数据集 | 1500 train 样本 6 task_type | `examples/tool_calling/` |
| P1-03 多 seed 评测 | `scripts/run_multi_seed.py` Dense + MoE dispatch | `tests/test_grpo_mvp.py` |
| P1-04 数据版本 D0/D1 | D1 数据集目录 + MANIFEST + filter 脚本 | `docs/protocols/p1-04-data-version-d0.md` |
| P1-05 八级分类器 | parse_success → final_answer_correct | `scripts/classify_tool_failure.py` |
| SFT 工具调用训练 MVP | Dense + MoE 训练管线 + 5 次实跑对比 | `stage-sft-tool-mvp.md` |
| P2 确定性 evaluator | reward_binary + reward_layered + reward_type 主导通道 | `stage-p2-evaluator.md` |
| P3 D2 多轮数据集 | 5000 样本（train 3500 / dev 750 / test 750） | `stage-p3-d2-multi-turn.md` |
| P4 GRPO MVP | GRPO 训练 + advantage + policy 更新闭环 | `stage-p4-grpo-mvp.md` |
| P4 GRPO 小规模正确性实验 | 3-seed 小规模实验 + round 9/10/11 修复 | `stage-p4-grpo-smoketest.md` |
| P5-02 Transformers 后端 | 5 公开模型 × D2 dev reward | `stage-p5-02-transformers-backend.md` |
| P5-03 vLLM 后端 | WSL2 smoke PASS | `stage-p5-03-vllm-feasibility.md` |
| P5-04 双后端基准对比 | 5 模型 × 2 backend × 2 batch × 90 样本 4 轴 | `stage-p5-04-backend-comparison.md` |
| **N13 GQA vs MLA vs MHA** | 自研 GQA + 简化 MLA + Dense MHA 三方对比 | `stage-n13-gqa-vs-mla-vs-mha.md` |
| **真实 OWT 评测** | 5 公开模型 × 277MB held-out per-token loss | `stage-owt-real-eval.md` |

阶段审查通过率 100%（`stage-n13-...` 等关键 stage 由独立 reviewer 通过）。


## 当前状态

当前项目已经具备一个完整的小规模 LLM 实验闭环：

```text
自研架构
Dense / MoE / GQA / simplified MLA
        ↓
OWT tokenizer + token cache
        ↓
原生 PyTorch training + checkpoint + validation
        ↓
公开模型 Transformers / vLLM backend
        ↓
工具调用 evaluator + reward + failure taxonomy
        ↓
JSON Schema + hash metadata + tests + 实验记录
```

## 后续可选方向（非本仓库主线）

- GQA `num_kv_heads` sweep；
- MLA `latent_dim` sweep；
- prefill/decode latency、throughput 和 peak memory benchmark；
- N13 多 seed 复现。

以上属于可选增强，不影响当前项目作为可复现实验闭环的“已完成”状态。
