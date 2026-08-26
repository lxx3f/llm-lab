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

这些阶段都已通过阶段审查（`minimax-cn/MiniMax-M3` reviewer），不允许回退。

## 当前阶段：P1 工具执行器

目标：在 N5 规模 sweep 之后，从架构/训练阶段跨到评测阶段，先建立工具执行器的最小闭环。子阶段：

| 子阶段 | 范围 | 退出条件 |
|---|---|---|
| P1-01 mock 执行器 | 无外部依赖的 mock 工具、structured result schema | schema 与 example 落盘 |
| P1-02 沙箱执行器 | subprocess / timeout / 资源限制 / 退出码 | schema 与 example 落盘 |
| P1-03 失败分类 | parse_success / schema_valid / execution_success / result_grounded / task_success 五级 | 分类 schema 与 mock 失败测试 |

任一子阶段失败必须先更新本路线图和 `open-issues.md`。

## 下一阶段：评测器、工具执行器、SFT

| 阶段 | 范围 | 触发条件 |
|---|---|---|
| P1 工具执行器 | mock / 沙箱执行；执行结果结构化；失败模式分类 | N2 完成 |
| P2 确定性 evaluator | 任务定义、reward schema、离线 reward 校验 | P1 完成 |
| P3 数据版本 D0/D1/D2 | sample 数、工具分布、split、hash、过滤规则 | P1 完成 |
| P4 SFT/GRPO 训练 | D0 SFT 跑通；reward offline 校验；GRPO MVP | P2+P3 完成 |
| P5 vLLM benchmark | Transformers + vLLM 双后端 inference 接口；deployment 实验 | P5 评测 MVP 完成 |

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