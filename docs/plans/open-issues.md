# 决策与已解决问题记录

> 本文是项目内部的决策记录。GitHub 公开版本仅作为参考资料；活跃问题已全部关闭，没有未关闭的 P0/P1/P2 项。
>
> 详细 stage review 见 `docs/plans/reviews/`；归档审查见 `docs/plans/reviews/archive/`。

## 已解决（按模块归并）

### 架构实验

- **P0-01 Dense 基线真实训练闭环**：通过 N4 / N11 / N12 交付，含 0.66M / 2.10M / 12M 多规模 5000-50000 步训练。
- **P0-02 MoE Top-1 完整训练 MVP**：通过 N1 + 后续阶段交付，含 active loss + aux loss + checkpoint + result schema。
- **P0-03 Dense 与 MoE 的比较协议不公平**：通过 N2 协议 A（相同 total params）和协议 B（相同 active params）解决。
- **P0-04 MoE routing statistics 污染 latency benchmark**：通过 N2 把 routing stats 和 latency 测量解耦解决。
- **P0-05 MoE capacity 与 prefill/decode cache 语义未定义**：通过 N2 解决，统一 capacity 策略。

### 数据和评测

- **P1-01 缺少统一 benchmark schema**：通过 N3 统一 benchmark/result metadata 解决。
- **P1-02 实验元数据和版本信息不完整**：通过 N3 解决，含 git commit / config hash / CUDA/PyTorch/GPU compute capability / seed。
- **P1-03 缺少多 seed 和统计区间**：通过 multi-seed sweep 和 sft-mvp 多 seed eval 解决。
- **P1-04 数据版本 D0/D1/D2 定义不够严格**：通过 D1.1 / D2 多轮数据集（5000 样本，train 3500 / dev 750 / test 750）解决。
- **P1-05 工具执行成功不等于任务完成**：通过 P1-05 八级分类器（parse_success → final_answer_correct）解决。
- **P1-06 JSON Schema 校验不能替代语义校验**：通过 schema + evaluator + manifest + 执行验证的组合解决。
- **P1-08 GRPO 依赖评测和执行器先稳定**：通过 P4 GRPO MVP 解决。
- **P1-09 vLLM 运行环境需要单独确认**：通过 WSL2 smoke PASS 解决。

### 文档工程

- **P2-01 计划目录和实际目录不同步**：通过目录统一解决（`docs/plans/` + `docs/protocols/` + `docs/experiments/`）。
- **P2-02 README 协议示例与真实 Schema 不一致**：通过 README schema example 与 schema 文件一致解决。
- **P2-03 缺少依赖锁定文件**：保留为已记录但显式延后（README 显式说明依赖通过 `pip install` 而非 lockfile 锁定）。
- **P2-04 文档产物目录已经统一，但链接需要持续维护**：通过归档不活跃文档（archive/）保持主目录可读。
- **P2-05 SFT held-out 评测 split 偏小**：通过 P3-01 D2 多轮 IID held-out split 解决。

### 后端与模型

- **P5-02 Transformers backend + 5 个公开 instruction-tuned 模型 × D2 dev reward**：通过 25 单测覆盖 + target-answer 泄漏修复解决。
- **P5-03 vLLM 后端接入**：通过 WSL2 smoke PASS 解决。
- **P5-04 双后端基准对比**：通过 20 个真实 GPU 组合（5 模型 × 2 backend × 2 batch size × 90 样本）解决。

## 当前活跃问题

无。所有 P0/P1/P2/P3/P4/P5 项目都已交付并通过阶段审查。

## 最终全链路审查

最新一轮 detached auditor 全部通过；详细 reviewer evidence 已归档至 `docs/archive/reports/`。
