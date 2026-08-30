# AGENTS.md

## 项目概述

`llm-lab` 是一个面向大语言模型研究与工程实践的实验项目，围绕以下闭环逐步建设：

```text
模型架构实现 → 数据构造与处理 → 训练/微调 → vLLM 部署 → 统一评测 → 失败分析
```

当前项目处于规划和初始化阶段，优先构建小而完整、可复现的实验，再逐步扩展功能。

## 目录约定

- `architecture_lab/`：基于 CS336 扩展 Dense Transformer、MoE、GQA 和 MLA 实验。
- `data-pipeline/`：工具调用和结构化指令数据的采集、清洗、生成、校验与版本管理。
- `evaluation-lab/`：模型推理后端、评测指标、任务执行器和失败案例分析。
- `serving/`：标准开源模型的 Transformers/vLLM 部署与性能测试。
- `configs/`：模型、数据、训练和评测配置。
- `scripts/`：实验运行和结果汇总脚本。
- `docs/`：计划、协议、环境说明、实验记录和中间文档产物。
- `README.md`：项目总体计划和实施方案。

自研架构实验与开源模型后训练可以使用不同模型，避免 MLA/MoE 实现或 vLLM 适配阻塞数据、训练和评测模块。

## 开发原则

- 先固定 Dense Transformer 基线，再增加 MoE、GQA 和简化 MLA。
- 自研模型优先使用原生 PyTorch 验证正确性和性能；标准开源模型优先接入 Transformers 和 vLLM。
- 数据项目必须记录来源、许可证、处理版本、校验结果和数据统计。
- 评测结果必须记录模型版本、数据版本、配置、推理后端和指标定义。
- 重要结论应有可复现实验、基线对比和失败案例支撑。
- 不把“使用开源框架”或“使用 vLLM 部署”本身当作项目成果，重点记录实际修改、验证过程和实验结果。
- 未经确认的模型指标、数据规模和性能结论不得写入正式报告或简历。
- 每完成一个阶段或 MVP，必须先进行一次计划、正确性、实验有效性和文档审查，再进入下一阶段；阶段审查必须由子 agent 执行，审查模型固定使用 `minimax-cn/MiniMax-M3`（MiniMax M3）；审查通过后自动创建一个阶段 Git commit。审查流程见 `docs/plans/review-process.md`，问题统一记录在 `docs/plans/open-issues.md`。
- 例外：**单审计轮 fix**（如 `audit round N fix` / `audit round N postfix`，范围限于文档、测试、生成器参数或 schema 扩展，且改动已在 `docs/plans/reviews/stage-*.md` 对应 Round 中完整记录）可省略 subagent reviewer，仅由 goal 框架的 detached auditor 单独负责；详见 `docs/plans/review-process.md` 中的“Stage reviewer 跳过条件”段。
- 如果 harness 无法确认审查模型身份，必须停止自动 commit 并报告阻塞；
- 不提交模型权重、API 密钥、内部数据或其他敏感信息。
- 不提交训练产物（checkpoint/tokenizer artifact/中间产物 JSON），只提交 config + 评测脚本 + docs；训练曲线/评测 JSON 是产物可重跑复现，以 `.gitignore` 覆盖（`artifacts/checkpoints/`, `artifacts/*.json`, `artifacts/tokenizers/`）。数据集（D1 / D1.1 / D2 等生成产物）统一以 `.gitignore` 覆盖，本地按需通过 `scripts/generate_*_dataset.py` 重新生成（其中 D1 由 `test_d1_failure.py::setUpClass` 自动重建，D1.1 / D2 需要预先调用对应生成器）。

## 当前阶段

1. 固定现有 Dense Transformer 基线。
2. 确定工具调用数据 schema 和评测结果 schema。
3. 实现最小评测器与统一推理接口。
4. 实现 MoE Top-1 MVP。
5. 实现 GQA（`num_kv_heads` 可配置）+ 简化 MLA（KV 压缩到 `latent_dim`），与 N4 Dense MHA 在同 OWT cache 同 5000 步下三方对比（**已交付 N13，2026-08-30**）。
6. 再逐步扩展数据管线、SFT/GRPO 和 vLLM benchmark。

## 检查与记录

项目测试使用三档命令，详见 `docs/protocols/testing.md`：

- `scripts/run_tests.py fast`：日常快速 API/schema/forward 检查；
- `scripts/run_tests.py module <data|training|architecture|tokenization>`：按模块回归；
- `scripts/run_tests.py full`：阶段审查和 commit 前的全量 unittest + schema examples。

新增代码后，应同步补充：

- 运行方式和依赖版本；
- 单元测试或正确性测试；
- 实验配置和随机种子；
- 指标定义与结果文件；
- README 或 `docs/` 中的复现说明。

## 沟通约定

项目文档和协作说明优先使用中文；代码、配置字段和公开模型/框架名称保留其通用英文名称。
