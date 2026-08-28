# 阶段 SFT 工具调用训练 MVP 审查

审查模型：`minimax-cn/MiniMax-M3`
审查 agent：`reviewer`（subagent dispatch 名称）
实际 provider/model：`PI_PROVIDER=minimax-cn`，`PI_MODEL=MiniMax-M3`
可选运行标识：`PI_AGENT_NAME` 未由 harness 注入
审查文件路径：`docs/plans/reviews/stage-sft-tool-mvp.md`
commit 候选变更：`docs/plans/roadmap.md`、`docs/plans/reviews/stage-sft-tool-mvp.md`、`docs/plans/open-issues.md`

## 阶段 sft-tool-mvp 审查

- 完成范围：
  - **D1 模板工具调用数据集**：126 样本 × 8 task_type；D0 manifest；MockExecutor 端到端验证；11 单测
  - **D1.1 LLM 生成数据集**：MiniMax-M3@minimax-cn 生成 126→1500 样本；per-file source provenance；aggregate sha256（双重 sha256+source）；MANIFEST 重建脚本
  - **P1-03 多 seed 评测协议**：`scripts/run_multi_seed.py`（Dense + MoE 自动 dispatch）；总体标准差（÷N）；N3 协议 metadata
  - **P1-04 数据版本 D0**：`docs/protocols/p1-04-data-version-d0.md`；`scripts/build_d0_manifest.py`；D0 manifest CLI
  - **P1-05 八级分类器**：parse_success → schema_valid → tool_name_correct → argument_value_correct → call_plan_matches → execution_success → result_grounded → final_answer_correct；含 `_norm_name`/`_norm_args` 鲁棒性；call_id 显式位置配对（multiset 匹配）；`scripts/classify_tool_failure.py`；44 单测
  - **SFT 训练管线（完整可用）**：
    - `architecture_lab/training/sft_training.py`：tool_calling_sample → 掩码序列；`### User`/`### Assistant`/`### Result` 模板；换行分隔 JSON 对象（匹配 tokenizer 合并的 `{"` token）；`_find_user_turn` 处理 system 消息
    - `architecture_lab/training/sft_moe_training.py`：MoE SFT；复用 Dense 数据构造 + MoETransformer + aux_loss
    - `architecture_lab/training/sft_augment.py`：确定性增强（城市/数字/查询替换 + mock 重算 expected_result）
    - `scripts/train_sft.py` / `scripts/train_sft_moe.py`：Dense + MoE 训练 CLI
    - `scripts/generate_sft_tool.py` / `scripts/eval_sft_tool.py`：自动检测 Dense/MoE 架构
  - **多 seed eval 聚合器**：`scripts/aggregate_d256_eval.py`；parse_success_rate mean ± pop_std + 8 级分类器分布；2 单测
  - **d_model=256 端到端实验**：OWT 5000 步预训练 + 3 seeds × 5000/20000 步 SFT；20k 使用已提交的 `configs/sft-d256-20k.example.yaml`
  - **MoE 工具调用 SFT**：MoE 4-expert + OWT long init + 2000 步

- 未完成范围：
  - P2 确定性 evaluator（任务定义 + reward schema）
  - D2 数据集（多轮对话 + 错误恢复）
  - GRPO 训练
  - P5-01/02/03/04（公开 instruction-tuned 模型的模型选择、Transformers backend、vLLM backend 与双后端 benchmark）
  - 大模型起点（≥20M instruction-tuned 开源模型）

- 测试结果：
  - `scripts/run_tests.py full` → Ran 185 tests OK（含 2 skipped）
  - 包含：
    - `tests/test_d1_llm.py` (8)
    - `tests/test_sft_moe_training.py` + `architecture_lab/training/tests/test_sft_training.py` (10)
    - `tests/test_d1_failure.py` (44)
    - `tests/test_aggregate_d256_eval.py` (2)
    - 其他 121（schema / tokenizer / batching / plots / multi-seed 等）

- 实验结果（5 次 SFT 实跑完整对比，详见 `docs/experiments/sft-tool-mvp/README.md`）：

  | # | 架构 | 参数 | 数据 | 步数 | val_min | D1 dev | D1.1 train 150 eval |
  |---|---|---|---|---|---|---|---|
  | 1 | Dense medium | 2.10M | 804 | 2k | 1.76 | 0/13 | — |
  | 2 | Dense large | 5.11M | 804 | 2k | 1.18 | 1/13 argument_value_correct | — |
  | 3 | Dense large | 5.11M | 5288 | 20k | 0.33 | 0/13（过拟合）| — |
  | 4 | Dense d256 | 12.0M | 1500 | 5k × 3 seeds | 0.6824 ± 0.0027 | 0/13 × 3 | 0.0067 ± 0.0094 |
  | 4b | Dense d256 | 12.0M | 1500 | **20k × 3 seeds** | 0.6866 ± 0.0029 | 0/13 × 3 | **0.0400 ± 0.0283** |
  | 5 | MoE 4-expert | ~2M active | 1500 | 2k + OWT init | 1.94 | 0/13 | — |

  **诚实负结果**：本项目自研 2M-12M 规模在当前数据集/任务上**未能学会可靠的工具调用**。五次实跑全 D1 dev 0/13 parse（除 v2 1/13 突破到 argument_value_correct 层）。

  **关键实证新发现**：D1 dev 13 样本 × 3 seeds = 39 eval 不足以区分变体；D1.1 train 50/seed × 3 = 150 eval 上 20k = 5k 的 6× 生成质量提升。**val_min 不是 eval 的完美预测器**。

- 新发现问题（按严重级别）：
  - **P0：无**。本阶段核心数据、训练、生成、评测链路均有测试和实跑证据。
  - **P1：自研 2M-12M base 模型未达到可靠工具调用**。D1 dev 上除 Dense large 2k 的 1/13 部分通过外，其余候选均未通过 parse；这阻塞 GRPO 作为下一步直接训练目标，但不阻塞公开 instruction-tuned 模型后端评测。
  - **P1：vLLM 平台约束**。当前开发环境为 Windows；后续 vLLM backend 应以 WSL/Linux/Docker 为运行目标，不能把原生 Windows 可用性作为既定前提。
  - **P2：D1 dev 样本数较小且 D1.1 train 50-sample 聚合不是独立 held-out split**。后续应在 P2/P3 阶段建立明确的 IID/compositional held-out split，避免把训练集抽样结果当作泛化质量。

- 计划调整：
  - 当前阶段：~~P1 工具执行器~~ → **P5 公开 instruction-tuned 模型推理后端接入**。
  - **P5-03 的 vLLM 范围明确限定为公开 instruction-tuned 模型的部署/推理 backend**（例如 Qwen2.5-0.5B-Instruct 或 LLaMA-3.2-1B-Instruct）；不包含被路线图暂缓的“自研模型完整 vLLM 适配”。
  - P5-03 的环境目标为 WSL/Linux/Docker；先完成 P5-02 Transformers backend，再验证 vLLM，避免 Windows 原生环境成为阻塞。
  - 下一阶段候选：P5-01/02/03/04；随后再处理 P2 确定性 evaluator、D2 数据集与 GRPO。
  - 阶段规模 + 路线图已同步更新（见 `docs/plans/roadmap.md`）。

- 是否允许进入下一阶段：是
- 下一步：
  1. P5-01：选定 Qwen2.5-0.5B-Instruct 或 LLaMA-3.2-1B-Instruct，记录模型卡、LICENSE、revision 和运行环境
  2. P5-02：先实现 Transformers backend + 同一 P1-05 八级分类器评测
  3. P5-03：在 WSL/Linux/Docker 中接入 vLLM（仅公开模型）并记录吞吐/延迟
  4. P5-04：同一样本、同一 prompt/template、同一分类器下做双后端对比
  5. P2/P3：补充确定性 reward evaluator 与独立 D2 多轮/错误恢复数据集，再进入 GRPO

## 审查结论

- 审查模型：`minimax-cn/MiniMax-M3`
- 审查 agent：`reviewer`
- 结论：**有条件通过 → 条件已由主 agent 修复**
- 允许创建阶段 commit：是
- 主 agent 修复动作：补齐审查元数据；将 vLLM 明确限定为公开模型 backend；在 `open-issues.md` 记录风险和 Windows 平台约束；明确 commit 候选文件范围。
