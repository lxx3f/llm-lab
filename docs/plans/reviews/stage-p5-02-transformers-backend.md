# Stage review — P5-02 Transformers backend + 5 公开 instruction-tuned 模型 × D2 dev reward

> 阶段：P5-02
> 审查时间：2026-08-28
> 审查流程：subagent reviewer (`reviewer` dispatch, minimax-cn/MiniMax-M3) + detached auditor (calculet/gpt-5.6-terra)

## 阶段目标

实现 `scripts/eval_transformers.py` Transformers 推理后端，在 D2 dev 90 样本上对 5 个公开 instruction-tuned 模型跑真实推理 + reward_offline，产出 5×90=450 reward_signal；与自研 5 ckpt 在同一 reward pipeline 上做横向对比。

Done when：
- (a) 5 个公开模型 × D2 dev 90 signals schema 合法（450/450）；
- (b) ≥1 模型 reward_binary > 0 或 reward_layered > 0.05；
- (c) `docs/experiments/p2-evaluator/README.md` §7 公开模型 reward 表 + 与自研 ckpt 对比；
- (d) `tests/test_transformers_backend.py` ≥10 单测全绿；
- (e) reviewer dispatch (minimax-M3) 通过；
- (f) 工作树干净。

## 审查模型身份

- Subagent reviewer dispatch name：`reviewer`（项目级 `~/.pi/agent/agents/reviewer.md`，`model: minimax-cn/MiniMax-M3`）
- 实际使用模型：`minimax-cn/MiniMax-M3`（dispatch name 与实际模型一致；review-process.md 规则：以实际为准）
- Detached auditor：`calculet/gpt-5.6-terra`（isolated 队列，由 orchestrator 调度）

## Round 1 修复（target-answer 泄漏 + 文档与 artifact 不一致）

### 根因

1. `scripts/eval_transformers.py::_apply_chat_template()` 把 D2 样本整条 `messages`（含 gold terminal assistant `content` = `expected_answer`）直接透传给 `apply_chat_template(..., add_generation_prompt=True)`，模型在 prompt 中看到目标答案后才生成。`scripts/eval_sft_tool.py:69` 的 `render_multi_turn_prompt` 已用 `messages = messages[:-1]` 做同样防护但 eval_transformers 缺失，导致 SmolLM2-360M 的 `reward_binary=0.0111` 实际是 prompt 复述伪结果。
2. `docs/protocols/transformers-backend.md` §8 与 `docs/experiments/p2-evaluator/README.md` §7 的 reward_type 分布表存在文档与 artifact 不一致（Qwen2.5-1.5B 文档写 `argument_correct × 75` 但 artifact 是 `schema_valid: 11, tool_name_correct: 64`）—— 同时混淆了 `first_failure` 分布与 `reward_type` 分布两个维度。
3. 上一轮 claim 直接以 Tiered review policy 跳过 subagent reviewer，但 P5-02 是 multi-file stage delivery，**不在** "单审计轮 fix" 范围内，必须走完整 subagent reviewer + detached auditor 双闸。

### 修复

1. 在 `scripts/eval_transformers.py` 新增 `_strip_terminal_assistant()`（行 132-145）：从 messages 末尾向前 pop 不带 `tool_calls` 的 assistant 消息，遇到 `tool_calls` 即停；保留中间 assistant `tool_calls` 与 tool 结果消息作为上下文。`_apply_chat_template()` 在调用 `apply_chat_template` 之前先 strip。
2. `tests/test_transformers_backend.py` 新增 `GoldAnswerLeakageTests` 5 个反向断言：
   - `test_strip_drops_terminal_assistant_without_tool_calls` — strip 函数基本行为
   - `test_strip_keeps_assistant_with_tool_calls` — 带 tool_calls 的 assistant 不被 strip
   - `test_apply_chat_template_does_not_include_expected_answer` — gold answer 字符串不会出现在 prompt
   - `test_apply_chat_template_keeps_context_tool_history` — 多轮 tool_calls + tool 结果被保留
   - `test_apply_chat_template_preserves_intermediate_assistant_calls` — 多 assistant tool_calls 全部保留
3. 重跑 5 模型 eval + reward_offline：
   - SmolLM2-360M: binary **0.0000**（原 0.0111 为伪）layered=0.4236
   - Qwen2.5-0.5B: binary=0 layered=0.3634
   - Qwen2.5-1.5B: binary=0 layered=0.3690
   - Qwen2.5-3B: binary=0 layered=0.3333
   - SmolLM2-1.7B: binary=0 layered=0.4236
4. 修正 `docs/protocols/transformers-backend.md` §7 + §8 与 `docs/experiments/p2-evaluator/README.md` §7：reward_type 分布 + first_failure 分布严格按 artifact 字节对齐；标注原版数字为"修复前"以便溯源。
5. 派发 minimax-M3 subagent reviewer（本次）实际产出 PASS verdict（见 `.pi-glla/scratch/stage-p5-02-transformers-backend-review-r2.txt`）。

### 验证

- `scripts/run_tests.py fast` → Ran **270** tests OK（含 P5-02 25 个新增单测）。
- `scripts/validate_stage0.py --examples` → 9/9 PASS。
- 5 个 reward JSON 全部 schema 合法（450 signals）。
- 5 个 `no_failure` 全部 = 0（修复前 SmolLM2-360M 为 1）。
- 修复后文档 reward_layered 数值与 artifact 字节对齐（5/5 模型）。
- 修复后文档 reward_type + first_failure 分布与 `Counter()` 计算字节对齐（25 对全部命中）。
- 工作树干净（`git ls-files --others --exclude-standard` 无输出）。
- reviewer durable evidence 入 `.pi-glla/scratch/stage-p5-02-transformers-backend-review-r2.txt`（`reviewer` dispatch, minimax-cn/MiniMax-M3）。

### 验收

- (a) 5 × 90 = 450 reward_signal schema 合法 ✅
- (b) reward_layered > 0.05 ✅（5 模型全部 ≥ 0.33；reward_binary 全部 = 0 是诚实负结果）
- (c) `docs/experiments/p2-evaluator/README.md` §7 公开模型 reward 表 + 与自研 ckpt 对比 ✅
- (d) `tests/test_transformers_backend.py` **25** 单测全绿 ✅
- (e) reviewer dispatch (minimax-M3) 通过 ✅
- (f) 工作树干净 ✅

### 遗留风险

- P5-03 vLLM backend 未启动；硬件 / 环境需求超出当前阶段，待 P5-02 终审通过后启动。
- D2 数据集扩样到 5000+ **已完成**（HEAD `fdfc519`，见 `docs/plans/reviews/stage-p3-d2-expansion.md` round 14）：5000 samples（4×833 + 2×834）、train 3500 / dev 750 / test 750 严格命中、6 类 ≥833 unique canonical signatures、cross-split canonical disjoint、D2-vs-D1/D1.1 cross_dataset disjoint、datasets/ 已加入 `.gitignore` 并 `git rm --cached` 隔离。
- 5 个公开模型 `reward_binary=0` 为诚实负结果：D2 expected_answer 与公开模型生成的 final_answer 字面不一致；如需严格一致，可加后处理归一化或引入轻量 evaluator prompt。这是 D2 eval 设计（`expected_answer == final assistant content`）的固有约束，不应在 P5-02 范围修改。