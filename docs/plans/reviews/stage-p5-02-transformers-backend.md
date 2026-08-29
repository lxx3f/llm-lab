# Stage review — P5-02 Transformers backend + 5 公开 instruction-tuned 模型 × D2 dev reward

> 阶段：P5-02
> 审查时间：2026-08-28
> 审查流程：subagent reviewer (`reviewer` dispatch, minimax-cn/MiniMax-M3) + detached auditor (calculet/gpt-5.6-terra)
> 当前 D2 契约：5000 样本 / train 3500 / dev 750 / test 750（HEAD：见当前 main；stage review 通过）。本 stage review 评测在 P5-02 benchmark evaluation subset（benchmark 子集规模详见协议 §3）上完成；自研 5 ckpt 在该 benchmark 子集上的历史评测（与本 stage 不同时间点）见 `docs/plans/open-issues.md` P3-01 段（line 769-841）。

## 阶段目标

实现 `scripts/eval_transformers.py` Transformers 推理后端，在 P5-02 benchmark evaluation subset（从当前扩样版 D2 dev 750 中采样的固定 benchmark 子集，**不是 D2 数据集规模或 split 契约**）上对 5 个公开 instruction-tuned 模型跑真实推理 + reward_offline，产出 reward_signal；与自研 5 ckpt 在同一 reward pipeline 上做公平对比。

Done when：
- (a) 5 个公开模型 × P5-02 benchmark evaluation subset signals schema 合法（reward_signal 总数 = 模型数 × benchmark 子集规模）；
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
3. 重跑 5 模型 eval + reward_offline（评测在 P5-02 benchmark evaluation subset 上，benchmark 子集规模详见 `docs/protocols/transformers-backend.md` §3，**不是 D2 数据集规模或 split 契约**）：
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
- 5 个 reward JSON 全部 schema 合法（公开 5 模型 × P5-02 benchmark evaluation subset reward_signal 已落盘；自研 5 ckpt 在同 benchmark 子集上的等量评测归档于 `docs/plans/open-issues.md` round-13 归档段）。
- 5 个 `no_failure` 全部 = 0（修复前 SmolLM2-360M 为 1）。
- 修复后文档 reward_layered 数值与 artifact 字节对齐（5/5 模型）。
- 修复后文档 reward_type + first_failure 分布与 `Counter()` 计算字节对齐（25 对全部命中）。
- 工作树干净（`git ls-files --others --exclude-standard` 无输出）。
- reviewer durable evidence 入 `.pi-glla/scratch/stage-p5-02-transformers-backend-review-r2.txt`（`reviewer` dispatch, minimax-cn/MiniMax-M3）。

### 验收

- (a) reward_signal schema 合法 ✅（P5-02 benchmark evaluation subset，规模详见协议 §3）
- (b) reward_layered > 0.05 ✅（5 模型全部 ≥ 0.33；reward_binary 全部 = 0 是诚实负结果）
- (c) `docs/experiments/p2-evaluator/README.md` §7 公开模型 reward 表 + 与自研 ckpt 对比 ✅
- (d) `tests/test_transformers_backend.py` **25** 单测全绿 ✅
- (e) reviewer dispatch (minimax-M3) 通过 ✅
- (f) 工作树干净 ✅

### 遗留风险

- P5-03 vLLM backend **已交付**（HEAD `bb61a3a` + `docs/plans/reviews/stage-p5-03-vllm-feasibility.md`，WSL2 Ubuntu-22.04 smoke PASS，3 个 WSL2 workarounds 已记录）。P5-02 阶段交付时该 review 记录的当时状态是“未启动”，此后 round 已交付并独立 audit PASS。
- 当前 D2 契约 5000/3500/750/750 已稳定（HEAD：见当前 main；见 `docs/plans/reviews/stage-p3-d2-expansion.md` round 14）：6 类各 833/834 unique canonical signatures、cross-split canonical disjoint、D2-vs-D1/D1.1 cross_dataset disjoint、MANIFEST 含 `build_count` 字段、`datasets/tool-calling-d2/` 加入 `.gitignore` 并 `git rm --cached` 隔离。
- 5 个公开模型 `reward_binary=0` 为诚实负结果：D2 expected_answer 与公开模型生成的 final_answer 字面不一致；如需严格一致，可加后处理归一化或引入轻量 evaluator prompt。这是 D2 eval 设计（`expected_answer == final assistant content`）的固有约束，不应在 P5-02 范围修改。