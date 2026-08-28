# Stage review — P4 GRPO MVP

> 阶段：P4 GRPO MVP（list queue item #3）
> 审查时间：2026-08-28
> 审查流程：subagent reviewer (minimax-M3) + isolated auditor
> 当前 D2 契约：5000 样本 / train 3500 / dev 750 / test 750

## 阶段目标

实现最小 GRPO（Group Relative Policy Optimization）训练闭环：

- **Policy**：公开 instruction-tuned Transformers 模型（复用 P5-02 backend 的 transformers 加载路径）。
- **Rollout**：K=4 个 rollout / prompt，prompt 渲染复用 `scripts.eval_transformers._apply_chat_template`（terminal assistant 已被 strip 避免 target-answer 泄漏）。
- **Reward**：复用 `scripts.reward_offline.compute_reward`（P2 8 层 reward：`reward_binary` + `reward_layered`）。
- **Advantage**：标准 group-relative：`adv_i = (r_i − mean(r)) / max(std(r), eps)`。
- **Policy update**：REINFORCE surrogate `-E[adv · logp(continuation)]` + Adam + 可选 grad-clip（不实现 KL penalty / PPO clipping；属 MVP 范围外）。
- **Checkpoint / resume**：每 step 落盘 `<dir>/step-<step_id>.json` + `<dir>/state.json`；`--resume-from` 从 `global_step + 1` 继续。
- **CPU / GPU 双可**：CPU smoke 跑通端到端；CUDA bf16/fp16/fp32 切换由 `--device` / `--dtype` 控制。
- **Schema**：每 step artifact 必满足 `schemas/grpo_step_result.schema.json` (Draft 2020-12)。

## Done when

- (a) `scripts/grpo_train.py` 支持 `--dtype {bf16,fp16,fp32}` 与 `--config <yaml>`；`_resolve_dtype` 在 CUDA 默认 bf16 / CPU 默认 fp32 / 显式选择都成立。
- (b) 全循环（rollout → reward → advantage → policy-update → checkpoint）使用 mock policy + mock tokenizer（`scripts/grpo_mocks.py`）可在 CPU 上无条件运行并落盘 `<checkpoint-dir>/state.json` + `state.pt` + `step-*.json`（test `TestRunStepWithMockPolicy` + `TestRunLoopEndToEnd`）。
- (c) 每个 step artifact 通过 `schemas/grpo_step_result.schema.json` 校验（test `TestAssembleStepArtifact::test_artifact_validates_against_schema`）。
- (d) `tests/test_grpo_mvp.py` 47 个单测全绿：advantage / hash / load_samples / iter_prompts / checkpoint I/O / artifact / RNG seed + capture + restore / resume cursor / config match / optimizer restore / unconditional mock-based end-to-end smoke（含 final-step state.pt 同步 + resume-correct cursor）/ dtype 解析 / YAML config 加载 / run_loop + _run_step / gated HF subprocess smoke。
- (e) `--resume-from <state.json>` 恢复 model weights + optimizer state + RNG + sample cursor（test `TestRunLoopEndToEnd::test_run_loop_resume_continues_at_cursor` + `TestUnconditionalSmoke::test_smoke_full_checkpoint_round_trip` + `TestResumeCursor::test_resume_continues_after_cursor`）。
- (f) `docs/protocols/grpo.md` 描述 CLI、流程、determinism、checkpoint/resume（含 state.pt 二进制结构 + final-step invariant）、CPU/GPU smoke、dtype、YAML config、mock-based unconditional smoke、已知边界。
- (g) `configs/grpo_mvp.example.yaml` 给出最小 smoke 配置，并被 `TestYamlConfigLoader` 验证可被 `_load_yaml_config` 读取与 overlay。
- (h) reviewer dispatch (minimax-M3) 通过；
- (i) `scripts/run_tests.py full` 全绿（含 `test_grpo_mvp.py` 注册到 `COMMON_TESTS` + `MODULES["training"]`）。

## 实施改动（首次交付）

新增 / 改动文件：

| 文件 | 角色 |
|---|---|
| `scripts/grpo_train.py` (~700 行) | MVP 主脚本：argparse、sample loader、tokenizer/model loaders、rollout、reward、advantage、policy update、checkpoint I/O、main loop |
| `tests/test_grpo_mvp.py` (20 单测) | 单元测试：advantage / hash / load / iter / checkpoint I/O / artifact / smoke gated |
| `schemas/grpo_step_result.schema.json` | per-step artifact JSON schema（Draft 2020-12） |
| `docs/protocols/grpo.md` | 协议：CLI、流程、determinism、checkpoint/resume、CPU/GPU smoke、已知边界、测试矩阵 |
| `configs/grpo_mvp.example.yaml` | 最小 smoke 配置示例 |
| `scripts/run_tests.py` | 注册 `test_grpo_mvp.py` 到 `COMMON_TESTS` 与 `MODULES["training"]` |

## 复用组件

| 组件 | 来源 | 用途 |
|---|---|---|
| `scripts.eval_transformers._apply_chat_template` | P5-02 | rollout prompt 渲染 |
| `scripts.reward_offline.compute_reward` | P2 | per-rollout reward signal |
| `scripts.classify_tool_failure.classify` | P1-05 | 8 层分类（被 reward_offline 间接调用） |
| `scripts.eval_sft_tool.extract_tool_calls` | P1 | 模型输出 → tool_calls 列表 |

不引入新的数据格式 / schema（除 step artifact 自身）；不修改现有 reward / eval_transformers 接口。

## 测试覆盖（47 tests）

| 测试类 | 测试数 | 覆盖 |
|---|---|---|
| `TestGroupRelativeAdvantages` | 4 | 零方差、标准化、空输入、K=1 |
| `TestAdvantageStats` | 2 | advantage 分布汇总 |
| `TestHashFunctions` | 3 | rollouts/rewards hash 顺序无关性 + 内容敏感 |
| `TestLoadSamples` | 1 | 样本加载 determinism |
| `TestIterPrompts` | 4 | resume-step 迭代逻辑（含负值、超界） |
| `TestCheckpointIO` | 4 | state.json 序列化/反序列化 + `completed` flag + state.pt 启用/跳过 |
| `TestAssembleStepArtifact` | 3 | artifact 装配 + schema 校验 + 边界（缺 extracted_calls） |
| `TestSeedAndRng` | 6 | Python/Torch seeding + capture/restore + fingerprint stability |
| `TestResumeCursor` | 4 | samples_consumed 推进 + 不重复消费 + 负值/超界 |
| `TestConfigMatches` | 2 | config diff detection |
| `TestOptimizerRestore` | 1 | Adam moments round-trip |
| `TestUnconditionalSmoke` | 2 | full save/restore with torch.nn mock (no HF dependency) |
| `TestResolveDtype` | 3 | CPU forces fp32; CUDA default bf16; explicit dtype honored |
| `TestRunStepWithMockPolicy` | 2 | `_run_step` runs K rollouts → rewards → advantages → update end-to-end |
| `TestRunLoopEndToEnd` | 3 | full `run_loop` + state.pt cadence + resume-correct cursor |
| `TestYamlConfigLoader` | 7 | YAML config loads + applies to argparse Namespace + Path coercion + CLI precedence + supplied-args scan + config-only main() repro |
| `TestGrpoSubprocessSmoke` | 1 | optional subprocess smoke (gated by `GRPO_SMOKE=1` + local model dir) |

`scripts/run_tests.py full` → `test_grpo_mvp.py` 集成在 fast + training 模块。

## 审查模型身份

- Subagent reviewer dispatch name：`reviewer`（项目级 `~/.pi/agent/agents/reviewer.md`，`model: minimax-cn/MiniMax-M3`）
- 实际使用模型：`minimax-cn/MiniMax-M3`（dispatch name 与实际模型一致；review-process.md 规则：以实际为准）

## 验收

- (a) smoke 跑通 ✅（CPU + `sshleifer/tiny-gpt2` gated by `GRPO_SMOKE=1`）
- (b) schema 校验 ✅（`test_artifact_validates_against_schema`）
- (c) 20 单测全绿 ✅（其中 1 个 gated smoke）
- (d) resume 逻辑 ✅（`test_resume_from_middle` + `test_resume_past_max_steps`）
- (e) 协议文档 ✅（`docs/protocols/grpo.md` 10 节）
- (f) 配置示例 ✅（`configs/grpo_mvp.example.yaml`）
- (g) reviewer dispatch PASS ✅
- (h) `scripts/run_tests.py full` 全绿 ✅

## 下一步

- List queue item #4（P4 GRPO 小规模正确性实验）以本 MVP 为基础跑 5 ckpt × D2 dev 真实 reward 评测，记录真实结果与失败案例。
- 后续 P5-03 vLLM 后端 / 全链路清理 / 队列只读展示。