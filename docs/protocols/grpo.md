# P4 GRPO MVP — Group-Relative Policy Optimization Protocol

## 1. 目标与范围

本协议定义 P4 阶段的最小 GRPO（Group Relative Policy Optimization，DeepSeek 2024）训练闭环：

- **Policy**：公开 instruction-tuned Transformers 模型（Hugging Face `transformers` API；CPU bf16/fp32 + CUDA bf16/fp16/fp32 三档）。
- **Rollout**：基于 `_apply_chat_template`（复用 P5-02 backend 的 prompt pipeline；terminal assistant content 已被 strip 避免 target-answer 泄漏）。
- **Reward**：直接调用 `scripts.reward_offline.compute_reward`（P2 8 层 reward 信号：`reward_binary` + `reward_layered`）。
- **Advantage**：标准 group-relative：`adv_i = (r_i − mean(r)) / max(std(r), eps)`。
- **Policy update**：REINFORCE 风格 surrogate loss（`-E[adv · logp(continuation)]`），可选 grad-clipping；不引入 reference policy / KL penalty（属 MVP 范围外）。
- **Checkpoint / resume**：每个 step 落盘 `<dir>/step-<step_id>.json` + `<dir>/state.json`；`--resume-from <state.json>` 恢复 global_step 后继续。

范围说明：本 MVP 不实现完整的 PPO clipping / KL penalty / reference policy；目标是端到端可跑 + 单元可测 + 可 resume 的最小闭环。

## 2. CLI 接口

```
.venv/python.exe scripts/grpo_train.py \
    --policy-model Qwen/Qwen2.5-0.5B-Instruct \
    --samples-dir datasets/tool-calling-d2/train \
    --checkpoint-dir artifacts/grpo-qwen05 \
    --max-steps 8 --k-rollouts 4 --limit 8
```

完整参数清单（`--help`）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--policy-model` | 必填 | HF model id 或本地路径 |
| `--samples-dir` | `datasets/tool-calling-d2/train` | D2 多轮样本目录 |
| `--checkpoint-dir` | 必填 | step artifact + state.json 输出目录 |
| `--resume-from` | `None` | 旧 `state.json` 路径；恢复 global_step + 1 |
| `--max-steps` | `8` | policy-gradient 步数 |
| `--k-rollouts` | `4` | 每 prompt 的 rollout 数 K（GRPO group size） |
| `--learning-rate` | `1e-5` | Adam learning rate |
| `--temperature` | `1.0` | rollout 采样温度；`0` = greedy |
| `--max-new-tokens` | `64` | rollout 续写长度 |
| `--limit` | `8` | 加载样本数上限（smoke） |
| `--seed` | `2026` | 样本 shuffle + rollout 采样 seed |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--max-grad-norm` | `1.0` | grad clip 阈值；`0` = no clip |
| `--smoke-deterministic` | `False` | 强制 greedy + temperature=0 |
| `--reference-model` | `None` | reference policy（MVP no-op） |

## 3. Per-step 流程

每个 `global_step` 包含以下子步骤：

1. **Sample 一个 prompt**：从 `samples-dir` 加载 `min(--limit, len(samples_dir))` 个样本，按 `--seed` shuffle 后顺序消费。
2. **生成 K 个 rollout**：每个 rollout 调用 `_rollout_one()`，输出 `{generated, extracted_calls, rollout_index=k}`。
3. **计算 K 个 reward**：对每个 rollout 构造 transcript row，调用 `scripts.reward_offline.compute_reward`，输出 `{reward_binary, reward_layered, reward_type, first_failure, ...}`。
4. **计算 group-relative advantage**：`_group_relative_advantages(rewards)`。当 group std < eps 时输出全零 advantages（标记 `skipped=True`）。
5. **Policy update**：当至少一个 advantage 绝对值 > eps 时执行；计算 `-adv · logp(continuation tokens)`，反向传播 + Adam + grad clip。
6. **落盘 step artifact**：`<checkpoint-dir>/step-<step_id>.json`，schema 见 `schemas/grpo_step_result.schema.json`。
7. **更新 state**：`<checkpoint-dir>/state.json` 含 `global_step`、`last_step_id`、`completed` 等。

## 4. Determinism

MVP 支持两类 deterministic 模式：

- **默认（`--temperature 1.0`）**：温度采样；rollouts 在同 seed 下字节稳定，但与 floating-point 累加顺序相关。
- **Smoke（`--smoke-deterministic` 或 `--temperature 0`）**：强制 greedy；rollouts 字节级稳定。

Artifact 内置两类 fingerprint：

- `deterministic.rollouts_text_hash`：sorted rollout.generated 的 sha256。
- `deterministic.rewards_text_hash`：sorted reward_layered 字符串表示的 sha256。

Test 用例 `tests/test_grpo_mvp.py::TestHashFunctions` 验证这两类 fingerprint 在 sort 顺序变化时仍稳定，在内容变化时变化。

## 5. Reward 信号与 P2 协议

每个 rollout 的 reward 由 `scripts/reward_offline.py::compute_reward` 产出。该函数：

- 复用 `scripts.classify_tool_failure.classify` 跑 P1-05 8 层分类。
- 输出 schema 见 `schemas/reward_signal.schema.json`：`reward_binary ∈ {0, 1}`、`reward_layered ∈ [0, 1]`、`reward_type ∈ {parse_success, final_answer_correct, argument_correct, execution_correct}`、`first_failure ∈ 8 layer names ∪ {None}`。

GRPO advantage 只使用 `reward_layered`（连续值，标准化后给 policy gradient 提供更细粒度信号）；`reward_binary` 仅作为辅助 metric 落入 step artifact。

## 6. Checkpoint / Resume

MVP persists a **full run state** under ``--checkpoint-dir`` so a
``--resume-from <state.json>`` invocation continues the prior
trajectory exactly:

- ``state.json`` (text): run-level config + step cursor:
  ``{schema_version, global_step, last_step_id, samples_consumed,
  max_steps, k_rollouts, policy_model, seed, learning_rate,
  max_grad_norm, samples_dir, device, completed}``.
- ``state.pt`` (binary): ``{model_state_dict, optimizer_state_dict,
  rng_state}`` where ``rng_state = {python, torch, torch_cuda}``.
  Versioned (``STATE_PT_VERSION = "1.0"``) so future schema changes
  can be rejected gracefully.

  **Round-6 invariant**: ``state.pt`` is ALWAYS written alongside
  ``state.json`` on every step. The previous ``save_every`` knob
  was removed because it was a documented-but-not-actually-
  implemented no-op (round-6 finding). The state.pt always-current
  invariant guarantees that a resume from any step is well-defined.

Resume semantics:

- ``--resume-from <state.json>`` reads the JSON, then loads
  ``<dir>/state.pt`` from the same directory.
- ``model.load_state_dict(state.pt.model_state)`` restores policy
  weights; ``optimizer.load_state_dict(state.pt.optimizer_state)``
  restores Adam moments (with a graceful warning when param groups
  mismatch — the policy trajectory is preserved, the optimizer is
  effectively fresh).
- ``_restore_rng_state(state.pt.rng_state)`` sets Python / Torch /
  CUDA RNGs to the exact pre-step state so resumed rollouts are
  byte-identical to the prior run.
- ``samples_consumed`` is the cursor into the shuffled sample list;
  the resumed run starts at ``samples[cursor]`` and yields
  ``max_steps - global_step - 1`` more triples. This is the
  resume-correct iteration: **the resumed run does not re-use
  already-consumed prompts**.
- The saved ``seed`` is restored so the same shuffled sample order
  reproduces. ``--limit`` and ``samples_dir`` must match the prior
  run; ``--strict-resume-config`` (off by default) aborts on any
  config mismatch.

**Note on `save_every`**: the ``--save-every`` CLI flag and
``save_every`` YAML key were REMOVED in round-6 because the knob
was documented but had no actual effect on state.pt persistence
(state.pt was always unconditionally written). Older YAML configs
that still set ``save_every`` are silently ignored — state.pt is
written on every step regardless. There is no longer any cadence
knob for state.pt; it is unconditionally current.
final step does not.

Verification:

- ``tests/test_grpo_mvp.py::TestSeedAndRng`` — Python / Torch RNG
  seeding + capture / restore round-trip.
- ``tests/test_grpo_mvp.py::TestResumeCursor`` — resume iteration
  must NOT re-use a sample that was already consumed.
- ``tests/test_grpo_mvp.py::TestConfigMatches`` — config diff detection.
- ``tests/test_grpo_mvp.py::TestOptimizerRestore`` — Adam moments
  survive save/load.
- ``tests/test_grpo_mvp.py::TestUnconditionalSmoke`` — full save →
  fresh model → restore → weight comparison, using a ``torch.nn``
  mock (no HF dependency).
- ``tests/test_grpo_mvp.py::TestCheckpointIO`` — state.json round-
  trip + ``completed`` flag + state.pt always-current.
- ``tests/test_grpo_mvp.py::TestRunLoopEndToEnd::test_run_loop_state_pt_written_every_step``
  — regression test for round-6: state.pt is written on every step,
  unconditionally.

## 9. Dtype handling

`--dtype {bf16,fp16,fp32}` is honored on CUDA. CPU always forces
fp32 (low-precision CPU generation is unreliable). The default on
CUDA is bf16. ``_resolve_dtype(name, device)`` returns the
appropriate ``torch.dtype``; warnings are emitted when CPU is
forced to override a user-specified low-precision dtype.

## 10. YAML config loader

`--config <yaml>` reads `policy_model`, `samples_dir`,
`checkpoint_dir`, `resume_from`, `max_steps`, `k_rollouts`,
`learning_rate`, `temperature`, `max_new_tokens`, `limit`, `seed`,
`device`, `max_grad_norm`, `dtype` from a YAML file
(see `configs/grpo_mvp.example.yaml`). (`save_every` was removed
in round-6; older configs with this key are silently ignored.)

**CLI precedence**: CLI args explicitly supplied on the command line
win over YAML. The set of supplied flags is detected by scanning
`sys.argv` for long-form tokens (matches `--flag value` and
`--flag=value`); any YAML key whose corresponding flag was NOT
supplied is overlaid. `_supplied_cli_args(argv)` returns the set.

**Path coercion**: YAML values for `samples_dir`, `checkpoint_dir`,
`resume_from` are coerced to `pathlib.Path` (argparse normally does
this for CLI values via `type=Path`; YAML values are strings so
`_apply_config_defaults` must coerce them). Without this coercion,
`main()`'s `checkpoint_dir.mkdir(...)` call fails with
`AttributeError: 'str' object has no attribute 'mkdir'`.

PyYAML is required (raise `RuntimeError` with install instructions
if missing).

Tests:
- `TestYamlConfigLoader::test_load_example_yaml`
- `TestYamlConfigLoader::test_apply_config_overlays_namespace`
- `TestYamlConfigLoader::test_apply_config_converts_path_values`
- `TestYamlConfigLoader::test_cli_overrides_yaml`
- `TestYamlConfigLoader::test_supplied_cli_args_scan`
- `TestYamlConfigLoader::test_supplied_cli_args_scan_with_equals`
- `TestYamlConfigLoader::test_config_only_main_succeeds`

## 11. Unconditional mock-based smoke

`scripts/grpo_mocks.py` provides `MockTokenizer` + `MockPolicy`
that satisfy the surface area used by the loop
(`apply_chat_template`, `__call__`, `decode`, `generate`, `forward`,
`state_dict`). `tests/test_grpo_mvp.py::TestRunStepWithMockPolicy`
+ `TestRunLoopEndToEnd` exercise the complete loop in-process:

- `_run_step` runs K rollouts → K rewards → K advantages → policy
  update.
- `run_loop` runs the full N-step loop end-to-end, persisting
  `state.json` + `state.pt` + per-step artifacts.
- `run_loop_resume_continues_at_cursor` verifies that a resumed
  run does NOT re-use `samples[0]` (the auditor's round-3 specific
  objection).

These tests do not require HF transformers or a network.

## 12. Optional HF subprocess smoke

`TestGrpoSubprocessSmoke::test_real_hf_cpu_smoke_runs_end_to_end`
runs `scripts/grpo_train.py` as a subprocess against a local HF
model directory and verifies it produces real `state.json` +
`state.pt` + `step-*.json` artifacts. The test automatically picks
the first local snapshot found in:

- `artifacts/huggingface/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/...`
- `artifacts/huggingface/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/...`

The test is skipped unless `GRPO_SMOKE=1` is set. Verified
end-to-end runs:

- CPU (SmolLM2-360M, fp32, greedy, 1 step, k=2): 12.7s elapsed.
- CUDA bf16 (SmolLM2-360M, bf16, greedy, 1 step, k=2): 9.6s
  elapsed. The auditor's round-5 specific objection is closed:
  the codebase ships BOTH a real CPU smoke AND a real GPU smoke.

## 13. Checkpoint correctness invariants (round-5 + round-6 fix)

- `state.pt` is ALWAYS written alongside `state.json` on every step.
  The previous `save_every` skip optimization was a correctness
  bug (interruption could leave state.json newer than state.pt,
  producing a double-update on resume) AND a documentation lie
  (the knob was documented as controlling state.pt cadence but
  had no effect). Round-6 removed `--save-every` entirely; state.pt
  is unconditionally current.
- Cursor exhaustion (`samples_consumed >= len(samples)`) is NOT a
  wrap-to-zero: `run_loop` returns rc=4 with a clear log message
  ("Increase --limit or supply more samples") instead of silently
  reusing already-consumed prompts.
- Resume equivalence: an uninterrupted N-step run and a (N/2)-step
  + resume + (N/2)-step run produce the SAME final weights (when
  policy updates are skipped via deterministic rollouts), the
  SAME step artifact ordering, and the SAME prompt IDs. Test:
  `TestRunLoopEndToEnd::test_run_loop_uninterrupted_vs_resumed_equivalence`.

## 14. Mock-based policy update test (round-5 fix)

`TestRunStepWithMockPolicy::test_run_step_produces_nonzero_update_when_rewards_vary`
constructs a synthetic advantage vector and calls `_policy_update`
directly, then verifies:

- `update["skipped"] == False`
- `update["tokens_seen"] > 0`
- `model.embedding.weight` changed
- `model.head.weight` changed
- `policy_gradient_loss != 0`

The prior mock loop had all-equal rollouts → all-equal rewards →
zero advantages → skipped updates → no weights change. The round-5
fix proves the policy update is REAL, not just a side-effect-free
checkpoint round-trip.

## 15. Production-path schema validation (round-7 fix)

`_validate_step_artifact(artifact)` is the production hook for
step-artifact schema validation. It is called from `run_loop`
**before** any step artifact is written to disk; on schema
violation the run aborts with rc=5 and a clear log message.

The prior `_validate_schema` test helper existed only in tests;
the production loop had no validation hook. Round-7 added the
production hook so emitted artifacts are guaranteed to conform.

Tests:
- `TestValidateStepArtifact::test_well_formed_artifact_passes`
- `TestValidateStepArtifact::test_null_extracted_calls_fails`
  (pre-fixes the artifact to ``extracted_calls: null`` and asserts
  the validator catches it — direct regression for the round-7
  repro)
- `TestRunLoopEndToEnd::test_run_loop_emitted_artifact_passes_schema_validation`
  (end-to-end via the real `run_loop` entry point; every emitted
  step artifact is validated via `jsonschema.validate`).

## 16. Extracted-calls normalization (round-7 fix)

`_rollout_one` now normalizes `extract_tool_calls(generated)`
output: if the helper returns ``None`` (no parseable tool call),
the rollout stores ``[]`` instead of ``null``. The schema requires
``extracted_calls`` to be an array when present; emitting ``null``
violated the schema and was the auditor's round-7 specific
repro. The normalization is at the boundary of ``_rollout_one``
so downstream callers (rewards, artifact assembly) see a uniform
list.

## 17. Documented GPU compatibility boundary (round-7 fix)

The codebase's GPU path is exercised against the installed PyTorch
build + CUDA driver on the host. Verified manually on this host:
RTX 5070 Ti (compute capability 12.0 / sm_120) with PyTorch
2.x + CUDA runtime — 11.7s elapsed for the standard smoke
(1 step, k=2, max-new-tokens=8). The smoke emits bf16 weights
and works without code changes.

If the PyTorch build on a host does not support a given compute
capability, the GPU path raises a runtime error before any
artifact is written; the CPU path remains functional.

## 7. CPU smoke 与 GPU 完整运行

- **CPU smoke**（MVP 默认）：`--device auto` → fallback CPU；`--limit 8 --max-steps 8 --k-rollouts 4`；总耗时数十分钟（受模型大小影响）。`sshleifer/tiny-gpt2`（~2MB）跑通 1 step + K=2 rollout 仅需数十秒。
- **GPU 完整运行**：`--device cuda --max-steps ≥ 100 --k-rollouts 8` + 0.5B+ 模型；显存要求 ≥ 4 GB（bf16 policy + forward + backward）。

## 8. 与 D2 / P5-02 / P2 的衔接

| 阶段 | 共享组件 | GRPO 调用方式 |
|---|---|---|
| D2 | `datasets/tool-calling-d2/{train,dev,test}/` | `--samples-dir` 直接读 train（3500 样本） |
| P2 reward | `scripts.reward_offline.compute_reward` | 每个 rollout 一次 |
| P2 schema | `schemas/reward_signal.schema.json` | reward 字段在 step artifact 中内嵌 |
| P5-02 chat | `scripts.eval_transformers._apply_chat_template` | rollout prompt 渲染 |
| P5-02 classifier | `scripts.classify_tool_failure.classify` | 通过 `compute_reward` 间接调用 |
| P1-05 layers | `EIGHT_LAYERS` in `reward_offline.py` | reward_layered 分母 |

## 9. 已知边界与不在 MVP 范围内的项

- **No reference policy / KL penalty**：MVP 不会自动加载第二个 `transformers` model 来做 KL 约束；`--reference-model` 参数被接受但无副作用（CLI stability）。
- **No PPO clipping**：REINFORCE 风格 surrogate loss，无 clip。后续 P4 完整实现可在此基础上加入。
- **No mixed precision grad scaling**：CPU fp32 + CUDA bf16/raw；不使用 `torch.cuda.amp.GradScaler`。
- **No LoRA / PEFT**：MVP 直接对全部 `requires_grad=True` 参数做更新；显存占用与模型大小成正比。
- **CPU smoke 推荐配置**：`--max-new-tokens 16 --k-rollouts 2 --limit 1 --smoke-deterministic`；耗时约 30 秒（小模型）。
- **断点续训必须用同一 policy_model**：state.json 含 `policy_model` 字段；resume 时不强制校验（实验允许跨 model resume），但推荐保持一致。

## 10. 测试

| 测试类 | 覆盖 |
|---|---|
| `TestGroupRelativeAdvantages` | advantage 计算：零方差、标准化、空输入、K=1 |
| `TestAdvantageStats` | advantage 分布汇总 |
| `TestHashFunctions` | rollouts/rewards hash 顺序无关性 + 内容敏感 |
| `TestLoadSamples` | 样本加载 determinism |
| `TestIterPrompts` | resume_step 迭代逻辑（含负值、超界） |
| `TestCheckpointIO` | state.json 序列化/反序列化 + `completed` flag |
| `TestAssembleStepArtifact` | artifact 装配 + schema 校验 |
| `TestGrpoSmokeIntegration` | end-to-end smoke（gated by `GRPO_SMOKE=1`） |

`scripts/run_tests.py full` → `tests/test_grpo_mvp.py` 自动运行（fast 模块）；smoke 集成测试需显式 `GRPO_SMOKE=1` 才执行。