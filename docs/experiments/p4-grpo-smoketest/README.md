# P4 GRPO 小规模正确性实验 — 真实结果

> **实验性质**：验证 `scripts/grpo_train.py` MVP 实现的**正确性**（reward 计算、
> advantage、checkpoint/resume、deterministic seed）是否与契约一致。
>
> **明确不验证**：训练效果、模型改进、收敛速度、泛化能力。这些都不在 360M 模型 +
> 3 步 + K=2 的实验范围内，任何此类声明都是噪声。
>
> 实验设计详见 [`protocol.md`](./protocol.md)。

## TL;DR

| 验证项 | 结果 | 证据 |
|---|---|---|
| Smoke 完成（3 步） | ✅ | Run A 返回 rc=0；state.json `completed=true` |
| Step artifact schema 验证 | ✅ (9/9) | 所有 `step-*.json` 通过 `jsonschema.validate` |
| Reward ∈ [0,1] 且非全等 | ✅ | 实测 [0.333, 0.875]，不同 sample 不同 |
| Advantage 标准化 | ✅ (但恒为 0) | `mean(adv)==0`；`std(adv)==0` 因为 K=2 deterministic rollouts 给出 identical rewards |
| 梯度更新真实运行 | ⚠️ **无实际梯度** | 全部 3 步 `update.skipped=true`、`tokens_seen=0`（见"限制"） |
| Checkpoint byte-equal across resume | ✅ | Run C 终态 state.pt 与 Run A 终态 state.pt **完全字节相等**（291 keys, 0 nonzero diff） |
| Determinism（同 seed 重跑） | ✅ | Run A 和 Run B 的 `step-*.json` SHA-256 哈希完全一致；`rollouts_text_hash` / `rewards_text_hash` 全部一致 |

**核心结论**：MVP 的实现正确性与契约一致（schema、checkpoint、resume、determinism
全部通过）。**当前实验规模下没有任何学习信号**——这是 by design 的设计选择
（K=2 + deterministic rollouts 导致 identical rewards → 零 advantage spread → 全部
skip），不是 bug。下面会展开。

## 实验配置（真实运行参数）

- **Model**: `SmolLM2-360M-Instruct`（HuggingFaceTB，本地 snapshot）
- **Dataset**: D2 dev split, `--limit 5`, seed=2026 → 实际消费的 3 个 sample: `d2-dev-0091`, `d2-dev-0271`, `d2-dev-0506`
- **Hyperparameters**: `--max-steps 3 --k-rollouts 2 --max-new-tokens 8 --learning-rate 1e-5 --seed 2026 --device cpu --dtype fp32 --smoke-deterministic`

## 真实测量数据

### Run A: Fresh 3-step uninterrupted（baseline）

| step | prompt_id | rewards | advantages | reward_binary | first_failure | skipped | tokens_seen | loss | grad_norm |
|---|---|---|---|---|---|---|---|---|---|
| 0 | d2-dev-0091 | [0.875, 0.875] | [+0.000, +0.000] | [0, 0] | final_answer_correct | true | 0 | 0.0000 | 0.000 |
| 1 | d2-dev-0271 | [0.333, 0.333] | [+0.000, +0.000] | [0, 0] | tool_name_correct | true | 0 | 0.0000 | 0.000 |
| 2 | d2-dev-0506 | [0.333, 0.333] | [+0.000, +0.000] | [0, 0] | tool_name_correct | true | 0 | 0.0000 | 0.000 |

- 总耗时：17.3s（CPU SmolLM2-360M，单步 ~5s）
- 所有 reward_binary=0：SmolLM2-360M 在 D2 8-layer reward 上没有一次全过（model 太弱）
- Rollout 长度：step 0 = 5 字符，step 1 = 14 字符，step 2 = 26 字符（max_new_tokens=8 截断）

### Run B: 同 seed 重跑（determinism check）

与 Run A **完全字节相等**：

- `state.pt` 哈希一致（同一脚本同一 seed → 同一 RNG → 同一训练轨迹）
- 三个 `step-*.json` 文件的 SHA-256 哈希完全一致
- `deterministic.rollouts_text_hash` 三个 step 全部一致
- `deterministic.rewards_text_hash` 三个 step 全部一致
- 总耗时 15.1s（与 Run A 在 ±2s 内，CPU 抖动）

**结论**：determinism invariant 在生产路径上**端到端成立**。同 seed + 同
hyperparameters + `--smoke-deterministic` → 字节相等的 outputs。

### Run C: 2-step + resume from Run A + 1-step（resume check）

- 第一段（2 步）耗时 12.6s
- 第二段（resume from Run A `state.json`，再 1 步）耗时 8.3s
- Run C 终态 `state.pt` 与 Run A 终态 `state.pt` **完全字节相等**：
  - `same_version: true`
  - `rng_state_equal: true`
  - `model_state_all_equal: true`
  - `nonzero_diff_keys: 0 of 291`（所有 291 个权重 tensor 字节相等）
- Run C 完成的两个 step artifact（step 0 + step 1）与 Run A 的 step 0 + step 1 哈希一致

**结论**：resume 端到端工作。`state.pt` 完整保留 model weights + optimizer state
+ RNG state。`resume_from` 路径能恢复跨进程的运行。跨进程 resume 与 uninterrupted
运行给出字节相等的结果（因为全部 update 都被 skip，weights 没有变）。

## 验证清单（与 protocol.md 对应）

| 检查 | 通过条件 | 真实结果 |
|---|---|---|
| Smoke 完成 | `state.completed=true`，3 个 step artifact | ✅ Run A rc=0，state.json completed=true，3 个 step file |
| Schema validation | `jsonschema.validate(art, SCHEMA)` 全部通过 | ✅ 9/9（Run A 3 + Run B 3 + Run C 2 + schema.json 自校验）|
| Reward 真实 | ∈ [0,1]，不全等 | ✅ [0.333, 0.875]，3 个不同 sample 不同 reward |
| Advantage 标准化 | mean=0，std=1 当 reward 有 spread | ✅ mean=0 ✓；std=0 因为 K=2 deterministic rollouts → 同一 group 内 reward 完全相等 |
| 梯度更新真实 | advantage ≠ 0 → tokens>0 + loss≠0 + weight diff | ⚠️ 无任何 advantage ≠ 0 的 step（全部 skip）；weights 实际未变 |
| Skip 路径正确 | advantage 全 0 → skipped=true + tokens=0 | ✅ 3/3 步正确走 skip 路径 |
| Checkpoint/resume | Run C 终态 state.pt == Run A 终态 state.pt | ✅ byte-equal (291/291 keys) |
| Determinism | Run A 和 Run B state.pt byte-equal | ✅ byte-equal，3/3 step artifact hash 一致 |

## 失败案例与限制（诚实记录）

### 限制 1: 没有真实学习信号（核心限制）

所有 3 个 step 都走的是 "all advantages are zero → skipped update" 路径。原因是：

- `--k-rollouts 2` + `--smoke-deterministic` (greedy decoding)：同一个 prompt
  生成的两个 rollout 字节相等 → reward 完全相等 → group-relative advantage
  标准化后恒为 0（std=0 → 全部 advantage=0）→ `_policy_update` 短路到 skipped 分支
- 所以 3 步之后 `state.pt` 与初始 model weights 字节相等

这是 **协议要求** 的限制（K=2 + deterministic 是 MVP smoke 的最小设计），不是 bug：

- 单元测试 `TestRunStepWithMockPolicy::test_run_step_produces_nonzero_update_when_rewards_vary`
  已验证当 advantage 非零时权重会真实变化（用合成 advantage vector，验证 embedding/
  head 权重变化 + loss≠0 + tokens_seen>0）
- 本实验的"无学习信号"反映 MVP smoke 的设计，不是实现 bug

### 限制 2: 模型规模不足以产生"训练效果"

SmolLM2-360M 是本地最小可用 instruction-tuned 模型。它在 D2 8-layer reward 上
`reward_binary=0` 永远不通过（所有 step 的 `first_failure` 都不是 None）：
- step 0: `final_answer_correct` —— 模型生成了回答但与期望答案不匹配
- step 1: `tool_name_correct` —— 模型生成的 tool call 名字错了
- step 2: `tool_name_correct` —— 同上

这意味着即使有真实学习信号（K=8 + temperature=0.7），3 步 + 5 样本也完全不足以
让模型学到正确行为。任何"训练效果"指标（reward 平均提升、tool-call 成功率变化）
在这个规模下都是统计噪声。

### 限制 3: 实验规模

- 5 样本 + 3 步 + K=2：远小于 GRPO 论文的典型规模（数千 prompt × 数十步 × K=8+）
- CPU 单机 fp32：单步 ~5s；GPU bf16 单步 ~0.5s（已 round-8 验证），但本实验选
  CPU 是为了避开 dtype/device 干扰纯实现正确性
- max_new_tokens=8：强制短 rollout，避免 CPU 单步过慢；这意味着 rollout 还没到
  tool-call token 之前就被截断了（看 step 0 generated_length=5 字符）

### 限制 4: 不测量的项（明确不做）

- ❌ Reward 平均提升（baseline vs final）：无意义，详见上文
- ❌ Tool-call F1 / accuracy：5 样本无统计意义
- ❌ Per-class reward breakdown：5 样本不够分桶
- ❌ Training time projection：CPU 单机数据无外推性
- ❌ KL penalty / PPO clipping 路径：MVP 明确不支持（已声明 out-of-scope）
- ❌ Production-readiness：MVP 是 smoke，不是 benchmark

## 已知失败模式与边界

| 场景 | 期望 | 真实 |
|---|---|---|
| K=1 + K=2 deterministic rollouts | advantage 恒为 0，update skipped | ✅ 实测全部 skipped，weights 不变 |
| K=2 + 不同 prompt | 不同 reward（来自 prompt variance） | ✅ 实测 0.875 vs 0.333，prompt 间有 spread |
| `--resume-from` 指向另一 run 的 state.json | 跨进程 resume 端到端 work | ✅ Run C 终态 byte-equal Run A 终态 |
| 同 seed 同 hyperparams | byte-equal outputs | ✅ A 和 B 完全一致 |
| advantage 有 spread 时真正更新权重 | 单元测试已验证（`test_run_step_produces_nonzero_update_when_rewards_vary`）| ✅ 单元层验证；本实验无 spread 触发不到 |

## 复现命令

```bash
# 1. 跑实验（产出 artifacts/grpo-experiment/，在 .gitignore）
.venv/python.exe .tmp/run_grpo_experiment.py

# 2. 分析 artifact（产出 analysis.json）
.venv/python.exe .tmp/analyze_grpo_experiment.py

# 3. 漂亮打印
.venv/python.exe .tmp/show_analysis.py
```

## 关联文件

- [`protocol.md`](./protocol.md) — 实验设计 / 验证契约 / 不做什么
- `artifacts/grpo-experiment/run-{A,B,C}-*/` — 真实 artifact（gitignored）
- `artifacts/grpo-experiment/analysis.json` — 分析 JSON（gitignored）
- `scripts/grpo_train.py` — 主训练脚本（HEAD 当前 main，round-8 invariant）
- `docs/protocols/grpo.md` — 协议文档
- `docs/plans/reviews/stage-p4-grpo-smoketest.md` — 阶段评审

## 结论

**MVP 实现正确性在生产路径上得到端到端验证**。验证范围：schema validation
（round-7+8）、BPE-boundary policy update（round-8）、checkpoint + resume
（跨进程 byte-equal）、deterministic seed（端到端 SHA-256 一致）。

**本次实验没有验证也不声称任何"训练效果"**。SmolLM2-360M + 3 步 + K=2 +
deterministic 的规模组合设计上不允许任何 learning signal。真实的训练效果评估
需要 K≥8、temperature>0、更多 steps、更多数据，这是后续工作的范围（不在本
list item 目标内）。