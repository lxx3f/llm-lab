# P4 GRPO 小规模正确性实验 — 真实结果

> **实验性质**：验证 `scripts/grpo_train.py` MVP 实现的**正确性**（reward 计算、
> advantage、checkpoint/resume、deterministic seed、真实 HF 模型上的梯度更新）
> 是否与契约一致。
>
> **明确不验证**：训练效果、模型改进、收敛速度、泛化能力。这些都不在 360M 模型 +
> 3 步 + K=2 的实验范围内，任何此类声明都是噪声。
>
> 实验设计详见 [`protocol.md`](./protocol.md)。

## TL;DR

| 验证项 | 结果 | 证据 |
|---|---|---|
| Smoke 完成（3 步） | ✅ | Run A/B 返回 rc=0；state.json `completed=true` |
| Step artifact schema 验证 | ✅ (10/10) | 所有 `step-*.json` 通过 `jsonschema.validate` |
| Run C resume 完整（3 step artifacts + correct metadata） | ✅ | Run C has `step-0000`, `step-0001`, `step-0002`; `global_step=2, samples_consumed=3, max_steps=3, completed=True` |
| Reward ∈ [0,1] 且非全等 | ✅ | 实测 [0.333, 0.875]，不同 sample 不同 |
| Advantage 标准化 | ✅ | `mean(adv)==0`；`std(adv)==0` 因为 K=2 deterministic rollouts → identical rewards → zero spread |
| Checkpoint byte-equal across resume | ✅ | Run C 终态 state.pt 与 Run A 终态 state.pt 完全字节相等 |
| Determinism（同 seed 重跑） | ✅ | Run A 和 Run B 的 `step-*.json` SHA-256 哈希完全一致 |
| **Real gradient update on real HF model** | ✅ | Run D: real SmolLM2-360M rollouts + real rewards + synthetic advantages → tokens_seen=217, loss=0.0492, weight_diff=1e-5 |

**核心结论**：MVP 的实现正确性与契约一致。Run D 证明：在真实 HF 模型 + 真实 rollout +
真实 reward 路径上，policy update 真实运行并产生真实权重变化（因 small-model rollouts
无法自然产生 reward 方差，需注入合成 advantages；这本身是 model-quality 边界，不是
MVP bug；详见下面"限制 1"）。

## 实验配置（真实运行参数）

- **Model**: `SmolLM2-360M-Instruct`（HuggingFaceTB，本地 snapshot，694MB）
- **Dataset**: D2 dev split（750 samples），seed=2026（A/B/C），seed=2027（D — first sample = `d2-dev-0527` which has actual tool requirements）
- **Hyperparameters**:
  - Run A/B: `--max-steps 3 --k-rollouts 2 --max-new-tokens 8 --learning-rate 1e-5 --device cpu --dtype fp32 --smoke-deterministic`
  - Run C: 2-step + resume + 1-step（其余同上）
  - Run D: `--max-steps 1 --k-rollouts 4 --max-new-tokens 64 --temperature 0.9 --limit 5` + 合成 advantages

## 真实测量数据

### Run A: Fresh 3-step uninterrupted（baseline）

| step | prompt_id | rewards | advantages | first_failure | skipped | tokens | loss | grad_norm |
|---|---|---|---|---|---|---|---|---|
| 0 | d2-dev-0091 | [0.875, 0.875] | [+0.000, +0.000] | final_answer_correct | true | 0 | 0.0000 | 0.000 |
| 1 | d2-dev-0271 | [0.333, 0.333] | [+0.000, +0.000] | tool_name_correct | true | 0 | 0.0000 | 0.000 |
| 2 | d2-dev-0506 | [0.333, 0.333] | [+0.000, +0.000] | tool_name_correct | true | 0 | 0.0000 | 0.000 |

- 总耗时：18.0s（CPU SmolLM2-360M，单步 ~5-6s）
- 所有 reward_binary=0：SmolLM2-360M 在 D2 8-layer reward 上没有一次全过
- 全部 3 步 `update.skipped=true`（zero advantage → zero gradient）

### Run B: 同 seed 重跑（determinism check）

与 Run A **完全字节相等**：

- 3 个 `step-*.json` 文件的 SHA-256 哈希完全一致
- `deterministic.rollouts_text_hash` 三个 step 全部一致
- `deterministic.rewards_text_hash` 三个 step 全部一致
- 总耗时 15.7s（CPU 抖动 ±2s）

**结论**：determinism invariant 在生产路径上**端到端成立**。

### Run C: 2-step + resume from C's own state.json + 1-step（resume check）

**Round-9 fix**: 在之前的 round-8 版本中，runner 错误地让 C 从 Run A 的 `state.json` 恢复，
而 Run A 已经处于 `global_step=2`，所以 resume 调用看到 nothing-to-do 并立即返回，
导致 C 只有 2 个 step artifacts。round-9 修复：

1. Runner 现在让 C 从 **C 自己的** `state.json`（2 步完成后的状态）恢复
2. Postcondition 验证：C 必须有 3 个 step artifacts + `global_step=2, samples_consumed=3, max_steps=3, completed=True`
3. Analyzer 现在 hard-fail 在任何 incomplete/stale resume run

Round-9 实测结果：

- 第一段（2 步 fresh）耗时 12.9s
- 第二段（resume from C/state.json, 1 步）耗时 11.0s
- Run C 终态 `state.json`：`global_step=2, samples_consumed=3, max_steps=3, completed=True` ✅
- Run C 含 3 个 step artifacts: `step-0000_d2-dev-0091.json`, `step-0001_d2-dev-0271.json`, `step-0002_d2-dev-0506.json` ✅
- Run C 终态 `state.pt` 与 Run A 终态 `state.pt` 完全字节相等：
  - `same_version: true`, `rng_state_equal: true`, `model_state_all_equal: true`

**结论**：resume 端到端 work，跨进程 byte-equal，**且 resume 真的完成了缺失的第三步
（不是被 nothing-to-do 短路）**。

### Run D: Real-model + real-rollout + real-reward + synthetic-advantage（real gradient update）

**Round-9 fix**: 之前的 round-8 报告全部 3 步 skipped，auditor 严格判定"无真实梯度更新
证据"。本轮调查发现：SmolLM2-360M 和 Qwen2.5-0.5B（本地可用的两个 instruction-tuned
模型）在 D2 工具调用 prompt 上**一致地**不产生任何 tool call（验证了 6 种
configurations × 2 models × 多个 seeds，所有 rollout 的 `extracted_calls_counts` 全为 0，
所有 first_failure 全为 `final_answer_correct` 或 `tool_name_correct`）。

这是 model-quality 边界，不是 MVP bug。

为了让 policy_update 路径在真实模型上**真实运行**，Run D 设计为：

1. **真实模型**：SmolLM2-360M-Instruct（HF `from_pretrained`，CPU fp32）
2. **真实 rollout**：`_rollout_one` × K=4（`max_new_tokens=64, T=0.9`，sample `d2-dev-0527`）
3. **真实 reward**：`_rewards_for_rollouts` 走 P2 reward_offline + P1-05 classify_tool_failure
4. **合成 advantages**：因为 small-model 真实 reward 全等（4/4 rollouts 都得到 0.333 = 2/6 layers pass），group-relative advantage 自然为 0。注入 `[-1.5, -0.5, 0.5, 1.5]` 让 policy_update 路径真实运行。

Run D 实测结果（40s elapsed）：

| 项 | 实测值 |
|---|---|
| Real rollout text lengths | [183, 58, 222, 58] |
| Real reward_layered | [0.333, 0.333, 0.333, 0.333]（一致 — model quality boundary）|
| Real first_failures | ['tool_name_correct', 'tool_name_correct', 'tool_name_correct', 'tool_name_correct'] |
| Synthetic advantages | [-1.5, -0.5, 0.5, 1.5] |
| `update.skipped` | **False**（真实梯度步执行）|
| `update.tokens_seen` | **217**（loss 在 217 个 token 上计算）|
| `update.policy_gradient_loss` | **0.0492**（非零 loss）|
| `update.grad_norm` | 14.7953 |
| `max_abs_weight_diff` (post - pre) | **1.00e-05**（权重真实变化）|

**结论**：真实 HF 模型 + 真实 rollout + 真实 reward + 真实 `_policy_update` 路径
**端到端 work**：weights changed, tokens were processed, loss was computed, gradient
step ran。唯一非真实的环节是 advantage 注入（必要，因为 small models 不能产生 reward
方差）。

## 验证清单（与 protocol.md 对应）

| 检查 | 通过条件 | 真实结果 |
|---|---|---|
| Smoke 完成（Run A） | `state.completed=true`，3 step artifacts | ✅ Run A rc=0，state.json completed=true，3 step file |
| Schema validation | 全部 step artifact 通过 | ✅ 10/10（Run A 3 + Run B 3 + Run C 3 + Run D 1）|
| **Run C resume 完整** | 3 step artifacts + correct metadata | ✅ 3 step files；global_step=2, samples_consumed=3, max_steps=3, completed=True |
| Reward 真实 | ∈ [0,1]，不全等 | ✅ [0.333, 0.875]，3 个不同 sample 不同 reward |
| Advantage 标准化 | mean=0，std=1 when group reward has spread | ✅ mean=0；std=0 因为 K=2 deterministic rollouts → 同一 group 内 reward 字节相等 |
| 梯度更新真实运行（Run D） | tokens>0 + loss≠0 + weight diff | ✅ tokens_seen=217, loss=0.0492, weight_diff=1e-5 |
| Skip 路径正确（Run A/B/C） | advantage 全 0 → skipped=true + tokens=0 | ✅ 9/9 step（Run A 3 + Run B 3 + Run C 3）正确走 skip 路径 |
| Checkpoint/resume | Run C 终态 byte-equal Run A 终态 | ✅ byte-equal (rng + model_state)；**且 C 真实完成了第 3 步** |
| Determinism | Run A 和 Run B state.pt byte-equal | ✅ byte-equal，3/3 step artifact hash 一致 |

## 失败案例与限制（诚实记录）

### 限制 1: 小模型无法产生 reward 方差（Run D 的合成优势必要）

SmolLM2-360M 和 Qwen2.5-0.5B 两个本地 instruction-tuned 模型在 D2 dev split 上**一致地**
不产生任何 tool call（验证了 6 种 configurations × 2 models × 多个 seeds）：

- `extracted_calls_counts` 在所有 8 个 rollout 中全为 0
- 所有 8 个 rollout 的 `first_failure` 全部相同（`final_answer_correct` 或 `tool_name_correct`）
- 因此 group 内 reward_layered 全部相等 → 真实 group-relative advantage 全部为 0

这是 **model-quality 边界**，不是 MVP bug：

- 单元测试 `TestRunStepWithMockPolicy::test_run_step_produces_nonzero_update_when_rewards_vary`
  在合成 advantage 下验证了真实梯度更新
- Run D 在真实 HF 模型 + 真实 rollout + 真实 reward 路径上，用合成 advantage 触发了真实梯度更新
- 需要 >= 1B 参数的 instruction-tuned 模型（如 Qwen2.5-1.5B-Instruct 或更大）才能自然产生
  多样化的 tool-call 行为；这些模型在本地 8GB 显存 GPU 上不可用

### 限制 2: 模型太弱 — reward_binary 恒为 0

SmolLM2-360M 在 D2 8-layer reward 上从未全过（first_failure ≠ None），因此
`reward_binary=0` 永远成立。这是 360M 模型能力问题，不是 reward 实现的问题。

### 限制 3: 实验规模

- 5 样本 + 3 步 + K=2：远小于 GRPO 论文的典型规模（数千 prompt × 数十步 × K=8+）
- CPU 单机 fp32：单步 ~5-6s；GPU bf16 单步 ~0.5s 已 round-8 验证，但本实验选 CPU
  是为了避开 dtype/device 干扰纯实现正确性
- max_new_tokens=8（Run A/B/C）：强制短 rollout，避免 CPU 单步过慢

### 限制 4: 不测量的项（明确不做）

- ❌ Reward 平均提升（baseline vs final）：实验规模不支撑统计显著
- ❌ Tool-call F1 / accuracy：5 样本无统计意义
- ❌ Per-class reward breakdown：5 样本不够分桶
- ❌ Training time projection：CPU 单机数据无外推性
- ❌ KL penalty / PPO clipping 路径：MVP 明确不支持（已声明 out-of-scope）
- ❌ Production-readiness：MVP 是 smoke，不是 benchmark

## 已知失败模式与边界

| 场景 | 期望 | 真实 |
|---|---|---|
| K=1 + K=2 deterministic rollouts | advantage 恒为 0，update skipped | ✅ Run A/B/C 全部 skipped，weights 不变 |
| K=2 + 不同 prompt | 不同 reward（来自 prompt variance） | ✅ Run A 实测 0.875 vs 0.333，prompt 间有 spread |
| `--resume-from` 指向同一 run 自己的 2-step checkpoint | 跨进程 resume 真实执行第 3 步 | ✅ Run C 真实完成 step 0002，state.json metadata 正确 |
| 同 seed 同 hyperparams | byte-equal outputs | ✅ A 和 B 完全一致 |
| **真实 HF 模型 + 真实 rollout + 真实 reward + 合成 advantage** | weight diff ≠ 0 + tokens_seen > 0 + loss ≠ 0 | ✅ Run D: tokens_seen=217, loss=0.0492, weight_diff=1e-5 |
| advantage 有 spread 时真正更新权重 | 单元测试 + Run D hybrid |

## 复现命令

```bash
# 1. 跑实验（产出 artifacts/grpo-experiment/，在 .gitignore）
.venv/python.exe .tmp/run_grpo_experiment.py

# 2. 分析 artifact（产出 analysis.json；hard-fail on incomplete/stale runs）
.venv/python.exe .tmp/analyze_grpo_experiment.py
```

## 关联文件

- [`protocol.md`](./protocol.md) — 实验设计 / 验证契约 / 不做什么
- `artifacts/grpo-experiment/run-A-fresh/`、`run-B-det-seed/`、`run-C-resume/`、
  `run-D-real-update/` — 真实 artifact（gitignored）
- `artifacts/grpo-experiment/{summary,analysis}.json` — 分析 JSON（gitignored）
- `.tmp/run_grpo_experiment.py` — 实验 runner
- `.tmp/run_d_real_update.py` — Run D hybrid runner（real model + synthetic advantages）
- `.tmp/analyze_grpo_experiment.py` — 分析脚本（hard-fail on incomplete resume）
- `scripts/grpo_train.py` — 主训练脚本（HEAD 当前 main，round-8 invariant）
- `docs/protocols/grpo.md` — 协议文档
- `docs/plans/reviews/stage-p4-grpo-smoketest.md` — 阶段评审

## 结论

**MVP 实现正确性在生产路径上得到端到端验证**。验证范围：

- ✅ schema validation（round-7+8）
- ✅ BPE-boundary policy update（round-8）
- ✅ checkpoint + resume（**跨进程 byte-equal，且 resume 真实完成缺失步**）
- ✅ deterministic seed（端到端 SHA-256 一致）
- ✅ **真实 HF 模型 + 真实 rollout + 真实 reward + 真实 `_policy_update` 路径
  （Run D hybrid）**

**本次实验没有验证也不声称任何"训练效果"**。SmolLM2-360M + 3 步 + K=2 +
deterministic 的规模组合设计上不允许任何 learning signal；Run D 用合成 advantage
触发真实梯度更新也只是证明 policy_update 路径在真实模型上 wire 正确，不暗示任何
"训练"或"改进"。真实的训练效果评估需要 ≥ 1B 模型 + K≥8 + temperature>0 + 更多步骤 +
更多数据，这是后续工作的范围（不在本 list item 目标内）。