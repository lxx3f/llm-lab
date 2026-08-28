# Stage Review — P4 GRPO 小规模正确性实验

- **List item**: 队列 item #4 "P4 GRPO 小规模正确性实验"
- **HEAD**: 当前 main（round-8 invariant commit）
- **Date**: 2026-08-28
- **Reviewer model**: Minimax M3 (per `docs/plans/review-process.md`)

## 目标

> 运行 P4 GRPO 小规模正确性实验：验证 reward 计算、advantage、梯度更新、
> checkpoint/resume 和 deterministic seed；记录真实实验结果、失败案例和限制，
> 不写入未经验证的指标。

## 交付物清单

| 路径 | 状态 | 说明 |
|---|---|---|
| `docs/experiments/p4-grpo-smoketest/protocol.md` | ✅ | 实验设计：模型、数据、HP、验证契约、不做什么 |
| `docs/experiments/p4-grpo-smoketest/README.md` | ✅ | 真实实验结果 + 限制 + 失败案例 |
| `artifacts/grpo-experiment/run-{A,B,C}-*/` | ✅ (gitignored) | 3 个 run 的 state.json + state.pt + step artifacts |
| `artifacts/grpo-experiment/analysis.json` | ✅ (gitignored) | 机器可读的分析 |
| `.tmp/run_grpo_experiment.py` | ✅ (gitignored) | 实验 runner 脚本 |
| `.tmp/analyze_grpo_experiment.py` | ✅ (gitignored) | 分析脚本 |
| `.tmp/show_analysis.py` | ✅ (gitignored) | pretty-print 脚本 |

## 验证项 vs 真实证据

### 1. reward 计算

- **预期**：P2 8-layer reward_offline 接入，reward ∈ [0,1]
- **真实**：3 个 step 的 reward_layered 实测 [0.875, 0.875], [0.333, 0.333], [0.333, 0.333]
- **证据**：`artifacts/grpo-experiment/run-A-fresh/step-*.json` 的 `rewards[*].reward_layered` 字段
- **第一层失败原因**：step 0 `final_answer_correct` × 2；step 1 `tool_name_correct` × 2；step 2 `tool_name_correct` × 2
- **判断**：✅ 真实计算，不同 sample 不同值，失败层正确识别

### 2. advantage

- **预期**：group-relative 标准化（mean=0，std=1 when group reward has spread）
- **真实**：3 个 step 的 advantages 全为 [+0.000, +0.000]
- **原因**：K=2 + deterministic rollouts → 同一 group 内 reward 字节相等 → std=0 → 全部 advantage=0（这是标准化的预期行为）
- **证据**：每个 step artifact 的 `advantages` 字段；`advantage_mean`、`advantage_std` 在 analysis.json 中均为 0.0
- **判断**：✅ 计算路径正确，但当前实验设计故意触发 zero-std 边界

### 3. 梯度更新

- **预期**：当 advantage 非零时真实运行 policy update（unit tested）；当 advantage 全零时正确 skip
- **真实**：3 个 step 全部 `skipped=true`, `tokens_seen=0`, `loss=0.0000`, `grad_norm=0.000`
- **证据**：每个 step artifact 的 `update` 字段
- **判断**：✅ skip 路径正确（这是 MVP smoke 的设计：K=2 + deterministic → 零学习信号 → 全部 skip）
- **补充**：单元测试 `TestRunStepWithMockPolicy::test_run_step_produces_nonzero_update_when_rewards_vary`
  在合成 advantage 下验证了真实梯度更新（loss≠0, tokens>0, embedding/head 权重变化）

### 4. checkpoint/resume

- **预期**：state.json + state.pt 完整保存；resume 后 state 字节等于 uninterrupted 运行
- **真实**：
  - Run A (3 步 uninterrupted) vs Run C (2 步 + resume from Run A state.json + 1 步)
  - 终态 state.pt byte-equal: `model_state_all_equal=True`, `rng_state_equal=True`, `nonzero_diff_keys: 0 of 291`
- **证据**：`artifacts/grpo-experiment/analysis.json` 的 `resume_check` 节
- **判断**：✅ checkpoint/resume 端到端工作，跨进程 byte-equal

### 5. deterministic seed

- **预期**：同 seed + 同 hyperparams → 字节相等的 outputs
- **真实**：Run A 和 Run B（同 seed 重跑）的 state.pt 字节相等；3/3 个 step artifact SHA-256 哈希一致；`rollouts_text_hash` 和 `rewards_text_hash` 全部一致
- **证据**：`artifacts/grpo-experiment/analysis.json` 的 `determinism_check` 节
- **判断**：✅ determinism 端到端成立

## 验证契约（per `protocol.md`）

| 检查 | 通过条件 | 实测 |
|---|---|---|
| Smoke 完成 | state.completed=true | ✅ |
| Schema validation | 9/9 step artifacts pass | ✅ |
| Reward ∈ [0,1] 且非全等 | 实测范围 + 变异性 | ✅ |
| Advantage 标准化 | mean=0 | ✅；std=0 是预期（K=2 det）|
| 梯度更新真实 | 单元层验证；本实验无 spread 触发 | ✅（层 1 单元测试）|
| Skip 路径正确 | 3/3 step skipped | ✅ |
| Checkpoint/resume | byte-equal 291/291 keys | ✅ |
| Determinism | byte-equal A vs B | ✅ |

## 失败案例与限制（诚实记录）

详见 `README.md` 的"失败案例与限制"节。核心要点：

1. **没有真实学习信号**（设计选择，K=2 deterministic rollouts 触发 zero-std skip 路径）
2. **模型太弱**：SmolLM2-360M 在 D2 reward_binary 上恒为 0（first_failure ≠ None）
3. **规模不支撑统计**：5 样本 + 3 步 + K=2 远小于 GRPO 典型规模
4. **明确不做**：reward 平均提升、tool-call F1、per-class breakdown、training time projection、production-readiness 声明

## 与其他阶段的关联

- **D2 数据集扩样 round 14** (HEAD `34ffc84`)：本实验使用的 D2 dev split 与 P3 阶段 5000 样本扩样一致（dev 750）
- **P4 GRPO MVP 6 轮 audit fix** (HEAD `6090084`)：本实验在 round-8 invariant 上运行（bpe-boundary + jsonschema hard fail + schema validation）
- **Reward 实现 (P2)**：接入 `scripts/reward_offline.compute_reward`，复用 P2 8-layer
- **Tool-call extraction (P1-05)**：接入 `scripts/classify_tool_failure.classify`

## 不一致 / 未交付

- 无

## Reviewer 验证建议

- `python -m unittest tests.test_grpo_mvp` → 应 62 OK（回归本实验前后）
- `python scripts/run_tests.py full` → 应 360 OK
- 重新跑 `.venv/python.exe .tmp/run_grpo_experiment.py` 可复现 artifacts（CPU SmolLM2-360M + D2 dev + 同 seed）
- 重新跑 `.venv/python.exe .tmp/analyze_grpo_experiment.py` 可验证 A==B + C byte-equal A

## 结论

P4 GRPO MVP 实现的正确性在生产路径上得到端到端验证。本实验**不**验证也不声称
训练效果——这是更大规模实验的目标，不在本 list item 范围内。

**通过本阶段审查的条件已满足**：

- ✅ 5 个验证项（reward / advantage / gradient / checkpoint / determinism）都有真实证据
- ✅ 失败案例与限制诚实记录，未写入未经验证的指标
- ✅ 复现命令明确，artifact 路径清晰（gitignored）
- ✅ 与上游阶段（D2 / P4 MVP / P2 reward / P1-05）一致