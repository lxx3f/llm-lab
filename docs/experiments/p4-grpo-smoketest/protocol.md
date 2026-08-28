# P4 GRPO 小规模正确性实验 — Protocol

## 目标

验证 P4 GRPO MVP（`scripts/grpo_train.py`）在**真实** D2 dev 样本 + 公开
instruction-tuned Transformers 模型上：

1. **reward 计算** 正确（P2 reward_offline 接入，8 层失败分类 + reward_layered + reward_binary）
2. **advantage** 标准化正确（group-relative, zero-variance → 0 advantages → skipped update）
3. **梯度更新** 真实运行（loss / tokens_seen / grad_norm 真实数据；BPE 边界对齐）
4. **checkpoint/resume** 端到端 work（state.json + state.pt + 恢复后权重 byte-equal）
5. **deterministic seed** 端到端 work（同 seed 重跑 → artifact 哈希一致）

明确**不**做什么：

- 不写入"训练效果"指标（reward 平均提升 / policy 改进 / 泛化能力等）——360M 模型 + 5 步 GRPO 不足以产生统计上有意义的训练效果
- 不声称算法正确性 —— 我们只验证**实现**与**契约**一致
- 不外推到"生产 GRPO 训练可行"——这是 MVP smoke，不是 benchmark

## 实验设计

### Model

- **SmolLM2-360M-Instruct**（HuggingFaceTB，本地 snapshot 在 `artifacts/huggingface/models--HuggingFaceTB--SmolLM2-360M-Instruct/`）
- 选用原因：360M 是本地最小可用的 instruction-tuned 模型；CPU 单步 1-3s 可控；行为可观察；真实 BPE tokenizer 暴露 round-8 边界问题
- Dtype：CPU fp32（避免 dtype / 精度干扰纯实现正确性验证）

### Dataset

- **D2 dev split**（750 samples，本地 `datasets/tool-calling-d2/dev/`）
- `--limit 5`：取前 5 个 shuffled 样本（seed=2026，与默认一致）
- 选用 dev 而非 train 的原因：dev 样本量小（750 vs 3500），更易审计；样本 shape 已知；不污染 train 的训练契约

### Hyperparameters

| 参数 | 值 | 说明 |
|---|---|---|
| `--max-steps` | 3 | 最小可验证步数（1 步无法测 resume，5+ 步会超过 GPU/CPU 时间预算） |
| `--k-rollouts` | 2 | GRPO group size K（最小有意义：1 时 advantage 恒为 0，3+ 时 CPU smoke 过慢） |
| `--max-new-tokens` | 8 | 缩短 rollout 时间（8 token ≈ 1-3s CPU） |
| `--learning-rate` | 1e-5 | Adam 默认 |
| `--smoke-deterministic` | true | greedy decoding，温度=0（保证 rollout 可复现） |
| `--device` | cpu | fp32，可控可审计 |
| `--dtype` | fp32 | CPU 强制 fp32 |

### 实验矩阵

3 个 run：

| Run | 路径 | 目的 | 关键测量 |
|---|---|---|---|
| A: Fresh | `artifacts/grpo-experiment/run-A-fresh/` | 基线 3 步 uninterrupted | reward, advantage, loss, tokens_seen, weights-change |
| B: Determinism | `artifacts/grpo-experiment/run-B-det-seed/` | 同 seed 重跑 run A 步骤 0-2 | artifact hash 字节相等（rollouts_text_hash, rewards_text_hash） |
| C: Resume | `artifacts/grpo-experiment/run-C-resume/` | 2 步 + resume + 1 步 | final state.json == run A 的 final state.json（除 samples_consumed） |

### 输出物

每个 run 产出：
- `<dir>/state.json`（run-level config + step cursor）
- `<dir>/state.pt`（model weights + optimizer state + RNG state，PyTorch binary）
- `<dir>/step-0000_*.json`、`step-0001_*.json`、`step-0002_*.json`（per-step artifact）

每个 artifact 包含：
- `rollouts` (K=2): `{generated, extracted_calls}` for each rollout
- `rewards` (K=2): `reward_binary`, `reward_layered`, `first_failure`, `layers`
- `advantages` (K=2): group-relative standardized
- `update`: `{policy_gradient_loss, tokens_seen, grad_norm, skipped, skip_reason}`
- `deterministic.rollouts_text_hash`, `rewards_text_hash`（SHA-256）

### 验证检查

| 检查 | 通过条件 |
|---|---|
| Smoke 完成 | `state.completed == true`; 3 个 `step-*.json` 都通过 schema validation |
| Reward 真实 | 每个 step 的 `rewards[*].reward_layered ∈ [0, 1]`；不全为 0 也不全为 1 |
| Advantage 标准化 | `mean(advantages) == 0`；`std(advantages) == 1` 当 group 内 reward 有 spread |
| 梯度更新真实 | 当 advantage 非全 0 时：`tokens_seen > 0`, `policy_gradient_loss != 0`; weight diff (`final - initial`) 非 0 |
| Skip 路径 | 当 advantage 全 0 时：`update.skipped == true`, `tokens_seen == 0`, grad_norm == 0 |
| Checkpoint/resume | Run C final state.pt byte-equal Run A final state.pt（用 `torch.equal` + `torch.allclose`） |
| Determinism | Run B final state.pt byte-equal Run A final state.pt（同 seed）; `rollouts_text_hash` 完全一致 |

### 不报告的指标（明确不做）

- ❌ Reward 平均提升（baseline vs final）：实验规模不支撑统计显著
- ❌ Tool-call F1 / accuracy：5 样本 + 3 步毫无意义
- ❌ Per-class reward breakdown：5 样本不分类
- ❌ Training time projection：CPU 单机数据无外推性
- ❌ 任何与 production-readiness 相关的声明

### 限制

1. **模型规模**：SmolLM2-360M 是本地最小可用模型；它没有足够能力学会 D2 工具调用。任何"训练效果"信号都是噪声。
2. **数据规模**：5 样本 + 3 步远小于 GRPO 论文的典型规模（数千 prompt × 数十步）。
3. **设备**：CPU 单机；无分布式、无大批量、无 fp16/bf16 影响。
4. **算法范围**：MVP 是 REINFORCE-style + group-relative advantage；不含 KL penalty、PPO clipping、reference model。这些都是已声明的 out-of-scope。
5. **奖励稳定性**：P2 8-layer reward 在 360M 模型的 low-quality 生成上可能恒为 0；这本身是 model quality 问题，不是 reward 实现的 bug。

## 工具

- `scripts/grpo_train.py`：主训练脚本（HEAD 当前 main，round-8 invariant：bpe 边界 + jsonschema hard fail）
- `docs/protocols/grpo.md`：协议文档
- `docs/plans/reviews/stage-p4-grpo-mvp.md`：阶段评审
- `datasets/tool-calling-d2/MANIFEST-dev.json`：D2 dev 契约
- `tests/test_grpo_mvp.py`：62 unit tests（baseline 正确性）

## 工作目录

- 实验 artifact 路径：`artifacts/grpo-experiment/run-{A,B,C}-*/`（在 `.gitignore` 的 `artifacts/**/*.json` 范围内，不入仓）
- 结果分析：`docs/experiments/p4-grpo-smoketest/README.md`（可入仓）
- Stage review：`docs/plans/reviews/stage-p4-grpo-smoketest.md`（可入仓）