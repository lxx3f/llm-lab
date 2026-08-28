# P2 离线 reward 校验：D1 dev 跨 5 个 SFT MVP checkpoint 评测

> 状态：阶段交付（2026-08-28）。
>
> 本实验在 D1 dev 13 样本上对 SFT MVP 的 5 个 checkpoint 跑 `scripts/reward_offline.py`，把 P1-05 八级分类器的结果映射为可训练 reward 信号（含 reward_type 主导通道 + reward_binary + reward_layered）。结果用于：(a) 验证 P2 协议的语义一致性；(b) 量化每个 checkpoint 在 held-out split 上的 reward 分布；(c) 作为 P4 GRPO 的前置信号校验。

## 1. 评测对象（5 × 13 = 65 个 reward_signal）

按 `docs/experiments/sft-tool-mvp/README.md` 的"五次实跑"明确收敛：

| # | Checkpoint 名 | 实际产物名 | 架构 / 规模 | 训练配置 | reward_binary | reward_layered | reward_type 分布 |
|---|---|---|---|---|---|---|---|
| 1 | `sft-tool-large-v1` | `artifacts/sft-large-v1-eval-d1dev-reward.json` | Dense large 5.11M | OWT long init + 804 aug k=3 + 2k steps | 0.0000 | 0.0288 | parse_success × 12, argument_correct × 1 |
| 2 | `sft-tool-large-night` | `artifacts/sft-large-night-eval-d1dev-reward.json` | Dense large 5.11M | OWT long init + 5288 × 20k steps | 0.0000 | 0.0385 | parse_success × 13 |
| 3 | `sft-tool-d256-5k-seed42` | `artifacts/sft-d256-5k-seed42-eval-d1dev-reward.json` | Dense d256 12.0M | OWT d256 init + 1500 × 5k steps seed42 | 0.0000 | 0.0000 | parse_success × 13 |
| 4 | `sft-tool-d256-20k-seed42` | `artifacts/sft-d256-20k-seed42-eval-d1dev-reward.json` | Dense d256 12.0M | OWT d256 init + 1500 × 20k steps seed42 | 0.0000 | 0.0769 | parse_success × 13 |
| 5 | `sft-moe-v1` | `artifacts/sft-moe-v1-eval-d1dev-reward.json` | MoE 4-expert (~2M active) | OWT long init + 1500 × 2k steps | 0.0000 | 0.0000 | parse_success × 13 |

> **reward_type 映射契约**：见 `scripts/reward_offline.py::_dominant_reward()`。`execution_correct` 仅在所有适用层通过（`first_failure is None`）时返回；`execution_success` / `result_grounded` 失败映射为 `argument_correct`（因为 argument chain 通过了，但 execution / grounding 失败）。本轮 5 ckpt 没有样本进入 execution / grounding failure 分支（所有 13 个 first_failure 都是 `parse_success`），所以表格里只出现 `parse_success` + 1 个 `argument_correct`（来自 sft-tool-large-v1，其 first_failure = `argument_value_correct`）。

> **Checkpoint 名与产物名**：表头使用 SFT MVP README 的 `sft-tool-*` 命名（说明模型身份）；实际产物文件名省略 `tool-`（如 `sft-large-v1-eval-d1dev-reward.json`）是 orchestrator 原始命名习惯。两者一一对应。

> **reward_binary = 0** 对全部 5 个 checkpoint 成立：没有任何样本的 P1-05 八级全部通过；所有样本的 first_failure 都是 `parse_success`。
> **reward_layered 0.0000 ~ 0.0769**：dense large / d256 20k 略高（部分样本 final_answer_correct 以外的其他层判定 True），但 parse_success 仍是 0/13。
> **reward_type 主导通道**：仅 sft-tool-large-v1 出现 1 个 `argument_correct`（其余 12 个仍为 parse_success）；其余 4 个 checkpoint 13 个样本全为 `parse_success`。

**5 × 13 = 65 个 reward_signal**，全部通过 `schemas/reward_signal.schema.json` jsonschema Draft202012 校验。

## 2. 解读

### 2.1 reward_binary 的一致性结论

`reward_binary` 与 P1-05 八级分类的 `first_failure is None` 完全等价；本批 5 ckpt × 13 样本 = **65 reward_signal 全部为 0**。这与 `docs/experiments/sft-tool-mvp/README.md` 的五次实跑 D1 dev 全 0/13 parse 一致。

### 2.2 reward_layered 的诊断价值

`reward_layered` 给出连续分布，可以区分"模型完全乱码"与"模型能输出最终回答但工具调用格式不对"：

- d256 20k seed42：0.0769
- d256 5k seed42：0.0000
- Dense large night (5.11M + 20k)：0.0385
- Dense large v1 (5.11M + 2k)：0.0288
- MoE v1：0.0000

但这种差异在 reward_binary 上看不见 → `reward_layered` 给 GRPO advantage 提供了连续信号。

### 2.3 reward_type 主导通道分布

| reward_type | ckpt × 样本 = signals | 含义 |
|---|---|---|
| `parse_success` | 5 × 13 几乎全占（64/65） | transcript 不是合法 list-of-dicts；模型输出 JSON 损坏 |
| `argument_correct` | 1 × 1 = 1（sft-tool-large-v1） | 解析后 tool name + args 通过，但 execution/result grounding 不通过 |
| `final_answer_correct` | 0 | 当前 5 ckpt 没有样本在该层首失败 |
| `execution_correct` | 0 | 当前 5 ckpt 没有样本最终全部 8 层通过 |

`argument_correct` 主导（仅 1 个样本）出现在 sft-tool-large-v1，意味着该模型曾经解析出可识别的 JSON call（结构 + 名字 + 参数都通过），但 MockExecutor 未能给出 grounded 结果——比"输出乱码"略进一步，但仍非完整工具调用。

### 2.4 已知边界

- **D1 dev 仅 13 样本，5 checkpoint × 13 = 65 signals 不是一个有统计意义的 reward benchmark**；
- 真正的 reward 验证需建立在**独立的 IID held-out split**（P3 阶段产出）；
- D1.1 train 50-sample 聚合不是 held-out split，不能用于正式 reward 分布对比；
- 5 ckpt 之外的多 seed 聚合（d256 5k × 3 seeds）属 SFT MVP 阶段的多 seed 协议（P1-03）产出，本表不重复列举。

## 3. 文件

| 文件 | 含义 |
|---|---|
| `schemas/reward_signal.schema.json` | reward_signal 结构定义（含 reward_type enum: parse_success / argument_correct / final_answer_correct / execution_correct）|
| `scripts/reward_offline.py` | 离线 reward 计算 + CLI（含 `_dominant_reward()` 选择 reward_type 主导通道）|
| `tests/test_reward_offline.py` | **35** 单测 = 8 层各 ≥ 3 例（共 25：parse_success 4 例 + 其他 7 层各 3 例）+ classifier 一致性（4 例）+ reward_type 映射（5 例）+ CLI 聚合（1 例）|
| `tests/test_stage0_schemas.py` | 新增 reward schema 正负例测试（full pass / parse fail / 缺 reward_type / 未知 reward_type / reward_binary 越界）|
| `examples/reward_signals/reward-sample-001.json` | reward_signal 正例样例（execution_correct）|
| `examples/reward_signals/reward-sample-002-parse-fail.json` | reward_signal parse-fail 样例（reward_type=parse_success）|
| `scripts/validate_stage0.py` | 注册 reward schema + 两个 examples |
| `docs/protocols/p2-evaluator.md` | P2 协议文档 |
| `docs/experiments/p2-evaluator/README.md` | 本文件 |
| `artifacts/sft-{large-v1,large-night,d256-5k-seed42,d256-20k-seed42,moe-v1}-eval-d1dev-reward.json` | 5 ckpt × D1 dev 13 = **65** reward_signal + aggregate（gitignored）|

## 4. 复现

```bash
# 对单个 checkpoint（5 个 CKPT 中的一个）
.venv/python.exe scripts/reward_offline.py \
    --samples-dir datasets/tool-calling-d1/dev \
    --transcripts artifacts/<ckpt>-eval-d1dev.json \
    --output artifacts/<ckpt>-eval-d1dev-reward.json \
    --checkpoint <ckpt-name>

# 单测（30 个 reward_offline + 5 个 stage0 reward case = 35 个）
.venv/python.exe -m unittest tests.test_reward_offline tests.test_stage0_schemas

# stage0 schema 验证（含 reward_signal 校验）
.venv/python.exe scripts/validate_stage0.py --examples
```

## 5. 下一步

1. **P3 数据版本 D2 多轮对话**：解决 held-out split 不足的问题，让 reward 分布对比有统计意义；
2. **P5-02 Transformers backend**：让公开 instruction-tuned 模型能跑同一 reward signal，做公平对比；
3. **P4 GRPO**：当 P3 + P5-02 就位后，把 `reward_binary` / `reward_layered` 作为 GRPO advantage 计算的输入。