# P2 离线 reward 校验：D1 dev 跨 5 个 SFT MVP checkpoint 评测

> 状态：阶段交付（2026-08-28）。
>
> **历史记录**：本实验在 **D1 dev 13 样本** 与 **D2 dev 90 样本（MVP 600 样本版）**两个 split（**MVP 600 样本版历史**，当前 D2 契约为扩样后的 dev 750）上对 SFT MVP 的 5 个 checkpoint 跑 `scripts/reward_offline.py`，把 P1-05 八级分类器的结果映射为可训练 reward 信号（含 reward_type 主导通道 + reward_binary + reward_layered）。P5-02 阶段在扩样后的 D2 dev（750 样本）上重跑 5 公开模型，详见 `docs/protocols/transformers-backend.md` §7-8 与 `docs/experiments/p2-evaluator/README.md` §7。结果用于：(a) 验证 P2 协议的语义一致性；(b) 量化每个 checkpoint 在 held-out split 上的 reward 分布；(c) 作为 P4 GRPO 的前置信号校验；(d) 验证 reward_offline 在多轮 transcript pipeline 上的可移植性。
>
> 当前 D2 契约（round 14，HEAD `fdfc519`）：5000 样本 / train 3500 / dev 750 / test 750；本 README §1-§6 描述的 MVP 600 样本版结果仅为 P2 阶段历史交付，后续 P5-02 / P4 / P5-03 均以 5000 样本版为准。

## 1. 评测对象（5 × 13 = 65 个 reward_signal）

按 `docs/experiments/sft-tool-mvp/README.md` 的"五次实跑"明确收敛：

| # | Checkpoint 名 | 实际产物名 | 架构 / 规模 | 训练配置 | reward_binary | reward_layered | reward_type 分布 |
|---|---|---|---|---|---|---|---|
| 1 | `sft-tool-large-v1` | `artifacts/sft-large-v1-eval-d1dev-reward.json` | Dense large 5.11M | OWT long init + 804 aug k=3 + 2k steps | 0.0000 | 0.0288 | parse_success × 12, argument_correct × 1 |
| 2 | `sft-tool-large-night` | `artifacts/sft-large-night-eval-d1dev-reward.json` | Dense large 5.11M | OWT long init + 5288 × 20k steps | 0.0000 | 0.0385 | parse_success × 13 |
| 3 | `sft-tool-d256-5k-seed42` | `artifacts/sft-d256-5k-seed42-eval-d1dev-reward.json` | Dense d256 12.0M | OWT d256 init + 1500 × 5k steps seed42 | 0.0000 | 0.0000 | parse_success × 13 |
| 4 | `sft-tool-d256-20k-seed42` | `artifacts/sft-d256-20k-seed42-eval-d1dev-reward.json` | Dense d256 12.0M | OWT d256 init + 1500 × 20k steps seed42 | 0.0000 | 0.0769 | parse_success × 13 |
| 5 | `sft-moe-v1` | `artifacts/sft-moe-v1-eval-d1dev-reward.json` | MoE 4-expert (~2M active) | OWT long init + 1500 × 2k steps | 0.0000 | 0.0000 | parse_success × 13 |

> **reward_type 映射契约**：见 `scripts/reward_offline.py::_dominant_reward()`。`execution_correct` 仅在所有适用层通过（`first_failure is None`）时返回；`execution_success` / `result_grounded` / `tool_name_correct` / `argument_value_correct` / `call_plan_matches` 失败映射为 `argument_correct`（因为 parse + schema + argument chain 的下一步受阳）。本轮 5 ckpt 中 `reward_type` 分布 = **64 parse_success + 1 argument_correct**（`sft-tool-large-v1`，其 `first_failure = argument_value_correct`）；零样本进入 execution / grounding failure 分支与 final_answer_correct 分支。

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
| `argument_correct` | 1 × 1 = 1（sft-tool-large-v1） | tool name 通过但 argument_value 不匹配；first_failure = `argument_value_correct` |
| `final_answer_correct` | 0 | 当前 5 ckpt 没有样本在该层首失败 |
| `execution_correct` | 0 | 当前 5 ckpt 没有样本最终全部 8 层通过 |

`argument_correct` 主导（仅 1 个样本）出现在 sft-tool-large-v1，其 `first_failure = argument_value_correct`（参数 `city` 不匹配）：模型输出了 JSON call 且能识别工具名，但 `arguments` 与样本期望值不同。比“输出乱码”略进一步，但仍非完整工具调用。本轮 5 ckpt 没有样本的 `first_failure` 处于 `execution_success` / `result_grounded` 分支（表明模型尚未跨过“参数对”门坎），说明该主导通道今后可能还会出现更多样本。

### 2.4 已知边界

- **D1 dev 仅 13 样本，5 checkpoint × 13 = 65 signals 不是一个有统计意义的 reward benchmark**；
- **MVP 600 样本版历史 D2 dev 90 样本** 是首个有统计意义的 held-out reward split（详见 `docs/protocols/d2-multi-turn.md`）。本节第 6 节已用 round 10 IID stratified shuffle 后的最终多样化数据（**MVP 600 样本版**，不是扩样后的 5000 样本）和真实模型推理完成 5 ckpt × 90 = 450 reward_signal：全部 `reward_binary=0.0`，但 large-v1 有 3 个样本越过解析层，`reward_layered=0.0099`；d256-20k / large-night / d256-5k / moe-v1 均为 0.0000。该结果仍是诚实负结果，且 **600** 个 canonical semantic content 全局唯一、三 split 无语义重叠，说明：(a) D2 dev 评测管线可跑通；(b) 单轮模型基本不具多轮工具调用能力；(c) 需 P5-02 公开 instruction-tuned 模型或自研多轮 SFT 才能获得有意义的 reward 分布。当前 D2 契约为扩样后的 5000 样本 / dev 750 样本（round 14，HEAD `fdfc519`），P5-02 在扩样后的 dev 子集 90 上重跑 5 公开模型（详见 §7 与 `docs/protocols/transformers-backend.md`）。
- `artifacts/d2-mock-reward-d2dev.json`（**MVP 600 样本版历史**：mock transcript 90/90 reward_binary=1.0）仅验证 reward_offline 能消费 D2 dev 多轮 transcript 的**管线兼容性**，不等同于模型推理 reward 评测；
- D1.1 train 50-sample 聚合不是 held-out split，不能用于正式 reward 分布对比；
- 5 ckpt 之外的多 seed 聚合（d256 5k × 3 seeds）属 SFT MVP 阶段的多 seed 协议（P1-03）产出，本表不重复列举；
- 下一阶段：P5-02 公开 instruction-tuned 模型可直接接受 D2 dev 多轮 messages，是首个能在 D2 dev 上获得非退化 reward 分布的场景。

## 3. 文件

| 文件 | 含义 |
|---|---|
| `schemas/reward_signal.schema.json` | reward_signal 结构定义（含 reward_type enum: parse_success / argument_correct / final_answer_correct / execution_correct；task_type enum 含 D1 + D2 六类多轮）|
| `scripts/reward_offline.py` | 离线 reward 计算 + CLI（含 `_dominant_reward()` 选择 reward_type 主导通道）|
| `tests/test_reward_offline.py` | **37** 单测 = 8 层各 ≥ 3 例（共 25：parse_success 4 例 + 其他 7 层各 3 例）+ classifier 一致性（4 例）+ reward_type 映射（7 例：5 主路径 + 2 鲁棒性）+ CLI 聚合（1 例）|
| `tests/test_stage0_schemas.py` | reward schema 正负例测试（full pass / parse fail / 缺 reward_type / 未知 reward_type / reward_binary 越界 / D2 task_type 接受 / 未知 task_type 拒绝）|
| `examples/reward_signals/reward-sample-001.json` | reward_signal 正例样例（execution_correct）|
| `examples/reward_signals/reward-sample-002-parse-fail.json` | reward_signal parse-fail 样例（reward_type=parse_success）|
| `scripts/validate_stage0.py` | 注册 reward schema + 两个 examples |
| `docs/protocols/p2-evaluator.md` | P2 协议文档 |
| `docs/experiments/p2-evaluator/README.md` | 本文件 |
| `artifacts/sft-{large-v1,large-night,d256-5k-seed42,d256-20k-seed42,moe-v1}-eval-d1dev-reward.json` | 5 ckpt × D1 dev 13 = **65** reward_signal + aggregate（gitignored）|
| `artifacts/sft-{large-v1,large-night,d256-5k-seed42,d256-20k-seed42,moe-v1}-eval-d2dev-reward.json` | **MVP 600 样本版历史**：5 ckpt × D2 dev 90 = **450** reward_signal + aggregate（gitignored）|
| `artifacts/d2-mock-reward-d2dev.json` | **MVP 600 样本版历史**：mock transcript pipeline 验证：reward_offline 在 D2 dev 90 多轮 transcript 上输出 90/90 reward_binary=1.0（gitignored）|

## 4. 复现

```bash
# 对单个 checkpoint（5 个 CKPT 中的一个）
.venv/python.exe scripts/reward_offline.py \
    --samples-dir datasets/tool-calling-d1/dev \
    --transcripts artifacts/<ckpt>-eval-d1dev.json \
    --output artifacts/<ckpt>-eval-d1dev-reward.json \
    --checkpoint <ckpt-name>

# 单测（37 个 reward_offline + 5 个 stage0 reward case = 42 个 P2 阶段新增）
.venv/python.exe -m unittest tests.test_reward_offline tests.test_stage0_schemas

# stage0 schema 验证（含 reward_signal 校验）
.venv/python.exe scripts/validate_stage0.py --examples
```

## 5. 下一步

1. **P3 数据版本 D2 多轮对话**：解决 held-out split 不足的问题，让 reward 分布对比有统计意义（本节第 6 节已完成 5 ckpt × D2 dev 真实推理评测）；
2. **P5-02 Transformers backend**：让公开 instruction-tuned 模型能跑同一 reward signal，做公平对比；
3. **P4 GRPO**：当 P3 + P5-02 就位后，把 `reward_binary` / `reward_layered` 作为 GRPO advantage 计算的输入。

## 6. D2 dev 90 样本（**MVP 600 样本版历史**）：5 ckpt 真实推理 reward 评测（2026-08-28，round 10 IID 后）

> **历史记录**：本节是 P2 阶段在 **MVP 600 样本版 D2 dev (90 样本)** 上对自研 5 ckpt 跑真实推理 reward 评测的结果。P5-02 阶段在扩样后的 D2 dev（750 样本）上重跑 5 公开 instruction-tuned 模型，详见 `docs/protocols/transformers-backend.md` §7-8 与 `docs/experiments/p2-evaluator/README.md` §7。当前 D2 契约以 5000 样本版为准（round 14，HEAD `fdfc519`）。

**MVP 600 样本版历史**：P3 阶段交付 D2 dev（90 多轮样本，独立 IID held-out split，与 D1.1 train 不重叠）后，按目标第 5 项对 SFT MVP 的 5 个 checkpoint 在 D2 dev 上跑真实模型推理 + reward_offline。

### 6.1 方法（**MVP 600 样本版历史**）

- 推理入口：`scripts/eval_sft_tool.py --prompt-mode multi_turn`。多轮提示把 D2 样本的完整 `messages` 历史按 SFT 模板序列化为 `### User` / `### Assistant` / `### Result` 轮次，末尾追加 `### Assistant\n`，贪婪生成续写；
- 每个 checkpoint 对 **MVP 600 样本版** D2 dev 90 样本生成 1 个 eval JSON（`artifacts/sft-<name>-eval-d2dev.json`）；
- 再用 `scripts/reward_offline.py` 计算 90 个 reward_signal（`artifacts/sft-<name>-eval-d2dev-reward.json`，**MVP 600 样本版历史**）；
- `schemas/reward_signal.schema.json` 的 task_type enum 已扩展 D2 六类，450 个新 signal 全部通过 Draft202012 校验。

### 6.2 结果（5 ckpt × 90 = 450 reward_signal，round 10 IID 重生成后）（**MVP 600 样本版历史**）

| # | Checkpoint | reward_binary | reward_layered | reward_type 分布 |
|---|---|---:|---:|---|
| 1 | `sft-tool-large-v1` | 0.0000 | 0.0099 | parse_success × 87；argument_correct × 3（argument_value_correct × 2，tool_name_correct × 1） |
| 2 | `sft-tool-large-night` | 0.0000 | 0.0000 | parse_success × 90 |
| 3 | `sft-tool-d256-5k-seed42` | 0.0000 | 0.0000 | parse_success × 90 |
| 4 | `sft-tool-d256-20k-seed42` | 0.0000 | 0.0000 | parse_success × 90 |
| 5 | `sft-moe-v1` | 0.0000 | 0.0000 | parse_success × 90 |

### 6.3 解读（诚实负结果）

- **450/450 reward_binary=0.0**：没有 checkpoint 在 D2 dev 上完整通过八级分类；整体仍是诚实负结果。
- round 10 引入 IID stratified split 后样本内容变化，4 个 ckpt 的 reward_layered 由原来的非零值变为 0.0000；large-v1 仍保留 3 个越过解析层的样本，最终 reward_layered=0.0099、reward_type=`argument_correct`。
- 自研单轮 SFT 模型面对 D2 多轮 transcript（含历史 assistant 工具调用 + tool 结果）仍无法完成正确的多轮工具调用；`reward_layered` 的非零值只表示诊断层部分通过，不代表任务成功。
- **MVP 600 样本版历史**：D2 dev 是首个有统计意义的 held-out split（90 样本 vs 13 样本）；round 10 引入分层 seeded shuffle 后，train/dev/test 三 split 在每个 task_type 上都是严格的 70/15/15 均匀抽样（参见 `docs/protocols/d2-multi-turn.md` §1.1），且 canonical semantic content 600/600 唯一、三 split 交集为空；因此该结果不再是连续语义 slice 的 proxy，而是真实 IID held-out 评测。
- 该结果不贬低 D2 数据集本身：D2 dev 的 expected_tool_calls / depends_on / multi-turn 结构均经 MockExecutor 与 schema 校验正确；round 10 进一步在生成器写盘前验证每个 assistant `tool_call.id` 与对应 tool 消息 `tool_call_id` 双向引用 + name/arguments 匹配 + ordering 正确（见 `docs/plans/reviews/stage-p3-d2-multi-turn.md` Round 10）。
- **后续动作**：P5-02 公开 instruction-tuned 模型（可直接消费多轮 messages）是首个能在 D2 dev 上获得非退化 reward 分布的场景；或先训练自研多轮 SFT（messages 含 assistant + tool 角色）。

### 6.4 复现

```bash
# 5 个 checkpoint 中的 1 个：D2 dev 多轮推理
.venv/python.exe scripts/eval_sft_tool.py \
    --checkpoint artifacts/checkpoints/<ckpt>.pt \
    --samples-dir datasets/tool-calling-d2/dev \
    --output artifacts/sft-<name>-eval-d2dev.json \
    --prompt-mode multi_turn

# reward_offline
.venv/python.exe scripts/reward_offline.py \
    --samples-dir datasets/tool-calling-d2/dev \
    --transcripts artifacts/sft-<name>-eval-d2dev.json \
    --output artifacts/sft-<name>-eval-d2dev-reward.json \
    --checkpoint sft-tool-<name>
```

## 7. P5-02 Transformers backend：5 个公开 instruction-tuned 模型 × D2 dev reward 评测（2026-08-28）

为了在同一 reward pipeline 上与自研 SFT ckpt 公平对比，新增 `scripts/eval_transformers.py` Transformers 推理后端。详细契约见 `docs/protocols/transformers-backend.md`；本节聚焦实验结果与自研对比。

### 7.1 模型矩阵

| # | Model | 规模 | bf16 显存 | 下载通道 |
|---|---|---|---|---|
| 1 | `HuggingFaceTB/SmolLM2-360M-Instruct` | 360 M | ~0.7 GB | HF mirror |
| 2 | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5 B | ~1.0 GB | HF mirror |
| 3 | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5 B | ~3.0 GB | HF mirror |
| 4 | `Qwen/Qwen2.5-3B-Instruct` | 3.0 B | ~6.0 GB | HF mirror |
| 5 | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7 B | ~3.5 GB | HF mirror |

### 7.2 结果（5 × 90 = 450 reward_signal；2026-08-28 下午修正）

| # | Model | reward_binary | reward_layered | reward_type 分布 | first_failure 分布 |
|---|---|---:|---:|---|---|
| 1 | SmolLM2-360M-Instruct | 0.0000 | 0.4236 | final_answer_correct × 15, argument_correct × 75 | tool_name_correct × 75, final_answer_correct × 15 |
| 2 | Qwen2.5-0.5B-Instruct | 0.0000 | 0.3634 | argument_correct × 85, final_answer_correct × 5 | tool_name_correct × 71, schema_valid × 8, argument_value_correct × 6, final_answer_correct × 5 |
| 3 | Qwen2.5-1.5B-Instruct | 0.0000 | 0.3690 | argument_correct × 84, final_answer_correct × 6 | tool_name_correct × 79, schema_valid × 5, final_answer_correct × 6 |
| 4 | Qwen2.5-3B-Instruct | 0.0000 | 0.3333 | argument_correct × 88, final_answer_correct × 2 | tool_name_correct × 67, schema_valid × 19, call_plan_matches × 1, argument_value_correct × 1, final_answer_correct × 2 |
| 5 | SmolLM2-1.7B-Instruct | 0.0000 | 0.4236 | final_answer_correct × 15, argument_correct × 75 | tool_name_correct × 75, final_answer_correct × 15 |

**当前扩样版 D2 dev (750 样本) 子集 90**：450 signals 全 schema 合法；与 `schemas/reward_signal.schema.json` jsonschema Draft202012 一致。

### 7.3 解读（修正后诚实负结果）

- **全部 5 个公开模型 reward_layered ≥ 0.33**，远高于自研 5 ckpt 的 0.0，说明公开模型在多轮工具调用结构化生成上仍优于自研 SFT 模型。
- **全部 reward_binary = 0**：原版 `SmolLM2-360M binary=0.0111` 是 target-answer 泄漏导致（`_apply_chat_template()` 把 D2 整条 `messages` 包含 gold terminal assistant content 透传给 prompt，模型从 prompt 中复述出 expected_answer）。修正后 `_strip_terminal_assistant()` 删除 terminal assistant content message，所有模型 `no_failure=0`。
- **tool_name_correct 仍是主要失败层**：D2 使用 `d1_*` 工具名，公开 instruction-tuned 模型未在 D2 数据集上微调。这是诚实预期的行为，不应解读为模型能力退化。
- **横向对比自研 SFT（**MVP 600 样本版历史**）**：自研 5 ckpt 90/90 reward_layered = 0（全部 parse_success 失败）；公开模型即使不做 D2 微调，reward_layered 仍 ≥ 0.33。说明 D2 难度梯度对自研模型过高（5 ckpt 没有学会按 SFT 模板输出 JSON tool-call），而 instruction-tuned 模型对 system + tools 的 chat template 理解更接近 D2 期望格式。
- **与原版结果对比**（详见 `docs/protocols/transformers-backend.md` §8 修正记录）：reward_layered 数值有少量变化（0.33–0.42 区间），证明原版数字虽方向正确但 reward_binary 数字有误；修正后所有指标严格对应到 artifact。

### 7.4 复现

```bash
export HF_ENDPOINT=https://hf-mirror.com
.venv/python.exe scripts/eval_transformers.py \
    --model <model-id> \
    --samples-dir datasets/tool-calling-d2/dev \
    --output artifacts/<safe-name>-eval-d2dev.json
.venv/python.exe scripts/reward_offline.py \
    --transcripts artifacts/<safe-name>-eval-d2dev.json \
    --samples-dir datasets/tool-calling-d2/dev \
    --checkpoint <model-id> \
    --transcript-kind model_generated \
    --output artifacts/<safe-name>-eval-d2dev-reward.json
```

5 模型产物位于 `artifacts/{qwen2.5,huggingfacetb-smollm2}-*-eval-d2dev.json` 与 `artifacts/{qwen2.5,huggingfacetb-smollm2}-*-eval-d2dev-reward.json`（gitignored）。