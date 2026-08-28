# P2 离线 reward 校验：D1 dev 跨 checkpoint 评测

> 状态：阶段交付（2026-08-28）。
>
> 本实验在 D1 dev 13 样例上对所有现存 SFT checkpoint 跑 `scripts/reward_offline.py`，把 P1-05 八级分类器的结果映射为可训练 reward binary 与 layered reward。结果用于：(a) 验证 P2 协议的语义一致性；(b) 量化每个 checkpoint 在 held-out split 上的 reward 分布；(c) 作为 P4 GRPO 的前置信号校验。

## 1. 评测对象

| Checkpoint | 训练配置 | 数据 | D1 dev reward_binary | D1 dev reward_layered | first_failure |
|---|---|---|---|---|---|
| sft-tool-large-v1 | Dense large 5.11M / OWT long init | 804 aug k=3 | 0.0000 | 0.0288 | parse_success × 13 |
| sft-tool-d256-5k-seed42 | Dense d256 12M / OWT d256 init | 1500 | 0.0000 | 0.0000 | parse_success × 13 |
| sft-tool-d256-5k-seed123 | 同上 seed123 | 1500 | 0.0000 | 0.0000 | parse_success × 13 |
| sft-tool-d256-5k-seed7 | 同上 seed7 | 1500 | 0.0000 | 0.0000 | parse_success × 13 |
| sft-tool-d256-20k-seed42 | Dense d256 12M / OWT d256 init / 20k 步 | 1500 | 0.0000 | 0.0769 | parse_success × 13 |
| sft-tool-d256-20k-seed123 | 同上 seed123 | 1500 | 0.0000 | 0.0769 | parse_success × 13 |
| sft-tool-d256-20k-seed7 | 同上 seed7 | 1500 | 0.0000 | 0.0385 | parse_success × 13 |
| sft-moe-v1 | MoE 4-expert / OWT long init | 1500 | 0.0000 | 0.0000 | parse_success × 13 |

> **reward_binary = 0** 对所有 8 个 checkpoint 全部成立：没有任何样本的 P1-05 八级全部通过。
> **reward_layered 0.0000 ~ 0.0769**：dense large / d256 20k 的 layered 略高（部分层判定为 True，例如 final_answer_correct 因 expected_answer 缺失为 None 而被跳过），但 parse_success 仍是 0/13 → reward_binary 自然为 0。

## 2. 解读

### 2.1 reward_binary 的一致性结论

`reward_binary` 与 P1-05 八级分类的 `first_failure is None` 完全等价；本批 13 样本 × 8 checkpoint = **104 个 reward_signal 全部为 0**。这与 `docs/experiments/sft-tool-mvp/README.md` 的五/六次实跑 D1 dev 全 0/13 parse 一致。

### 2.2 reward_layered 的诊断价值

`reward_layered` 给出连续分布，可以区分“模型完全乱码”与“模型能输出最终回答但工具调用格式不对”：

- d256 20k × 3 seeds：0.0769（部分样本执行 final_answer_correct 以外的其他层 True）
- d256 5k × 3 seeds：0.0000（更彻底崩塌）
- Dense large v1：0.0288
- MoE v1：0.0000

但这种差异在 reward_binary 上看不见 → reward_layered 给 GRPO advantage 提供了连续信号。

### 2.3 已知边界

- D1 dev 仅 13 样本，**不是一个有统计意义的 reward benchmark**；
- 真正的 reward 验证需建立在**独立的 IID held-out split**（P3 阶段产出）；
- D1.1 train 50-sample 聚合不是 held-out split，不能用于正式 reward 分布对比。

## 3. 文件

| 文件 | 含义 |
|---|---|
| `schemas/reward_signal.schema.json` | reward signal 结构定义 |
| `scripts/reward_offline.py` | 离线 reward 计算 + CLI |
| `tests/test_reward_offline.py` | 5 个单测（full-pass / parse-fail / no_tool / partial-pass / CLI 聚合）|
| `docs/protocols/p2-evaluator.md` | P2 协议文档 |
| `docs/experiments/p2-evaluator/README.md` | 本文件 |
| `artifacts/reward-d1dev-*.json` | 跨 checkpoint 的 reward signal 与 aggregate（gitignored）|

## 4. 复现

```bash
# 对单个 checkpoint
.venv/python.exe scripts/reward_offline.py \
    --samples-dir datasets/tool-calling-d1/dev \
    --transcripts artifacts/<checkpoint>-eval-d1dev.json \
    --output artifacts/<checkpoint>-eval-d1dev-reward.json \
    --checkpoint <checkpoint-name>

# 单测
.venv/python.exe -m unittest tests.test_reward_offline
```

## 5. 下一步

1. **P3 数据版本 D2 多轮对话**：解决 held-out split 不足的问题，让 reward 分布对比有统计意义；
2. **P5-02 Transformers backend**：让公开 instruction-tuned 模型能跑同一 reward signal，做公平对比；
3. **P4 GRPO**：当 P3 + P5-02 就位后，把 `reward_binary` 作为组内 advantage 计算的输入。