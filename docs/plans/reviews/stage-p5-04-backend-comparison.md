# Stage Review — P5-04 双后端基准对比（Transformers vs vLLM）

- **List item**: D
- **状态**: ✅ PASS；本记录为 round-6 correction 后的最终 stage-review record
- **日期**: 2026-08-29
- **当前提交指针**: 使用抽象 HEAD 指针；审核时运行 `git rev-parse HEAD` 核验，不在 active verification 中固化 SHA
- **Canonical reviewer**: `minimax-cn/MiniMax-M3`，符合 `docs/plans/review-process.md`

## 1. Objective 与范围

完成 5 个公开 instruction-tuned 模型 × 2 个推理后端 × 2 个 batch size 的双后端基准对比，共 **20 个真实 GPU 组合**。评测必须使用 P5-02 已使用的 benchmark evaluation subset，而不是任意 D2 前缀子集。

本阶段最终固定为：

- **样本集合**: `datasets/tool-calling-d2/dev/d2-dev-0001.json` 到 `d2-dev-0090.json`，按文件名排序，共 90 个样本；固定 ID 清单保存在 `docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset-ids.txt`；
- **P5-02 对齐证据**: P5-02 artifact `artifacts/huggingfacetb-smollm2-360m-instruct-eval-d2dev.json` 的 row ID 为 `d2-dev-0001`…`d2-dev-0090`，与上述 manifest 精确一致；
- **模型**: `HuggingFaceTB/SmolLM2-360M-Instruct`、`HuggingFaceTB/SmolLM2-1.7B-Instruct`、`Qwen/Qwen2.5-0.5B-Instruct`、`Qwen/Qwen2.5-1.5B-Instruct`、`Qwen/Qwen2.5-3B-Instruct`；
- **后端**: Transformers greedy bf16 与 vLLM 0.27.1 greedy bfloat16；
- **batch size**: 1、4；
- **生成设置**: `max_new_tokens=64`、decoder-only tokenizer `padding_side="left"`；
- **指标**: `per_sample_latency_ms`、`throughput_samples_per_s`、`reward_binary`、`reward_layered`，以及同模型同 batch 的 vLLM 相对 Transformers Δ%；
- **边界**: 不修改自研模型、不重跑 P5-02/P5-03 训练、不做 vLLM serving 或多 GPU 扩展。

## 2. 交付物与文件范围

任务要求的当前交付文件为唯一新增脚本与 `docs/*`：

| 路径 | 状态 | 说明 |
|---|---|---|
| `scripts/eval_backend_comparison.py` | ✅ | Backend Protocol、TransformersBackend、VLLMBackend、批处理、计时、reward 聚合、Δ% 计算、CSV/JSON 输出、`--selftest` |
| `docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset-ids.txt` | ✅ | 90 个 P5-02 benchmark sample ID 的固定 manifest |
| `docs/protocols/backend-comparison.md` | ✅ | 输入/输出契约、Backend tuple、指标公式、复现命令、边界与缓存说明 |
| `docs/experiments/p5-04-backend-comparison/README.md` | ✅ | 90 样本实际结果、四轴表格、同模型 Δ% 表格、关键发现与限制 |
| `docs/plans/reviews/stage-p5-04-backend-comparison.md` | ✅ | 本最终 stage-review record、实时验证命令与 reviewer verdict |

round-1 曾临时添加的独立测试文件已删除；当前树不再跟踪该越出范围的文件，测试逻辑保留在脚本 `--selftest` 子命令中。

## 3. 实际 GPU 运行证据

运行命令：

```bash
wsl -d Ubuntu-22.04 -- bash -c "cd /mnt/c/Users/23236/repositories/llm-lab && \
  python3 scripts/eval_backend_comparison.py \
    --samples-dir datasets/tool-calling-d2/dev \
    --output-dir artifacts/p5-04-backend-comparison/full \
    --limit 90 --batch-sizes 1 4 --max-new-tokens 64"
```

实际结果：

- `artifacts/p5-04-backend-comparison/full/` 有 **20** 个 `run_*.json`；
- 每个 run 有 90 个 row，总计 **1800 rows**；
- `comparison.csv` 有 20 个数据行；
- `comparison_delta.csv` 有 10 个 `(model, batch_size)` 数据行；
- 20 个 run 的 vLLM metadata 记录 version `0.27.1`；
- 所有 1800 个 row 的 `user_turn` 非空；
- 所有 run 的 sample ID 顺序与 `p5-02-benchmark-subset-ids.txt` 一致；
- artifacts 按项目规则仅保留本地，不纳入源码提交。

可复核命令：

```bash
ls artifacts/p5-04-backend-comparison/full/run_*.json | wc -l
# 20

python - <<'PY'
import glob, json
from pathlib import Path
base = Path('artifacts/p5-04-backend-comparison/full')
files = sorted(base.glob('run_*.json'))
manifest = Path('docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset-ids.txt').read_text(encoding='utf-8').splitlines()
expected = [f'd2-dev-{i:04d}' for i in range(1, 91)]
assert manifest == expected
all_rows = []
for path in files:
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert set(payload) == {'summary', 'rows'}
    assert payload['summary']['samples'] == 90
    assert [row['sample_id'] for row in payload['rows']] == expected
    assert all(row['user_turn'] for row in payload['rows'])
    all_rows.extend(payload['rows'])
assert len(files) == 20
assert len(all_rows) == 1800
print('20 runs / 1800 rows / exact ordered P5-02 manifest / non-empty user_turn: PASS')
PY

python - <<'PY'
from pathlib import Path
assert sum(1 for _ in Path('artifacts/p5-04-backend-comparison/full/comparison.csv').open(encoding='utf-8')) - 1 == 20
assert sum(1 for _ in Path('artifacts/p5-04-backend-comparison/full/comparison_delta.csv').open(encoding='utf-8')) - 1 == 10
print('comparison.csv=20 rows / comparison_delta.csv=10 rows: PASS')
PY
```

## 4. 结果摘要

结果来源：

- `artifacts/p5-04-backend-comparison/full/comparison.csv`
- `artifacts/p5-04-backend-comparison/full/comparison_delta.csv`
- `docs/experiments/p5-04-backend-comparison/README.md`

同模型 Δ% 公式：

```text
(vllm_value - transformers_value) / abs(transformers_value) * 100
```

90 样本下：

- 10/10 `(model, batch_size)` pairs 的 vLLM latency 更低，Δ% 为 -36.64% 到 -53.53%；
- 10/10 pairs 的 vLLM throughput 更高，Δ% 为 +57.83% 到 +115.20%；
- 20 个 run 的 `reward_binary` 均为 0.0，符合 P5-02 round-2 target-answer leakage 修复后的诚实负结果；
- `reward_layered` 最大 backend 差异为 Qwen2.5-0.5B batch=4 的 -15.52%，已在 README 中明确记录；
- README 已包含 latency、throughput、reward_binary、reward_layered 四轴表，以及 latency/throughput/reward-layered Δ% 表。

## 5. 正确性与回归验证

### 5.1 Embedded self-test

```bash
python scripts/eval_backend_comparison.py --selftest
```

结果：**95 个断言全部 PASS**。覆盖 Backend tuple 接口、terminal assistant removal、timing 边界、mocked run pipeline、setup/teardown、batch/error fallback、JSON/CSV writer、CLI 参数和 Δ% 边界。

### 5.2 Full project tests

```bash
python scripts/run_tests.py full
```

结果：**386 tests OK，skipped=3**，无项目回归。

## 6. 审核时间线

| Round | 历史提交 | 结果 |
|---|---|---|
| 1 | `a56c9be` | 统一 backend scaffold 与文档初稿 |
| 2 | `ccf9ade` | 修复 `user_turn` 空值并加入 Δ% 计算 |
| 3 | `3f95bbe` | 修复 dtype、offline revision、HF cache 与 batched padding |
| 4 | `cea6850` | 同步协议、README 与 stage-review 初稿 |
| 5 | `41e4641` | 删除超出任务范围的独立测试文件、嵌入 `--selftest` |
| 6 | 当前 correction | 固定 P5-02 精确 90 样本 manifest，完成 20 组合正式重跑，清理 stale prose |

历史提交 SHA 仅描述过去变更；当前树状态由本文件中的 live commands 核验。

## 7. Detached reviewer verdict

- **Agent**: `reviewer`
- **PI_PROVIDER**: `minimax-cn`
- **PI_MODEL**: `MiniMax-M3`
- **Dispatch**: 本轮 correction 提交后独立 read-only 核验
- **Verdict**: **PASS**
- **Critical findings**: none
- **Warnings**: none
- **Suggestions**: 两项非阻塞文档/代码可读性建议，均不影响目标完成

本轮 reviewer 核验了：

- `git rev-parse HEAD` = 当前 correction commit，工作树 clean；
- `git ls-files tests/test_eval_backend_comparison.py` 为空；
- `p5-02-benchmark-subset-ids.txt` 为严格有序的 90 个 ID：`d2-dev-0001`…`d2-dev-0090`；
- 20 个真实 GPU run artifacts，每个 90 rows，顺序与 manifest 完全一致；
- `comparison.csv` 20 行、`comparison_delta.csv` 10 行；
- 1800/1800 个 `user_turn` 非空；
- `python scripts/eval_backend_comparison.py --selftest` 全部 PASS；
- `python -m py_compile scripts/eval_backend_comparison.py` 通过；
- `compute_delta_percentages()` 公式和正负号约定正确；
- protocol、README 与 stage review 没有 stale active 30-sample/test-file/PENDING claim。

### Reviewer findings

#### Critical
none.

#### Warnings
none.

#### Suggestions
- `compute_delta_percentages()` 的 `vl_val is None` 防御判断可进一步前置；当前逻辑已经正确，非阻塞。
- README 中约 30 分钟 wall-clock 说明出现多次；当前内容一致，非阻塞。

## 8. Final disposition

**PASS / complete**：P5-04 的代码入口、双 backend 实现、20 个真实组合、P5-02 精确 benchmark subset、四轴指标、同模型 Δ%、reader-facing README、protocol 与 stage-review record 均已交付。剩余事项（更大数据集、服务化、多 GPU、自研模型 vLLM 适配）明确属于后续阶段，不阻塞本目标。
