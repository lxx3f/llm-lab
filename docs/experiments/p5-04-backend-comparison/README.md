# P5-04 双后端基准对比实验 (Transformers vs vLLM)

> **状态**: 进行中 (list item D 第一轮)。本 README 将在实际 GPU runs 完成 + reward offline 复算后填充最终结果表。
>
> **协议**: `docs/protocols/backend-comparison.md`
> **代码**: `scripts/eval_backend_comparison.py`
> **测试**: `tests/test_eval_backend_comparison.py` (19 单测，全部 PASS)
> **Stage review**: `docs/plans/reviews/stage-p5-04-backend-comparison.md` (PENDING detached auditor dispatch)

## 范围

5 公开 instruction-tuned 模型（P5-02 §8 canonical set）× 2 后端（Transformers bf16 greedy + vLLM 0.27.1）× 2 batch size（1 + 4）= 20 个组合；在 P5-02 benchmark evaluation subset (D2 dev 750 采样) 上跑同一样本；输出 4 轴对比表（latency / throughput / reward_binary / reward_layered）。

## 不做什么（明确边界）

- ❌ 修改自研 Dense Transformer / MoE 模型的 vLLM 适配
- ❌ 训练 / LoRA / 全参数微调
- ❌ vLLM serving / FastAPI / Triton integration
- ❌ Multi-GPU scaling / concurrent request handling / speculative decoding
- ❌ 量化（INT8/INT4 部署优化）
- ❌ 重新训练 P5-02 5 模型 / P5-03 smoke 重新跑

## 5 个公开模型

| 模型 | 来源 | 参数 | dtype | 备注 |
|---|---|---|---|---|
| SmolLM2-360M-Instruct | HuggingFaceTB | 360M | bf16 | 最小模型，对比 baseline |
| SmolLM2-1.7B-Instruct | HuggingFaceTB | 1.7B | bf16 | SmolLM2 中等规模 |
| Qwen2.5-0.5B-Instruct | Qwen | 0.5B | bf16 | Qwen2.5 最小 |
| Qwen2.5-1.5B-Instruct | Qwen | 1.5B | bf16 | Qwen2.5 中等 |
| Qwen2.5-3B-Instruct | Qwen | 3B | bf16 | Qwen2.5 最大；约 6 GB bf16 |

## 测试覆盖 (mocked, no GPU)

`tests/test_eval_backend_comparison.py` — 19 单测，全部 PASS：

- **Backend 抽象**: Protocol typing + TransformersBackend / VLLMBackend metadata
- **Pipeline**: `_strip_terminal_assistant` + `_aggregate_timing` + `_slugify`
- **Run logic**: 单组合 setup/teardown + chat_generate batching + error fallback
- **Output**: per-run artifact writer + aggregate CSV/JSON writer + 空数据 fallback
- **CLI**: default + override + --help exit code

```text
$ pytest tests/test_eval_backend_comparison.py -v
============================= 19 passed in 3.65s ==============================
```

## 实际 GPU runs (TODO)

本节将在 list item D 第二轮 + 实际 GPU runs 完成后填充。

### 计划运行步骤

1. **Smoke (1 model, batch=1, limit=4)**:
   - `python scripts/eval_backend_comparison.py --models HuggingFaceTB/SmolLM2-360M-Instruct --backends transformers --samples-dir datasets/tool-calling-d2/dev --output-dir artifacts/p5-04-backend-comparison/smoke --limit 4`
   - 验证：per-run artifact 落盘 + comparison.csv 1 行 + 数字合理

2. **Full (5 models × 2 backends × 2 batch sizes)**:
   - `python scripts/eval_backend_comparison.py --output-dir artifacts/p5-04-backend-comparison/full --batch-sizes 1 4`
   - 验证：20 per-run artifacts + comparison.csv 20 行

### 预期结果维度

| 模型 | 后端 | batch | latency (ms/sample) | throughput (samples/s) | reward_binary | reward_layered |
|---|---|---|---|---|---|---|
| SmolLM2-360M | transformers | 1 | TBD | TBD | 0 (诚实负) | 0.42 (P5-02 baseline) |
| SmolLM2-360M | vllm | 1 | TBD | TBD | TBD | TBD |
| ... | ... | ... | ... | ... | ... | ... |

(具体数字待 GPU 跑完后填)

## 4 轴对比表 (TODO, 待 GPU runs)

### Latency (ms / sample, 越低越好)

| 模型 | transformers b=1 | transformers b=4 | vllm b=1 | vllm b=4 |
|---|---|---|---|---|
| SmolLM2-360M | TBD | TBD | TBD | TBD |
| ... |

### Throughput (samples / s, 越高越好)

| 模型 | transformers b=1 | transformers b=4 | vllm b=1 | vllm b=4 |
|---|---|---|---|---|
| ... |

### Reward binary (越高越好; 与 P5-02 round-2 fix 后口径一致)

| 模型 | transformers b=1 | vllm b=1 | Δ |
|---|---|---|---|
| ... |

### Reward layered (越高越好; 8 层平均通过率)

| 模型 | transformers b=1 | vllm b=1 | Δ |
|---|---|---|---|
| ... |

## 与其他阶段的关联

- **P5-02 (eval_transformers.py)**: 本协议的 transformers 后端直接复用其 `_build_eval_row` + `_strip_terminal_assistant`；reward_offline 复算路径一致。
- **P5-03 (vllm_smoke)**: 本协议的 vLLMBackend 复用 P5-03 验证的 3 个 WSL2 workarounds；不重新探索可行性。
- **P2 evaluator (reward_offline)**: 4 轴中 reward_binary / reward_layered 直接走 reward_offline.compute_reward；保证与 P5-02 横向对比口径一致。
- **架构项目 (architecture-lab/)**: 本协议不涉及自研模型 vLLM 适配（明确边界）。

## 失败案例与限制

### 限制 1: 单 host 单 GPU 不能 `--parallel`

vLLM 启动时占用 ~6 GB；5 模型 × 2 后端 × 2 batch sizes 串行跑约 30-60 分钟（取决于 `--limit`）。`--parallel` 在单 GPU 上反而 OOM 概率高；默认 sequential。

### 限制 2: vLLM + flashinfer 卸载是硬要求

不卸载 flashinfer 会让 vLLM EngineCore 启动失败（详见 P5-03 README §限制 1）。本协议 `VLLMBackend.setup()` 自动 setdefault 三个 env vars 但不卸载包；用户需先 host 端手动 `pip3 uninstall flashinfer-python`。

### 限制 3: Reward fallback 数字仅供测试

正式结论必须来自完整 `reward_offline.compute_reward`。若 reward_offline 因 vLLM 进程占满 GPU 不可用，本协议自动 fallback 到 layer 直读；该 fallback 数字仅作为 placeholder，不作为正式结论（README 中明确标注 "FALLBACK"）。

## 复现命令

详见 `docs/protocols/backend-comparison.md` §9。

## 结论 (TODO, 待 GPU runs 完成 + stage review 触发后填)
