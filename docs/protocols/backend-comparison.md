# P5-04 双后端基准对比协议 (Transformers vs vLLM)

> 状态：阶段交付（2026-08-29，list item D round-6 correction）。

本协议固定 `scripts/eval_backend_comparison.py` 的输入 / 输出契约、backend 接口、四轴对比指标与 reduced-precision 复用策略。

## 1. 目标与定位

在 P5-02（Transformers 推理后端）与 P5-03（vLLM 可行性 smoke）已交付的前提下，做 5 公开 instruction-tuned 模型 × 2 后端 × 2 batch size 的横向基准对比，输出 4 轴对比表（latency / throughput / reward_binary / reward_layered）。该协议是 P5-04 list item D 的核心交付。

不包含的内容（明确边界）：

- 不训练 LoRA / 全参数微调；本协议只覆盖推理评测（eval-time only）；
- 不替代 P5-02 (eval_transformers) 与 P5-03 (vllm_smoke)；本协议在两者之上做统一调度与对比聚合；
- 不修改自研 Dense Transformer / MoE 模型的 vLLM 适配（明确 out-of-scope，参考 P5-03 §严格不做）；
- 不接入 vLLM serving / FastAPI / Triton；只使用 vLLM `LLM` Python API；
- 不做 Multi-GPU scaling / concurrent request handling / speculative decoding。

## 2. Backend 抽象接口

```python
class Backend(Protocol):
    name: str

    def setup(self, model_id: str, **kwargs: Any) -> None: ...
    def chat_generate(
        self,
        messages_batch: list[list[dict[str, Any]]],
        tools_batch: list[list[dict[str, Any]] | None],
    ) -> tuple[list[str], list[str]]:
        """Run chat completion for each sample.

        Returns ``(generations, prompt_previews)``:
        - ``generations[i]``: full generated text for sample ``i`` (NOT truncated).
        - ``prompt_previews[i]``: rendered prompt's first 200 chars for sample ``i``;
          the comparison CLI uses this to populate the ``user_turn`` field of
          the persisted row so artifacts match ``eval_transformers._build_eval_row`` exactly.
        """
    def teardown(self) -> None: ...
    def metadata(self) -> dict[str, Any]: ...
```

两个真实实现：`TransformersBackend`（封装 `eval_transformers._greedy_generate` 等价逻辑 + left-padding 适配 batched generation）与 `VLLMBackend`（封装 vLLM `LLM.generate` + 同一 chat template 渲染 + P5-03 WSL2 workarounds）；测试通过脚本嵌入的 `--selftest` 子命令（24 测试点，95 断言）验证。

## 3. CLI 接口

```text
.venv/python.exe scripts/eval_backend_comparison.py \\
    --models <HF id 1> <HF id 2> ... \\
    --backends transformers vllm \\
    --samples-dir datasets/tool-calling-d2/dev \\
    --output-dir artifacts/p5-04-backend-comparison/ \\
    --batch-sizes 1 4 \\
    [--limit N] [--max-new-tokens 256] [--dtype bf16|fp16|fp32]
```

| Flag | 默认 | 含义 |
|---|---|---|
| `--models` | 5 个 P5-02 canonical models | Hugging Face model ids |
| `--backends` | transformers, vllm | 启用的后端列表 |
| `--batch-sizes` | 1 | 每个 backend 调用内部的 sample 数 |
| `--samples-dir` | `datasets/tool-calling-d2/dev` | D2 多轮样本目录；只接受 `*.json` 文件 |
| `--output-dir` | `artifacts/p5-04-backend-comparison/` | 写出 per-run + aggregate |
| `--limit` | 0（=全部） | 评测样本上限（smoke 用） |
| `--max-new-tokens` | 256 | greedy / vLLM sampling budget |
| `--dtype` | bf16 | transformers 用 bf16/fp16/fp32；vLLM 用 bf16/fp16（不支持 fp32） |
| `--device` | auto | transformers 用（cuda/cpu/auto）；vLLM 强制 cuda |
| `--vllm-gpu-mem-util` | 0.85 | vLLM `gpu_memory_utilization` 参数 |
| `--parallel` | off | 跨 model+backend 组合的线程并行（仅限组合之间；同一组合内 batch 仍走串行） |

## 4. 输入数据契约

读 `datasets/tool-calling-d2/<split>/*.json`；每行必须包含：

- `id` — 唯一样本 id（与 D2 schema 一致）；
- `messages` — 多轮消息数组；评估时 `_strip_terminal_assistant()` 会移除末位 assistant 避免 target-answer 泄漏（与 P5-02 round-2 fix 一致）；
- `tools` — OpenAI tool schema 列表；可空（no-tool 任务）；
- `metadata.task_type` — 用于行 schema 兼容性；
- `expected_tool_calls` / `expected_answer` — 用于 P1-05 八级分类器与 reward offline 复算。

## 5. 输出契约

### 5.1 Per-run artifact

每个 (model, backend, batch) 组合写一个 `run_<model_slug>__<backend>__b<batch>.json`：

```json
{
  "summary": {
    "model": "<HF id>",
    "backend": "transformers|vllm",
    "batch_size": 1,
    "samples": <int>,
    "elapsed_s": <float>,
    "per_sample_latency_ms": <float>,
    "throughput_samples_per_s": <float>,
    "reward_binary": <float 0..1>,
    "reward_layered": <float 0..1>,
    "parse_success_count": <int>,
    "parse_success_rate": <float 0..1>,
    "first_failure_distribution": {"<failure>": <count>, ...},
    "backend_metadata": {<version / dtype / device / max_new_tokens>},
    "samples_dir": "<path>"
  },
  "rows": [
    {
      "sample_id": "<id>",
      "task_type": "<task_type>",
      "user_turn": "<prompt preview, ≤200 chars>",
      "generated": "<full generated text>",
      "generated_preview": "<first 200 chars>",
      "extracted_calls": [...],
      "layers": {<P1-05 8 layers>},
      "first_failure": "<failure or None>"
    },
    ...
  ]
}
```

`rows[*]` 与 P5-02 / `eval_sft_tool` 完全一致，可直接喂 `scripts/reward_offline.py --transcripts`。

### 5.2 Aggregate comparison

`--output-dir/comparison.json`：

```json
{
  "runs": [<summary 列表，按 execution 顺序>]
}
```

`--output-dir/comparison.csv`：

```csv
model,backend,batch_size,samples,elapsed_s,per_sample_latency_ms,throughput_samples_per_s,reward_binary,reward_layered,parse_success_rate
HuggingFaceTB/SmolLM2-360M-Instruct,transformers,1,3,2.0,666.6,1.5,0.0,0.42,0.66
HuggingFaceTB/SmolLM2-360M-Instruct,vllm,1,3,1.0,333.3,3.0,0.0,0.42,0.66
```

## 6. 四轴对比指标

| 指标 | 公式 | 含义 |
|---|---|---|
| `per_sample_latency_ms` | `elapsed_s * 1000 / samples` | 单样本端到端耗时（含 model load 仅计入该 batch 内 amortized 时间） |
| `throughput_samples_per_s` | `samples / elapsed_s` | 端到端吞吐（含 prompt render + tokenize + generate + detokenize + classify） |
| `reward_binary` | `(first_failure is None).mean()` | 8 层全过的样本比例；与 P5-02 round-2 fix 后口径一致 |
| `reward_layered` | `mean(sum(layers True) / 8)` | 8 层平均通过率；与 P2 reward offline / P5-02 完全一致 |

`reward_binary` / `reward_layered` 通过 `scripts/reward_offline.compute_reward` 复算，确保与 P5-02 + reward_offline 流水线一致；若 reward_offline 不可用，则自动 fallback 到 layer 直读（仅供占位，不作为正式结论）。

## 7. vLLM 依赖与 workarounds

按 `docs/experiments/p5-03-vllm-feasibility/README.md` §三个必需 workarounds，本协议在 `VLLMBackend.setup()` 内自动应用：

1. `VLLM_WSL2_ENABLE_PIN_MEMORY=1`（WSL2 GPU passthrough）
2. 卸载 `flashinfer-python`（硬要求；运行时通过 `import vllm` 前 setdefault 该 env var 触发降级）
3. `VLLM_USE_FLASHINFER_SAMPLER=0` + `VLLM_ATTENTION_BACKEND=TORCH_SDPA`

启动 vLLM 前请先确认 host 已卸载 flashinfer：

```bash
wsl -d Ubuntu-22.04 -- bash -c "pip3 show flashinfer-python"
# 期望: WARNING: Package(s) not found
```

## 8. 与 P5-02 / P5-03 的关系

| 阶段 | 交付 | P5-04 是否复用 |
|---|---|---|
| P5-02 (eval_transformers.py) | transformers 后端 CLI + round-2 target-answer fix + 5 模型横向评测 | ✅ 直接 import `_build_eval_row` + `_strip_terminal_assistant` |
| P5-02 (reward_offline.py) | P1-05 8 层 → reward_binary/reward_layered 映射 | ✅ 直接 import `compute_reward` + `load_samples` |
| P5-03 (vllm_smoke/smoke.py) | vLLM WSL2 workarounds + 6 GB peak GPU mem | ✅ `VLLMBackend.setup()` 复用 workaround env vars |
| P5-03 (eval_vllm.py) | 不存在 | N/A — P5-04 通过 `VLLMBackend` 内置 vLLM 调用，无独立 eval_vllm.py |

## 9. 复现命令

```bash
# Smoke (mocked, no GPU): in-process self-tests
.venv/python.exe scripts/eval_backend_comparison.py --selftest

# Smoke (real, 1 model + 4 samples):
.venv/python.exe scripts/eval_backend_comparison.py \
    --models HuggingFaceTB/SmolLM2-360M-Instruct \
    --backends transformers \
    --samples-dir datasets/tool-calling-d2/dev \
    --output-dir artifacts/p5-04-backend-comparison/smoke \
    --limit 4

# Full (5 models × 2 backends × 2 batch sizes = 20 runs; exact P5-02 subset = 90 samples):
.venv/python.exe scripts/eval_backend_comparison.py \
    --samples-dir datasets/tool-calling-d2/dev \
    --output-dir artifacts/p5-04-backend-comparison/full \
    --limit 90 \
    --batch-sizes 1 4
```

## 10. 已知边界

- **vLLM 与 transformers 不能在同 GPU 同时运行**：单 host 单 GPU 上 `--parallel` 反而 OOM 概率高；默认 sequential。`--parallel` 仅在多 GPU host 才有意义。
- **Reward fallback 仅供测试**：当 `reward_offline` 因环境问题不可用时（如 vLLM 进程占满 GPU），本协议 fallback 到 layer 直读；该 fallback 数字仅作为 placeholder，正式结论必须来自完整 `reward_offline`。
- **`first_failure_distribution` 在 mock 测试中可能为空**：mock 不一定触发所有 8 个 failure class；正式对比表会按真实生成结果填充。
- **CSV 字段顺序固定**：添加新指标需同时更新 `write_aggregate_comparison` 与本协议；保持 schema version 向前兼容。
