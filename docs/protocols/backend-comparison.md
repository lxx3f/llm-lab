# P5-04 双后端基准对比协议 (Transformers vs vLLM)

> 状态：阶段交付（2026-08-29，list item D round-10）。Manifest-driven 90 样本 + per-sample batch retry + generation-failure tracking + reward 计算以 manifest sample 为准。
>
> 范围决策（用户确认）：P5-02 benchmark evaluation subset = P5-02 历史真实运行并冻结的 90 样本（6 task type × 15，source commit `b4fd879`）。当前 D2 dev split 的 750 是父 split 总规模，不是本阶段已发布 benchmark 的运行规模。

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

两个真实实现：`TransformersBackend`（封装 `eval_transformers._greedy_generate` 等价逻辑 + left-padding 适配 batched generation）与 `VLLMBackend`（封装 vLLM `LLM.generate` + 同一 chat template 渲染 + P5-03 WSL2 workarounds）；测试通过脚本嵌入的 `--selftest` 子命令（~42 测试点，240 断言 PASS：round-12 manifest 干净 checkout 复原 + round-13 两后端 revision metadata 对称 + round-14 full-aggregate 完整性 6 个 + round-15 README/CSV 同步 65 个 + round-16 section-aware row/column binding + 负向 4 个）验证。

## 3. CLI 接口

```text
.venv/python.exe scripts/eval_backend_comparison.py \\
    --samples-manifest docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset.manifest.json \\
    --output-dir artifacts/p5-04-backend-comparison/ \\
    --batch-sizes 1 4 [--limit N] [--max-new-tokens 64] [--dtype bf16]
```

| Flag | 默认 | 含义 |
|---|---|---|
| `--samples-manifest` | 无（可选）| tracked manifest JSON;推荐走本路径以保证审计可复现 |
| `--samples-dir` | `datasets/tool-calling-d2/dev` | 不推荐：仅供 smoke / ad-hoc 评测使用 |
| `--models` | 5 个 P5-02 canonical models | Hugging Face model ids |
| `--backends` | transformers, vllm | 启用的后端列表 |
| `--batch-sizes` | 1 | 每个 backend 调用内部的 sample 数 |
| `--output-dir` | `artifacts/p5-04-backend-comparison/` | 写出 per-run + aggregate |
| `--limit` | 0（=全部）| 评测样本上限（manifest 模式下默认取全部 90 个） |
| `--max-new-tokens` | 256 | greedy / vLLM sampling budget；P5-04 实测 64 |
| `--dtype` | bf16 | transformers 用 bf16/fp16/fp32；vLLM 用 bf16/fp16（不支持 fp32） |
| `--device` | auto | transformers 用（cuda/cpu/auto）；vLLM 强制 cuda |
| `--vllm-gpu-mem-util` | 0.85 | vLLM `gpu_memory_utilization` 参数 |
| `--parallel` | off | 跨 model+backend 组合的线程并行（仅限组合之间；同一组合内 batch 仍走串行） |

## 3.1 Manifest 路径、SHA 校验与 90/750 术语契约

- **D2 dev split = 750**：当前扩样版数据集父 split 的总样本数。
- **P5-02 benchmark evaluation subset = 90**：P5-02 阶段从 D2 dev 冻结并真实运行的固定子集（6 task type × 15）；五个历史 P5-02 transcript 均为 90 rows，`b4fd879:MANIFEST-dev.json` 的 `count=90`。
- **P5-04 = 复用上述 90**：本阶段按内容、顺序、逐文件 SHA 和 aggregate SHA 严格复用 P5-02 baseline。评测全部 750 样本会形成新实验，不能被 relabel 为历史 P5-02/P5-04 benchmark。

### Manifest 校验步骤

`--samples-manifest <path>` 加载逻辑：

1. 读 manifest 为 JSON（schema 见 `docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset.manifest.json`）：
   - `source_commit`：`b4fd879`（与 P5-02 §8 round-2 target-answer 修复同一提交）；
   - `source_aggregate_sha256`：`d44fa149af7d1b229305016711d148642fa1f4bf86f5dae3aab4fd0846b07282`（与 `b4fd879:MANIFEST-dev.json` 一致）；
   - `samples[*]`：每行含 `sample_id` / `path` / `sha256` / `task_type`，共 90 条。
2. 对每个 `path`，`hashlib.sha256` 读文件后与 manifest `sha256` 比对，**不匹配立即 `ValueError`**（防止任何 stale / 篡改样本进入）。
3. 按 `sample_id` 排序后累加 `sha256(f"{sid}\n".encode())` 得到 aggregate，与 manifest `recomputed_aggregate_sha256` 比对。
4. 校验通过后按 manifest 顺序返回 90 个 sample 对象。

manifest 路径的 samples 文件位于 `datasets/tool-calling-d2/p5-02-benchmark/`，gitignored 但本地与 manifest 完全一致；本轮 runner 会以 `git show b4fd879:<path>` 提取（以 CRLF 规范化保存，因生成脚本 `_sha256_file` 是按 Windows CRLF 写入的）。

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
    "generation_failure_count": <int>,
    "generation_failed_sample_ids": [<sample_id>...],
    "first_failure_distribution": {"<failure>": <count>, ...},
    "backend_metadata": {<version / dtype / device / max_new_tokens>},
    "samples_dir": "<path>",
    "samples_manifest": "<optional manifest path>"
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

## 6. Generation failure semantics (round-7)

- 原来：batch 异常 → `generated = [""] * len(chunk)` → 该 row 被后续 P1-05 分类器当作 `parse_success=True`（因为 `tool_not_available` 样本正确生成就是空 JSON）。这遮蔽了 OOM / 内存碎片等真实生成错误。
- 现在：batch 异常 → 逐 sample 用 batch=1 重试 → 仍失败的 row 写 `first_failure = "generation_failed"` + `layers.parse_success = False` + `row.generation_error = <repr>`；汇总中记录 `generation_failure_count` + `generation_failed_sample_ids`。
- 该 round 20 个组合 × 90 样本实测 `generation_failure_count=0`。

## 6.1 reward_layered 口径说明 (round-8 fix)

P5-04 采用 P5-02 §8 round-2 (commit `b4fd879`) 后 `eval_transformers.py` 的 8 层口径；`scripts.reward_offline.compute_reward()` 接受 `(sample, row)` 计算 8 层 `parse_success` → `final_answer_correct` 并归一化到 `reward_layered ∈ [0, 1]`。

**重要**: 必须传入 manifest 加载出的 sample 对象 (`args.samples_by_id`)。`args.samples_dir` 是变量赋值, 仍然是 manifest 父目录（用于日志报告路径），但 reward 计算不再从那里 `glob("*.json")` —— manifest 父目录里只有 manifest 本身。round-7 时代码仍然走 `reward_offline.load_samples(args.samples_dir)`，导致 `samples_by_id={}` → signals=[] → 全部 20 组合 `reward_layered = 0.0` (silently zero input)。round-8 改为 `main()` 加载 manifest 后把 `args.samples_by_id = {s['id']: s for s in samples}` 挂到 args; `run_one_combination()` 直接传给 `compute_reward()`，不再调用 `reward_offline.load_samples()`。修复后取值集中在 0.36–0.42，与 P5-02 §8 round-2 历史表 (0.33–0.43) 吻合。

负向断言：若 `samples_by_id` 为空 → raises `RuntimeError`；若 transcript-by-id 无 overlap → raises `RuntimeError`，绝不 silent zero。

| 指标 | 公式 | 含义 |
|---|---|---|
| `per_sample_latency_ms` | `elapsed_s * 1000 / samples` | 单样本端到端耗时（含 model load 仅计入该 batch 内 amortized 时间） |
| `throughput_samples_per_s` | `samples / elapsed_s` | 端到端吞吐（含 prompt render + tokenize + generate + detokenize + classify） |
| `reward_binary` | `(first_failure is None).mean()` | 8 层全过的样本比例；与 P5-02 round-2 fix 后口径一致 |
| `reward_layered` | `mean(sum(layers True) / 8)` | 8 层平均通过率；与 P2 reward offline / P5-02 完全一致 |

`reward_binary` / `reward_layered` 通过 `scripts/reward_offline.compute_reward` 复算，确保与 P5-02 + reward_offline 流水线一致；若 reward_offline 不可用，则自动 fallback 到 layer 直读（仅供占位，不作为正式结论）。

## 6.2 两后端 immutable revision 对称 (round-12 fix)

detached auditor round-12 指出 `TransformersBackend.setup()` 未读取 `kwargs["revision"]`，造成 vLLM 与 Transformers 加载同一 HF id 时 revision 可能不一致。本轮加固：

- `TransformersBackend.setup()` 读取 `kwargs.get("revision", "main")` 并同时传入 `AutoTokenizer.from_pretrained(model_id, revision=revision, …)` 与 `AutoModelForCausalLM.from_pretrained(model_id, revision=revision, …)`；
- `self.revision` 与 `backend_metadata["revision"]` 都记录传入的 40-hex commit；
- `VLLMBackend.setup()` 原本就传 `revision`，本轮改为 `self.revision = str(revision)` 并写入 metadata；
- `run_one_combination()` 调用 `backend.setup(model_id, revision=revision)` 对两后端使用同一份 DEFAULT_MODELS 中的 canonical revision；
- selftest `test_transformers_records_revision` / `test_transformers_metadata_has_revision` / `test_vllm_records_revision` / `test_vllm_metadata_has_revision` 验证 metadata 完整。

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
    --models 'HuggingFaceTB/SmolLM2-360M-Instruct=a10cc1512eabd3dde888204e902eca88bddb4951' \
    --backends transformers \
    --samples-dir datasets/tool-calling-d2/dev \
    --output-dir artifacts/p5-04-backend-comparison/smoke \
    --limit 4

# Full (5 models × 2 backends × 2 batch sizes = 20 runs; exact P5-02 subset = 90 samples via manifest):
.venv/python.exe scripts/eval_backend_comparison.py \
    --samples-manifest docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset.manifest.json \
    --output-dir artifacts/p5-04-backend-comparison/full \
    --batch-sizes 1 4

# Offline-friendly WSL2 invocation (used to produce the 20 published runs):
wsl -d Ubuntu-22.04 -- bash -c "\
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
         HF_HUB_CACHE=/mnt/c/.../artifacts/huggingface \
         HF_HOME=/mnt/c/.../artifacts/huggingface \
         VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
         VLLM_USE_FLASHINFER_SAMPLER=0 \
         VLLM_ATTENTION_BACKEND=TORCH_SDPA && \
  cd /mnt/c/.../llm-lab && \
  python3 scripts/eval_backend_comparison.py \
    --samples-manifest docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset.manifest.json \
    --output-dir artifacts/p5-04-backend-comparison/full \
    --batch-sizes 1 4 --max-new-tokens 64"
```

### 9.1 干净 checkout 复原 + tracked archive 验证 (round-12 fix)

```bash
# 1) 干净 checkout 不需要额外脚本, 直接跑 full 命令; 90 个 gitignored
#    sample 文件会从 b4fd879 自动重建并逐文件 SHA256 校验。
rm -rf datasets/tool-calling-d2/p5-02-benchmark
python3 scripts/eval_backend_comparison.py \
    --samples-manifest docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset.manifest.json \
    --output-dir .tmp/p5-04-round12-cleanchk --batch-sizes 1 \
    --models 'HuggingFaceTB/SmolLM2-360M-Instruct=a10cc1512eabd3dde888204e902eca88bddb4951' \
    --max-new-tokens 8 --dtype fp32 --device cpu --backends transformers --limit 3
# 期望 stdout: [p5-04] 3 samples loaded from manifest ...
# 期望 stderr: aggregate_sha256=d44fa149af7d1b22... (与 manifest source_aggregate_sha256 一致)

# 2) 实测 20-run 产物被跟踪为 immutable archive + SHA manifest;
#    detached auditor 可在干净 checkout 中验证 archive SHA 与每个 entry SHA:
python3 -c "
import hashlib, json, tarfile
from pathlib import Path
arch=Path('docs/experiments/p5-04-backend-comparison/audit-artifacts/p5-04-runs.tar.gz')
meta=json.loads(Path('docs/experiments/p5-04-backend-comparison/audit-artifacts/p5-04-runs.archive-manifest.json').read_text())
print('archive_sha_match', hashlib.sha256(arch.read_bytes()).hexdigest() == meta['archive_sha256'])
# Round-13: verify every run JSON carries backend_metadata.revision matching
# the canonical DEFAULT_MODELS revision (same model, transformers+vllm identical).
with tarfile.open(arch,'r:gz') as t:
    for e in meta['entries']:
        if not e['path'].startswith('artifacts/p5-04-backend-comparison/full/run_'):
            continue
        m=t.extractfile(e['path'].split('/')[-1])
        d=json.loads(m.read().decode())
        print(e['path'].split('/')[-1], 'rev', d['summary']['backend_metadata'].get('revision'))
"
# 期望: archive_sha_match=True + 24 entries sha_match=True + 20/20 run 僅报 revision + comparison.csv rows=20 (both backends) + comparison_delta.csv rows=10 (全 non-null latency delta)。
```

## 10. 已知边界

- **vLLM 与 transformers 不能在同 GPU 同时运行**：单 host 单 GPU 上 `--parallel` 反而 OOM 概率高；默认 sequential。`--parallel` 仅在多 GPU host 才有意义。
- **Reward fallback 仅供测试**：当 `reward_offline` 因环境问题不可用时（如 vLLM 进程占满 GPU），本协议 fallback 到 layer 直读；该 fallback 数字仅作为 placeholder，正式结论必须来自完整 `reward_offline`。
- **`first_failure_distribution` 在 mock 测试中可能为空**：mock 不一定触发所有 8 个 failure class；正式对比表会按真实生成结果填充。
- **CSV 字段顺序固定**：添加新指标需同时更新 `write_aggregate_comparison` 与本协议；保持 schema version 向前兼容。
