# Stage Review — P5-04 双后端基准对比 (Transformers vs vLLM)

- **List item**: D (active list item)
- **Date**: 2026-08-29
- **Reviewer model**: per `docs/plans/review-process.md` = `minimax-cn/MiniMax-M3`

## 范围（Scope）

5 公开 instruction-tuned 模型 × 2 后端 × 2 batch size = 20 组合的双后端基准对比；在 P5-02 benchmark evaluation subset (D2 dev 750 采样) 上跑同一样本；输出 4 轴对比表（latency / throughput / reward_binary / reward_layered）。

不在本阶段范围内：

- 实际 GPU 跑（GPU runs 计划在第二轮，本 stage review 提交时仅有 mocked 测试 + protocol + experiment README）；
- 修改自研 Dense / MoE 模型架构；
- 训练 / LoRA / 全参数微调；
- vLLM serving / FastAPI / Triton integration；
- Multi-GPU scaling / speculative decoding；
- 重新训练 P5-02 / 重跑 P5-03 smoke。

## 修改 summary（第一轮交付）

| 文件 | 类型 | 说明 |
|---|---|---|
| `scripts/eval_backend_comparison.py` | NEW | 254 行；Backend Protocol + TransformersBackend + VLLMBackend + run_one_combination + write_run_artifact + write_aggregate_comparison + main CLI |
| `tests/test_eval_backend_comparison.py` | NEW | 14 KB；19 单测覆盖 backend interface / pipeline / output schema / CLI |
| `docs/protocols/backend-comparison.md` | NEW | 7 KB；10 节协议（目标 / 接口 / 输入 / 输出 / 4 轴 / vLLM workarounds / 与 P5-02 关系 / 复现 / 边界） |
| `docs/experiments/p5-04-backend-comparison/README.md` | NEW | 4 KB；实验 README (TODO 表待 GPU runs 完成) |

不修改现有代码 / schema / config / tests；纯新增文件。

## 设计要点

### Backend 抽象

```python
class Backend(Protocol):
    name: str
    def setup(self, model_id: str, **kwargs: Any) -> None: ...
    def chat_generate(messages_batch, tools_batch) -> list[str]: ...
    def teardown(self) -> None: ...
    def metadata() -> dict[str, Any]: ...
```

两真实实现 + 一 mock：

- **TransformersBackend** — 封装 `AutoModelForCausalLM` + greedy generation；与 P5-02 `_greedy_generate` 等价但合并多 batch 调用；
- **VLLMBackend** — 封装 vLLM `LLM.generate` + 同一 chat template 渲染（保证两后端 per-row 输出可比）；自动应用 P5-03 三个 workarounds；
- **MockBackend** — 测试用，deterministic generation，无 GPU 依赖。

### Chat template 渲染一致性

两后端都用 `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=...)`；fallback 到 `<role>: <content>` 拼接。两后端 prompt 渲染路径完全一致，保证 per-row 对比的语义公平。

### Target-answer 泄漏修复继承

`_strip_terminal_assistant()` 在两后端共用，移除末位 assistant message（避免 D2 样本 gold answer 泄漏）。这是 P5-02 round-2 fix (`b4fd879`) 的等价实现。

### 4 轴对比指标

| 指标 | 公式 | 与 P5-02 / P2 关系 |
|---|---|---|
| `per_sample_latency_ms` | `elapsed_s * 1000 / samples` | 端到端 (含 prompt render + tokenize + generate + detokenize + classify) |
| `throughput_samples_per_s` | `samples / elapsed_s` | 同上 |
| `reward_binary` | `(first_failure is None).mean()` | 通过 `reward_offline.compute_reward` 复算；与 P5-02 round-2 修复后口径一致 |
| `reward_layered` | `mean(sum(layers True) / 8)` | 同上；与 P2 reward offline 完全一致 |

### Reward 复算路径

`run_one_combination()` 内 `compute_reward()` 直接 import 自 `scripts/reward_offline.py`（既有模块），保证数字与 P5-02 横向对比一致；若 reward_offline 不可用（vLLM 占满 GPU），fallback 到 layer 直读（仅作 placeholder，不作为正式结论）。

### 输出 schema 兼容性

`rows[*]` schema 与 P5-02 (`eval_transformers`) + `eval_sft_tool` 完全一致，可直接喂 `reward_offline --transcripts`。`{summary, rows}` 双层结构 + `first_failure_distribution` + 8 层 `layers` dict 全部对齐。

### Mocked 测试覆盖

19 单测 (no GPU / no model download / no WSL 依赖)：

- Backend 抽象 + TransformersBackend / VLLMBackend metadata
- `_strip_terminal_assistant` 边界（仅 assistant / 无 assistant）
- `_aggregate_timing` (basic / empty / zero-elapsed)
- `run_one_combination` (mocked factory + setup/teardown 计数 + error fallback)
- `write_run_artifact` + `write_aggregate_comparison` (JSON + CSV + empty)
- CLI (defaults + overrides + `--help`)

```text
$ pytest tests/test_eval_backend_comparison.py -v
============================= 19 passed in 3.65s ==============================
```

## 不在第一轮范围内的项

- **实际 GPU runs**: 本 stage review 仅交付 mocked 完整 pipeline + protocol + experiment README。第二轮 list item D 提交时跑实际 GPU runs + 填充结果表。
- **Reward fallback 数字作为正式结论**: 协议明确 fallback 仅供测试，正式数字必须来自 `reward_offline.compute_reward`。
- **N12 / N13 long curve / 简化 MLA**: 候选 4 / 候选 5，独立 list items，不在本协议范围。

## 验证 contract

abstract HEAD pointer — auditor runs `git rev-parse HEAD` to verify current value。

- **live command 1** (clean): `git status --short` → empty。
- **live command 2** (only doc + new files): `git diff --name-only f1fe61c..HEAD` 中本 stage review 的 commit 应仅含：
  - `scripts/eval_backend_comparison.py` (NEW)
  - `tests/test_eval_backend_comparison.py` (NEW)
  - `docs/protocols/backend-comparison.md` (NEW)
  - `docs/experiments/p5-04-backend-comparison/README.md` (NEW)
  - `docs/plans/reviews/stage-p5-04-backend-comparison.md` (NEW)
- **live command 3** (mocked tests pass): `pytest tests/test_eval_backend_comparison.py` → 19 passed。
- **live command 4** (CLI imports OK): `python -c "from scripts.eval_backend_comparison import Backend, DEFAULT_MODELS, DEFAULT_BACKENDS"` → no error。
- **live command 5** (协议 / experiment README 存在): `ls docs/protocols/backend-comparison.md docs/experiments/p5-04-backend-comparison/README.md` → both exist。

**expected live verification outputs**: 上述 5 个命令在 round-N 后保持一致；新 commits 不应破坏 mocked tests + import path。

## 关联文件

- 修改: (none — 全部 NEW)
- 新增:
  - `scripts/eval_backend_comparison.py` (Backend Protocol + TransformersBackend + VLLMBackend + run_one_combination + write_run_artifact + write_aggregate_comparison + main)
  - `tests/test_eval_backend_comparison.py` (19 mocked 单测)
  - `docs/protocols/backend-comparison.md` (10 节协议)
  - `docs/experiments/p5-04-backend-comparison/README.md` (实验 README; TODO 表待 GPU runs)
  - `docs/plans/reviews/stage-p5-04-backend-comparison.md` (本文件，stage review record)

## 审查 reviewer 身份与详细 verdict

- detached auditor agent name: `reviewer` (project-level subagent reviewer, dispatched via Agent tool)
- **detached auditor verdict**: **PENDING independent review** (status: stage review record created, awaiting detached auditor subagent dispatch per `docs/plans/review-process.md` §"提交后触发 stage review" 流程)
- **detached auditor provider/model**: pending dispatch
- **dispatch timestamp**: pending

### Reviewer findings summary

(待 detached auditor subagent dispatch 后填充)
