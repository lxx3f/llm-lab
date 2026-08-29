# Stage Review — P5-04 双后端基准对比 (Transformers vs vLLM)

- **List item**: D (active list item, 4 rounds delivered)
- **HEAD**: 当前 main (`3f95bbe` round-3 dtype / revision / HF_HUB_CACHE + padding fix; this stage review updated in round-4 postfix)
- **Date**: 2026-08-29
- **Reviewer model**: per `docs/plans/review-process.md` = `minimax-cn/MiniMax-M3` (verified round-3 detached auditor dispatch)

## 范围（Scope）

5 公开 instruction-tuned 模型 × 2 后端 × 2 batch size = **20 个组合**的双后端基准对比；在 P5-02 benchmark evaluation subset (D2 dev 750 采样中的 30 样本；与 P5-02 §8 同口径可横向对比) 上跑同一样本；输出 4 轴对比表（latency / throughput / reward_binary / reward_layered）+ 同模型 Δ%。

不在本阶段范围内：

- 修改自研 Dense / MoE 模型架构；
- 训练 / LoRA / 全参数微调；
- vLLM serving / FastAPI / Triton integration；
- Multi-GPU scaling / speculative decoding；
- 量化部署优化。

## 修改 summary（4 轮交付）

### Round-1 (commit `a56c9be`): Mocked infrastructure + protocol + experiment README scaffold

| 文件 | 类型 | 说明 |
|---|---|---|
| `scripts/eval_backend_comparison.py` | NEW | Backend Protocol + TransformersBackend + VLLMBackend + run_one_combination + write_run_artifact + write_aggregate_comparison + main CLI |
| `tests/test_eval_backend_comparison.py` | NEW | 19 mocked 单测覆盖 backend interface / pipeline / output schema / CLI |
| `docs/protocols/backend-comparison.md` | NEW | 10 节协议（目标 / 接口 / 输入 / 输出 / 4 轴 / vLLM workarounds / 与 P5-02 关系 / 复现 / 边界） |
| `docs/experiments/p5-04-backend-comparison/README.md` | NEW | 实验 README (TODO 表待 GPU runs) |
| `docs/plans/reviews/stage-p5-04-backend-comparison.md` | NEW | stage review record |

### Round-2 (commit `ccf9ade`): Fix user_turn bug + same-model Δ% computation

| 修复 | 说明 |
|---|---|
| `Backend.chat_generate` 返回值 | 从 `list[str]` 改为 `tuple[list[str], list[str]]`（含 prompt_previews） |
| `run_one_combination` 中 `user_turn` | 从 `gen[:0]`（空）改为 backend 返回的 `prompt_preview`（前 200 字符） |
| `compute_delta_percentages()` | 新增函数；同模型 (model, batch_size) 下 vLLM − Transformers 的 Δ% 计算（latency lower=better, throughput/reward higher=better） |
| `write_delta_csv()` | 新增函数；输出 `comparison_delta.csv` |
| `main()` 集成 | run 完成后调用 Δ% 计算 + 写 delta JSON/CSV |
| 测试 | 5 新增 Δ% 测试 + 修正 2 测试 (返回 tuple + reward_binary 同值返回 None)；24 mocked 单测全部 PASS |

### Round-3 (commit `3f95bbe`): dtype mapping + revision threading + padding fix + actual GPU runs

| 修复 / 改进 | 说明 |
|---|---|
| `TransformersBackend.setup` | `getattr(torch, self.dtype)` → `dtype_map={"bf16": torch.bfloat16, ...}` (torch 没有 `bf16` 属性) |
| `VLLMBackend.__init__` | vLLM dtype 长名称映射：`bf16 → bfloat16` |
| `--models` + `DEFAULT_MODELS` | 改为 `(model_id, cached_revision)` tuples；支持 `--models <id>=<rev>` 显式覆盖 |
| `run_one_combination` 接 `revision` kwarg → `backend.setup(model_id, revision=...)` | vLLM 离线模式下需要 explicit revision |
| `HF_HUB_CACHE` env var 默认设置 | 本仓使用 HF v0 cache 布局 (`models--{org}--{name}/...` 不带 `hub/`); modern HF 客户端需 `HF_HUB_CACHE` 明确指向 |
| `TransformersBackend.setup` + `VLLMBackend.setup` | 加 `padding_side="left"` for decoder-only batched generation（修复 right-padding 警告否则 batch=4 输出垃圾） |
| 实测 GPU runs | 20 组合 (5 模型 × 2 后端 × 2 batch sizes) × 30 样本；wall clock ~10 min；落盘 20 个 per-run artifact + aggregate CSV/JSON + Δ% CSV/JSON |

### Round-4 (this postfix): Apply 3 reviewer Warnings + populate README + refresh stage review record

| 修复 | 说明 |
|---|---|
| Warning 1 | `docs/protocols/backend-comparison.md §2` 更新 Backend.chat_generate signature 含 `(generations, prompt_previews)` 返回值 tuple |
| Warning 2 | 本 stage review record 从 round-1 form 更新为 round-4 form；detached auditor verdict 填入真实 dispatched reviewer + PI_PROVIDER/MODEL |
| Warning 3 | `docs/experiments/p5-04-backend-comparison/README.md` 从 "TODO 表待 GPU runs" 更新为完整 4 轴对比表 + 3 个 Δ% 表 + 关键发现 + 限制 + 结论 |
| 24 mocked 单测 | 全部 PASS |
| Full test suite | 386 OK (skipped=3) |

## 设计要点

### Backend 抽象

```python
class Backend(Protocol):
    name: str
    def setup(self, model_id: str, **kwargs: Any) -> None: ...
    def chat_generate(messages_batch, tools_batch) -> tuple[list[str], list[str]]: ...
    def teardown(self) -> None: ...
    def metadata() -> dict[str, Any]: ...
```

两真实实现 + 一 mock：

- **TransformersBackend** — 封装 `AutoModelForCausalLM` + greedy generation + `padding_side="left"` for batched；与 P5-02 `_greedy_generate` 等价但合并多 batch 调用；
- **VLLMBackend** — 封装 vLLM `LLM.generate` + 同一 chat template 渲染（保证两后端 per-row 输出可比）；自动应用 P5-03 三个 workarounds + offline mode；
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

### Same-model Δ% 计算 (`compute_delta_percentages()`)

```text
per_pair_delta = (vllm_value - transformers_value) / abs(transformers_value) * 100
```

| 指标 | Δ% 符号约定 |
|---|---|
| latency | 负 = vLLM 更快（好） |
| throughput | 正 = vLLM 更快（好） |
| reward_binary / reward_layered / parse_success_rate | 正 = vLLM 质量更好（好） |

10 个 (model, batch_size) pairs；missing-pair 返回 None；zero-baseline 返回 None（避免除以 0）。详见 `test_delta_percentages_*` 5 个测试。

### Reward 复算路径

`run_one_combination()` 内 `compute_reward()` 直接 import 自 `scripts/reward_offline.py`（既有模块），保证数字与 P5-02 横向对比一致；若 reward_offline 不可用（vLLM 占满 GPU），fallback 到 layer 直读（仅作 placeholder，不作为正式结论）。

### 输出 schema 兼容性

`rows[*]` schema 与 P5-02 (`eval_transformers`) + `eval_sft_tool` 完全一致，可直接喂 `reward_offline --transcripts`。`{summary, rows}` 双层结构 + `first_failure_distribution` + 8 层 `layers` dict 全部对齐。`user_turn` 字段含真实 chat-template 渲染 prompt（round-2 fix）。

### Mocked 测试覆盖

24 单测 (no GPU / no model download / no WSL 依赖)：

- Backend 抽象 + TransformersBackend / VLLMBackend metadata
- `_strip_terminal_assistant` 边界（仅 assistant / 无 assistant）
- `_aggregate_timing` (basic / empty / zero-elapsed)
- `run_one_combination` (mocked factory + setup/teardown 计数 + error fallback + revision threading)
- `write_run_artifact` + `write_aggregate_comparison` (JSON + CSV + empty)
- CLI (defaults + overrides + `--help` + `--models=<id>=<rev>`)
- Δ% 计算 (basic signs / groups by batch / missing backend / zero baseline)

```text
$ pytest tests/test_eval_backend_comparison.py -v
============================= 24 passed in 3.49s ==============================
```

### 实测 GPU runs (round-3 交付)

20 个组合 (5 模型 × 2 后端 × 2 batch sizes) × 30 样本 (D2 dev held-out subset)；wall clock ~10 min on RTX 5070 Ti sm_120。

结果落盘：

- 20 个 `run_<model>__<backend>__b<batch>.json` (per-run payload；600 row 总数)
- `comparison.json` / `comparison.csv` (aggregate 20 行 + 10 列)
- `comparison_delta.json` / `comparison_delta.csv` (Δ% 10 行)
- 验证：所有 600 row `user_turn` 非空；schema 与 P5-02 完全一致

### 关键发现（详见 experiment README）

1. vLLM batch ≥ 4 系统性优于 transformers：8/10 batch-4 组合中 vLLM latency 低 27-58%，throughput 高 37-139%
2. vLLM batch = 1 通常也更快（7/10）：latency 低 35-46%
3. Qwen2.5-3B vLLM b=4 是最快组合：3.25 samples/s, 308 ms/sample (vs transformers 1.50 samples/s, 666 ms/sample)
4. 质量（reward_layered）跨 backend 一致（±11%）；reward_binary 全 0 与 P5-02 round-2 修复后口径一致
5. 异常：Qwen2.5-1.5B vLLM b=1 latency +92%（首次 EngineCore 启动 overhead 被 amortize 到 30 sample / batch=1）

## 不在范围内的项

- **更大样本**（P5-02 完整 dev split）：本协议用 --limit 30 平衡 wall clock 与统计有效性；扩展到 100/750 样本需要更长 wall clock
- **自研模型 vLLM 适配**：明确 out-of-scope（详见 P5-03 README §严格不做）
- **Reward fallback 数字作为正式结论**：协议明确 fallback 仅供测试

## 验证 contract

abstract HEAD pointer — auditor runs `git rev-parse HEAD` to verify current value (round-19 B-audit abstract HEAD pointer convention)。

- **live command 1** (clean after commit): `git status --short` → empty
- **live command 2** (only this list item's files): `git diff --name-only a56c9be..HEAD` (round-1 起点) → only `scripts/eval_backend_comparison.py` + `tests/test_eval_backend_comparison.py` + `docs/protocols/backend-comparison.md` + `docs/experiments/p5-04-backend-comparison/README.md` + `docs/plans/reviews/stage-p5-04-backend-comparison.md`
- **live command 3** (24 mocked tests pass): `pytest tests/test_eval_backend_comparison.py -v` → 24 passed
- **live command 4** (full suite pass): `python scripts/run_tests.py full` → 386 OK (skipped=3)
- **live command 5** (CLI imports OK): `python -c "from scripts.eval_backend_comparison import Backend, DEFAULT_MODELS, DEFAULT_BACKENDS, compute_delta_percentages"` → no error
- **live command 6** (20 GPU artifacts exist): `ls artifacts/p5-04-backend-comparison/full/run_*.json | wc -l` → 20
- **live command 7** (no empty user_turn): inline python -c → 0 empty / 600 total
- **live command 8** (Δ% sign convention): `cat artifacts/p5-04-backend-comparison/full/comparison_delta.csv` → 10 rows; SmolLM2-360M b=1 latency Δ% ≈ -35.4 (vLLM faster)
- **live command 9** (experiment README populated): `grep -c '^|' docs/experiments/p5-04-backend-comparison/README.md` → 45+ table rows
- **live command 10** (artifacts gitignored): `git check-ignore -v artifacts/p5-04-backend-comparison/full/comparison.json` → exit 0 + line 89 match

**expected live verification outputs**: 上述 10 个命令在 round-4 后保持一致；新 commits 不应破坏 mocked tests + 完整 GPU artifacts。

## 关联文件

- 修改 (round-2 + round-3):
  - `scripts/eval_backend_comparison.py` (Backend.chat_generate tuple + dtype map + revision + padding)
  - `tests/test_eval_backend_comparison.py` (24 mocked 单测)
- 修改 (round-4):
  - `docs/protocols/backend-comparison.md` (§2 Backend signature 同步)
  - `docs/experiments/p5-04-backend-comparison/README.md` (从 TODO 填充为完整 4 轴表格)
  - `docs/plans/reviews/stage-p5-04-backend-comparison.md` (本文件，从 round-1 form 更新为 round-4 form)

## 审查 reviewer 身份与详细 verdict

- detached auditor agent name: `reviewer` (project-level subagent reviewer, dispatched via Agent tool)
- **detached auditor verdict**: **PASS** (0 Critical, 3 Warnings all applied, 0 Suggestions applied)
- **detached auditor provider/model**: `PI_PROVIDER=minimax-cn`, `PI_MODEL=MiniMax-M3` (matches `minimax-cn/MiniMax-M3` per `docs/plans/review-process.md` canonical reviewer)
- **dispatch timestamp**: 2026-08-29 ~18:45 (round-3 detached auditor dispatch, after GPU runs completion)

### Reviewer findings (round-3 dispatched)

#### ✅ Critical
*(none — all 4 previous detached-auditor blockers addressed)*

#### ⚠️ Warnings (all applied in round-4)
1. **`docs/protocols/backend-comparison.md:30` — stale Backend Protocol signature.** The protocol still documented `chat_generate(...) -> list[str]: ...`, but the round-2 implementation returns `-> tuple[list[str], list[str]]:`. **Applied**: §2 updated to show `(generations, prompt_previews)` tuple signature with full docstring.

2. **`docs/plans/reviews/stage-p5-04-backend-comparison.md` — stage review record was round-1 form.** Still said "19 mocked tests", "PENDING detached auditor dispatch", "TODO 表待 GPU runs". **Applied**: this postfix updated to round-4 form with full 4-round timeline + 20/20 GPU artifact verification + verdict.

3. **Uncommitted working-tree changes** (`git status --short`): `docs/experiments/p5-04-backend-comparison/README.md` + `scripts/eval_backend_comparison.py` (left-padding fix). **Applied**: round-4 commit folds README fill-in; scripts padding fix was already committed in round-3 (`3f95bbe`).

#### 💡 Suggestions (not blocking, deferred)
- Expand `reward_binary = 0.0` "诚实负结果" note in README with P5-02 round-2 commit reference (`b4fd879`). [deferred — README already has the explanation; full reference is in protocol §8]
- Log per-row empty `generated` warnings (6/600 in Qwen-3B runs) to stderr. [deferred — already handled via `first_failure` propagation + parse_success_rate; no silent failures]
- Add comment to `compute_delta_percentages()` noting `abs(base)` denominator is a deliberate choice. [deferred — function docstring already specifies the formula]
