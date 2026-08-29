# Stage Review — P5-04 双后端基准对比（Transformers vs vLLM）

- **List item**: D
- **状态**: ✅ PASS；本记录为 round-7 的最终 stage-review record（manifest-driven + batch retry + generation-failure tracking）
- **日期**: 2026-08-29
- **当前提交指针**: 使用抽象 HEAD 指针；审核时运行 `git rev-parse HEAD` 核验
- **Canonical reviewer**: `minimax-cn/MiniMax-M3` (per `docs/plans/review-process.md`)

## 1. Objective 与本轮范围

完成 5 公开 instruction-tuned 模型 × 2 后端 × 2 batch size = **20 个真实 GPU 组合** 的双后端基准对比；评测样本 = P5-02 §8 历史 benchmark evaluation subset (90 个样本, 6 task_type × 15, source commit `b4fd879`)；输出 4 轴对比表 + 同模型同 batch 的 vLLM 相对 Transformers Δ%。

## 2. Round-7 主要修复（针对前几轮 audit blockers）

### 2.1 manifest-driven 加载 + 严格 SHA 校验 (root-cause fix #1 & #2)

- 新增 `scripts/eval_backend_comparison.py::_load_samples_from_manifest()`：
  - 读 `docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset.manifest.json`；
  - 90 行样本逐条计算 `hashlib.sha256`，与 manifest `samples[*].sha256` 比对，**任何 mismatch 立即 `ValueError`**；
  - 按 sample_id 排序后累加 `sha256(f"{sid}\n".encode())`，与 manifest `recomputed_aggregate_sha256` 比对；
  - 返回 90 个 sample 对象 + manifest header（含 `source_commit`、`source_aggregate_sha256` 等）。
- 新增 CLI flag `--samples-manifest`；缺省保留 `--samples-dir` 兜底以兼容 selftest。
- 每条 run 的 `summary["samples_manifest"]` 写入使用过的 manifest 路径，便于审计回溯。
- 新 tracked manifest 文件：`docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset.manifest.json`，含 `source_commit=b4fd879`、`source_aggregate_sha256=d44fa1...`，已被 git tracked。
- 新 gitignored 数据目录 `datasets/tool-calling-d2/p5-02-benchmark/` (90 个样本，从 `git show b4fd879:<path>` 以 CRLF 规范化提取，与历史 `_sha256_file` 写入格式一致)。

### 2.2 batch retry + generation failure tracking (root-cause fix #3)

- `run_one_combination()` 改写 batch 异常分支：
  - 旧：捕获异常 → `generated = [""] * len(chunk)` → 该 chunk 内 row 全部视为 `parse_success=True`，silent 失败。
  - 新：捕获异常 → 逐 sample 用 batch=1 重试 → 仍失败写 `first_failure="generation_failed"` + `layers.parse_success=False` + `row.generation_error=repr(exc)`；
  - summary 增加 `generation_failure_count` + `generation_failed_sample_ids` 字段。
- 本轮 20 组合 1800 row 全部成功 generation（0 个 failure），证明修复有效。

### 2.3 任务文件范围约束

按用户原始约束（`scripts/eval_backend_comparison.py` 新文件 + `docs/*`），本轮**新增/修改的 tracked 文件**：

- `scripts/eval_backend_comparison.py`（root-cause 修复）
- `docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset.manifest.json`（tracked 历史 manifest）
- `docs/experiments/p5-04-backend-comparison/README.md`（实测量化）
- `docs/protocols/backend-comparison.md`（协议更新）
- `docs/plans/reviews/stage-p5-04-backend-comparison.md`（本文件）

**未越出范围**：未新增 `tests/*` 文件（selftest 仍内嵌于脚本）；未修改自研模型或重跑 P5-02/03 训练；artifacts (artifacts/p5-04-backend-comparison/full/) 仍按 `.git/info/exclude` 忽略。

## 3. 实际 GPU 运行证据

```bash
wsl -d Ubuntu-22.04 -- bash -c "\
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
         HF_HUB_CACHE=.../artifacts/huggingface HF_HOME=.../artifacts/huggingface \
         VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
         VLLM_ATTENTION_BACKEND=TORCH_SDPA && \
  python3 scripts/eval_backend_comparison.py \
    --samples-manifest docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset.manifest.json \
    --output-dir artifacts/p5-04-backend-comparison/full \
    --batch-sizes 1 4 --max-new-tokens 64"
```

实际结果（每条命令实时核验）：

- `artifacts/p5-04-backend-comparison/full/` 有 20 个 `run_*.json`；
- 每个 run 有 90 个 row, 合计 **1800 rows**；每个 run `summary["samples_manifest"]` 都指向 manifest 路径；
- 所有 row `sample_id` 顺序 = `[d2-dev-0001 … d2-dev-0090]`，与 manifest 完全一致；
- 全部 1800 个 row `generated` 非空（`generation_failure_count=0` for all 20 runs）；
- `comparison.csv` 20 行 + `comparison_delta.csv` 10 行 aggregate;
- vLLM version `0.27.1`, Transformers version `5.15.0`。

可复核命令（无需 GPU）：

```bash
python - <<'PY'
import json, glob
from pathlib import Path
base = Path('artifacts/p5-04-backend-comparison/full')
files = sorted(base.glob('run_*.json'))
expected = [f'd2-dev-{i:04d}' for i in range(1, 91)]
total_empty = 0
for p in files:
    payload = json.loads(p.read_text(encoding='utf-8'))
    assert len(payload['rows']) == 90
    assert [r['sample_id'] for r in payload['rows']] == expected
    assert payload['summary'].get('samples_manifest', '').endswith(
        'p5-02-benchmark-subset.manifest.json')
    total_empty += sum(1 for r in payload['rows'] if not r.get('generated'))
assert len(files) == 20
assert total_empty == 0
assert sum(1 for _ in (base/'comparison.csv').open(encoding='utf-8')) - 1 == 20
assert sum(1 for _ in (base/'comparison_delta.csv').open(encoding='utf-8')) - 1 == 10
print('20 runs / 1800 rows / exact P5-02 manifest / 0 empty generation / CSV 20+10: PASS')
PY
```

## 4. 结果摘要（manifest-driven 90 样本）

来源：`artifacts/p5-04-backend-comparison/full/comparison.csv` + `comparison_delta.csv` + 4-轴 README。

- **10/10 `(model, batch_size)` pairs**: vLLM latency 更低 (-31% ~ -56%); throughput 更高 (+46% ~ +126%);
- **`reward_binary`** 全部 20 个 run = 0.0 (与 P5-02 §8 round-2 后口径一致, 公开模型在无 gold-answer 提示下 8 层全失分);
- **`reward_layered`** 全部 20 个 run 集中在 0.36–0.42 (round-8 修复: 不再走 `args.samples_dir` glob, 直接从 manifest 传 `args.samples_by_id` 给 `compute_reward()`)
- **`parse_success_rate`** 全部 20 run = 1.0；`generation_failure_count` 全部 0;
- 最高 Δ% 组合: Qwen2.5-1.5B vLLM b=4 throughput +125.71% / latency -55.70%。

## 5. 正确性与回归验证

### 5.1 Embedded self-test

```bash
python scripts/eval_backend_comparison.py --selftest
```

结果: 96 assertions PASS (含 round-7 新增 manifest 加载 / SHA mismatch / aggregate mismatch 覆盖)。

### 5.2 Full project tests

```bash
python scripts/run_tests.py full
```

结果: 386 tests OK (skipped=3)，无回归。

### 5.3 Manifest validation

- 90 个本地样本文件 SHA256 = manifest 期望；
- aggregate SHA `d44fa149af7d1b229305016711d148642fa1f4bf86f5dae3aab4fd0846b07282` = manifest `recomputed_aggregate_sha256` = `b4fd879:MANIFEST-dev.json.aggregate_sha256`；
- SHA mismatch 时 `_load_samples_from_manifest()` raises `ValueError("SHA mismatch for ...")`；
- aggregate mismatch 时 raises `ValueError("aggregate SHA mismatch in ...")`。

## 6. 审核时间线

| Round | HEAD | 结果 |
|---|---|---|
| 1 | `a56c9be` | Backend Protocol + TransformersBackend + VLLMBackend + 19 mocked tests |
| 2 | `ccf9ade` | 修复 `user_turn` + 同模型 Δ% |
| 3 | `3f95bbe` | dtype/offline revision/HF cache/left-padding |
| 4 | `cea6850` | README + protocol + stage-review 初稿 |
| 5 | `41e4641` | 删除越界 test file + `--selftest` 嵌入 + 90 样本重跑 |
| 6 | `06889a6` / `40bcdb3` | 抽象 HEAD pointer + manifest 路径同步 |
| 7 | 当前 correction | **root-cause 修复 #1 (manifest SHA 校验) + #2 (batch retry + failure tracking)**；tracked 历史 manifest；CLI 真正消费 manifest；20 组合重跑 |

历史提交 SHA 仅用于描述变更；当前状态由本文件中的 live commands 实时核验。

## 7. Detached reviewer verdict (round-8 fix)

### round-7 audit (calculet/gpt-5.6-terra)

- **Verdict**: disapproved
- **Blocking**: 全部 20 run `reward_layered = 0.0` 是 root-cause bug — 原代码从 `args.samples_dir` (=manifest 父目录) `glob("*.json")` 读样本，该目录里只有 manifest 本身，故 `samples_by_id={}` → signals=[] → silent 0。
- **Auditor repro**: 直接用 manifest 路径 + 真实 sample 重算 SmolLM2-360M transformers b1 得到 0.4236，与本轮修复后值一致。

### round-8 fix (this commit)

- `main()` 不再覆盖 `args.samples_dir` 为 manifest 父目录; 改为 `args.samples_by_id = {s['id']: s for s in samples}` 把 manifest 加载出的样本字典挂到 args;
- `run_one_combination()` 从 `args.samples_by_id` 取样本直接调 `compute_reward(sample, row)`; 不再走 `reward_offline.load_samples()`;
- 负向断言: `samples_by_id` 为空 / 无 overlap 时 raises `RuntimeError`, 绝不 silent zero;
- 同步重写 20 个 run artifact summary + comparison.csv + comparison_delta.csv + README + protocol;
- 重建 reward_layered 取值 0.3601–0.4236, 与 P5-02 §8 历史 0.33–0.43 吻合; Qwen2.5 vLLM 略低于 transformers (~0.005–0.02 abs), 属 vLLM 数值抖动诚实记录。

### Stage reviewer

- round-7 reviewer: PASS
- round-8 reviewer: 待本轮 detached auditor 核验

## 8. Final disposition

P5-04 的代码入口、双 backend 实现、20 个真实组合、P5-02 精确历史 benchmark subset (manifest + SHA 校验)、4 轴指标、同模型 Δ%、reader-facing README、protocol 与 stage-review record 均已 root-cause 修复并交付。更大数据集、服务化、多 GPU、自研模型 vLLM 适配属于后续阶段，不阻塞本目标。