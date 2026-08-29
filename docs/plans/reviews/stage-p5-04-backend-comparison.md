# Stage Review — P5-04 双后端基准对比（Transformers vs vLLM）

- **List item**: D
- **状态**: ✅ PASS；本记录为 round-10 的最终 stage-review record（用户明确确认历史 P5-02 90 样本范围；manifest-driven + batch retry + generation-failure tracking + reward 计算以 manifest sample 为准）
- **日期**: 2026-08-29
- **当前提交指针**: 使用抽象 HEAD 指针；审核时运行 `git rev-parse HEAD` 核验
- **Canonical reviewer**: `minimax-cn/MiniMax-M3` (per `docs/plans/review-process.md`)

## 1. Objective 与本轮范围

完成 5 公开 instruction-tuned 模型 × 2 后端 × 2 batch size = **20 个真实 GPU 组合** 的双后端基准对比；评测样本 = P5-02 §8 历史 benchmark evaluation subset (90 个样本, 6 task_type × 15, source commit `b4fd879`)；输出 4 轴对比表 + 同模型同 batch 的 vLLM 相对 Transformers Δ%。

### 1.1 90/750 范围决策（round-10，用户确认）

用户已在 goal decision 中明确选择：本目标按**历史 P5-02 benchmark evaluation subset 的 90 个 manifest + SHA 校验样本**完成，不扩展为 D2 dev 全部 750 样本。

证据与术语：

- 当前扩样版 **D2 dev split = 750**，这是父 split 数据集规模；
- 历史 **P5-02 benchmark evaluation subset = 90**，五个已交付 P5-02 transcript 均有 90 rows (`d2-dev-0001`…`d2-dev-0090`)；
- `git show b4fd879:datasets/tool-calling-d2/MANIFEST-dev.json` 返回 `count=90`、aggregate SHA `d44fa149af7d1b229305016711d148642fa1f4bf86f5dae3aab4fd0846b07282`；
- P5-04 复用该历史 baseline 才能保证同样本公平比较；全 750 评测属于独立的新实验，不能覆盖或 relabel 现有历史结果。

本 decision 解决了 objective 旧括号“D2 dev 750 采样”的歧义：含义是“90 benchmark subset 来源于 750 父 dev split”，不是“本阶段每个组合运行 750 rows”。

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

结果: 96 assertions PASS (含 round-7 新增 manifest 加载 / SHA mismatch / aggregate mismatch 覆盖 + round-8 新增 reward wiring 覆盖)。

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
| 6 | `06889a6` / `40bcdb3` | 抽象 HEAD pointer + 历史 90 样本 subset 路径同步 |
| 7 | `8c868a6` | **root-cause 修复 #1 (manifest SHA 校验) + #2 (batch retry + failure tracking)**；tracked 历史 manifest；CLI 真正消费 manifest；20 组合重跑 |
| 8 | `b14e732` | **root-cause 修复 #3 (reward wiring)**；`args.samples_by_id` 直传 `compute_reward()`；重算 20 个 run 的 reward metrics |
| 9 | `bbb6e75` | doc-only：protocol Full command 改用 manifest；round/status/fix description 同步 |
| 10 | 当前 correction | doc-only：用户确认 90/750 范围术语；roadmap 状态同步；明确 90 是历史 P5-02 benchmark，750 是父 split |

历史提交 SHA 仅用于描述变更；当前状态由本文件中的 live commands 实时核验。

## 7. Detached auditor history and round-10 scope resolution

### round-9 audit (calculet/gpt-5.6-terra)

- **Verdict**: disapproved; auditor interpreted objective parenthetical “D2 dev 750 采样” as requiring 20 × 750 rows.
- **Independent repository evidence**: all five historical P5-02 transcripts have exactly 90 rows; `b4fd879:MANIFEST-dev.json` has count=90; the published P5-02 benchmark is therefore a 90-sample child subset of the current 750-sample dev split.
- **User decision**: explicitly selected the historical 90-sample interpretation and requested continuation on that path. This is now the authoritative goal scope; no 750-sample rerun is required for this list item.

### round-10 doc sync (this commit)

- README / protocol / stage review all state the user-confirmed 90/750 terminology contract;
- roadmap no longer says P5-04 is “尚未启动”; it records P5-04 as delivered and distinguishes parent split 750 from benchmark subset 90;
- full repository stale-prose scan performed (`.tmp/p5-04-round10-stale-scan.txt`); relevant roadmap stale text corrected.

## 8. Prior detached auditor history

### round-8 audit (calculet/gpt-5.6-terra)

- **Verdict**: disapproved (3 blocking doc consistency issues)
  1. Protocol §9 "Full" command still used `--samples-dir datasets/tool-calling-d2/dev --limit 90` — would silently select a different non-SHA-verified sample set.
  2. Stage review (round-8) wrongly claimed "no longer overwrites `args.samples_dir` as the manifest parent" — the code at line 913 still does `args.samples_dir = Path(args.samples_manifest).parent`; the actual fix is the *direct use of* `args.samples_by_id` for `compute_reward()`.
  3. README / protocol / stage review still described the work as "round-7" while the committed root-cause fix is round-8 and the audit history shows prior rounds were disapprovals.
- **Auditor verified substantive items**: 20 runs / 1800 rows / 0 empty generations / 0 SHA mismatches / 96 selftest assertions PASS / independently recomputed `reward_binary` and `reward_layered` from the manifest samples matched all 20/20 summaries bit-for-bit.

### round-9 doc sync (this commit)

- Protocol §9 "Full" command now uses `--samples-manifest docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset.manifest.json` (the actual command used to produce the 20 published runs); also adds the WSL2 offline-friendly form with the same flag.
- README status header + 5 inline references switched "round-7" → "round-8" (or split into round-7 / round-8 contributions where the fix spans two rounds).
- Protocol status header + §3.1 / §6 / §6.1 sub-section titles + §6.1 wording sync'd.
- Stage review §7 rewritten to accurately describe the round-8 fix (passing `args.samples_by_id` directly to `compute_reward()`; `args.samples_dir` is retained for log path only — NOT used by the reward code path).
- Round-7 audit block retained as historical evidence (auditor's repro is the contract we satisfied in round-8).

### round-7 audit (calculet/gpt-5.6-terra)

- **Verdict**: disapproved
- **Blocking**: 全部 20 run `reward_layered = 0.0` 是 root-cause bug — 原代码从 `args.samples_dir` (=manifest 父目录) `glob("*.json")` 读样本，该目录里只有 manifest 本身，故 `samples_by_id={}` → signals=[] → silent 0。
- **Auditor repro**: 直接用 manifest 路径 + 真实 sample 重算 SmolLM2-360M transformers b1 得到 0.4236。

### round-8 fix (commit b14e732)

- `main()` 把 `args.samples_by_id = {s['id']: s for s in samples}` 挂到 args; `args.samples_dir` 仍然被覆盖为 `Path(args.samples_manifest).parent`（仅用于 summary 报告路径, 不参与 reward 计算）。
- `run_one_combination()` 从 `args.samples_by_id` 取样本直接调 `compute_reward(sample, row)`; 不再走 `reward_offline.load_samples()`。
- 负向断言: `samples_by_id` 为空 / 无 overlap 时 raises `RuntimeError`, 绝不 silent zero; `RuntimeError` 不再被 fallback `except Exception` 吞掉。
- 同步重写 20 个 run artifact summary + comparison.csv + comparison_delta.csv + README + protocol。
- 重建 reward_layered 取值 0.3601–0.4236, 与 P5-02 §8 历史 0.33–0.43 吻合。

### Stage reviewer

- round-7 reviewer: PASS
- round-8 reviewer: PASS (subagent dispatched, `PI_PROVIDER=minimax-cn`, `PI_MODEL=MiniMax-M3`)
- round-9 (this commit) reviewer: 待本轮 detached auditor 核验

## 9. Final disposition

P5-04 的代码入口、双 backend 实现、20 个真实组合、P5-02 精确历史 benchmark subset (manifest + SHA 校验)、4 轴指标、同模型 Δ%、reader-facing README、protocol 与 stage-review record 均已 root-cause 修复并交付。更大数据集、服务化、多 GPU、自研模型 vLLM 适配属于后续阶段，不阻塞本目标。