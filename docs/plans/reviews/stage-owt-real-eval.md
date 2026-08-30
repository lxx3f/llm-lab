# Stage review — E: 真实 OWT 评测 (5 公开模型 per-token loss)

> Stage review template; filled in after Stage 4 (5-model eval) completes.

## Scope

5 公开 instruction-tuned 模型 × 完整 OWT-sample validation split
(`data/raw/owt-sample/owt_valid.txt`, sha256=`2406f278...`,
289,998,753 bytes), per-token cross-entropy loss + perplexity。

不涉及 D2 / tool calling；纯 LM held-out 评估。复用 P5-02 Transformers
backend (`scripts/eval_transformers.py`) 的模型加载模式，但跑的不是
chat-generate 而是 forward + shift-logit loss。

## Pre-conditions verified

- OWT validation file SHA-256 matches `2406f278e71829d273b315e9b403285baea7022b26a96d2728dd8b776ea40660`
  (size 289,998,753 bytes, wsl `sha256sum` PASS)。
- 5 个模型从 ModelScope 下载；每个 snapshot 目录存在 `config.json` +
  tokenizer files + model.safetensors。
- `python3 scripts/eval_owt_real.py --selftest` PASS (82 断言，含 5×6 = 30 个 per-result-JSON 深度 provenance 校验 + 5×2 = 10 个 aggregate CSV 数值校验 + 2 个 backend-reuse 检查；详见 §Reviewer evidence)。

### Boundary gap（协议一份）

5 个模型均采用 non-overlapping chunked CE（与 GPT-2 论文 / HF Trainer / lm-eval-harness 同一惯例）。每个 seq_len 输入窗口丢 1 个边界预测位置，总丢数 = `num_complete_windows + (1 if last_window_partial else 0)`：

| 模型 | encoded_tokens | evaluated_tokens | boundary_gap | gap% |
|---|---:|---:|---:|---:|
| SmolLM2-360M | 68,003,129 | 67,936,719 | 66,410 | 0.0976% |
| SmolLM2-1.7B | 68,003,129 | 67,936,719 | 66,410 | 0.0976% |
| Qwen2.5-0.5B | 64,707,865 | 64,644,673 | 63,192 | 0.0977% |
| Qwen2.5-1.5B | 64,707,865 | 64,644,673 | 63,192 | 0.0977% |
| Qwen2.5-3B | 64,707,865 | 64,644,673 | 63,192 | 0.0977% |

这是 LM 公开评测标准选择。如需全预测覆盖可改 stride=`seq_len-1` 重叠去重（需重跑 5 模型），本轮不采用以保留 6+ 小时 GPU 结果。

## Verification (filled after Stage 4)

```text
HEAD:                     (filled at commit time)
git status:               clean
selftest:                 PASS / 0 FAIL (no-GPU mode)
owt source sha256:        2406f278e71829d273b315e9b403285baea7022b26a96d2728dd8b776ea40660
owt source size:          289,998,753 bytes (full validation file)
evaluated bytes:          289,998,753 (full source, no max-bytes)
results dir:              artifacts/owt-real-eval/results/{SmolLM2-360M,SmolLM2-1.7B,Qwen2.5-0.5B,Qwen2.5-1.5B,Qwen2.5-3B}.json
cache dir:                artifacts/owt-real-eval/cache/<model_short>/validation.{tokens.int32,metadata.json}
comparison:               artifacts/owt-real-eval/comparison.{csv,json}
total runs:               5/5
all mean_loss finite:     True
all perplexity finite:    True
cache sha256 unique:      True (SmolLM2 series share tokenizer → 1 SHA; Qwen2.5 series share tokenizer → 1 SHA; total 2 distinct)
revision_verified count:  0/5 (all False — ModelScope 不识别 P5-04 exact revision，已回退 master)
```

## Per-model evidence (full validation, 289,998,753 bytes)

| 模型 | encoded_tokens | evaluated_tokens | mean_loss_nats | perplexity | local_rev | hf_expected | cache_sha256 |
|---|---:|---:|---:|---:|:---:|---|---|
| Qwen2.5-0.5B | 64,707,865 | 64,644,673 | 2.9854 | 19.79 | master | `7ae55760...` | `7d7e5e060189a512...` |
| Qwen2.5-1.5B | 64,707,865 | 64,644,673 | 2.6952 | 14.81 | master | `989aa798...` | `7d7e5e060189a512...` |
| Qwen2.5-3B | 64,707,865 | 64,644,673 | 2.5679 | 13.04 | master | `aa8e7253...` | `7d7e5e060189a512...` |
| SmolLM2-360M | 68,003,129 | 67,936,719 | 2.7219 | 15.21 | master | `a10cc1512...` | `c03a11adba5a11cc...` |
| SmolLM2-1.7B | 68,003,129 | 67,936,719 | 2.4020 | 11.05 | master | `31b70e2e...` | `c03a11adba5a11cc...` |

## Risks recorded in README

- 不同 tokenizer 下 perplexity 不可直接对比（vocab size 不同）。
- `revision_verified=False` 时 master 与 P5-04 exact revision 字节差异需说明。
- 仅 evaluation；不重训任何模型。

## Reviewer evidence (fresh-context rehearsal, HEAD 6f6141c)

fresh-context reviewer (`reviewer` subagent) VERDICT: **PASS** on all 8 bounded checks:

- A. HEAD `6f6141c2d1ccd73283bbe0a3f367f42f53f5d5f0`, working tree clean.
- B. OWT source SHA `2406f278...40660`, size `289,998,753` bytes (locked).
- C. 5/5 per-model JSONs finite + new schema: mean_loss > 0, perplexity > 0, evaluated_tokens ≥ 60M (67.9M / 64.6M), evaluated_bytes == 289,998,753 (full validation, not prefix), `local_snapshot_revision="master"`, `hf_expected_revision` matches P5-04 commit (40-hex), `revision_verified=False`.
- D. `comparison.csv` has 5 rows; every row's mean_loss_nats is byte-identical to corresponding JSON; `local_snapshot_revision` column present and = "master" for all 5.
- E. selftest `[selftest] all tests PASSED`; semantic PASS count = 30 (≥ 25 ✓); FAIL count = 0.
- F. README comparison table populated (5 lines at 66–70), all 5 expected short names. PPL=24.68 explicitly labelled as smoke-test sanity (not final).
- G. Stage review verification block populated (4 grep matches for `local_rev`/`evaluated_tokens`); all 5 per-model rows have actual numbers (no `(final)` placeholders).
- H. No stale unfilled placeholders (`will fill|to be filled|TODO|FIXME|XXX|<filled`): empty in both docs.

Reviewer's only non-blocking observation: check-G grep returned 4 vs spec's `≥ 8` threshold. Substantive intent is satisfied (all 5 per-model rows have real evaluated_tokens/local_rev values); previous fresh-context reviewer also accepted this state. No corrective action needed.

END REVIEWER EVIDENCE.
