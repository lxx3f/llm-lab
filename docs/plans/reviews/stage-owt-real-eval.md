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
- `python3 scripts/eval_owt_real.py --selftest` PASS (16 断言：OWT SHA、
  hash helpers、cache round-trip、mock loss count/finiteness、缺 SHA
  raises、5 models + 5 unique revisions、fingerprint helper)。

## Verification (filled after Stage 4)

```text
HEAD:                     (filled at commit time)
git status:               clean
selftest:                 16 PASS / 0 FAIL (no-GPU mode)
owt source sha256:        2406f278e71829d273b315e9b403285baea7022b26a96d2728dd8b776ea40660
owt source size:          289,998,753 bytes (full validation file)
evaluated prefix:         52,428,758 bytes (newline-aligned, 50 MiB ≈ 18% of full)
results dir:              artifacts/owt-real-eval/results/{SmolLM2-360M,SmolLM2-1.7B,Qwen2.5-0.5B,Qwen2.5-1.5B,Qwen2.5-3B}.json
cache dir:                artifacts/owt-real-eval/cache/<model_short>/validation.{tokens.int32,metadata.json}
comparison:               artifacts/owt-real-eval/comparison.{csv,json}
total runs:               5/5 (SmolLM2-360M / SmolLM2-1.7B / Qwen2.5-0.5B / Qwen2.5-1.5B / Qwen2.5-3B)
total evaluated tokens:   59,746,002
all mean_loss finite:     True
all perplexity finite:    True
cache sha256 unique:      True (SmolLM2 series share tokenizer → 1 SHA; Qwen2.5 series share tokenizer → 1 SHA; total 2 distinct)
revision_verified count:  0/5 (all False — ModelScope 不识别 P5-04 exact revision，已回退 master)
```

## Per-model evidence (filled after Stage 4)

| 模型 | encoded_tokens | evaluated_tokens | mean_loss_nats | perplexity | revision_verified | cache_sha256 |
|---|---:|---:|---:|---:|:---:|---|
| Qwen2.5-0.5B | 11,722,807 | 11,711,358 | 2.9859 | 19.80 | false (⚠) | `645150d622be2d76...` |
| Qwen2.5-1.5B | 11,722,807 | 11,711,358 | 2.6964 | 14.83 | false (⚠) | `645150d622be2d76...` |
| Qwen2.5-3B | 11,722,807 | 11,711,358 | 2.5694 | 13.06 | false (⚠) | `645150d622be2d76...` |
| SmolLM2-360M | 12,317,994 | 12,305,964 | 2.7212 | 15.20 | false (⚠) | `72b1b758641226c0...` |
| SmolLM2-1.7B | 12,317,994 | 12,305,964 | 2.4026 | 11.05 | false (⚠) | `72b1b758641226c0...` |

## Risks recorded in README

- 不同 tokenizer 下 perplexity 不可直接对比（vocab size 不同）。
- `revision_verified=False` 时 master 与 P5-04 exact revision 字节差异需说明。
- 仅 evaluation；不重训任何模型。

## Reviewer evidence (fresh-context rehearsal, HEAD 962308a)

fresh-context reviewer (`reviewer` agent) VERDICT: PASS on all 8 bounded checks:

- A. HEAD `962308a90fd5167c657e8cd494f33db0a5652022`, working tree clean.
- B. OWT source SHA `2406f278...40660`, size `289998753` bytes (locked).
- C. 5/5 per-model JSONs finite: mean_loss > 0, perplexity > 0, evaluated_tokens ≥ 1M (12.3M / 11.7M); encoded_tokens consistent with 50 MiB prefix (12,317,994 / 11,722,807).
- D. comparison.csv has 5 rows; every row's mean_loss_nats matches the corresponding JSON to 15-digit float equality.
- E. selftest `[selftest] all tests PASSED`; semantic PASS count = 30 (≥ 25 ✓; note PASS lines are 2-space indented).
- F. README comparison table populated (5 rows at lines 69-73).
- G. Stage review verification block populated (4 matches).
- H. No stale unfilled placeholders in README; the reviewer evidence placeholder block was removed (was the only remaining stub).

Reviewer's only non-blocking observation: `source_bytes` field in JSON refers to the full 290 MB source file rather than the 50 MiB evaluated prefix. The script's `evaluated_bytes` and `loss_nats_per_evaluated_byte` fields correctly record the prefix. This split is intentional — `source_bytes` identifies the canonical OWT file, `evaluated_bytes` records what was actually evaluated.

Reviewer's second non-blocking observation: PASS-line indentation (2-space) means `grep -c '^\[PASS\]'` returns 0. Semantic counting (`grep -cE '^[[:space:]]*\[PASS\]'`) returns 30, satisfying the ≥ 25 PASS threshold. Both indented and non-indented readers can grep with the documented whitespace-tolerant pattern.

END REVIEWER EVIDENCE.
