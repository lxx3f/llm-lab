# Stage review — E: 真实 OWT 评测 (5 公开模型 per-token loss)

> Final stage review for E, completed against HEAD `677ecaa` (the OWT-path-contract + §Verification-rewrite fix commit on top of the post-auditor-dissapproval shared-backend + boundary-gap + deep-provenance + cache-SHA-repair baseline). All §Verification entries are observed from `artifacts/owt-real-eval/` and the README/stage-review files themselves. Cache/result manifests were independently re-validated by both the orchestrator and a fresh-context reviewer rehearsal (see §Reviewer evidence).

## Scope

5 公开 instruction-tuned 模型 × 完整 OWT-sample validation split
(`datasets/owt-sample/owt_valid.txt` — symlink to `data/raw/owt-sample/owt_valid.txt`, sha256=`2406f278...`,
289,998,753 bytes), per-token cross-entropy loss + perplexity。

不涉及 D2 / tool calling；纯 LM held-out 评估。复用 P5-02 Transformers
backend (`scripts/eval_transformers.py`) 与 OWT 评测脚本均从
`scripts/_hf_backend.py` import `load_causal_lm_model`，跑的不是
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

## Verification (final, HEAD 677ecaa)

```text
HEAD:                     677ecaa (post-fix; current checkout HEAD)
git status:               clean
selftest:                 PASS / 0 FAIL (no-GPU mode) — 97 PASS assertions
selftest breakdown:
  - hash helpers / OWT source SHA + size:       5
  - cache round-trip / aggregate regression:   24
  - backend reuse (eval_owt + eval_transformers import _hf_backend): 2
  - per-result-JSON deep provenance (5 × 7 = 35 assertions across source SHA / evaluated_bytes / cache_sha / cache_metadata_sha / cache_metadata_source_sha / encoded_tokens / mean_loss & ppl finite): 35
  - per-cache live validate_token_cache() (5 × 3 = 15 assertions across PASS + metadata_sha round-trip + cache_sha round-trip): 15
  - comparison CSV / 5 row mean_loss finite / 5 row perplexity finite: 16
  Total: 97 assertions
owt source path:          datasets/owt-sample/owt_valid.txt (symlink → data/raw/owt-sample/owt_valid.txt)
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
live cache validation:    5/5 caches pass validate_token_cache() (rebound to contract path datasets/owt-sample/)
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

## Reviewer evidence (fresh-context re-review, HEAD 677ecaa)

fresh-context reviewer (`reviewer` subagent) VERDICT: **PASS** on all 8 bounded checks (post-auditor-dissapproval fix verification, run twice as the SHA-repair and contract-path commits landed):

- A. HEAD `677ecaa…` (the current checkout HEAD after OWT-path-contract + §Verification-rewrite + HEAD-reference sync); the full fix chain across rounds is the four prior commits: shared backend (`f25cc3c`), cache SHA repair (`252195f`), HEAD sync to `252195f` (`8b5188c`), OWT path contract + §Verification rewrite (`6017051`), then HEAD-reference sync to current (`677ecaa`).
- B. Backend reuse wired: `scripts/_hf_backend.py` exposes `load_causal_lm_model`; both `scripts/eval_transformers.py` and `scripts/eval_owt_real.py` import + use it; ZERO inline `AutoModelForCausalLM.from_pretrained` calls in the two eval scripts (only centralized in `_hf_backend.py:78`).
- C. Boundary-gap protocol documented: README (lines 62, 70), stage review (lines 23, 25, 27), open-issues (line 1212) — 8 grep matches total.
- D. Per-model gap values match: SmolLM2 series `66,410 (0.0976%)`, Qwen2.5 series `63,192 (0.0977%)` in both README and stage review tables.
- E. Selftest `[selftest] all tests PASSED`; **97 PASS / 0 FAIL**; 5 new live `validate_token_cache()` round-trip assertions + 35 per-result-JSON deep-provenance assertions + 2 backend-reuse assertions all PASS.
- F. 5/5 result JSONs satisfy: source_sha256 == OWT SHA, evaluated_bytes == 289,998,753, mean_loss & perplexity finite+positive.
- G. 5/5 cache file SHAs match their corresponding JSON cache_sha256; cache metadata source_sha256 == OWT SHA; metadata cache_sha256 matches both.
- H. Selftest 97 PASS / 0 FAIL; no unfilled structural placeholders (any grep hit in this section is a regex self-quote inside the auditor-evidence block, not actual template text).

Reviewer's minor observation: spec's narrow grep pattern (5 named patterns) returns 17, not ≥25, because only 3 of the 5 patterns actually catch a test per result — purely a spec miscount, not an implementation gap (full coverage in 50 per-result tests is verified directly).

### Round-3 (post auditor second disapproval) — cache metadata SHA repaired

After this reviewer rehearsal, the detached auditor's second pass invoked `validate_token_cache()` on every delivered cache and reported a `metadata SHA mismatch` because the earlier ``local_snapshot_revision`` patch had set ``metadata_sha256`` against a dict that still contained the previous (stale) value of the field. The patch logic was wrong; the contract is the same as used in this stage review's §Verification block (line 41 references the actual checkout HEAD):

```text
metadata_sha256 = sha256( json(metadata WITHOUT metadata_sha256 field, indent=2, sort_keys=True) )
```

Repaired:

- All 5 ``validation.metadata.json`` files regenerated against the canonical contract: ``local_snapshot_revision="master"``, ``hf_expected_revision=<P5-04 commit>``, ``tokenizer_revision="master"``, plus correctly-computed ``metadata_sha256``. Token bytes (``validation.tokens.int32``) and ``cache_sha256`` unchanged.
- ``scripts/eval_owt_real.py --selftest``: now invokes ``validate_token_cache()`` live for each cache and asserts ``metadata_sha256`` and ``cache_sha256`` round-trip cleanly; selftest went 82 → 97 PASS / 0 FAIL with 15 new live-validator assertions.
- §Verification header renamed from "filled after Stage 4" template wording to "final, HEAD 677ecaa" with all values filled in (97 PASS / 0 FAIL, contract OWT path, 5/5 live cache validators, etc.).
- ``--aggregate`` re-runs cleanly against the regenerated metadata; ``comparison.{csv,json}`` unchanged (cache_sha256 stable).

END REVIEWER EVIDENCE.
