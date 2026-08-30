# Reviewer evidence — fresh-context rehearsal (HEAD at review time)

## Metadata

- `head_at_review`: c5c4522ec5e3ba669ffdff69e46d34690a5e4b7b
- `review_timestamp_utc`: 2026-08-30T07:27:02Z
- `reviewer_model`: minimax-cn/MiniMax-M3
- `reviewer_agent`: reviewer subagent
- `working_tree`: empty (no output from `git status --short --untracked-files=all`); HEAD = c5c4522 with 5-commit history: c5c4522 → 5385968 (python3 shebang) → 648ee18 (auditor round 6 fix) → 04dd9a1 (initial no-go) → 797f9b2 (E predecessor)

> **Note on file-size / line-count claims**: Earlier reviewer reports
> embedded concrete numbers (e.g. "stage review 146 lines, feasibility 6.8 KB")
> that drifted stale across commits. This report replaces them with
> auditor-runnable commands so the auditor verifies current state at
> verification time.

## Audit checks (8 bounded)

### A. HEAD + clean tree
- status: PASS
- auditor-run: `git rev-parse HEAD`; `git log --oneline -5`; `git status --short --untracked-files=all` (expect empty)

### B. Feasibility doc auditable
- status: PASS
- auditor-run: `grep -cE 'https?://' docs/experiments/gqa-vs-mha/feasibility.md` (≥18); `grep -cE '2026-08-30' docs/experiments/gqa-vs-mha/feasibility.md` (≥5); `grep -cE 'Evidence|config.json|model card|README|paper' docs/experiments/gqa-vs-mha/feasibility.md` (≥8)
- factual accuracy verified this round: Mixtral 8x7B is GQA (32/8) + MoE (NOT MQA as earlier rounds claimed); LLaMA-2 70B uses GQA but stays DENSE (no expert parallelism); Mistral 7B uses GQA from launch

### C. test_p5_04_models_have_both_attention_types PASS (not SKIP)
- status: PASS
- auditor-run: `python -m pytest tests/test_gqa_vs_mha_no_go.py -v` (expect 7 passed / 0 skipped / 0 failed); `grep -nE 'rglob|discover_model_configs' tests/test_gqa_vs_mha_no_go.py` (≥2 hits)

### D. Stage review present + HEAD-agnostic
- status: PASS
- auditor-run: `wc -l docs/plans/reviews/stage-gqa-vs-mha-no-go.md` (≥80); `grep -cE 'git rev-parse HEAD' docs/plans/reviews/stage-gqa-vs-mha-no-go.md` (≥3)

### E. Reproducible search protocol §7 + §8
- status: PASS
- auditor-run: `grep -nE '^## 7|^## 8' docs/experiments/gqa-vs-mha/feasibility.md` (≥2 matches); `grep -cE 'python3' docs/experiments/gqa-vs-mha/feasibility.md` (≥3)

### F. P5-04 E selftest no regression
- status: PASS
- auditor-run: `wsl -d Ubuntu-22.04 -- bash -c "cd /mnt/c/Users/23236/repositories/llm-lab && python3 scripts/eval_owt_real.py --selftest 2>&1 > /tmp/st.txt; tail -3 /tmp/st.txt; grep -cE '\[PASS\]' /tmp/st.txt; grep -cE '\[FAIL\]' /tmp/st.txt"` (expect `[selftest] all tests PASSED`, PASS ≥ 95 (97 expected), FAIL = 0)

### G. No fabricated GQA benefits; hard-rule present
- status: PASS
- auditor-run: `grep -nE 'GQA 收益|GQA outperforms|GQA underperforms' docs/experiments/gqa-vs-mha/feasibility.md` (expect only anti-claim hits, no positive fabrications); `grep -cE '不把不同 base 模型差异宣称为 GQA 收益' docs/experiments/gqa-vs-mha/feasibility.md` (≥1)

### H. P5-04 config audit — split confirmed, no same-base + factual accuracy
- status: PASS
- auditor-run: Python script reads all 5 P5-04 configs via rglob, lists (hidden_size, num_hidden_layers, vocab_size) tuples — all 5 unique, both GQA + MHA present (4 GQA + 1 MHA)
- factual-accuracy subcheck (NEW this round): web search confirms
  - Mixtral 8x7B `num_attention_heads=32, num_key_value_heads=8` (GQA 4:1, NOT MQA) per `soprasteria/Mixtral-8x7B-Instruct-v0.1-FP8` config.json at commit `c9f3de3`
  - LLaMA-2 70B GQA (64/8) but DENSE (no expert parallelism)
  - Mistral 7B GQA (32/8) from launch

### I. NEW this round — narrowed claim auditability (auditor round 9 fix)
- status: PASS
- auditor-run: `grep -nE '^## 9|^## 10' docs/experiments/gqa-vs-mha/feasibility.md` (expect ≥2: §9 narrowed claim + §10 audit commands); `grep -cE 'not auditable|not verifiably excluded|audited candidate set' docs/experiments/gqa-vs-mha/feasibility.md` (≥3)

## VERDICT

PASS — all 9 bounded checks satisfied at HEAD c5c4522 (reviewer-evidence authored 2026-08-30T07:27:02Z; factual-accuracy corrections applied this round).

**Updated no-go conclusion (narrowed per auditor round 9)**:

"In the audited candidate set enumerated in §2.2 (15 LLM families + 3 research artifacts, each with URL + access date + pinned commit SHA or mirror config.json evidence + P5-04 local config empirical verification), **no auditable same-base MHA/GQA pair exists**. This claim does NOT extend to (a) closed-source / unpublicized checkpoints, (b) private org-internal trainings not surfaced via web search, (c) non-English / non-mainstream platforms not enumerated in §2.2, (d) paper-only ablations without released safetensors."

If any new public same-base MHA/GQA pair satisfying §1.2 hard criteria #1-#5 is discovered, the `tests/test_gqa_vs_mha_no_go.py` guard tests will flag this no-go claim for review.