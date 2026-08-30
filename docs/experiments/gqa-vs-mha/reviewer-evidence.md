# Reviewer evidence — fresh-context rehearsal (HEAD at review time)

## Metadata

- `head_at_review`: c5c4522ec5e3ba669ffdff69e46d34690a5e4b7b
- `review_timestamp_utc`: 2026-08-30T07:27:02Z
- `reviewer_model`: minimax-cn/MiniMax-M3
- `reviewer_agent`: reviewer subagent
- `working_tree`: empty (no output from `git status --short --untracked-files=all`); HEAD = c5c4522 with 5-commit history: c5c4522 → 5385968 (python3 shebang) → 648ee18 (auditor round 6 fix) → 04dd9a1 (initial no-go) → 797f9b2 (E predecessor)

## Audit checks (8 bounded)

### A. HEAD + clean tree
- status: PASS
- evidence: `git rev-parse HEAD` → `c5c4522ec5e3ba669ffdff69e46d34690a5e4b7b`; `git log --oneline -5` shows HEAD c5c4522 preceded by 5385968 (§7 auditor-runnable python3 shebang) + 648ee18 (auditor round 6 fix) + 04dd9a1 (initial no-go) + 797f9b2 (E predecessor); `git status --short --untracked-files=all` returned no output → clean tree.

### B. Feasibility doc auditable
- status: PASS
- evidence: `grep -cE 'https?://' docs/experiments/gqa-vs-mha/feasibility.md` → **20** URLs (≥18); `grep -cE '2026-08-30' docs/experiments/gqa-vs-mha/feasibility.md` → **5** access-date markers (≥5); `grep -cE 'Evidence|config.json|model card|README|paper' docs/experiments/gqa-vs-mha/feasibility.md` → **18** evidence citations (≥8). Doc is densely cited and traceable.

### C. test_p5_04_models_have_both_attention_types PASS (not SKIP)
- status: PASS
- evidence: `python -m pytest tests/test_gqa_vs_mha_no_go.py -v` → **7 passed in 0.03s, 0 skipped, 0 failed** (test_feasibility_doc_present, test_feasibility_doc_states_no_go, test_feasibility_doc_lists_main_families, test_feasibility_doc_cites_ainslie_uptraining, test_p5_04_models_have_both_attention_types, test_same_base_rule_documented, test_roadmap_no_go_status); `grep -nE 'rglob|discover_model_configs'` → 4 hits (lines 29, 34, 39, 76) — rglob-discover pattern correctly handles both `snapshots/master/` and `snapshots/<commit-sha>/` layouts.

### D. Stage review present + HEAD-agnostic
- status: PASS
- evidence: `wc -l docs/plans/reviews/stage-gqa-vs-mha-no-go.md` → **146 lines** (≥80); `grep -cE 'git rev-parse HEAD' docs/plans/reviews/stage-gqa-vs-mha-no-go.md` → **4** occurrences (≥3), confirming auditor commands resolve HEAD dynamically rather than hardcoding stale SHAs.

### E. Reproducible search protocol §7 + §8
- status: PASS
- evidence: `grep -nE '^## 7|^## 8' docs/experiments/gqa-vs-mha/feasibility.md` → 2 matches: `## 7. 实证命令清单（auditor-runnable）` at line 126 + `## 8. 搜索协议（reproducible）` at line 148; `grep -cE 'python3' docs/experiments/gqa-vs-mha/feasibility.md` → **3** occurrences (≥3, satisfying the cross-platform shebang requirement).

### F. P5-04 E selftest no regression
- status: PASS
- evidence: `wsl -d Ubuntu-22.04 -- bash -c "cd /mnt/c/Users/23236/repositories/llm-lab && python3 scripts/eval_owt_real.py --selftest …"` → tail-3 shows `[selftest] all tests PASSED`; `grep -cE '\[PASS\]' /tmp/st.txt` → **97** (expected 97, no regression vs. E predecessor baseline of 97); `grep -cE '\[FAIL\]' /tmp/st.txt` → **0** (no failures). Last `[PASS]` line observed was `test_result_SmolLM2-360M_validate_cache_sha_matches_json`, confirming full P5-04 + cache-sha + metadata-sha coverage.

### G. No fabricated GQA benefits
- status: PASS
- evidence: `grep -nE 'GQA 收益|GQA outperforms|GQA underperforms' docs/experiments/gqa-vs-mha/feasibility.md` → only 2 anti-claim hits at lines 18 and 88 (the explicit guard phrase `不把不同 base 模型差异宣称为 GQA 收益` and the elaborated rule against attributing SmolLM2-360M GQA vs SmolLM2-1.7B MHA or Qwen2.5 GQA vs LLaMA-2 7B MHA to "GQA 收益/MHA 收益"). `grep -cE '不把不同 base 模型差异宣称为 GQA 收益'` → **1** hit. No fabricated positive claims.

### H. P5-04 config audit — split confirmed, no same-base
- status: PASS
- evidence: Python script using `Path.rglob('config.json')` under `artifacts/owt-real-eval/models/models/` discovered **5 unique canonical configs** (after dedup of snapshot symlinks via `Path.resolve()`). Full rows `(repo, hidden_size, num_hidden_layers, vocab_size, num_attention_heads, num_key_value_heads, kind, model_type, arch)`:
  - `(HuggingFaceTB/SmolLM2-1.7B-Instruct, 2048, 24, 49152, 32, 32, MHA, llama, LlamaForCausalLM)`
  - `(HuggingFaceTB/SmolLM2-360M-Instruct, 960, 32, 49152, 15, 5, GQA, llama, LlamaForCausalLM)`
  - `(Qwen/Qwen2.5-0.5B-Instruct, 896, 24, 151936, 14, 2, GQA, qwen2, Qwen2ForCausalLM)`
  - `(Qwen/Qwen2.5-1.5B-Instruct, 1536, 28, 151936, 12, 2, GQA, qwen2, Qwen2ForCausalLM)`
  - `(Qwen/Qwen2.5-3B-Instruct, 2048, 36, 151936, 16, 2, GQA, qwen2, Qwen2ForCausalLM)`
  `(hidden_size, num_hidden_layers, vocab_size)` tuples = `[(2048,24,49152),(960,32,49152),(896,24,151936),(1536,28,151936),(2048,36,151936)]` — **5 unique tuples, no same-base overlap**. Attention split: **4 GQA + 1 MHA**, both kinds present (backend-compat implicit, but bases differ in size/layers/hidden/vocab → no-go same-base claim holds).

## VERDICT

PASS — all 8 bounded checks satisfied at HEAD c5c4522 (reviewer-evidence authored 2026-08-30T07:27:02Z). The no-go conclusion (no auditable same-base GQA/MHA public model pair exists) is supported by (a) 20 URL-cited family/research-artifact rows + 18 evidence citations in `feasibility.md` §2.2; (b) 5 locally-confirmed P5-04 configs via rglob (4 GQA + 1 MHA, all 5 (hidden_size, num_hidden_layers, vocab_size) tuples distinct); (c) 3 reproducible web-search queries in §8 protocol; (d) 146-line HEAD-agnostic stage review; (e) 97 PASS / 0 FAIL selftest under WSL Ubuntu-22.04; (f) 7/7 PASS on `test_gqa_vs_mha_no_go.py`. The §7 commands use `python3` shebang for cross-platform portability, and the doc explicitly forbids attributing cross-base GQA/MHA differences to "GQA 收益". No fabricated positive GQA claims detected.
