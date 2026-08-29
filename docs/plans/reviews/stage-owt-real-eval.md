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
HEAD:                     <commit SHA>
git status:               clean
selftest:                 <pass count> PASS / 0 FAIL
owt source sha256:        2406f278e71829d273b315e9b403285baea7022b26a96d2728dd8b776ea40660
owt source size:          289,998,753 bytes
results dir:              artifacts/owt-real-eval/results/{SmolLM2-360M,SmolLM2-1.7B,Qwen2.5-0.5B,Qwen2.5-1.5B,Qwen2.5-3B}.json
cache dir:                artifacts/owt-real-eval/cache/<model_short>/
comparison:               artifacts/owt-real-eval/comparison.{csv,json}
total runs:               5/5 (SmolLM2-360M / SmolLM2-1.7B / Qwen2.5-0.5B / Qwen2.5-1.5B / Qwen2.5-3B)
total evaluated tokens:   <sum>
all mean_loss finite:     True
all perplexity finite:    True
all 5 cache sha256 unique:<True/False>
all 5 model_local_dir:    <paths>
all revision_verified:    <True/False counts>
```

## Per-model evidence (filled after Stage 4)

| 模型 | encoded_tokens | evaluated_tokens | mean_loss_nats | perplexity | revision_verified | cache_sha256 |
|---|---:|---:|---:|---:|:---:|---|

## Risks recorded in README

- 不同 tokenizer 下 perplexity 不可直接对比（vocab size 不同）。
- `revision_verified=False` 时 master 与 P5-04 exact revision 字节差异需说明。
- 仅 evaluation；不重训任何模型。

## Reviewer evidence placeholder

`<filled by reviewer rehearsal>`
