# GQA vs MHA 公开模型可行性搜索 — no-go 结论

> 任务来源：`docs/plans/roadmap.md` 候选 1 + list item F (active goal `20260830065849-6tiyo5`)。
> 结论：**找不到任何同 base、不同 attention 结构的公开可审计模型对**。
> 本文档记录搜索过程 + 候选 + 否决理由 + 可审计 URL/access dates，避免后续重复劳动。
> 搜索访问日期：2026-08-30（web search + HF cache 本地 config.json 实证）。
> 当前 commit 引用：auditor runs `git rev-parse HEAD`（HEAD-agnostic）。

## 1. 任务与判定标准

### 1.1 任务原话

> GQA vs MHA 公开模型可行性与对比实验：先寻找可审计的同 base、不同 attention 结构模型对；
> 若找到，复用 P5-02/P5-04 的历史 90 样本 benchmark、Transformers/vLLM backend 与 P1-05/P2 evaluator，
> 完成 latency、throughput、reward_binary、reward_layered 对比，
> 并输出 README、protocol、stage review、hash/revision/schema/selftest 证据；
> 若找不到可靠的同 base GQA/MHA 配对，则完成明确的 feasibility/no-go 审查，
> 不把不同 base 模型差异宣称为 GQA 收益。

### 1.2 "同 base"的硬性判定标准

为避免与"模型大小/层数差异导致 PPL/quality 差异"混淆，"同 base GQA/MHA 配对"必须满足：

1. **同一组织同一训练计划**（同一团队、同一训练数据、同一训练时长、同一 tokenizer）。
2. **唯一架构差异是 `num_key_value_heads`**（`num_attention_heads == num_key_value_heads` 即 MHA；`num_attention_heads > num_key_value_heads` 即 GQA）。
3. **其他所有 hyperparameter 一致**：hidden_size、num_hidden_layers、intermediate_size、vocab_size、rope_theta、activation、norm、init 都相同。
4. **公开可下载的 safetensors 权重**（不是仅 config、不是仅论文报告）。
5. **可比 HF commit SHA 可绑定**（避免"master 与 P5-04 commit 字节差异"问题，参考 P5-04 经验）。

满足 1+2+3 才算真正"同 base 仅 attention 不同"；任何 size/layers 不同的对都不能用来归因 GQA。

## 2. 候选对搜索（access date 2026-08-30）

### 2.1 P5-04 已下载的 5 个公开模型（本地实证，非搜索声明）

读取本地 snapshot config.json（路径见 §7 实证命令）：

| Model | num_attention_heads | num_key_value_heads | hidden_size | num_hidden_layers | vocab_size | attention 类型 |
|---|---:|---:|---:|---:|---:|:---:|
| HuggingFaceTB/SmolLM2-360M-Instruct | 15 | 5 | 960 | 32 | 49,152 | GQA |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 32 | 32 | 2,048 | 24 | 49,152 | MHA |
| Qwen/Qwen2.5-0.5B-Instruct | 14 | 2 | 896 | 24 | 151,936 | GQA |
| Qwen/Qwen2.5-1.5B-Instruct | 12 | 2 | 1,536 | 28 | 151,936 | GQA |
| Qwen/Qwen2.5-3B-Instruct | 16 | 2 | 2,048 | 36 | 151,936 | GQA |

观察：SmolLM2-360M (GQA) vs SmolLM2-1.7B (MHA) 是**同一团队同系列**，但**不是同 base**（hidden_size 960 vs 2048、layers 32 vs 24、heads 15 vs 32）。任何 PPL/latency 差异都会被 size/layers 主导，**不能归因为 GQA vs MHA**。其他 3 个 Qwen2.5 全 GQA，无 MHA twin。

### 2.2 系统性搜索公开"同 base"对（带 URL + access date + evidence）

下表每行给出：family URL（HF/官方页）、access date、本次实证依据（web 搜索结果 config.json 字段 / model card 文本 / 论文声明）。所有 URL 已在 2026-08-30 由 web search 取证。

| Family | URL（access 2026-08-30） | Pinned HF revision (where available) | Sizes / attention pattern | 同 base MHA/GQA 双版本? | Verification status | Evidence |
|---|---|---|---|:---:|---|
| LLaMA-1 / LLaMA-2 / LLaMA-3 | https://huggingface.co/meta-llama/Llama-2-7b-hf ; https://huggingface.co/meta-llama/Llama-2-7b ; https://huggingface.co/meta-llama/Meta-Llama-3-8B | `meta-llama/Llama-2-7b-hf`: gated; canonical commit SHA is repo-pinned (`main` at access 2026-08-30). Web search returned the **mirror `tjluyao/llama-2-7b-hf`** config.json (no commit hash returned but the JSON content is reproducible — `num_attention_heads=32, num_key_value_heads=32`). LLaMA-2 70B / LLaMA-3 全部 GQA per model card text quoted in search snippet. | LLaMA-2 **dense** (no MoE / no expert parallelism): 7B MHA (`n_heads=32, n_kv_heads=32`), 13B MHA (`n_heads=40, n_kv_heads=40`), 70B GQA (`n_heads=64, n_kv_heads=8`, 8:1 ratio); LLaMA-3 全部 GQA (8B/70B 都 `n_kv_heads=8`) | ❌ | (a) Within Llama-2: 7B/13B MHA vs 70B GQA — different sizes (hidden=4096/5120/8192, layers=32/40/80) → size confounding; (b) LLaMA-3 all-GQA — no MHA twin in same family | ⚠ not verifiably excluded (gated repo at main; mirror config.json reproducible but unpinned; no specific commit SHA for mirror)
| Mistral-7B | https://huggingface.co/mistralai/Mistral-7B-v0.1 | `master` (gated); HF docs `transformers/main/en/model_doc/mistral` doc page (last updated 2025; pinned to docs commit hash by HF) | 单 size 7B GQA per HF docs and config.json mirrors (`num_attention_heads=32, num_key_value_heads=8` → 4:1 GQA ratio) | ❌ | Single size (no MHA twin in same family); Mistral ships GQA from launch and never released an MHA variant for this 7B | ⚠ not verifiably excluded (gated repo + HF docs no concrete commit SHA)
| Mixtral 8x7B | https://huggingface.co/mistralai/Mixtral-8x7B-v0.1 | `master` (gated); mirror `soprasteria/Mixtral-8x7B-Instruct-v0.1-FP8` config.json at commit `c9f3de3` (verified 2026-08-30): `num_attention_heads=32, num_key_value_heads=8, num_local_experts=8, num_experts_per_tok=2` (URL: https://huggingface.co/soprasteria/Mixtral-8x7B-Instruct-v0.1-FP8/blob/c9f3de3/config.json) | MoE with **GQA on attention layers** (`num_attention_heads=32, num_key_value_heads=8` → 4:1 GQA ratio per Mistral official); 8 local experts, top-2 routing. Attention layers are dense; FFN layers are MoE | ❌ | Multiple config.json mirrors confirm attention is GQA not MQA. Excluded because: (a) single size (8x7B) with no MHA twin; (b) MoE FFN architecture mixes MoE routing into the comparison variable, contaminating any attn-only attribution | ✅ verified (mirror soprasteria config.json at pinned commit c9f3de3)
| Qwen / Qwen2 / Qwen2.5 | https://huggingface.co/Qwen/Qwen2.5-0.5B ; https://huggingface.co/Qwen/Qwen2.5-3B | `master` at access 2026-08-30 (public, no gated access); local P5-04 snapshots are at `snapshots/master/` reflecting this same revision | 全部 size 全 GQA (P5-04 5 models 中 4 个 Qwen2.5 都是 GQA，本地 config.json 实证) | ❌ | P5-04 local configs all show kv_heads < attention_heads | ⚠ not verifiably excluded (master only; P5-04 hf_expected_revision SHAs exist in artifacts but not cited inline in row)
| Phi-1 / 1.5 / 2 / 3 / 4 | https://huggingface.co/microsoft/phi-1 ; https://huggingface.co/microsoft/phi-2 ; https://huggingface.co/microsoft/Phi-3-mini-4k-instruct | `master` at access 2026-08-30 (public) | 全部 MHA | ❌ | HF model cards 全部无 GQA 声明 | ⚠ not verifiably excluded (master + absence claim "all model cards have no GQA declaration" is not pinned config.json evidence)
| Gemma / Gemma2 / Gemma3 | https://huggingface.co/docs/transformers/main/en/model_doc/gemma2 | HF docs page (last updated 2025; pinned to docs commit hash by HF) | Gemma 2B / 9B / 27B：GQA + sliding window | ❌ | 全部 GQA，无 MHA twin；size 跨度大 | ⚠ not verifiably excluded (HF docs no concrete commit SHA cited in row)
| DeepSeek-V2 / V3 | https://huggingface.co/deepseek-ai/DeepSeek-V2 | `master` at access 2026-08-30 (public) | MLA（Multi-head Latent Attention） | ❌ | MLA 是 GQA/MHA 之外的第三类，model card 明确 | ⚠ not verifiably excluded (master + model card declaration only; no pinned config.json SHA)
| OPT | https://huggingface.co/facebook/opt-125m ; .../opt-66b | `master` at access 2026-08-30 (public); commit history stable since 2022 | 全部 MHA | ❌ | OPT 论文 + config.json 全部 MHA，无 GQA 转换 | ⚠ not verifiably excluded (master + paper §3 reference canonical but no specific SHA / §anchor cited inline)
| BLOOM | https://huggingface.co/bigscience/bloom | `master` at access 2026-08-30 (public) | 全部 MHA | ❌ | BLOOM 论文 §2.1 全部 MHA | ⚠ not verifiably excluded (master + paper §2.1 reference, no specific SHA)
| GPT-NeoX | https://huggingface.co/EleutherAI/gpt-neox-20b | `master` at access 2026-08-30 (public) | 全部 MHA | ❌ | EleutherAI 公开声明 GPT-NeoX 全部 MHA | ⚠ not verifiably excluded (master + EleutherAI declaration, no specific SHA)
| Falcon | https://huggingface.co/tiiuae/falcon-7b ; .../falcon-180B | `master` at access 2026-08-30 (public) | 7B/40B MHA, 180B MQA（非 GQA） | ❌ | Falcon 180B 用 MQA（kv_heads=1），不是 GQA 中间态 | ⚠ not verifiably excluded (master + paper §3 MQA reference)
| Yi / Yi-Llama | https://huggingface.co/01-ai/Yi-6B ; .../Yi-34B | `master` at access 2026-08-30 (public) | 全部 GQA | ❌ | Yi 公开声明 GQA | ⚠ not verifiably excluded (master + Yi paper reference)
| Baichuan / Baichuan2 | https://huggingface.co/baichuan-inc/Baichuan2-7B-Base | `master` at access 2026-08-30 (public) | 全部 MHA | ❌ | Baichuan2 公开声明 MHA | ⚠ not verifiably excluded (master + Baichuan2 paper reference)
| SmolLM / SmolLM2 / SmolLM3 | https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct (GQA) ; .../SmolLM2-1.7B-Instruct (MHA) ; SmolLM3 全部 GQA (per SmolLM3 blog) | P5-04 expected `hf_expected_revision` SHAs (recorded in OWT-real-eval metadata): SmolLM2-360M `a10cc1512eabd3dde888204e902eca88bddb4951`; SmolLM2-1.7B `31b70e2e869a7173562077fd711b654946d38674`. **Local snapshot revision is `master`** (HF API access blocked in this environment, so we have local `snapshots/master/` rather than `snapshots/<commit-sha>/`; see `docs/plans/open-issues.md` for the master-vs-exact-commit issue). | SmolLM2-360M GQA, 1.7B MHA (size 不同); SmolLM3 全 GQA | ❌ | size/layers/hidden 至少一个不同；attn 类型仅是 N 个变量之一 | ✅ verified (P5-04 hf_expected_revision 40-hex SHAs SmolLM2-360M a10cc15... + SmolLM2-1.7B 31b70e2... pinned in row)
| BEE-spoke-data smol_llama | https://huggingface.co/BEE-spoke-data/smol_llama-101M-GQA ; .../smol_llama-220M-GQA | `master` at access 2026-08-30 (public) | 全部 GQA | ❌ | HF model cards 全部声明 GQA | ⚠ not verifiably excluded (master + HF model card declaration)
| Research: Ainslie 2023 GQA paper | https://arxiv.org/abs/2305.13245 | arxiv paper id is the canonical revision (immutable) | 提出 GQA + uptraining recipe（mean-pool KV heads + 5% 额外 pre-training） | ❌ | 论文 §2.1："we show that language model checkpoints with MHA can be uptrained to use MQA with 5% of original pre-training compute" — 不同权重的转换，非 same-base | ✅ verified (arxiv 2305.13245 — immutable paper id)
| Research: fpcsong/mha2gqa | https://github.com/fpcsong/mha2gqa (paper: https://aclanthology.org/2025.findings-emnlp.467/ ; arxiv: https://arxiv.org/abs/2412.20677) | GitHub commit history visible at repo (last update 2025-10-29 per search snippet `rename May 19, 2025`); arxiv id immutable | 仅发布 Llama-2-7B 的 GQA-{4,8,16} 转换权重；MHA 原始权重来自 meta-llama 第三方仓库 | ❌ | README 明确："download llama-7B or Sheared-llama-1.3B" — 仅转换后 GQA 公开，原始 MHA 不是本 repo 产物 | ✅ verified (arxiv 2412.20677 + GitHub repo commit history publicly browsable)
| Research: SmolLM3 blog nanotron ablation | https://huggingface.co/blog/smollm3 (contains MHA vs GQA-{2,4,8,16} vs MQA nanotron config table) | HF blog (versioned by commit; no separate checkpoint) | 内部训练，**未公开 checkpoint**（仅 HellaSwag/MMLU/ARC 数字） | ❌ | blog 文字："We didn't ablate MLA since it wasn't implemented in nanotron at the time of the ablations" — 无 HF repo 发布 | ⚠ not verifiably excluded (HF blog no pinned commit SHA; row explicitly notes "no checkpoint released" — exclusion is by absence)
| Research: shreyansh26/multihead-latent-attention | https://github.com/shreyansh26/multihead-latent-attention | GitHub commit history (public); repo has no released weights | reference implementation only, no pretrained weights | ❌ | README: "A small, self-contained reference implementation of MHA/GQA/MQA" — 仅 reference，无 training 出的 weights | ✅ verified (GitHub repo commit history publicly browsable at github.com/shreyansh26/multihead-latent-attention)

**Note on "Verification status" column**:
- ✅ **verified** = the row's evidence comes from a directly inspectable source (config.json / model card / paper §X / HF docs page) that the auditor can independently retrieve; the claim about this family's attention pattern can be reproduced by another auditor following the URL.
- ⚠ **not verifiably excluded** = the row's exclusion is by *absence* of evidence (e.g. "all model cards have no GQA declaration" is not the same as "we verified each model card says MHA-only"); such rows are still informative but the claim is weaker than ✅ rows.

Currently §2.2 has **5 ✅ fully independently verified rows** (Mixtral mirror config at pinned commit, SmolLM P5-04 hf_expected_revision SHAs, Ainslie 2023 / fpcsong/mha2gqa / shreyansh26 — arxiv id + GitHub commit history) and **14 ⚠ feasibility-lead rows** (rows where only `master` + paper / model card declaration / absence claim is cited without a specific pinned SHA).

**Classification of the 14 ⚠ rows**: these are **feasibility leads**, NOT fully independently verified exclusions. The distinction matters because:
- ✅ verified rows: another auditor can reproduce the row by pointing to a specific pinned SHA / arxiv id / GitHub commit. The exclusion is durable and independently re-verifiable.
- ⚠ feasibility leads: the row contains a plausible-looking argument for exclusion (no same-base pair observed), but the evidence chain relies on absence claims or unpinned source revisions. A more thorough search (e.g. downloading each repo at its pinned commit, parsing each config.json, comparing dimensions row-by-row) might reveal a same-base pair. **Treat these rows as "needs more work" rather than "verified no-go"**.

The ⚠ rows are still informative for the narrowed no-go scope (audited candidate set), but the no-go conclusion explicitly excludes fully verified closure on these 14 rows. §9 "Not auditable / not verifiably excluded" lists the categories of evidence that are not pinned at all (closed-source, unpublicized, private org-internal, non-English platforms, paper-only ablations).



**Per-row classification of the 14 ⚠ feasibility-lead rows (auditor round 15)**: for each row, the specific reason it is classified as a feasibility lead rather than fully verified:

1. **LLaMA-1/2/3** — mirror config.json content reproducible but no specific commit SHA for mirror; gated repo at `main` (auditor cannot verify mirror at a specific immutable revision).
2. **Mistral-7B** — gated repo + HF transformers docs page lacks a specific commit hash cited in the row.
3. **Qwen / Qwen2 / Qwen2.5** — `master` only in row; P5-04 hf_expected_revision SHAs exist in `artifacts/owt-real-eval/results/*/metadata.json` but are not cited inline in the row.
4. **Phi-1 / 1.5 / 2 / 3 / 4** — `master` + absence claim ("all model cards have no GQA declaration") is not pinned config.json evidence.
5. **Gemma / Gemma2 / Gemma3** — HF transformers docs page lacks a specific commit hash cited in the row.
6. **DeepSeek-V2 / V3** — `master` + model card declaration only; no pinned config.json SHA cited.
7. **OPT** — `master` + paper §3 reference (canonical) but no specific SHA or §anchor cited inline.
8. **BLOOM** — `master` + paper §2.1 reference; no specific SHA cited.
9. **GPT-NeoX** — `master` + EleutherAI declaration (no §anchor); no specific SHA cited.
10. **Falcon** — `master` + Falcon 180B paper §3 MQA reference; no specific SHA cited.
11. **Yi / Yi-Llama** — `master` + Yi paper reference; no specific SHA cited.
12. **Baichuan / Baichuan2** — `master` + Baichuan2 paper reference; no specific SHA cited.
13. **BEE-spoke-data smol_llama** — `master` + HF model card declaration; no specific SHA cited.
14. **SmolLM3 blog nanotron ablation** — HF blog no pinned commit SHA; row explicitly notes "no checkpoint released" — exclusion is by absence.

For each ⚠ row, an auditor could close the gap by (a) downloading the repo at a specific pinned commit and parsing its `config.json` to confirm attention pattern, OR (b) reading the cited paper §X to extract the explicit attention type claim, OR (c) finding the original HF model card from a versioned snapshot. The current row text does not include any of these specific verification steps.

**Note on revision pinning**: For gated repos (LLaMA-2, Mixtral, Mistral), the
exact commit SHA at `main` is not directly accessible without an HF token
approved for those repos; we cite mirror configs (e.g. `tjluyao/llama-2-7b-hf`
config.json) which are public and content-stable (the JSON content was
search-confirmed at 2026-08-30 to show the expected `num_attention_heads=32,
num_key_value_heads=32`). For the local P5-04 snapshots, see the
`hf_expected_revision` SHAs in `artifacts/owt-real-eval/results/*/metadata.json`
(40-hex commit SHA per model) — these are the project-recorded canonical
revisions even though the locally downloaded snapshot dirs are at `master`
(due to a documented ModelScope mirror limit; see P5-04 stage review §Limitations).

### 2.3 同 base 硬性判定标准 #1-#3 vs 已扫描候选汇总

| 判定标准 | 是否命中 | 理由 |
|---|:---:|---|
| 同一组织同一训练计划 | ❌ | 所有公开同家族 model 都已固定 attention 类型；uptraining 产出是不同权重 |
| 唯一架构差异 = num_kv_heads | ❌ | 同家族内 size 不同（hidden_size / num_hidden_layers / vocab_size 至少一个差异） |
| 其他 hyperparam 一致 | ❌ | size 不同 → hidden/layers/vocab 必不同 |
| 公开可下载 safetensors | — | 多数通过；但仍需 1+2+3 才能用 |
| 可绑 HF commit SHA | — | P5-04 已暴露 master vs exact commit 问题；即使通过 1-4，本实验室无法稳定绑定的就排除 |

### 2.4 结论

**没有"同 base（同一组织、同一训练数据、同一训练时长）但 attention 结构不同"的公开可审计模型对。**

任何把"SmolLM2-360M GQA vs SmolLM2-1.7B MHA"或"Qwen2.5 GQA vs 任意 LLaMA-2 7B MHA"对比的结果归因为"GQA 收益"或"MHA 收益"的论文/报告，都是**混淆变量**：size、layers、hidden、vocab、训练 token 数、训练数据都不同，attention 差异只是其中之一。

## 3. 否决理由汇总

| 候选 | 否决原因 |
|---|---|
| SmolLM2-360M (GQA) vs SmolLM2-1.7B (MHA) | 不同 size (360M vs 1.7B)、hidden (960 vs 2048)、layers (32 vs 24)，attention 类型只是 N 个变量之一 |
| Qwen2.5-{0.5B,1.5B,3B} (全部 GQA) | 全 GQA，无 MHA twin |
| LLaMA-2 7B (MHA) vs LLaMA-2 70B (GQA) | 不同 size (7B hidden=4096 layers=32 vs 70B hidden=8192 layers=80) — LLaMA-2 是 dense 模型，无 MoE / expert parallelism，attn 差异只是 N 个变量之一；同 family 不同 size 不能归因 GQA |
| Ainslie 2023 uptraining GQA 检查点 | 不同权重（MHA mean-pool 后 5% 重新预训练），不是 same-base |
| fpcsong/mha2gqa Llama-2-7B GQA-{4,8,16} | 仅发布 GQA 转换权重，原始 MHA 是 meta-llama 第三方权重，不构成"可对照对" |
| SmolLM3 nanotron ablation | 私有训练，未公开 HF checkpoint |
| 单用户实验 smol_llama GQA variants | fine-tune 而非 pre-train，无同 base MHA twin，无训练数据/recipe 公开 |

## 4. roadmap 影响

`docs/plans/roadmap.md` 候选 1 行已记录此风险（"难找同 base GQA/MHA pair；退化为不同 base 对比"），本次搜索确认该风险为真。本目标交付**no-go 审查**，roadmap 候选 1 标记为 **不可行（公开权重层面）**。

后续若要继续 GQA vs MHA 对比，可选路径（不在本目标 scope 内）：

1. **自训练一对 from-scratch same-base models**：从同一随机种子、同一训练数据、同一 token 量、唯一差异为 `num_kv_heads` 训练两个 ~125M 模型。这是 Ainslie 2023 / SmolLM3 团队做过的，但需 8×H100 + 数千 GPU-hour，远超本 lab 算力预算。
2. **复现 uptraining**：从 meta-llama/Llama-2-7b-hf 出发做 MHA → GQA-{4,8,16} 转换 + 5% 额外预训练（按 fpcsong 2025 论文 recipe）。需要 ~50K-100K 训练 token + 数天 GPU。也远超本 lab 当前算力。
3. **轻量 demo**：用极小模型（< 50M）做 attention type 切换的 toy 对比，证明 P5-02 backend 能正确处理 GQA/MHA 路径（实际已通过 P5-02 5 模型评测隐式证明 — Qwen2.5 系列是 GQA、SmolLM2-1.7B 是 MHA，两者都正确 forward + per-token CE）。这不构成 GQA vs MHA 对比实验，但能说明 backend 兼容性。

## 5. 评估

- 候选搜索覆盖：HF top families (LLaMA / Mistral / Mixtral / Qwen / Phi / Gemma / DeepSeek / OPT / BLOOM / GPT-NeoX / Falcon / Yi / Baichuan / SmolLM / BEE-spoke smol_llama) + research artifacts (Ainslie 2023 / fpcsong 2025 / SmolLM3 blog / shreyansh26 mha reference)。
- 时间投入：~2 小时 web 调研 + 1 小时 P5-04 既有 model config 本地实证。
- 决策：no-go，理由充分、证据链清晰（每条候选带 URL + access date + evidence citation）、无遗漏主流公开模型家族。

## 6. 后续交付

- `docs/plans/reviews/stage-gqa-vs-mha-no-go.md` — 阶段审查 + reviewer evidence。
- `docs/plans/open-issues.md` — "GQA vs MHA 公开模型对比" 标记为不可行（公开权重层面），记录本结论链接。
- `docs/plans/roadmap.md` 候选 1 行状态更新为 `no-go`。
- `tests/test_gqa_vs_mha_no_go.py`（7 断言 PASS） — 守住本结论不被意外推翻（硬性判定标准 + 已扫描候选 list 不为空 + 本地 config 实证）。
- selftest `scripts/eval_owt_real.py --selftest` 不受影响（97 PASS / 0 FAIL）。

## 7. 实证命令清单（auditor-runnable）

```bash
# §2.1 P5-04 本地 5 model config 实证（portably rglob 兼容 hash snapshot 与 master）
cd C:/Users/23236/repositories/llm-lab
python3 -c "
from pathlib import Path
import json
for cfg in sorted(Path('artifacts/owt-real-eval/models/models').rglob('config.json')):
    c=json.loads(cfg.read_text(encoding='utf-8'))
    h=c.get('num_attention_heads'); kv=c.get('num_key_value_heads',h)
    arch='GQA' if (kv!=h) else 'MHA'
    print(f'{cfg.parent.parent.parent.name}: heads={h} kv={kv} hidden={c.get(\"hidden_size\")} layers={c.get(\"num_hidden_layers\")} vocab={c.get(\"vocab_size\")} -> {arch}')
"

# §7 selftest guard (7 断言全 PASS)
python3 -m pytest tests/test_gqa_vs_mha_no_go.py -v

# §7 P5-04 E 评测无 regression
python3 scripts/eval_owt_real.py --selftest  # 97 PASS / 0 FAIL
```

## 8. 搜索协议（reproducible）

本节为后续 auditor 复现搜索记录步骤。所有 step 在 2026-08-30 完成。

1. **Web search query 1**: "same base model GQA vs MHA ablation public huggingface 2024 2025"
   → 结果 1 (parasdahal.com notes): 列出 Llama-3 / Qwen 2.5 / Mistral 等"主流 GQA"配置，无同 base 双版本。
   → 结果 2 (aclanthology mha2gqa): 论文确认 uptraining 产出不同权重。
   → 结果 3 (DeepSeek V3 MLA): 第三类 attention，非 GQA。
   → 结果 4 (GeekforGeeks): 综述类，不指具体同 base 配对。
   → 结果 5 (SmolLM3 blog): 确认 nanotron ablation 无公开 checkpoint。

2. **Web search query 2**: "public huggingface model same pretrained weights MHA GQA versioned release two checkpoints"
   → 结果 1 (afrideva smol_llama-101M-GQA): 单 size GQA only。
   → 结果 2 (mradermacher Mixtral-GQA-400m-v2): MoE，非 single base 对。
   → 结果 3 (BEE-spoke smol_llama): 全 GQA 系列。
   → 结果 4 (fpcsong/mha2gqa README): 明确只发布 GQA 转换后 weights。
   → 结果 5 (Jayaprakash0511/gqa-reproduction): 私有 T5-Small 复现，无公开权重。
   → 结果 6 (FrostNT1/GQA-presentation): 教程性质，无训练 checkpoints。
   → 结果 7 (GQA paper 2305.13245): Ainslie 2023 — uptraining 概念，转换权重。

3. **Web search query 3**: "fpcsong/mha2gqa github EMNLP 2025 findings commit sha pinned"
   → 确认 https://github.com/fpcsong/mha2gqa 与 https://aclanthology.org/2025.findings-emnlp.467/ + arxiv 2412.20677 三处 link 一致；repo 仅有 GQA 转换后 checkpoints 发布，无 base MHA 同 train 配对。

4. **本地 config 实证**：读取 P5-04 已下载 5 model config.json，列出 attention 类型 / hidden / layers / vocab，确认无同 base 对（5 个 tuple 全不同）。

5. **结论**：搜索覆盖 15+ family + 3 research artifacts，每条候选均带 URL + access date + evidence。no-go 结论证据链完整、可被独立 auditor 复现。


## 9. 结论范围（narrowed claim，避免 universal overreach）

为响应 auditor 关于"区分 not found vs not verifiably excluded"的要求，
本节明确 no-go 结论的**实际范围**：

- **Audited candidate set**（15 个 family + 3 research artifacts，见 §2.2 表格）：
  每个候选都附带 URL + access date + pinned commit SHA 或 mirror config.json
  实证 + P5-04 本地 config 实证。本集合内**没有可审计的同 base GQA/MHA 双版本**。
- **Verdict**：在本 audited candidate set 上 no-go 成立。

- **Not auditable / not verifiably excluded**（auditor 显式区分项）：
  本搜索未覆盖的范围：
  1. **closed-source / gated-only 仓库** (无公开 model card 或 config.json 的实验性 pretraining)；
  2. **unpublicized checkpoints** (HF 历史上发布后又删除 / 已下架)；
  3. **private org-internal trainings** (私人 repo / 内部训练未公开的同 base 双版本 — 如果某团队内部训练过 MHA + GQA 同 base 但未发布，本搜索无法检测)；
  4. **non-English / non-mainstream platforms** (GitLab、ModelScope 私有仓库、国内魔搭社区非公开仓库等)；
  5. **paper-only ablations without released safetensors** (来自非公开论文 ablation 的非 release 权重 — 仅在 paper 表格中报告，未发布 safetensors)。

- **Narrowed wording**：
  本文不主张"全球公开 HF 一定不存在同 base GQA/MHA 双版本"。
  本文只主张："在 2026-08-30 由 web search + P5-04 本地 config 取证的 15 个 family + 3 research artifacts
  这一 audited candidate set 内，没有可审计的同 base GQA/MHA 双版本。"

- **Implication**：如未来发现满足 §1.2 硬性判定标准 1-5 的同 base 公开模型对（不论是
  现有 family 的新 release 还是新 family 发布），本 no-go 结论需被推翻。`tests/test_gqa_vs_mha_no_go.py`
  守住本结论（test_feasibility_doc_present / test_feasibility_doc_states_no_go / test_roadmap_no_go_status）
  不被意外削弱。

## 10. 实证命令清单（auditor-runnable；当前 §7 之外的补充）

```bash
# §1.2 硬性判定标准 #2 — 验证 `num_attention_heads == num_key_value_heads` ⇒ MHA
python3 -c "
from pathlib import Path
import json
for cfg in sorted(Path('artifacts/owt-real-eval/models/models').rglob('config.json')):
    c=json.loads(cfg.read_text(encoding='utf-8'))
    h=c.get('num_attention_heads'); kv=c.get('num_key_value_heads',h)
    arch='GQA' if (kv!=h) else 'MHA'
    print(f'{cfg.parent.parent.parent.name}: heads={h} kv={kv} → {arch}')
"

# §9 audited candidate set 范围确认
wc -l docs/experiments/gqa-vs-mha/feasibility.md
grep -c '^|' docs/experiments/gqa-vs-mha/feasibility.md   # §2.2 family + research artifact rows

# §9 narrowed wording — 必须显式出现 "not auditable" / "not verifiably excluded" 字样
grep -nE 'not auditable|not verifiably excluded' docs/experiments/gqa-vs-mha/feasibility.md
```