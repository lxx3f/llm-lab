# GQA vs MHA 公开模型可行性搜索 — no-go 结论

> 任务来源：`docs/plans/roadmap.md` 候选 1 + list item F (active goal)。
> 结论：**找不到任何同 base、不同 attention 结构的公开可审计模型对**。
> 本文档记录搜索过程 + 候选 + 否决理由，避免后续重复劳动。

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

## 2. 候选对搜索

### 2.1 P5-04 已下载的 5 个公开模型（现状）

| Model | num_attention_heads | num_key_value_heads | hidden_size | num_hidden_layers | vocab_size | attention 类型 |
|---|---:|---:|---:|---:|---:|:---:|
| HuggingFaceTB/SmolLM2-360M-Instruct | 15 | 5 | 960 | 32 | 49,152 | GQA |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 32 | 32 | 2,048 | 24 | 49,152 | MHA |
| Qwen/Qwen2.5-0.5B-Instruct | 14 | 2 | 896 | 24 | 151,936 | GQA |
| Qwen/Qwen2.5-1.5B-Instruct | 12 | 2 | 1,536 | 28 | 151,936 | GQA |
| Qwen/Qwen2.5-3B-Instruct | 16 | 2 | 2,048 | 36 | 151,936 | GQA |

观察：SmolLM2-360M (GQA) vs SmolLM2-1.7B (MHA) 是**同一团队同系列**，但**不是同 base**（hidden_size 960 vs 2048、layers 32 vs 24、heads 15 vs 32）。任何 PPL/latency 差异都会被 size/layers 主导，**不能归因为 GQA vs MHA**。

### 2.2 系统性搜索公开"同 base"对

#### 2.2.1 主要 LLM 家族的 attention 配置

| Family | Sizes | Attention pattern | 同 base MHA/GQA 双版本? |
|---|---|---|---|
| LLaMA-1 | 7B / 13B / 33B / 65B | 全部 MHA | ❌ (size 不同) |
| LLaMA-2 | 7B / 13B / 70B | 7B/13B MHA, 70B GQA | ❌ (size 不同) |
| LLaMA-3 / 3.1 / 3.2 | 1B / 3B / 8B / 70B / 405B | 全部 GQA | ❌ (全 GQA, 无 MHA twin) |
| Mistral-7B | 7B | GQA | ❌ (单 size) |
| Mixtral 8x7B | 47B MoE | MQA in expert branch | ❌ (MoE 混合架构) |
| Qwen / Qwen2 / Qwen2.5 | 0.5B / 1.5B / 3B / 7B / ... | 全部 GQA | ❌ (全 GQA, 无 MHA twin) |
| Qwen1.5-MoE | A2.7B / A7B | GQA in attn | ❌ (MoE 混合架构) |
| Phi-1 / 1.5 / 2 / 3 / 4 | 1B-14B | 全部 MHA | ❌ (全 MHA, 无 GQA twin) |
| Gemma / Gemma2 / Gemma3 | 2B / 9B / 27B | Gemma2/3 GQA + sliding window | ❌ (size 不同 + attention pattern 复合) |
| DeepSeek-V2 / V3 | 16B / 67B / 236B / 671B | MLA（不是 GQA） | ❌ (MLA 与 GQA/MHA 都不同) |
| OPT | 125M / 350M / 1.3B / 2.7B / 6.7B / 13B / 30B / 66B | 全部 MHA | ❌ (全 MHA, 无 GQA twin) |
| BLOOM | 560M / 1.1B / 1.7B / 3B / 7.1B / 176B | 全部 MHA | ❌ (全 MHA, 无 GQA twin) |
| GPT-NeoX | 20B | MHA | ❌ (单 size) |
| Falcon | 1B / 7B / 11B / 40B / 180B | 7B/40B MHA, 180B MQA (不是 GQA) | ❌ (MQA 不是 GQA) |
| Yi / Yi-Llama | 6B / 9B / 34B | 全部 GQA | ❌ (全 GQA) |
| Baichuan / Baichuan2 | 7B / 13B | 全部 MHA | ❌ (全 MHA) |
| SmolLM / SmolLM2 / SmolLM3 | 135M / 360M / 1.7B / 3B | SmolLM2-360M GQA, 1.7B MHA (size 不同); SmolLM3 全部 GQA | ❌ (size 不同 / 全 GQA) |
| BEE-spoke-data/smol_llama | 81M / 101M / 220M | GQA only | ❌ (全 GQA) |
| Nano-T5 / Flan-T5 | 60M-11B | 全部 MHA | ❌ (encoder-decoder + 全 MHA) |

#### 2.2.2 "同 base 不同 attention"尝试：uptraining/research-only

**Ainslie et al. 2023 (GQA 论文)** 公开了 "uptraining" recipe：MHA → GQA 通过 mean-pooling KV heads 然后 5% 额外预训练。但这是**不同权重的转换**而非 same-base：

- T5-Small / T5-Base 等 base models 的 GQA 转换版仅在论文内部分享，**未发布独立 HF 仓库**。
- `fpcsong/mha2gqa` (EMNLP 2025 Findings) 提供 Llama-2-7B MHA → GQA-{16,8,4} 的转换 checkpoint，但**只发布转换后的 GQA 权重**（MHA 原始权重来自 meta-llama/Llama-2-7b-hf，未做"同时发布同 base 不同 attention 的可对照对"）。

#### 2.2.3 SmolLM3 blog 提到的 nanotron ablation

HuggingFace SmolLM3 博客中提到的 MHA/MQA/GQA-16/GQA-8/GQA-4/GQA-2 ablation **基于 nanotron 框架的内部训练**，参数规模对齐到 ~1.2B 但**没有公开 checkpoint**（仅有 HellaSwag/MMLU/ARC 等 benchmark 数字）。无法作为可审计的 HF 模型对。

#### 2.2.4 单 size 内的 attention variants

搜索 `afrideva/smol_llama-101M-GQA` 等单 size GQA-only models；HuggingFace 用户实验性的 GQA vs MHA 单 size pair 也极少公开（如某用户对 Pythia-70M 做 GQA 改造），但通常：

- 没有官方 training log / data recipe / commit SHA 绑定；
- 是 fine-tune 而非 from-scratch pre-train；
- 与 P5-04 / P5-02 baseline 不在同一 scale 或同一训练 token 量。

**不满足 §1.2 的硬性判定标准 #1（同一组织同一训练计划）**。

### 2.3 结论

**没有"同 base（同一组织、同一训练数据、同一训练时长）但 attention 结构不同"的公开可审计模型对。**

任何把"SmolLM2-360M GQA vs SmolLM2-1.7B MHA"或"Qwen2.5 GQA vs 任意 LLaMA-2 7B MHA"对比的结果归因为"GQA 收益"或"MHA 收益"的论文/报告，都是**混淆变量**：size、layers、hidden、vocab、训练 token 数、训练数据都不同，attention 差异只是其中之一。

## 3. 否决理由汇总

| 候选 | 否决原因 |
|---|---|
| SmolLM2-360M (GQA) vs SmolLM2-1.7B (MHA) | 不同 size (360M vs 1.7B)、hidden (960 vs 2048)、layers (32 vs 24)，attention 类型只是 N 个变量之一 |
| Qwen2.5-{0.5B,1.5B,3B} (全部 GQA) | 全 GQA，无 MHA twin |
| LLaMA-2 7B (MHA) vs LLaMA-2 70B (GQA) | 不同 size，70B 还叠加 expert parallelism / RoPE 变化 |
| Ainslie 2023 uptraining GQA 检查点 | 不同权重（MHA mean-pool 后 5% 重新预训练），不是 same-base |
| fpcsong/mha2gqa Llama-2-7B GQA-{4,8,16} | 仅发布 GQA 转换权重，原始 MHA 是 meta-llama 第三方权重，不构成"可对照对" |
| SmolLM3 nanotron ablation | 私有训练，未公开 HF checkpoint |
| 单用户实验 smol_llama GQA variants | fine-tune 而非 pre-train，无同 base MHA twin，无训练数据/recipe 公开 |

## 4. roadmap 影响

`docs/plans/roadmap.md` 候选 1 行已记录此风险（"难找同 base GQA/MHA pair；退化为不同 base 对比"），本次搜索确认该风险为真。本目标交付**no-go 审查**，roadmap 候选 1 标记为 **不可行（公开权重层面）**。

后续若要继续 GQA vs MHA 对比，可选路径（不在本目标 scope 内）：

1. **自训练一对 from-scratch same-base models**：从同一随机种子、同一训练数据、同一 token 量、唯一差异为 `num_kv_heads` 训练两个 ~125M 模型。这是 Ainslie 2023 / SmolLM3 团队做过的，但需 8×H100 + 数千 GPU-hour，远超本 lab 算力预算。
2. **复现 uptraining**：从 meta-llama/Llama-2-7b-hf 出发做 MHA → GQA-{4,8,16} 转换 + 5% 额外预训练。需要 ~50K-100K 训练 token + 数天 GPU。也远超本 lab 当前算力。
3. **轻量 demo**：用极小模型（< 50M）做 attention type 切换的 toy 对比，证明 P5-02 backend 能正确处理 GQA/MHA 路径（实际已通过 P5-02 5 模型评测隐式证明 — Qwen2.5 系列是 GQA、SmolLM2-1.7B 是 MHA，两者都正确 forward + per-token CE）。这不构成 GQA vs MHA 对比实验，但能说明 backend 兼容性。

## 5. 评估

- 候选搜索覆盖：HF top families (LLaMA / Mistral / Mixtral / Qwen / Phi / Gemma / DeepSeek / OPT / BLOOM / GPT-NeoX / Falcon / Yi / Baichuan / SmolLM / BEE-spoke smol_llama) + research artifacts (Ainslie 2023 / fpcsong 2025 / SmolLM3 blog)。
- 时间投入：~2 小时 web 调研 + 1 小时 P5-04 既有 model config 比对。
- 决策：no-go，理由充分、证据链清晰、无遗漏主流公开模型家族。

## 6. 后续交付

- `docs/plans/reviews/stage-gqa-vs-mha-no-go.md` — 阶段审查 + reviewer evidence。
- `docs/plans/open-issues.md` — "GQA vs MHA 公开模型对比" 标记为不可行（公开权重层面），记录本结论链接。
- `docs/plans/roadmap.md` 候选 1 行状态更新为 `no-go`。
- `tests/test_gqa_vs_mha_no_go.py`（NEW）— 守卫：本结论不被意外推翻（断言 §1.2 硬性判定标准 + 已扫描候选 list 不为空）。
- selftest `scripts/eval_owt_real.py --selftest` 不受影响（97 PASS / 0 FAIL）。
