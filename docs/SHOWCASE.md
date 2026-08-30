# LLM Lab SHOWCASE — 简历项目摘要

> **目的**：把仓库里分散在 27 个 `docs/experiments/` 下的实验 + 421 个测试 + 11 个 JSON Schema 浓缩成可直接用于简历、面试演示和 GitHub 主页 README 的素材。详细的方法、protocol 和数字见各 `docs/experiments/<name>/` 子目录。

> **TL;DR**：自研实现 Dense / MoE / GQA / MLA 4 种注意力路径，从零训练多个 0.66M-12M 模型在真实 OpenWebText 上；在 5 个公开 instruction-tuned 模型（360M-3B）上完成真实 OWT held-out 评测 + Transformers/vLLM 双后端对比；数据管线 + 工具调用 8 级分类器 + SFT + GRPO 全闭环。

---

## 1. 数字一览（5 行概览）

| # | 实验 | 关键 metric | 数字 | 详细 |
|---|---|---|---|---|
| 1 | **N13** 自研 GQA + MLA 与 Dense MHA 5000 步 OWT 对比 | val_min | Dense 7.058 / GQA 7.106 / MLA 7.122 (差距 ≤0.07 nats) | [`docs/experiments/n13-gqa-vs-mla-vs-mha/`](experiments/n13-gqa-vs-mla-vs-mha/) |
| 2 | **E** 5 个公开 instruction-tuned 模型真实 OWT 评测（277MB held-out） | mean_loss_nats / PPL | SmolLM2-1.7B **PPL 11.05** 最优 / Qwen2.5-3B PPL 13.04 | [`docs/experiments/owt-real-eval/`](experiments/owt-real-eval/) |
| 3 | **P5-04** Transformers vs vLLM 双后端 × 5 模型 × 2 batch × 90 样本 | 4 轴 (latency / throughput / reward_binary / reward_layered) | 20 组合 0 generation failure；Transformers 在 360M-1.5B 段更快 | [`docs/experiments/p5-04-backend-comparison/`](experiments/p5-04-backend-comparison/) |
| 4 | **MoE Top-1** 5000 步 OWT 训练曲线（2.10M total / 1.51M active） | train_loss / val_min | 117.90 → 6.79；val_min **7.22 @ step 4600** | [`docs/experiments/moe-owt-formal-curve/`](experiments/moe-owt-formal-curve/) |
| 5 | **N2** Dense vs MoE 公平对比协议（同 total params / 同 active params） | benchmark schema + aux_loss | 4 配置 smoke PASS；protocol 双维度严格区分 active vs total | [`docs/experiments/n2-dense-moe-fairness/`](experiments/n2-dense-moe-fairness/) |
| 6 | **N4-N9** 5 个 Dense ablation sweep（scale / dropout / rope_base / n_heads / d_ff） | val_min overlay | 5 张 overlay PNG，每 sweep 3-4 点 | [`docs/experiments/n4-dense-formal-curve/`](experiments/n4-dense-formal-curve/) + N5/N6/N7/N8/N9 |

---

## 2. 推荐简历 bullet（4 条，按对面试官的吸引力排序）

### Bullet A — 真实跨模型 LM 评测方法学（最有技术深度）

> **在 5 个公开 instruction-tuned 模型（SmolLM2 360M/1.7B、Qwen2.5 0.5B/1.5B/3B）上完成 277MB held-out OpenWebText 真实评测，建立跨 tokenizer 公平的 PPL 指标体系（per-token + per-byte 双维度），识别了 SmolLM2-1.7B PPL 11.05 最优 / Qwen2.5-3B PPL 13.04 的型号差异，并明确指出不同 vocab size 下 perplexity 不可直接对比的方法学陷阱。**

### Bullet B — 自研 3 种 attention 路径 + 三方对比（最有架构深度）

> **基于 PyTorch 从零实现 Dense MHA / GQA (`num_kv_heads` 可配置，KV `repeat_interleave` 共享) / 简化 MLA (K/V 联合压缩到 `latent_dim`，KV cache 存压缩 latent 替代 full K/V) 三种 attention 路径；与 N4 Dense MHA baseline 在完全相同 OWT 正式 cache + 同 seed=42 + 同 5000 步 + 同 bf16 + 同 2.10M 规模下做三方对比：val_min 7.058 / 7.106 / 7.122（差距 ≤0.07 nats 在小规模属训练噪声），GQA 与 MLA 的 KV cache 仅为 Dense MHA 的 1/4（256B vs 1024B / 单 batch 单 token float32 per layer）。**

### Bullet C — 推理后端工程对比（最有工程实用性）

> **在 5 模型 × 2 后端 (Transformers vs vLLM) × 2 batch size = 20 组合 × 90 样本固定 benchmark 上做 4 轴（latency / throughput / reward_binary / reward_layered）对比；commit-level binding + manifest-driven 加载 + section-aware README 同步守护（含 4 个 NEGATIVE test）保证 comparison.csv 与 README 数字严格一致；实测 0 个 generation failure。**

### Bullet D — MoE + Dense 训练实证（最有全栈完整性）

> **实现 MoE Top-1 (Router + Load Balancing Loss + Capacity Factor + Dropped Token) 与 Dense Transformer，在 OWT 正式 cache (512 MiB train / 64 MiB validation) 上完成 5000 步训练，val_min 7.22 @ step 4600（2.10M total / 1.51M top-1 active）；同步实现 N2 公平对比协议（同 total params / 同 active params 双协议）保证架构对比不被参数总量差异污染。**

---

## 3. Plot Gallery（5 张关键图）

| 实验 | 图 | 内容 |
|---|---|---|
| **N13** | [`docs/experiments/n13-gqa-vs-mla-vs-mha/dense-vs-gqa-vs-mla-curves.png`](experiments/n13-gqa-vs-mla-vs-mha/dense-vs-gqa-vs-mla-curves.png) | Dense MHA / GQA / MLA 5000 步 train_loss log-scale + val_loss linear 双子图 3-way overlay |
| **N4** | [`artifacts/dense-owt-formal-curve-overlay.png`](../artifacts/dense-owt-formal-curve-overlay.png) | 3 个 Dense 模型规模（small / medium / large）训练曲线对比 |
| **N2 + MoE OWT** | [`artifacts/moe-vs-dense-medium-curve.png`](../artifacts/moe-vs-dense-medium-curve.png) | MoE Top-1 vs Dense medium 同规模训练曲线对比 |
| **N8** | [`artifacts/dense-owt-formal-curve-medium-heads-sweep.png`](../artifacts/dense-owt-formal-curve-medium-heads-sweep.png) | Dense medium `n_heads` 2/4/8 ablation |
| **N12** | [`artifacts/dense-owt-formal-curve-ultra-vs-long.png`](../artifacts/dense-owt-formal-curve-ultra-vs-long.png) | Dense 长训练曲线（baseline + medium × 50000 步） |

---

## 4. 项目完整性指标

| 维度 | 数字 |
|---|---|
| Git commits | **290+** |
| Python 源文件 | **104** |
| Pytest 收集 | **421 tests**（运行 `python -m pytest`，最近一次 46/46 PASS） |
| 实验 README | **27**（每个 stage 一个 doc） |
| Stage reviews | **38**（`docs/plans/reviews/`） |
| JSON Schemas | **11**（含 tool calling / model output / eval result / dense training / MoE / N2 benchmark / N2 routing / reward signal / GRPO step / eval result 等） |
| 自研模型架构 | 4（Dense MHA / MoE Top-1 / GQA / 简化 MLA） |
| 已交付工具调用数据集 | D1 (126) / D1.1 (1500) / D2 (5000 多轮对话) |
| 真实 OWT 评测覆盖 | 5 个公开 instruction-tuned 模型 × 277MB held-out |

---

## 5. 技术栈（按出现频次）

- **模型实现**：PyTorch 2.13 (sm_120 / RTX 5070 Ti)
- **数据 / 评测**：Transformers 5.15, vLLM 0.27.1 (TORCH_SDPA)
- **训练框架**：原生 PyTorch + 自研训练循环 + AMP bf16 + warmup_cosine scheduler
- **Schema 校验**：jsonschema 库 + 11 个手写 schema 文件
- **测试**：pytest
- **可复现基础设施**：git commit hash 绑定 + config sha256 + token cache sha256 + dataset hash + seed metadata

---

## 6. 同 base 硬性判定标准（架构对比通用）

任何"Dense vs MoE / MHA vs GQA / 不同 base 模型"对比必须满足 5 项：

1. **同 tokenizer**（sha256 校验）
2. **同训练数据 cache**（sha256 校验）
3. **同训练超参**（lr / batch / seq / steps / scheduler / AMP）
4. **同模型规模**（active params 或 total params 一致，差距 ≤5%）
5. **同 seed + 同 AMP dtype**

不满足任一项 → 该对比结论不成立。本项目所有架构对比实验都满足 5/5。

---

## 7. 关联文档

- 项目总体：`README.md` + `AGENTS.md`
- 项目路线图：`docs/plans/roadmap.md` + `docs/plans/open-issues.md`
- 阶段审查：`docs/plans/reviews/`（38 个 stage-*.md）
- 实验详情：见 `docs/experiments/` 下 27 个子目录
- N13（新交付）：`docs/experiments/n13-gqa-vs-mla-vs-mha/` + `docs/plans/reviews/stage-n13-gqa-vs-mla-vs-mha.md`
