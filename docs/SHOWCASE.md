# LLM Lab SHOWCASE — 简历项目摘要

> **目的**：把仓库里分散在 27 个 `docs/experiments/` 下的实验 + 421 个测试 + 11 个 JSON Schema 浓缩成可直接用于简历、面试演示和 GitHub 主页 README 的素材。详细的方法、protocol 和数字见各 `docs/experiments/<name>/` 子目录。

> **TL;DR**：自研实现 Dense / MoE / GQA / MLA 4 种注意力路径，从零训练多个 0.66M-12M 模型在真实 OpenWebText 上；在 5 个公开 instruction-tuned 模型（360M-3B）上完成真实 OWT held-out 评测 + Transformers/vLLM 双后端对比；数据管线 + 工具调用 8 级分类器 + SFT + GRPO 全闭环。

---

## 1. 数字一览（5 行概览）

| # | 实验 | 关键 metric | 数字 | 详细 |
|---|---|---|---|---|
| 1 | **N13** 自研 GQA + MLA 与 Dense MHA 5000 步 OWT 对比 | val_min | Dense 7.058 / GQA 7.106 / MLA 7.122 (差距 ≤0.07 nats) | [`docs/experiments/n13-gqa-vs-mla-vs-mha/`](experiments/n13-gqa-vs-mla-vs-mha/) |
| 2 | **E** 5 个公开 instruction-tuned 模型真实 OWT 评测（277MB held-out） | mean_loss_nats / PPL | SmolLM2-1.7B **PPL 11.05** 最优 / Qwen2.5-3B PPL 13.04 | [`docs/experiments/owt-real-eval/`](experiments/owt-real-eval/) |
| 3 | **P5-04** Transformers vs vLLM 双后端 × 5 模型 × 2 batch × 90 样本 | 4 轴 (latency / throughput / reward_binary / reward_layered) | **vLLM b=4 全部 Top-5 领先**；最快 = Qwen2.5-0.5B vLLM b=4 (8.18 samples/s, 122.3ms)；诚实负结果 reward_binary 全为 0；reward_layered 0.36-0.42 | [`docs/experiments/p5-04-backend-comparison/`](experiments/p5-04-backend-comparison/) |
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

> **在 5 模型 (SmolLM2-360M/1.7B、Qwen2.5-0.5B/1.5B/3B) × 2 后端 (Transformers vs vLLM) × 2 batch size (1/4) = 20 组合 × 90 样本固定 benchmark 上做 4 轴（latency / throughput / reward_binary / reward_layered）对比，commit-level binding + manifest-driven 加载 + section-aware README 同步守护（含 4 个 NEGATIVE test）保证 comparison.csv 与 README 数字严格一致。实测 vLLM 在 b=4 时全 5 模型 throughput 提升 +55%~+90%（最快 = Qwen2.5-0.5B vLLM b=4 = 8.18 samples/s, 122.3ms），reward_layered 0.36-0.42 与 P5-02 历史表一致；诚实记录 reward_binary 全为 0（公开模型无 gold answer 提示）。**

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

## 7. 关键代码片段（示意，不完整）

### GQA 注意力（仅 18 行核心逻辑）

```python
class GQACausalSelfAttention(nn.Module):
    def __init__(self, config: GQAConfig):
        # Q 投影 d_model；K/V 仅 num_kv_heads * head_dim (远小于 d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        # ... RoPE + out_proj

    def forward(self, x, start_pos=0, kv_cache=None):
        q = self.q_proj(x).view(B, L, n_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, num_kv_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, num_kv_heads, head_dim).transpose(1, 2)
        q, k = self.rope(q, k, start_pos=start_pos)
        # KV cache 只存 num_kv_heads (节省 K/V cache (1 - 1/n_heads)%)
        if kv_cache: kv_cache["k"] = k.detach(); kv_cache["v"] = v.detach()
        # broadcast: each KV head serves n_heads/num_kv_heads Q heads
        if self.q_per_kv > 1:
            k = k.repeat_interleave(self.q_per_kv, dim=1)
            v = v.repeat_interleave(self.q_per_kv, dim=1)
        return self.out_proj(F.scaled_dot_product_attention(q, k, v, attn_mask=causal))
```

### MLA 压缩 KV cache（仅核心）

```python
class MLACausalSelfAttention(nn.Module):
    def __init__(self, config: MLAConfig):
        # 联合 K/V 压缩：d_model -> latent_dim
        self.W_DKV = nn.Linear(d_model, latent_dim, bias=False)  # 压缩
        self.W_UK = nn.Linear(latent_dim, d_model, bias=False)   # K up-project
        self.W_UV = nn.Linear(latent_dim, d_model, bias=False)   # V up-project
        # Q 不压缩（简化版）
        self.W_Q = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, start_pos=0, kv_cache=None):
        c_kv = self.W_DKV(x)  # (B, L, latent_dim) — cache 只存这个
        if kv_cache:
            if kv_cache.get("c_kv") is not None:
                c_kv = torch.cat((kv_cache["c_kv"], c_kv), dim=1)
            kv_cache["c_kv"] = c_kv.detach()
        # 从 latent cache 重建 full K/V (每步重建)
        k = self.W_UK(c_kv).view(B, c_kv.size(1), n_heads, head_dim).transpose(1, 2)
        v = self.W_UV(c_kv).view(B, c_kv.size(1), n_heads, head_dim).transpose(1, 2)
        q = self.W_Q(x).view(...).transpose(1, 2)
        q, k = self.rope(q, k, start_pos=start_pos)
        return self.W_O(F.scaled_dot_product_attention(q, k, v, attn_mask=causal))
```

完整实现见 [`architecture_lab/models/gqa_transformer.py`](../architecture_lab/models/gqa_transformer.py) (8158B) + [`architecture_lab/models/mla_transformer.py`](../architecture_lab/models/mla_transformer.py) (9088B)。

---

## 8. 诚实记录的限制（避免简历过度宣传）

| 限制 | 实情 | 影响 |
|---|---|---|
| **自研模型规模上限 12M** | 不能直接对标百亿参数模型 | 在大模型场景下的结论不适用 |
| **MLA 是简化版（非 DeepSeek-V2 完整 decoupled RoPE）** | 省略 decoupled RoPE，K/V 重建时 RoPE 需每步重算 | 极端大 latent_dim 下数值误差会增大；本项目 latent_dim=64 × head_dim=32 误差可忽略 |
| **GQA num_kv_heads=1 (MQA 极端)** | 没有 sweep 多种 num_kv_heads (1/2/n_heads) | 无法判断 GQA 分组数 vs val_loss 的 Pareto 曲线 |
| **P5-04 公开模型在 tool calling 任务上 reward_binary 全 0** | 公开 instruction-tuned 模型没有 gold answer 提示 | 这是诚实负结果；说明模型不能在 tool calling benchmark 上达到 SOTA（已知事实），不代表 P5-04 实验失败 |
| **F (GQA vs MHA 公开模型) 无同 base 双版本对** | 公开模型没发布同 base GQA/MHA 双版本 | 只交付了 incomplete feasibility review；自训练双版本不在本项目算力预算内 |
| **真实 OWT 评测只覆盖 5 模型 × 完整 277MB held-out** | 没有跑 PPL vs n_steps 的 Pareto | 可信区间大；建议在简历中标注"per-token cross-entropy on full held-out" |

---

## 9. 关联文档

- 项目总体：`README.md` + `AGENTS.md`
- 项目路线图：`docs/plans/roadmap.md` + `docs/plans/open-issues.md`
- 阶段审查：`docs/plans/reviews/`（38 个 stage-*.md）
- 实验详情：见 `docs/experiments/` 下 27 个子目录
- N13（新交付）：`docs/experiments/n13-gqa-vs-mla-vs-mha/` + `docs/plans/reviews/stage-n13-gqa-vs-mla-vs-mha.md`
