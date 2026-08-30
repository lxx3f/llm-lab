# N13: 自研 Dense MHA vs GQA vs MLA 三方对比 (5000 步 OWT)

## 目的

补齐 README §项目目标 1 中缺失的 **MoE + GQA + 简化 MLA** 三件套中的 GQA 和 MLA 部分。原 N4 baseline 只有 Dense MHA（val_min 7.06）；本次新增 GQA 与简化 MLA 实现，并在**完全相同的 OWT 正式 cache + 训练配置**下做三方对比。

**同 base 硬性判定标准**（避免不同 base 模型差异污染结论）：
1. 同 tokenizer (OWT BPE v0.2.0, vocab=8192)
2. 同 OWT token cache (`data/processed/owt-sample/`, sha256 校验)
3. 同训练超参 (lr=3e-4, weight_decay=0.01, batch=8, seq=64, 5000 步, warmup_cosine)
4. 同模型规模 (d_model=128, n_heads=4, n_layers=4, d_ff=512, ~2.10M params)
5. 同 seed (42) + 同 AMP (bf16)

## 三个架构

| 架构 | 注意力路径 | 总参数 |
|---|---|---|
| **Dense MHA** (N4 baseline) | `qkv` 投影 3×d_model，n_heads=4 KV heads | 2,098,304 |
| **GQA** (`num_kv_heads=1`) | Q 投影 d_model，K/V 投影 1 head_dim，repeat_interleave 到 4 Q heads | 2,000,000 |
| **Simplified MLA** (`latent_dim=64`) | Q 投影 d_model；K/V 联合压缩 d_model→64，up-project 到 d_model per head | 2,065,536 |

实现细节：
- `architecture_lab/models/dense_transformer.py`（既有，未改动）
- `architecture_lab/models/gqa_transformer.py`（新增）
- `architecture_lab/models/mla_transformer.py`（新增）
- `architecture_lab/models/__init__.py`（统一导出）
- `architecture_lab/training/dense_training.py`（dispatch by `architecture` 字段）

KV cache 大小对比（同 seq_len）：
- Dense MHA: per layer = B × L × 2 × n_heads × head_dim × 4 bytes
- GQA:      per layer = B × L × 2 × num_kv_heads × head_dim × 4 bytes（按 num_kv_heads/n_heads 倍数缩小）
- MLA:      per layer = B × L × latent_dim × 4 bytes（按 2×n_heads×head_dim / latent_dim 倍数缩小）

## 关键结果（val_min）

| 架构 | train_loss first → last | val_min | 总参数 |
|---|---|---|---|
| Dense MHA | 104.69 → 6.56 | **7.058** | 2,098,304 |
| GQA (kv=1) | 103.84 → 6.71 | 7.106 | 2,000,000 |
| MLA (latent=64) | 111.72 → 6.68 | 7.122 | 2,065,536 |

**结论**：
- 三个架构收敛到相近的 val_loss（差距 ≤0.07 nats）
- Dense MHA 略优，**不把差距归因于 attention 路径差异**——差距在同规模小模型 + 5000 步范围内属于训练噪声
- GQA 与 MLA 都用更少/相近参数达到相近 loss，证明**两种 attention 路径在小规模 OWT 上同样 work**

## 文件

- `protocol.md`：实验协议 + 同 base 硬性判定
- `result.md`：详细 3-way 对比表 + KV cache 大小计算
- `dense-vs-gqa-vs-mla-curves.png`：train_loss + val_loss overlay
- 训练产物：
  - `artifacts/dense-owt-formal-curve-medium-result.json`
  - `artifacts/gqa-owt-formal-curve-medium-result.json`
  - `artifacts/mla-owt-formal-curve-medium-result.json`

## Stage review

`docs/plans/reviews/stage-n13-gqa-vs-mla-vs-mha.md`
