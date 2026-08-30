# N13: 协议 — Dense MHA vs GQA vs MLA 三方对比

## 1. 实验目的

补齐 CS336 扩展三件套中的 GQA + MLA，与 N4 Dense MHA baseline 在同 base 条件下对比。

## 2. 同 base 硬性判定标准（5 项）

| # | 标准 | 验证方式 |
|---|---|---|
| 1 | 同 tokenizer | `artifacts/tokenizers/owt-bpe/v0.2.0/tokenizer.json`，vocab=8192 |
| 2 | 同 OWT token cache | `data/processed/owt-sample/{train,validation}.tokens.uint16`，sha256 校验 |
| 3 | 同训练超参 | lr=3e-4 / wd=0.01 / batch=8 / seq=64 / 5000 步 / warmup_cosine |
| 4 | 同模型规模 | d_model=128 / n_heads=4 / n_layers=4 / d_ff=512，~2.10M params |
| 5 | 同 seed + AMP | seed=42 / bf16 |

不满足任一项则结论无效。

## 3. 三个架构的训练配置

**Dense MHA baseline**（既有，N4）：
- `configs/dense_training.owt-formal-curve-medium.example.yaml`
- `architecture: DenseTransformer`
- 2,098,304 params
- 已有 result: `artifacts/dense-owt-formal-curve-medium-result.json`（HEAD 4c239a6 之前的产物）

**GQA**（新增）：
- `configs/gqa-owt-formal-curve-medium.example.yaml`
- `architecture: GQATransformer`
- `num_kv_heads: 1`（4 个 Q head share 1 个 KV head；极端 MQA）
- 2,000,000 params（K/V 投影 channel 减少 → 节省 98K 参数）
- 训练产物: `artifacts/gqa-owt-formal-curve-medium-result.json`

**MLA simplified**（新增）：
- `configs/mla-owt-formal-curve-medium.example.yaml`
- `architecture: MLATransformer`
- `latent_dim: 64`（d_model/n_heads = 128/4 = 32 → latent_dim=64 是 2× head_dim 的压缩）
- 2,065,536 params
- 训练产物: `artifacts/mla-owt-formal-curve-medium-result.json`

## 4. KV cache 大小（per layer, batch=1, seq=1, float32）

| 架构 | KV cache 公式 | 单 token 单 layer bytes |
|---|---|---|
| Dense MHA | B × L × 2 × n_heads × head_dim × 4 | 2 × 4 × 32 × 4 = 1024 |
| GQA (kv=1) | B × L × 2 × num_kv_heads × head_dim × 4 | 2 × 1 × 32 × 4 = 256 |
| MLA (latent=64) | B × L × latent_dim × 4 | 64 × 4 = 256 |

**GQA 与 MLA cache 均为 Dense 的 1/4**（在 batch=1, num_kv_heads=1, latent_dim=64 配置下）。

## 5. 不做什么

- ❌ 多种 num_kv_heads sweep（GQA 只用 1）
- ❌ 多种 latent_dim sweep（MLA 只用 64）
- ❌ 长序列 benchmark（seq=64 而非 1024+）
- ❌ 推理 latency benchmark（仅训练 + val_loss 对比）
- ❌ 真实 LLM 工具调用/生成质量评估（仅 per-token cross-entropy）

## 6. Audit commands（auditor-runnable）

```bash
# 三个 result 都存在 + 参数合理
ls artifacts/{dense,gqa,mla}-owt-formal-curve-medium-result.json

# 三个 result 都是 schema-valid
python -c "
import json, jsonschema
schema = json.load(open('schemas/dense_training_result.schema.json', encoding='utf-8'))
for p in ['artifacts/dense-owt-formal-curve-medium-result.json',
         'artifacts/gqa-owt-formal-curve-medium-result.json',
         'artifacts/mla-owt-formal-curve-medium-result.json']:
    d = json.load(open(p, encoding='utf-8'))
    jsonschema.validate(d, schema)
    print(f'{p}: VALID arch={d[\"model\"][\"architecture\"]}')
"

# 三个架构都用同一 tokenizer
python -c "
import json
for p in ['artifacts/dense-owt-formal-curve-medium-result.json',
         'artifacts/gqa-owt-formal-curve-medium-result.json',
         'artifacts/mla-owt-formal-curve-medium-result.json']:
    d = json.load(open(p, encoding='utf-8'))
    print(p, 'tokenizer sha:', d['data']['tokenizer']['artifact_sha256'][:16])
"

# 三个架构都用同一 train cache
python -c "
import json
for p in ['artifacts/dense-owt-formal-curve-medium-result.json',
         'artifacts/gqa-owt-formal-curve-medium-result.json',
         'artifacts/mla-owt-formal-curve-medium-result.json']:
    d = json.load(open(p, encoding='utf-8'))
    print(p, 'train cache sha:', d['data']['train_cache']['token_file_sha256'][:16])
"

# 三个架构 seed 都 = 42
python -c "
import json
for p in ['artifacts/dense-owt-formal-curve-medium-result.json',
         'artifacts/gqa-owt-formal-curve-medium-result.json',
         'artifacts/mla-owt-formal-curve-medium-result.json']:
    d = json.load(open(p, encoding='utf-8'))
    print(p, 'seed:', d['training']['seed'])
"

# 模型单元测试全 PASS
python -m pytest tests/test_gqa_mla_models.py -v

# P5-04 E selftest 无 regression（97 PASS）
python3 scripts/eval_owt_real.py --selftest

# 三方对比 plot 生成
python scripts/plot_n13_comparison.py
```
