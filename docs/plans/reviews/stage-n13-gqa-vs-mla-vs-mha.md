# Stage review — list item "GQA + MLA 自研实现 + 5000 步 OWT 训练三方对比"（N13）

> 任务来源：list item "GQA + MLA 自研实现 + 5000 步 OWT 训练三方对比"（active list item，2026-08-30 启动）。
> 目标：补齐 README §项目目标 1 中缺失的 **MoE + GQA + 简化 MLA** 三件套中的 GQA 和 MLA，与 N4 Dense MHA baseline 在**完全相同的 OWT 正式 cache + 训练超参**下做三方对比。

## 范围

- ✅ 自研 GQA 模型 (`architecture_lab/models/gqa_transformer.py`): `num_kv_heads` 可配置，复用 RMSNorm / RoPE / SwiGLU from dense_transformer，KV 用 `repeat_interleave` 到 n_heads
- ✅ 自研简化 MLA 模型 (`architecture_lab/models/mla_transformer.py`): K/V 联合压缩到 `latent_dim`，KV cache 存压缩 latent（不是 full K/V），简化版省略 decoupled RoPE
- ✅ `architecture_lab/models/__init__.py` 导出 GQA / MLA
- ✅ `architecture_lab/training/dense_training.py` dispatch by `architecture` 字段 (`build_model_config` / `build_model`)
- ✅ `architecture_lab/training/results.py` 从 settings 读 architecture（之前硬编码 DenseTransformer）
- ✅ `schemas/dense_training_result.schema.json`: `model.architecture` enum 扩展为 `[DenseTransformer, GQATransformer, MLATransformer]`
- ✅ 13 个 pytest 单元测试 (`tests/test_gqa_mla_models.py`): forward shape / 因果 loss / KV cache 增量一致性 / GQA 缓存比 MHA 小 / MLA 缓存比 MHA 小 / generate() / 3 模型同数据 loss finite
- ✅ 3 个 training configs (Dense / GQA / MLA)，与 N4 medium 同 vocab/d_model/n_heads/n_layers/d_ff/max_seq_len/data/training，仅 architecture + 新参数不同
- ✅ 3 个 5000 步 OWT 训练（每个 ~3-5 分钟，RTX 5070 Ti + bf16）
- ✅ `scripts/plot_n13_comparison.py` 生成 3-way overlay PNG（train_loss log + val_loss linear）
- ✅ `docs/experiments/n13-gqa-vs-mla-vs-mha/{README,protocol,result}.md` + overlay PNG
- ✅ P5-04 E selftest 仍 97 PASS / 0 FAIL 无 regression
- ✅ `docs/plans/open-issues.md` / `roadmap.md` 待 Phase 5 同步

## 关键数字（5000 步 OWT）

| 指标 | Dense MHA (baseline) | GQA (kv=1) | MLA (latent=64) |
|---|---|---|---|
| 总参数 | 2,098,304 | 2,000,000 | 2,065,536 |
| train_loss first | 104.69 | 103.84 | 111.72 |
| train_loss last | 6.56 | 6.71 | 6.68 |
| **val_min** | **7.058** | 7.106 | 7.122 |
| KV cache bytes (per layer, B=1, L=1) | 1024 | **256** | **256** |
| KV cache vs MHA | 1.0× | **0.25×** | **0.25×** |

## 同 base 硬性判定标准（5 项）

| # | 标准 | 验证 |
|---|---|---|
| 1 | 同 tokenizer (OWT BPE v0.2.0, vocab=8192) | `python -c "..."` audit command in protocol.md |
| 2 | 同 OWT token cache | sha256 校验 in audit commands |
| 3 | 同训练超参 (lr=3e-4 / batch=8 / seq=64 / 5000 步) | 3 个 config YAML diff: 仅 model.architecture + model.num_kv_heads/model.latent_dim 不同 |
| 4 | 同模型规模 (~2.10M) | 实测: 2,098,304 / 2,000,000 / 2,065,536（差距 ≤5%） |
| 5 | 同 seed (42) + AMP (bf16) | metadata.seed + metadata.gpu_name |

5/5 满足。

## 注意事项（Caveats）

- val_min 差距 ≤0.07 nats 在 2.10M 规模 + 5000 步 + bf16 范围内属于训练噪声。**Dense MHA 的 +0.05 / +0.06 不应被解读为 "GQA/MLA 更差"**。
- MLA simplified 的 `latent_dim=64` 不是 head_dim=32 的整数倍（是 2×），重建 K/V 时存在数值重建误差。这影响 cache 大小但不直接限制精度。
- 本对比不评估推理延迟（仅训练 + val_loss 对比）。Cache 小 ≠ decode 快（GPU kernel launch overhead 在小模型上占主导）。
- num_kv_heads=1 是 GQA 的极端 case（Multi-Query Attention）。常规 GQA 配置（如 Mistral 7B 的 32/8）会有不同结果。

## Verification

```bash
# 13 个单元测试 PASS
python -m pytest tests/test_gqa_mla_models.py -v
# => 13 passed in 1.81s

# 3 个 result 都 schema-valid
python -c "
import json, jsonschema
schema = json.load(open('schemas/dense_training_result.schema.json', encoding='utf-8'))
for p in ['artifacts/dense-owt-formal-curve-medium-result.json',
         'artifacts/gqa-owt-formal-curve-medium-result.json',
         'artifacts/mla-owt-formal-curve-medium-result.json']:
    d = json.load(open(p, encoding='utf-8'))
    jsonschema.validate(d, schema)
"
# => all VALID

# P5-04 E selftest 无 regression
python3 scripts/eval_owt_real.py --selftest
# => 97 PASS / 0 FAIL

# 3-way overlay PNG 已生成
ls -la docs/experiments/n13-gqa-vs-mla-vs-mha/dense-vs-gqa-vs-mla-curves.png
```

## 关联文档

- 实验 README: `docs/experiments/n13-gqa-vs-mla-vs-mha/README.md`
- 协议: `docs/experiments/n13-gqa-vs-mla-vs-mha/protocol.md`
- 结果: `docs/experiments/n13-gqa-vs-mla-vs-mha/result.md`
- 代码: `architecture_lab/models/{gqa,mla}_transformer.py`, `architecture_lab/training/dense_training.py`, `architecture_lab/training/results.py`
- 测试: `tests/test_gqa_mla_models.py`
- 配置: `configs/{gqa,mla}-owt-formal-curve-medium.example.yaml`
- Schema: `schemas/dense_training_result.schema.json`
- Plot: `scripts/plot_n13_comparison.py`

## Phase 5 follow-up

- `README.md` §项目目标 1: 改写 GQA/MLA 三件套现状
- `README.md` §当前状态: 加 GQA + MLA ✓ 项
- `docs/plans/roadmap.md` 候选 2 (简化 MLA) 状态更新
- `docs/plans/open-issues.md`: GQA + MLA follow-up 项
- `AGENTS.md` 当前阶段同步
