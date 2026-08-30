# GQA vs MHA 公开模型对比 — 搜索协议 (no-go path)

> **本 protocol 是 `feasibility.md` §8 的独立拆分版本**，便于 auditor 单独
> 复现（§8 的内容已经在 feasibility.md 内 inline 完整保留）。两份内容必须保持
> 一致；如修改本文件，请同步更新 feasibility.md §8。

## 目标

在 2026-08-30 由 web search + P5-04 本地 config 取证的 15 个 LLM family +
3 research artifact 范围内，搜索可审计的同 base GQA/MHA 公开模型对。

## 硬性判定标准（feasibility.md §1.2）

"同 base GQA/MHA 双版本" 必须满足：

1. 同一组织同一训练计划
2. 唯一架构差异是 `num_key_value_heads`
3. 其他所有 hyperparameter 一致 (hidden_size / num_hidden_layers /
   intermediate_size / vocab_size / rope_theta / activation / norm / init)
4. 公开可下载的 safetensors 权重
5. 可比 HF commit SHA 可绑定

满足 1+2+3 才算真正 "同 base 仅 attention 不同"。

## 搜索步骤

### Step 1: 系统性搜索主要 LLM family

对每个候选 family，按以下顺序收集 evidence：

1. HF Hub 主页（`https://huggingface.co/<org>/<model>`）— 看 model card 文本
   声明 attention 类型。
2. HF config.json（`https://huggingface.co/<org>/<model>/resolve/main/config.json`）
   — 验证 `num_attention_heads` / `num_key_value_heads` 字段。Gated repo 可能
   需要 HF token + 接受 license；如果 gated，使用 mirror（如
   `tjluyao/llama-2-7b-hf` 镜像 `meta-llama/Llama-2-7b-hf` config.json）。
3. 论文 / 技术报告（如果存在）— 验证 model 描述与 config.json 一致。
4. GitHub repo commit history（如果模型有公开训练代码）— 验证训练数据 +
   token 数 + commit SHA。

### Step 2: 已下载本地 config 实证

P5-04 任务已下载 5 个公开模型到 `artifacts/owt-real-eval/models/models/`：

```
artifacts/owt-real-eval/models/models/
├── HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/<commit-or-master>/config.json
├── HuggingFaceTB--SmolLM2-1.7B-Instruct/snapshots/<commit-or-master>/config.json
├── Qwen--Qwen2.5-0.5B-Instruct/snapshots/<commit-or-master>/config.json
├── Qwen--Qwen2.5-1.5B-Instruct/snapshots/<commit-or-master>/config.json
└── Qwen--Qwen2.5-3B-Instruct/snapshots/<commit-or-master>/config.json
```

每个 config.json 含 `num_attention_heads` 和 `num_key_value_heads` 字段。
对 P5-04 而言，5 个模型都有 master snapshot dir；每个 repo 的 canonical
commit 记录在 `artifacts/owt-real-eval/results/<model>/metadata.json` 的
`hf_expected_revision` 字段。

### Step 3: 跨基线判断（同 base 双版本是否成立）

对任意两个候选模型，按以下 rubric 判断是否构成"同 base 仅 attention 不同"对：

| 维度 | 必须一致 | 否则 |
|---|---|---|
| hidden_size | 同 | size confounding |
| num_hidden_layers | 同 | depth confounding |
| vocab_size | 同 | tokenizer confounding |
| intermediate_size | 同 | FFN confounding |
| rope_theta | 同 | position embedding confounding |
| training data (per paper / model card) | 同 | data confounding |
| training token count (per paper / model card) | 同 | compute confounding |
| tokenizer type | 同 | vocab confounding |
| norm (RMSNorm vs LayerNorm) | 同 | norm confounding |
| activation (silu vs gelu vs relu) | 同 | activation confounding |

如果任意一个维度不同，则该 pair 不构成"同 base GQA/MHA 对"。

## 当前结论（access 2026-08-30）

按 §3 rubric 检查 P5-04 5 个本地模型：

| Pair | hidden | layers | vocab | 同 base? |
|---|---|---|---|:---:|
| SmolLM2-360M vs SmolLM2-1.7B | 960 vs 2048 | 32 vs 24 | 49152 vs 49152 | ❌ hidden + layers 不同 |
| 任意 Qwen2.5 pair | 896 / 1536 / 2048 | 24 / 28 / 36 | 151936 | ❌ hidden + layers 不同 |
| SmolLM2 vs Qwen2.5 | 960-2048 vs 896-2048 | 24-36 vs 24-36 | 49152 vs 151936 | ❌ vocab 不同 + training data 不同 |

无 pair 满足所有 §3 维度均一致的硬性判定标准 #1-#5。

## 实证命令（auditor-runnable）

```bash
# §1 P5-04 5 model config 实证
python -c "
from pathlib import Path
import json
for cfg in sorted(Path('artifacts/owt-real-eval/models/models').rglob('config.json')):
    c = json.loads(cfg.read_text(encoding='utf-8'))
    h = c.get('num_attention_heads'); kv = c.get('num_key_value_heads', h)
    arch = 'GQA' if kv != h else 'MHA'
    print(f'{cfg.parent.parent.parent.name}: heads={h} kv={kv} hidden={c.get(\"hidden_size\")} layers={c.get(\"num_hidden_layers\")} vocab={c.get(\"vocab_size\")} -> {arch}')
"

# §2 §2.2 family + research artifact 数量 + URL count + access date count
python -m pytest tests/test_gqa_vs_mha_audit_consistency.py -v
# Expected: 19 passed (7 keyword/family smoke tests + 12 structural audit-consistency tests)

# §3 narrowed claim wording — 必须显式出现 "audited candidate set" + "not auditable / not verifiably excluded"
grep -nE 'audited candidate set|not auditable|not verifiably excluded' docs/experiments/gqa-vs-mha/feasibility.md
```

## Cross-reference

- `feasibility.md` §1 — 硬性判定标准
- `feasibility.md` §2 — 候选 family 表格（含每行 Verification status）
- `feasibility.md` §3 — 否决理由汇总
- `feasibility.md` §7 + §10 — 实证命令清单
- `feasibility.md` §8 — 搜索步骤（与本文件同步）
- `feasibility.md` §9 — 结论范围 (narrowed claim)

如修改本 protocol，必须同步修改 `feasibility.md` §8 保持一致；auditor 会交叉比对。

## Schema evidence (cross-reference)

本 protocol 配套 schema artifact 在 `feasibility.md` §11：`schemas/evaluation_result.schema.json` + `examples/evaluation_results/sample-no-go-result.json` + auditor-runnable validation command + recorded output。
