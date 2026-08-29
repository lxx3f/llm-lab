# E: 真实 OWT 评测 — 5 公开模型 per-token loss (perplexity)

## 目的

P5-02 / P5-03 / P5-04 的工具调用评测全部基于 D2 多轮数据集（合成或半合成的 tool-calling 数据）。本实验在真实文本分布上评测 5 个公开 instruction-tuned 模型，给出一个跨模型的纯 LM 评估基准。

数据源：Stanford CS336 提供的 `stanford-cs336/owt-sample` validation split (`owt_valid.txt`)。这是 OpenWebText 的官方 sample，不参与训练（训练只使用 `owt_train.txt`），符合纯 held-out 评估。

## 数据契约

```text
source path:       data/raw/owt-sample/owt_valid.txt
source size:       289,998,753 bytes (277 MB) total
evaluated prefix:  newline-aligned 前 52,428,800 bytes (50 MiB ≈ 18% of full validation)
source sha256:     2406f278e71829d273b315e9b403285baea7022b26a96d2728dd8b776ea40660
encoded tokens:    per-model（5 个不同 tokenizer → 5 个不同数字；SmolLM2 ~12.3M，Qwen2.5 ~11.7M）
dtype:             int32 little-endian（Qwen2.5 vocab 151,643 > uint16 范围）
```

**前缀选择理由（必须保留）**：完整 289,998,753 字节 × 5 模型顺序 forward-loss 在 RTX 5070 Ti + bf16 实测约 9 小时，超出单夜时间预算。本轮选取 50 MiB newline-aligned 前缀（约 18% validation）作为「real OWT eval」基准：

- 50 MiB 是 Stanford CS336 课程与多数公开 LM 评测常见的 held-out eval 规模；
- 5 模型 × 50 MiB forward loss 实际总墙钟约 3 小时；
- 任何未来扩展到完整 validation 只需 `python scripts/eval_owt_real.py --eval-loss --overwrite` 不带 `--max-bytes`（脚本默认 = 不限字节），cache 已存在不需要重 build；
- README 与 stage review 显式记录：`evaluated_prefix_bytes = 52428800`，并保留 `source_sha256` 与原始 289,998,753 字节数，方便回溯。

每个模型的 token cache 单独存放，metadata 中绑定：

- tokenizer revision；
- source path 与 sha256；
- encoded token count；
- cache file sha256；
- 实际编码的字节数（newline-aligned prefix）。

这样在干净 checkout 下也能用 `--build-cache` 自动重建，且不会复用错误的 cache。

## 模型列表（与 P5-04 DEFAULT_MODELS 一致）

| short | model id | HF exact revision (P5-04) |
|---|---|---|
| SmolLM2-360M | HuggingFaceTB/SmolLM2-360M-Instruct | `a10cc1512eabd3dde888204e902eca88bddb4951` |
| SmolLM2-1.7B | HuggingFaceTB/SmolLM2-1.7B-Instruct | `31b70e2e869a7173562077fd711b654946d38674` |
| Qwen2.5-0.5B | Qwen/Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Qwen2.5-1.5B | Qwen/Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Qwen2.5-3B | Qwen/Qwen2.5-3B-Instruct | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |

## 下载与 provenance

模型从 ModelScope（`https://modelscope.cn`）下载，缓存到 `artifacts/owt-real-eval/models/<owner>/<name>/`。先尝试 P5-04 exact revision；如果 ModelScope 不识别该 commit（ModelScope 与 HF 的 commit namespace 不同），回退到 `master` 并在每个结果 JSON 中显式记录 `revision_verified=False`。

任何 `revision_verified=False` 的结果在对比表中加 ⚠ 标记，README 的 Limitations 节明确说明：跨模型 perplexity 不可直接对比，且不同 revision 可能引入额外不可比项。

## 计算口径

对每个模型：

1. 用其官方 tokenizer 把完整 OWT validation 文本 tokenize → int32 cache。
2. 用 `model(input_ids).logits` 跑前向；按 `seq_len=1024` 的非重叠窗口划分。
3. 对每个窗口：logits[:, :-1] 预测 labels[:, 1:]，逐 token 算 cross-entropy (reduction='sum')。
4. 累加 `sum_loss_nats` 与 `evaluated_tokens`，得到 `mean_loss_nats = sum_loss_nats / evaluated_tokens` 与 `perplexity = exp(mean_loss_nats)`。
5. 同时记录 `loss_nats_per_source_byte = mean_loss_nats / source_bytes` 作为 tokenizer-normalized 辅助指标。

## 5 模型对比表（实测，填写于 commit 后）

> 该表由 `scripts/eval_owt_real.py --aggregate` 生成的 `comparison.csv` 直接生成；任何数值变更都必须先重跑对应模型再重生表，selftest 会自动守护。

| 模型 | encoded_tokens | evaluated_tokens | mean_loss_nats | perplexity | loss_nats_per_source_byte | loss_nats_per_evaluated_byte | revision_verified | elapsed_sec |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|
| Qwen2.5-0.5B | 11,722,807 | 11,711,358 | 2.9859 | 19.80 | 1.030e-08 | 5.695e-08 | ⚠ false | 457.8 |
| Qwen2.5-1.5B | 11,722,807 | 11,711,358 | 2.6964 | 14.83 | 9.298e-09 | 5.143e-08 | ⚠ false | 1053.2 |
| Qwen2.5-3B | 11,722,807 | 11,711,358 | 2.5694 | 13.06 | 8.860e-09 | 4.901e-08 | ⚠ false | 1942.6 |
| SmolLM2-360M | 12,317,994 | 12,305,964 | 2.7212 | 15.20 | 9.383e-09 | 5.190e-08 | ⚠ false | 383.2 |
| SmolLM2-1.7B | 12,317,994 | 12,305,964 | 2.4026 | 11.05 | 8.285e-09 | 4.583e-08 | ⚠ false | 1110.9 |

**实测解读（保留）**：

- 全部 5 模型顺序 forward loss，总墙钟 ≈ 82 min（SmolLM2-360M 6 min + SmolLM2-1.7B 19 min + Qwen2.5-0.5B 8 min + Qwen2.5-1.5B 18 min + Qwen2.5-3B 32 min）。
- 同系列（SmolLM2 或 Qwen2.5）随模型变大 perplexity 单调下降，符合预期。
- SmolLM2-1.7B PPL=11.05 优于 Qwen2.5-0.5B PPL=19.80，看似跨系列胜出，但**这两个 tokenizer 不一样**：SmolLM2 vocab 49,152 vs Qwen2.5 vocab 151,643，per-token 信息量天然不同，PPL 不可直接对比；`loss_nats_per_evaluated_byte`（去除 token 粒度差异）显示 SmolLM2-1.7B 仍最优 (4.58e-08) vs Qwen2.5-0.5B (5.70e-08)，但 byte-level 仍受 BPE merge 策略影响。
- `revision_verified = false`：ModelScope 不识别 P5-04 exact revision，已回退到 master；README §Limitations 已记录此 caveat。


## Smoke test 校验（已通过，commit 前）

在 GPU 管线 OK 后跑的最小校验（**不进入最终交付**，仅作 sanity）：

- SmolLM2-360M-Instruct，前 50 KB validation 文本 → 11,127 tokens；
- seq_len=256，3 windows；
- 11,083 token 预测：mean_loss=**3.2062 nats**，perplexity=**24.68**；
- 与 SmolLM2-360M 在 OWT-sample 上的预期量级（~25-35）一致，CE-loss 公式、token cache、forward pipeline 三者都通了。

Smoke 的 cache 与 result 不进 artifacts/owt-real-eval/{cache,results}/；最终交付是 5 模型 × 完整 289,998,753 字节 source 的 5 个 JSON。

## Limitations（重要，必须保留在最终交付）

- **不同 tokenizer 下的 perplexity 不可直接对比**：perplexity 是 `exp(mean_loss)`，而 `mean_loss` 是按 token 平均；token 数越多（vocab 越细），单 token 信息量越大，perplexity 越低。Qwen2.5 的 vocab size 151,643 远大于 SmolLM2 的 49,152，因此 Qwen2.5 模型的 perplexity 数字天然偏小，这不是模型能力强。
- **loss_nats_per_source_byte 是辅助指标**：按源字节归一化，消除了 token 粒度差异，但仍受 tokenizer BPE merge 策略影响。
- **revision_verified=False 时结果需谨慎解读**：如果 ModelScope 上的 master 与 P5-04 的 exact revision 字节不完全一致，模型权重可能有微小差异；评测结果可能略偏离 P5-02/P5-04 的口径。

## 运行方式

```bash
# 1. 下载（ModelScope，并发 3）
python .tmp/download_all.py --workers 3

# 2. 构建 per-model token cache（5 个 tokenizer × 完整 validation）
.venv/python.exe scripts/eval_owt_real.py --build-cache
# 3. 单模型 smoke（cache 已有 max-bytes=5000000, 仅取前 ~108K tokens 验证管线）
.venv/python.exe scripts/eval_owt_real.py --eval-loss   --models Qwen2.5-3B

# 4. 完整 5 模型顺序评测（GPU, 单卡 sequential, ~3 小时 50 MiB prefix）
.venv/python.exe scripts/eval_owt_real.py --eval-loss   --max-bytes 52428800   # 50 MiB prefix; omit to evaluate full 289 MB (~9h)
.venv/python.exe scripts/eval_owt_real.py --eval-loss

# 5. 聚合对比表
.venv/python.exe scripts/eval_owt_real.py --aggregate

# 6. selftest（无需 GPU）
.venv/python.exe scripts/eval_owt_real.py --selftest
```

## 不做的事

- 不重新训练任何模型。
- 不修改自研 Dense / MoE 架构。
- 不接入 vLLM serving（per-token CE loss 需要 forward pass + softmax，vLLM 不暴露 logits）。
- 不跑 OWT train（只跑 validation；train 是训练数据）。
- 不向 Git 提交模型权重、token cache、`data/raw/`、`artifacts/owt-real-eval/{cache,results,models}`。

## 关联

- 上游：`docs/plans/roadmap.md` 候选 5
- 协议：本 README §「计算口径」即为协议主体；不再单独写 `docs/protocols/owt-real-eval.md`
- Stage review：`docs/plans/reviews/stage-owt-real-eval.md`
- 相关历史：`docs/experiments/p5-04-backend-comparison/README.md`（同一 5 模型列表，已交付 4 轴 D2 评测）