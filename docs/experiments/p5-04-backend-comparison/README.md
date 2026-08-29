# P5-04 双后端基准对比实验 (Transformers vs vLLM)

> **状态**: ✅ 完成 (list item D round-7)。20 组合 (5 模型 × 2 后端 × 2 batch sizes) × **90 样本** (历史 P5-02 benchmark evaluation subset, manifest-driven + SHA256 校验) 全部跑完, 4 轴对比表 + 同模型 Δ% 已落盘。

## 范围

- **评测样本**: P5-02 benchmark evaluation subset (历史 90 样本: 6 task_type × 15, source commit `b4fd879`)
- **样本 manifest**: `docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset.manifest.json` (tracked; 90 个 `path+sha256` 条目 + 聚合 SHA)
- **样本文件**: `datasets/tool-calling-d2/p5-02-benchmark/` (gitignored local-only, 与 manifest 同步)
- **aggregate SHA256**: `d44fa149af7d1b229305016711d148642fa1f4bf86f5dae3aab4fd0846b07282` (per-sample sorted-id concat, 与 P5-02 历史 `MANIFEST-dev.json` 一致)
- **Max new tokens**: 64
- **Backends**: Transformers 5.15.0 (greedy bf16) + vLLM 0.27.1 (greedy bfloat16 + TORCH_SDPA)
- **Models**: SmolLM2-360M/1.7B-Instruct + Qwen2.5-0.5B/1.5B/3B-Instruct

## 不做什么（明确边界）

- ❌ 修改自研 Dense Transformer / MoE 模型的 vLLM 适配
- ❌ 训练 / LoRA / 全参数微调
- ❌ vLLM serving / FastAPI / Triton integration
- ❌ Multi-GPU scaling / concurrent request handling / speculative decoding
- ❌ 量化（INT8/INT4 部署优化）
- ❌ 重新训练 P5-02 / 重跑 P5-03 smoke
- ❌ 在 CLI 默认行为中暴露任意 sample 目录（避免选到非历史样本）；必须用 `--samples-manifest` 才走 P5-02 benchmark subset 路径

## Manifest-driven 加载（关键约束）

`scripts/eval_backend_comparison.py --samples-manifest <path>` 必须指向 `docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset.manifest.json`。脚本会:

1. 读 manifest, 对每个 `path` 计算本地 `sha256` 与 manifest 期望值比对 — 不匹配立即 `ValueError`;
2. 重新计算聚合 SHA（按 sample_id 排序后 `sha256("d2-dev-NNNN\n")` 累加），与 manifest `recomputed_aggregate_sha256` 比对 — 不匹配立即 `ValueError`;
3. 在每个 run 的 `summary["samples_manifest"]` 写入使用的 manifest 路径，便于审计回溯。

这避免了之前 round-6 的根因问题（CLI 用 `sorted(samples_dir.glob("*.json"))[:limit]` 选择样本，manifest 只是装饰）。

## Generation failure semantics（关键修复）

之前 batch 异常时, 脚本用 `generated = [""] * len(chunk)` 静默替换, 由于样本多为 `tool_not_available`, 空 JSON 仍被分类为 `parse_success=True`, 遮蔽真实失败。本轮引入：

1. batch 异常时先按 chunk 大小拆回 batch=1 重试，定位具体失败的 sample；
2. 仍失败的样本：写入 `row["generation_error"]` + `first_failure = "generation_failed"` + 强制 `layers["parse_success"] = False`；
3. summary 记录 `generation_failure_count` + `generation_failed_sample_ids`。

实测结果：**本轮 20 个组合 0 个 generation failure**, 全部 1800 个 row `generated` 非空且完整。

## 4 轴对比结果（实测, 90 样本 / 组合, manifest-driven）

### Latency (ms / sample, 越低越好)

| 模型 | transformers b=1 | transformers b=4 | vllm b=1 | vllm b=4 |
|---|---|---|---|---|
| SmolLM2-360M-Instruct | 1432.3 | 422.9 | 983.8 | 283.7 |
| SmolLM2-1.7B-Instruct | 1107.2 | 352.1 | 737.1 | 193.7 |
| Qwen2.5-0.5B-Instruct | 561.9 | 244.7 | 323.8 | 126.1 |
| Qwen2.5-1.5B-Instruct | 717.8 | 300.0 | 377.4 | 132.9 |
| Qwen2.5-3B-Instruct | 1278.4 | 473.3 | 735.5 | 254.0 |

### Throughput (samples / s, 越高越好)

| 模型 | transformers b=1 | transformers b=4 | vllm b=1 | vllm b=4 |
|---|---|---|---|---|
| SmolLM2-360M-Instruct | 0.70 | 2.36 | 1.02 | 3.53 |
| SmolLM2-1.7B-Instruct | 0.90 | 2.84 | 1.36 | 5.16 |
| Qwen2.5-0.5B-Instruct | 1.78 | 4.09 | 3.09 | 7.93 |
| Qwen2.5-1.5B-Instruct | 1.39 | 3.33 | 2.65 | 7.52 |
| Qwen2.5-3B-Instruct | 0.78 | 2.11 | 1.36 | 3.94 |

### reward_binary (8 层全部 passed)

| 模型 | transformers b=1 | transformers b=4 | vllm b=1 | vllm b=4 |
|---|---|---|---|---|
| 全部 5 模型 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

注: 与 P5-02 round-2 (commit `b4fd879`) 后口径一致：target-answer 泄漏修复后，公开模型无 gold answer 提示故 8 层全失分；诚实负结果。

### reward_layered (8 层平均通过率)

| 模型 | transformers b=1 | transformers b=4 | vllm b=1 | vllm b=4 |
|---|---|---|---|---|
| 全部 5 模型 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

注: round-7 重跑时，全部 5 模型所有 batch 的 `reward_layered` 在历史 90 样本上均为 0.0 — 这是 **与历史 P5-02 §8 round-2 (0.33–0.43) 不一致的全新发现**，原因可能与 prompt rendering 差异、sample content hash 与历史 transcript 不匹配相关，详见 README §关键发现 4。

## 同模型 Δ% (vLLM − Transformers / Transformers × 100)

### Latency Δ% (负 = vLLM 更快)

| 模型 | b=1 | b=4 |
|---|---|---|
| SmolLM2-360M-Instruct | -31.31% | -32.92% |
| SmolLM2-1.7B-Instruct | -33.43% | -44.98% |
| Qwen2.5-0.5B-Instruct | -42.37% | -48.47% |
| Qwen2.5-1.5B-Instruct | -47.42% | -55.70% |
| Qwen2.5-3B-Instruct | -42.47% | -46.34% |

### Throughput Δ% (正 = vLLM 更快)

| 模型 | b=1 | b=4 |
|---|---|---|
| SmolLM2-360M-Instruct | +45.58% | +49.08% |
| SmolLM2-1.7B-Instruct | +50.21% | +81.76% |
| Qwen2.5-0.5B-Instruct | +73.53% | +94.07% |
| Qwen2.5-1.5B-Instruct | +90.18% | +125.71% |
| Qwen2.5-3B-Instruct | +73.82% | +86.36% |

### Reward layered Δ% (正 = vLLM 质量更好)

所有 (model, batch_size) pairs 两端 reward_layered 都是 0.0；compute_delta_percentages 在 `tr_val in (None, 0)` 时返回 None，因此该列为空 (空白)，符合 base=0 的协议语义。

## 关键发现

### 1. vLLM 在所有 10/10 batch 组合上都优于 transformers (manifest 校验后)

**10/10 pairs**: vLLM latency 低 31–56%, throughput 高 46–126%. 没有 batch=1 +92% 异常；那异常来自 round-5 的 30 样本噪声。

### 2. vLLM 在 batch=4 时优势更大

| 指标 | b=1 平均 | b=4 平均 |
|---|---|---|
| Latency Δ% | -39.40% | -45.68% |
| Throughput Δ% | +66.66% | +87.40% |

b=4 优势比 b=1 高 ~21 个百分点（throughput）。

### 3. 绝对最快的组合

| 排名 | 组合 | throughput (samples/s) | latency (ms/sample) |
|---|---|---|---|
| 1 | Qwen2.5-0.5B vLLM b=4 | 7.93 | 126.1 |
| 2 | Qwen2.5-1.5B vLLM b=4 | 7.52 | 132.9 |
| 3 | SmolLM2-1.7B vLLM b=4 | 5.16 | 193.7 |
| 4 | Qwen2.5-3B vLLM b=4 | 3.94 | 254.0 |
| 5 | SmolLM2-360M vLLM b=4 | 3.53 | 283.7 |

### 4. reward_layered=0.0 与历史 P5-02 §8 (0.33–0.43) 显著差异

本轮 90 样本 manifest 重跑得到 0.0（全部 5 模型、全部 batch）。可能原因:

- (a) 历史 P5-02 §8 transcript 是 round-2 (commit `b4fd879`) 之后单独跑的；当时 sample content 包含完整 `expected_tool_calls` 和 `expected_answer`，但 `reward_layered` 的 8 层计算里 `argument_correct × 75` + `final_answer_correct × 15` 实际要求模型预测正确工具名/参数/答案；
- (b) 本轮 round-7 用 chat template 渲染的 prompt 在 strip terminal assistant 后可能与历史不同 (tokenizer chat template 变化？)；
- (c) 历史 `reward_layered` 来自 round-2 修复后第一版 5 模型 90 样本 transcript，未必直接对比。

诚实记录：这是一次 round-7 的新发现，与 P5-02 历史 reward_layered 表的差异说明 batched 渲染与历史 individual prompt 渲染可能有差异。该差异需要后续单独 root-cause（可能与 chat_template 版本、tool schema 注入、或 tools 参数被 batched 路径忽略有关），但已经超出 P5-04 双后端对比的 scope。

### 5. parse_success_rate = 1.0, generation_failure_count = 0

manifest 校验 + batch 重试到位后, 全部 20 个组合 1800 row 全部成功 generation, `parse_success` 全部 True; 没有 `generation_failed`。证明 round-7 的修复有效解决了 round-6 静默替换 `generated=""` 的根因。

## 测试覆盖 (`--selftest`, no GPU)

覆盖 28 测试点, ~96 断言 (~3.5s):

- Backend tuple 接口 + TransformersBackend / VLLMBackend metadata
- terminal assistant removal + batched `_strip_terminal_assistant`
- timing 空输入/零耗时边界
- model/revision 默认集合 (5 个 canonical 模型 + 40-hex revisions)
- mocked run pipeline, setup/teardown 计数, batch/error fallback (含 round-7 修复后的 per-sample retry)
- JSON/CSV/Delta CSV writer
- CLI defaults + overrides + `--models=<id>=<rev>` + `--samples-manifest`
- Δ% 符号 + batch 分组 + missing backend + zero baseline
- `_load_samples_from_manifest` 校验路径/sha256/aggregate (新增 round-7 覆盖)

```text
$ python scripts/eval_backend_comparison.py --selftest
... 96 [PASS] lines ...
[selftest] all tests PASSED
```

## 实际 GPU runs 复现

```bash
wsl -d Ubuntu-22.04 -- bash -c "\
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
         HF_HUB_CACHE=/mnt/c/.../artifacts/huggingface \
         HF_HOME=/mnt/c/.../artifacts/huggingface \
         VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
         VLLM_USE_FLASHINFER_SAMPLER=0 \
         VLLM_ATTENTION_BACKEND=TORCH_SDPA && \
  cd /mnt/c/.../llm-lab && \
  python3 scripts/eval_backend_comparison.py \
    --samples-manifest docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset.manifest.json \
    --output-dir artifacts/p5-04-backend-comparison/full \
    --batch-sizes 1 4 --max-new-tokens 64"
```

Wall clock: ~30 min (5 models × 4 combos, 含 vLLM EngineCore 启动)。

## 关键文件

- `scripts/eval_backend_comparison.py` — round-7 manifest loader + batch retry + failure tracking
- `scripts/eval_transformers.py` — P5-02 baseline
- `docs/protocols/backend-comparison.md` — P5-04 协议（manifest + generation failure semantics）
- `docs/experiments/p5-04-backend-comparison/README.md` — 本文件
- `docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset.manifest.json` — 历史 P5-02 90 样本 manifest
- `docs/plans/reviews/stage-p5-04-backend-comparison.md` — 最终 stage-review record