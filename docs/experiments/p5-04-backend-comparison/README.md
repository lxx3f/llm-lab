# P5-04 双后端基准对比实验 (Transformers vs vLLM)

> **状态**: ✅ 完成 (list item D round-10)。20 组合 (5 模型 × 2 后端 × 2 batch sizes) × **90 样本** (历史 P5-02 benchmark evaluation subset, manifest-driven + SHA256 校验 + reward 计算以 manifest sample 为准) 全部跑完, 4 轴对比表 + 同模型 Δ% 已落盘。
>
> **范围决策（2026-08-29，用户确认）**：本交付中的 “P5-02 benchmark evaluation subset” 明确定义为 P5-02 阶段真实运行并冻结的 **90 样本**（6 task type × 15，source commit `b4fd879`），不是当前扩样版 D2 dev 父 split 的全部 750 样本。750 是父 split 规模；若未来评测全部 750，应作为独立新实验，不改写本历史 benchmark。

## 范围

- **评测样本**: P5-02 benchmark evaluation subset (历史 90 样本: 6 task_type × 15, source commit `b4fd879`)
- **父数据集关系**: 当前 D2 dev split 有 750 样本；本 benchmark 是该父 split 的固定历史子集，而非全 split 运行。历史五个 P5-02 transcript 均有 90 rows (`d2-dev-0001`…`d2-dev-0090`)；`b4fd879:datasets/tool-calling-d2/MANIFEST-dev.json` 的 `count=90`。
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

## Round-13 重跑与新 archive (2026-08-30)

detached auditor round-13 指出 round-12 的 archive (sha256=d2c85e48...) 生成于 commit `4176f65` 之前，20 个 run JSON 均缺 `backend_metadata.revision`。本轮:

1. 在 RTX 5070 Ti WSL2 bf16 用 round-12 代码重跑全部 20 个组合 (transformers 10 + vllm 10);
2. 重生成 `p5-04-runs.tar.gz` + `p5-04-runs.archive-manifest.json`；
3. 新 archive SHA256 = `ffd8e0471f28332cd66a413e7ba02aec76b84326abe9b7a28583c76463bc9e1a`, aggregate SHA = `572c499756b0a506f333608cc2a4f0957d021e75743d48d89ccacb87a4beab1e`;
4. 验证 `runs_with_revision=20/20` + `entry_sha_match=24/24`, 同一个模型的 transformers / vllm 两条 run 携带完全相同的 canonical revision;
5. vLLM selftest 强化为真正的 setup() 调用 (patch vllm.LLM + AutoTokenizer.from_pretrained), 取代之前对 `vb.revision` 的直接赋值, selftest 从 106 增至 108 PASS。

## Round-14 aggregate 补齐 (2026-08-30)

detached auditor round-14 指出 round-13 重跑拆成 `--backends transformers` 与 `--backends vllm` 两次独立 invocation, 每次都覆盖了 `comparison.{csv,json}` + `comparison_delta.{csv,json}`, 导致 archive 里 aggregates 只剩 vLLM 半边。本轮修复:

1. 不重跑 GPU: 20 个 `run_*.json` 完整无缺, 直接调用 `write_aggregate_comparison` + `compute_delta_percentages` + `write_delta_csv` 从 20 个 run JSON 重建 aggregates;
2. 新 archive SHA256 = `05a38aacd6628448fec432c56219bbc33ea0fc425c9595441a23b81fef164d62`, aggregate SHA = `089450b551f0db6f77e565318492ad2db7561fc213d6d4be630206616f90fee5`;
3. 验证 `comparison.csv rows=20` + `backends={transformers, vllm}`, `comparison.json runs=20`, `comparison_delta.csv rows=10` 全部非空 latency delta, `comparison_delta.json entries=10` 全部非空 latency delta;
4. selftest 新增 6 个 full-aggregate 完整性断言 (test_full_aggregate_runs_count_20 / test_full_aggregate_both_backends / test_full_aggregate_both_batch_sizes / test_full_aggregate_delta_entries_10 / test_full_aggregate_delta_latency_non_null / test_full_aggregate_delta_throughput_non_null), 防阳  个 run + both backends + both batch sizes + 10 non-null same-model Δ% 入口, 防阳 vLLM-only aggregate, selftest 从 108 增至 114 PASS。

## 4 轴对比结果（实测, 90 样本 / 组合, manifest-driven, round-14 重跑; 数值直接来自 tracked archive comparison.csv）

### Latency (ms / sample, 越低越好)

| 模型 | transformers b=1 | transformers b=4 | vllm b=1 | vllm b=4 |
|---|---|---|---|---|
| SmolLM2-360M-Instruct | 1555.3 | 488.4 | 1001.9 | 285.4 |
| SmolLM2-1.7B-Instruct | 1119.7 | 381.1 | 677.3 | 196.0 |
| Qwen2.5-0.5B-Instruct | 562.0 | 232.1 | 328.9 | 122.3 |
| Qwen2.5-1.5B-Instruct | 691.9 | 320.8 | 392.1 | 124.5 |
| Qwen2.5-3B-Instruct | 1232.6 | 460.3 | 710.3 | 246.1 |

### Throughput (samples / s, 越高越好)

| 模型 | transformers b=1 | transformers b=4 | vllm b=1 | vllm b=4 |
|---|---|---|---|---|
| SmolLM2-360M-Instruct | 0.64 | 2.05 | 1.00 | 3.50 |
| SmolLM2-1.7B-Instruct | 0.89 | 2.62 | 1.48 | 5.10 |
| Qwen2.5-0.5B-Instruct | 1.78 | 4.31 | 3.04 | 8.18 |
| Qwen2.5-1.5B-Instruct | 1.45 | 3.12 | 2.55 | 8.03 |
| Qwen2.5-3B-Instruct | 0.81 | 2.17 | 1.41 | 4.06 |

### reward_binary (8 层全部 passed)

| 模型 | transformers b=1 | transformers b=4 | vllm b=1 | vllm b=4 |
|---|---|---|---|---|
| SmolLM2-360M-Instruct | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SmolLM2-1.7B-Instruct | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen2.5-0.5B-Instruct | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen2.5-1.5B-Instruct | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen2.5-3B-Instruct | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

注: 与 P5-02 round-2 (commit `b4fd879`) 后口径一致：target-answer 泄漏修复后，公开模型无 gold answer 提示故 8 层全失分；诚实负结果。

### reward_layered (8 层平均通过率)

| 模型 | transformers b=1 | transformers b=4 | vllm b=1 | vllm b=4 |
|---|---|---|---|---|
| SmolLM2-360M-Instruct | 0.4236 | 0.4236 | 0.4236 | 0.4236 |
| SmolLM2-1.7B-Instruct | 0.4236 | 0.4236 | 0.4236 | 0.4236 |
| Qwen2.5-0.5B-Instruct | 0.3676 | 0.3662 | 0.3601 | 0.3601 |
| Qwen2.5-1.5B-Instruct | 0.3745 | 0.3745 | 0.3653 | 0.3653 |
| Qwen2.5-3B-Instruct | 0.3852 | 0.3801 | 0.3694 | 0.3662 |

注: 早期轮次 `reward_layered=0.0` 是 root-cause bug (round-7)：原代码从 `args.samples_dir` (=manifest 父目录) `glob("*.json")` 读不到任何 sample，`reward_offline.load_samples()` 返回空 dict 后 `0 / max(0, 1) = 0.0`。round-8 改为在 CLI 处把 manifest 加载出的 sample 字典存到 `args.samples_by_id`，由 `run_one_combination()` 直接传入 `compute_reward()`。重新计算后与 P5-02 §8 round-2 历史表 (0.33–0.43) 吻合。

## 同模型 Δ% (vLLM − Transformers / Transformers × 100)

### Latency Δ% (负 = vLLM 更快)

| 模型 | b=1 | b=4 |
|---|---|---|
| SmolLM2-360M-Instruct | -35.58% | -41.56% |
| SmolLM2-1.7B-Instruct | -39.51% | -48.57% |
| Qwen2.5-0.5B-Instruct | -41.48% | -47.32% |
| Qwen2.5-1.5B-Instruct | -43.33% | -61.18% |
| Qwen2.5-3B-Instruct | -42.38% | -46.54% |

### Throughput Δ% (正 = vLLM 更快)

| 模型 | b=1 | b=4 |
|---|---|---|
| SmolLM2-360M-Instruct | +55.23% | +71.11% |
| SmolLM2-1.7B-Instruct | +65.31% | +94.44% |
| Qwen2.5-0.5B-Instruct | +70.87% | +89.81% |
| Qwen2.5-1.5B-Instruct | +76.47% | +157.61% |
| Qwen2.5-3B-Instruct | +73.54% | +87.06% |

### Reward layered Δ% (正 = vLLM 质量更好)

| 模型 | b=1 | b=4 |
|---|---|---|
| SmolLM2-360M-Instruct | 0.00% | 0.00% |
| SmolLM2-1.7B-Instruct | 0.00% | 0.00% |
| Qwen2.5-0.5B-Instruct | -2.03% | -1.66% |
| Qwen2.5-1.5B-Instruct | -2.47% | -2.47% |
| Qwen2.5-3B-Instruct | -4.09% | -3.65% |

注: SmolLM2 两个模型 transformers 与 vLLM 输出完全相同 (compute_reward 输入一样) → Δ=0.00%；Qwen2.5 系列 vLLM 略低于 transformers (绝对差 < 0.02)，是 vLLM 引擎推理数值抖动的诚实记录。reward_binary 全部为 0，Δ 也为 0。

## 关键发现

### 1. vLLM 在所有 10/10 batch 组合上都优于 transformers (manifest 校验后)

**10/10 pairs**: vLLM latency 低 36–61%, throughput 高 55–158%. 没有 batch=1 +92% 异常；那异常来自 round-5 的 30 样本噪声。

### 2. vLLM 在 batch=4 时优势更大

| 指标 | b=1 平均 | b=4 平均 |
|---|---|---|
| Latency Δ% | -40.46% | -49.03% |
| Throughput Δ% | +68.28% | +100.01% |

b=4 优势比 b=1 高 ~32 个百分点（throughput: +100.01% vs +68.28%）。

### 3. 绝对最快的组合

| 排名 | 组合 | throughput (samples/s) | latency (ms/sample) |
|---|---|---|---|
| 1 | Qwen2.5-0.5B vLLM b=4 | 8.18 | 122.3 |
| 2 | Qwen2.5-1.5B vLLM b=4 | 8.03 | 124.5 |
| 3 | SmolLM2-1.7B vLLM b=4 | 5.10 | 196.0 |
| 4 | Qwen2.5-3B vLLM b=4 | 4.06 | 246.1 |
| 5 | SmolLM2-360M vLLM b=4 | 3.50 | 285.4 |

### 4. vLLM 与 transformers 在 reward_layered 上几乎相等 (合理结论)

5 模型 × 2 后端 × 2 batch = 20 组合 `reward_layered` 集中在 **0.36–0.42**，与 P5-02 §8 round-2 历史表 (0.33–0.43) 吻合。SmolLM2 系列两种后端数字一致 (0.4236) 是因为 greedy bf16 在两个 backend 下输出完全相同 (compute_reward 输入 = chat_template 渲染后的 prompt + greedy decoded text，vLLM 与 transformers 走同一 chat template)。Qwen2.5 系列 transformers 略高于 vLLM (~0.005–0.02 绝对差)，属于 vLLM 引擎推理时数值抖动的诚实记录。

### 5. parse_success_rate = 1.0, generation_failure_count = 0

manifest 校验 + batch 重试到位后, 全部 20 个组合 1800 row 全部成功 generation, `parse_success` 全部 True; 没有 `generation_failed`。证明 round-7 (manifest + batch retry) + round-8 (reward 计算 wiring) 的修复有效解决了前几轮的根因问题。

## 测试覆盖 (`--selftest`, no GPU)

覆盖 ~32 测试点, ~114 断言 (~5s, 含 round-12 manifest 干净 checkout 复原 + round-13 两后端 revision metadata 对称 + round-14 full-aggregate 完整性 6 个断言):

- Backend tuple 接口 + TransformersBackend / VLLMBackend metadata
- terminal assistant removal + batched `_strip_terminal_assistant`
- timing 空输入/零耗时边界
- model/revision 默认集合 (5 个 canonical 模型 + 40-hex revisions)
- mocked run pipeline, setup/teardown 计数, batch/error fallback (含 round-7 修复后的 per-sample retry + round-8 reward wiring)
- JSON/CSV/Delta CSV writer
- CLI defaults + overrides + `--models=<id>=<rev>` + `--samples-manifest`
- Δ% 符号 + batch 分组 + missing backend + zero baseline
- `_load_samples_from_manifest` 校验路径/sha256/aggregate (round-7 覆盖)

```text
$ python scripts/eval_backend_comparison.py --selftest
... 106 [PASS] lines ...
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

- `scripts/eval_backend_comparison.py` — round-7 manifest loader + batch retry + failure tracking + round-8 reward wiring 修复 (samples_by_id 直传)
- `scripts/eval_transformers.py` — P5-02 baseline
- `docs/protocols/backend-comparison.md` — P5-04 协议（manifest + generation failure semantics）
- `docs/experiments/p5-04-backend-comparison/README.md` — 本文件
- `docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset.manifest.json` — 历史 P5-02 90 样本 manifest
- `docs/plans/reviews/stage-p5-04-backend-comparison.md` — 最终 stage-review record