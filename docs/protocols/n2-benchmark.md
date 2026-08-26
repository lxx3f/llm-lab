# N2 runs a short configured training protocol before latency measurement.
# The reported train/validation losses are from that executed protocol, not a
# standalone random-initialization forward diagnostic.
# The N2 smoke uses a deliberately short decode sample; routing values are linkage checks only.

## 固定条件

N2 使用单 seed、单 batch/sequence 配置和 PyTorch eager 路径。每次 benchmark 先按 `token_budget` 执行短训练协议，再在训练后模型上测量 inference latency：

```text
seed=42
batch_size=2
sequence_length=32
token_budget=128
warmup_steps=1
measured_steps=2
collect_stats=false（latency benchmark）
```

Dense 与 MoE 必须绑定相同 tokenizer、train/validation token cache、optimizer、scheduler、AMP 和 gradient accumulation。benchmark 会实际加载并校验两个 token cache；N2 result JSON 记录 metadata hash 和 token-file hash。这些 binding 在结果中显式记录，完整环境/commit/config 元数据归 N3。

## 协议

### A：相同 total parameters

以 Dense `d_ff=D` 为目标，在整数候选中选择 MoE expert `d_ff`，最小化实际 total parameter 差值。由于 MoE router 参数和整数 hidden width，允许非零 delta；结果必须记录 target、delta 和 relative error。

当前 smoke：Dense `d_ff=64`，MoE expert `d_ff=16`。

### B：相同 active parameters

以 Dense active parameters 为目标，当前协议令 Dense `d_ff=D_act`、MoE expert `d_ff=D_act`，再报告实际 Top-1 active parameter delta。该协议不将非零 router 差值宣称为严格相等。

当前 smoke：Dense `d_ff=64`，MoE expert `d_ff=64`。

## 测量口径

- train loss：短训练协议最后一个 optimizer update 的 LM loss；
- validation loss：训练完成后从 validation cache 执行的 validation batch 平均 LM loss；
- optimizer、scheduler、AMP 和 gradient accumulation 在该短训练协议中实际执行；
- prefill：固定输入 token batch 的 forward mean latency 和 tokens/s；
- decode：先建立同一 KV cache，再测一个 token 增量 forward 的 mean latency 和 tokens/s；
- GPU 使用 `torch.cuda.max_memory_allocated` 记录 phase peak memory；CPU 环境显式记录 null；
- benchmark 不采集 routing stats，避免 CPU `.item()`/`.cpu()` 同步影响 latency；
- MoE benchmark 显式传入 prefill `capacity_factor=1.0`、decode `capacity_factor=2.0`。

## Routing stats

`scripts/run_n2_routing_stats.py` 独立使用 `collect_stats=true`，分别运行 prefill/decode 并记录：

- assigned/kept token per expert；
- dropped token ratio；
- load imbalance；
- per-layer stats；
- capacity factor 和 capacity。

## 结果和限制

- `schemas/n2_benchmark_result.schema.json` 校验四个 Dense/MoE benchmark JSON；
- `schemas/n2_routing_stats.schema.json` 校验 routing stats JSON；
- smoke 结果只证明协议、绑定和指标落盘正确，不代表正式模型质量、正式性能、多 seed 统计或置信区间；
- N2 不包含分布式训练、aux/capacity sweep、精确 FLOPs、profiler、N3 元数据统一和 N4 长训练曲线。
