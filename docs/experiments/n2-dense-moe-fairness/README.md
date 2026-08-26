# N2 Dense/MoE 公平对比实验

## 范围

N2 在同一 OWT BPE tokenizer、同一 train cache、同一 seed、batch、sequence length 和 token budget 下，运行两套 Dense/MoE Top-1 对比协议：

- **协议 A：相同 total parameters**。Dense `d_ff=64`，MoE 4 experts、每 expert `d_ff=16`；实际 total parameter 差值显式记录。
- **协议 B：相同 active parameters**。Dense `d_ff=64`，MoE 每 expert `d_ff=64`；实际 Top-1 active parameter 差值显式记录。

N2 只运行单 seed、单 batch/sequence 配置和 PyTorch eager mean latency smoke，不产生正式模型质量、多 seed统计或置信区间结论。当前随机初始化 smoke 的 loss 数值可能显著偏离 `log(vocab_size)`，不应解释为模型质量；loss 仅作为 forward 链路诊断。decode routing stats 仅覆盖单 token，用于链路验证，不用于均衡性结论。

## 配置和入口

配置：

```text
configs/n2_benchmark.example.yaml
```

Dense/MoE benchmark：

```bash
.venv/python.exe scripts/run_n2_benchmark.py \
  --config configs/n2_benchmark.example.yaml \
  --protocol A --architecture DenseTransformer \
  --output artifacts/n2-a-dense.json

.venv/python.exe scripts/run_n2_benchmark.py \
  --config configs/n2_benchmark.example.yaml \
  --protocol A --architecture MoETransformer \
  --output artifacts/n2-a-moe.json

.venv/python.exe scripts/run_n2_benchmark.py \
  --config configs/n2_benchmark.example.yaml \
  --protocol B --architecture DenseTransformer \
  --output artifacts/n2-b-dense.json

.venv/python.exe scripts/run_n2_benchmark.py \
  --config configs/n2_benchmark.example.yaml \
  --protocol B --architecture MoETransformer \
  --output artifacts/n2-b-moe.json
```

routing stats 独立入口：

```bash
.venv/python.exe scripts/run_n2_routing_stats.py \
  --config configs/n2_benchmark.example.yaml \
  --architecture MoETransformer \
  --output artifacts/n2-moe-routing-stats.json
```

## 统一结果字段

Dense/MoE 结果由 `schemas/n2_benchmark_result.schema.json` 校验，记录：

- 协议和参数对齐 basis、target、delta、relative error；
- total / active parameters 和 active parameter definition；
- tokenizer/cache hash binding、seed、batch、sequence、token budget；
- optimizer、scheduler、AMP 配置；
- train loss、aux loss、total loss；
- prefill/decode mean latency、throughput、peak memory；
- `collect_stats=false` 和 MoE prefill/decode capacity factor。

routing 结果由 `schemas/n2_routing_stats.schema.json` 校验，独立记录：

- `collect_stats=true`；
- expert load；
- kept/assigned token；
- dropped token ratio；
- load imbalance；
- prefill/decode 两个 phase 的 per-layer stats。

latency benchmark 不采集 CPU routing stats，避免 `.cpu()`/`.item()` 同步污染测量；routing stats 另行执行。

## Smoke 结果

| 协议 | 模型 | total params | active params | 对齐差值 |
|---|---|---:|---:|---:|
| A | Dense | 581,952 | 581,952 | 0 |
| A | MoE (`d_ff=16`) | 582,464 | 564,032 | 512 total |
| B | Dense | 581,952 | 581,952 | 0 |
| B | MoE (`d_ff=64`) | 656,192 | 582,464 | 512 active |

正式 smoke 配置使用：

```text
seed=42
batch_size=2
sequence_length=32
measured_steps=2
warmup_steps=1
token_budget=128
```

这些结果用于验证协议、绑定和指标落盘，不代表 Dense/MoE 的正式模型质量或性能结论。

## 边界

N2 不包含：

- 多 seed、均值/标准差、p50/p95 和置信区间；
- aux loss/capacity sweep；
- 分布式训练和 Expert Parallel；
- N3 的完整 git/config/environment 元数据统一；
- N4 Dense 正式长训练曲线；
- FLOPs 精确计数、kernel profiler、vLLM/Transformers backend；
- 正式模型质量结论。
