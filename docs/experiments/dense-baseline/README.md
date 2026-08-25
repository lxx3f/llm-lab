# Dense Transformer 基线记录

## 实现范围

当前基线是用于架构实验的最小 decoder-only Transformer，包含：

- token embedding；
- RMSNorm；
- RoPE；
- causal self-attention；
- SwiGLU FFN；
- tied input embedding / LM head；
- causal language modeling loss；
- KV Cache；
- greedy / temperature generation。

代码：`architecture_lab/models/dense_transformer.py`

## 固定配置

配置文件：`configs/dense_baseline.example.yaml`

| 配置 | 值 |
|---|---:|
| vocab size | 256 |
| max sequence length | 128 |
| d_model | 128 |
| attention heads | 4 |
| layers | 2 |
| FFN hidden size | 512 |
| dropout | 0.0 |
| seed | 42 |
| benchmark device | CUDA (`sm_120`) |

该配置是小规模 smoke baseline，不代表最终训练配置。项目当前使用 `.venv` 环境中的 PyTorch `2.10.0+cu128`，已验证支持 RTX 5070 Ti Laptop GPU 的 `sm_120`。

## 正确性测试

测试文件：`architecture_lab/tests/test_dense_transformer.py`

覆盖：

- forward 输出 shape；
- loss 有限且可反向传播；
- full forward 与逐 token incremental decode 一致；
- KV Cache 的 shape 和长度；
- embedding 与 LM head 权重共享；
- 非法模型配置拒绝。

运行：

```bash
.venv/python.exe -m unittest discover -s architecture_lab/tests -p "test_*.py" -v
.venv/python.exe scripts/run_dense_baseline.py
```

## Smoke benchmark

运行：

```bash
.venv/python.exe scripts/run_dense_baseline.py
```

默认输出：

```text
docs/experiments/dense-baseline/smoke-result.json
```

记录内容包括：

- 参数量；
- forward latency；
- 单 token decode latency；
- forward/decode tokens per second；
- 随机输入上的 loss；
- device、dtype、配置和随机种子；
- CUDA 环境下的峰值显存。

随机输入上的 loss 只用于确认计算链路可运行，不能作为模型效果结论。

## 当前 smoke 结果

结果文件：`docs/experiments/dense-baseline/smoke-result.json`

当前环境：Python 3.12.13、PyTorch 2.10.0+cu128、RTX 5070 Ti Laptop GPU、CUDA、float32。一次已记录运行得到：

- 参数量：557,696；
- batch size：4；
- sequence length：64；
- forward latency：约 3.80 ms；
- 单 token decode latency：约 10.54 ms；
- forward tokens per second：约 67,399；
- decode tokens per second：约 380；
- 随机输入 loss：约 126.57；
- 峰值显存：约 13.41 MB。

这些数字只用于确认 benchmark 能稳定产出结构化结果。它们会受硬件、线程、系统负载和 PyTorch 版本影响，不能作为正式模型性能结论。
