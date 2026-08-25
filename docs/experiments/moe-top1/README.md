# MoE Top-1 MVP 记录

## 实现范围

当前 MoE MVP 在 Dense Transformer 的 FFN 位置加入 Top-1 Router：

- Router 为线性层；
- 每个 token 选择一个概率最高的 expert；
- expert 使用 SwiGLU FFN；
- 支持 capacity factor；
- 超出 expert capacity 的 token 被 dropped；
- residual path 保留 dropped token 的输入；
- 记录 expert token 分配和 dropped token 统计；
- 提供 Switch-style load balancing auxiliary loss；
- 支持 KV Cache 和增量解码。

代码：`architecture_lab/models/moe_transformer.py`

## 固定配置

配置文件：`configs/moe_top1.example.yaml`

| 配置 | 值 |
|---|---:|
| vocab size | 256 |
| max sequence length | 128 |
| d_model | 128 |
| attention heads | 4 |
| layers | 2 |
| FFN hidden size | 512 |
| experts | 4 |
| capacity factor | 1.0 |
| auxiliary loss weight | 0.01 |
| seed | 42 |
| device | CUDA (`sm_120`) |

## Loss 定义

模型 forward 返回：

```python
logits, lm_loss, aux_loss = model(input_ids, labels=input_ids)
```

训练时总 loss 为：

```python
total_loss = lm_loss + aux_loss_weight * aux_loss
```

其中 auxiliary loss 使用：

```text
num_experts × sum(mean_router_probability × mean_assignment_fraction)
```

用于鼓励 router 在 experts 之间保持更均衡的负载。

## 正确性测试

测试文件：`architecture_lab/tests/test_moe_transformer.py`

覆盖：

- Top-1 输出 shape；
- router 统计信息；
- capacity 限制；
- dropped token 统计；
- LM loss、auxiliary loss 和 total loss；
- router 与 expert 的 backward 梯度；
- full forward 与 incremental decode 一致；
- 参数量统计。

运行：

```bash
.venv/Scripts/python.exe -m unittest architecture_lab.tests.test_moe_transformer -v
```

## Smoke benchmark

运行：

```bash
.venv/Scripts/python.exe scripts/run_moe_top1.py
```

默认输出：

```text
docs/experiments/moe-top1/smoke-result.json
```

当前一次 CUDA 运行记录：

- 参数量：1,738,368；
- batch size：4；
- sequence length：64；
- forward latency：约 6.19 ms；
- forward tokens per second：约 41,343；
- LM loss：约 127.56；
- auxiliary loss：约 1.006；
- dropped token rate：各层约 6.64% 和 7.03%；
- 峰值显存：约 16.29 MB。

随机输入 loss 和该 smoke benchmark 只用于验证计算链路，不代表训练效果或架构性能结论。

## 当前边界

本 MVP 暂不包含：

- Top-2 routing；
- expert parallel；
- 分布式 dispatch/combine；
- router z-loss；
- 训练 loop 和 checkpoint；
- 专门的 overflow token 策略比较。

另外，capacity overflow 会影响 prefill 与 incremental decode 的严格等价性；当前测试使用充足 capacity 验证 decode 数值一致，capacity 行为由独立测试覆盖。
