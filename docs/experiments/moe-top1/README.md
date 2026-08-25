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

### Forward MVP 配置

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

### N1 训练 MVP 配置

配置文件：`configs/moe_training.example.yaml` / `configs/moe_training.owt-formal.example.yaml`

| 配置 | 值 |
|---|---:|
| vocab size | 8192 |
| max sequence length | 64 |
| d_model | 64 |
| attention heads | 4 |
| layers | 2 |
| FFN hidden size per expert | 64 |
| experts | 4 |
| capacity factor | 1.0 |
| auxiliary loss weight | 0.01 |
| seed | 42 |
| tokenizer | `owt-bpe/v0.2.0` |

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
- 参数量统计；

N1 训练闭环测试：`tests/test_moe_training.py`

覆盖：

- 独立 MoE result schema 校验；
- checkpoint/resume 与 MoE config binding；
- total / Top-1 active parameter 统计；
- 非法结果写入前拒绝。

运行：

```bash
.venv/Scripts/python.exe -m unittest architecture_lab.tests.test_moe_transformer -v
.venv/Scripts/python.exe -m unittest tests.test_moe_training -v
```

## MoE 训练闭环 smoke

1MiB 配置：

```bash
.venv/Scripts/python.exe scripts/train_moe.py \
  --config configs/moe_training.example.yaml \
  --output artifacts/moe-owt-mvp-result.json
```

正式 cache 配置：

```bash
.venv/Scripts/python.exe scripts/train_moe.py \
  --config configs/moe_training.owt-formal.example.yaml \
  --output artifacts/moe-owt-formal-cache-result.json
```

两次运行均为 100 optimizer-step smoke，使用 `collect_stats=false`，并将训练后的短 generation 记录为 prefill `capacity_factor=1.0`、decode `capacity_factor=2.0`。正式 cache smoke 只覆盖连续 token stream 前缀，不代表完整 OWT 训练质量。

正式 cache smoke 结果：

```text
step 50 validation lm_loss: 60.627878
step 100 validation lm_loss: 58.920591
step 100 validation aux_loss: 1.042519
step 100 validation total_loss: 58.931017
```

结果写入前通过 `schemas/moe_training_result.schema.json` 校验，checkpoint 与结果 JSON 位于被 Git 忽略的 `artifacts/`。

## 当前边界

本 MVP 暂不包含：

- Dense/MoE 公平 benchmark（roadmap N2）；
- routing stats 分析脚本和 latency 统计（N2）；
- aux loss / capacity sweep；
- Top-2 routing；
- expert parallel；
- 分布式 dispatch/combine；
- 严格 prefill/decode 等价性；
- 多 seed 正式实验和统一 benchmark 元数据（N3）。
