# N4 Dense 训练曲线协议

## 目的

固定 Dense Transformer 在 OWT 正式 token cache 上**更长训练**的曲线契约：

- `train_losses` 序列：以 `log_interval` 步长采样 train loss
- `validation_losses` 序列：以 `validation_interval` 步长计算 val loss
- `curve_summary`：一次性摘要（first/last/min/delta）

不是模型质量评估；不是 perplexity 排名；不与 MoE 比较（MoE 单独走 N2 协议）。

## 范围

- Dense Transformer（`DenseTransformer`），MoE 走 N2 协议
- OWT 正式 token cache（512 MiB train / 64 MiB validation），OWT BPE tokenizer v0.2.0
- 不复用 N3 smoke config（`dense_training.owt-formal.example.yaml`，100-step）；N4 新 config（`dense_training.owt-formal-curve.example.yaml` 与 `…-medium.example.yaml`）

## Schema 契约

`schemas/dense_training_result.schema.json` v1.1：

```text
metrics.required = [
    "last_train_loss",       # float | null
    "validation_losses",     # dict[str(step), float]
    "train_losses",          # list[{step:int, loss:float, lr:float}]  ← v1.1 新增
    "curve_summary",         # dict 9 字段                              ← v1.1 新增
]
```

### train_losses 序列

| 字段 | 类型 | 说明 |
|---|---|---|
| `step` | int ≥ 1 | 采样 step（`log_interval` 边界） |
| `loss` | float | 该 step 的 train loss（micro-batch 平均） |
| `lr` | float | 该 step 末的 learning rate |

采样规则：每 `log_interval` 步一个采样点；5000 步 + `log_interval=50` → 100 个采样点。**不**每步记录（避免 5000 元素的 JSON 膨胀）。

### validation_losses 序列

`dict[str, float]`，key 为 step 字符串；value 为该 step 的 val loss。

### curve_summary 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `train_loss_first` | float \| null | `train_losses` 第一个采样点 |
| `train_loss_last` | float \| null | `train_losses` 最后一个采样点 |
| `val_loss_min` | float \| null | `validation_losses` 中最小 loss |
| `val_loss_min_step` | int \| null | `val_loss_min` 对应 step |
| `val_loss_last` | float \| null | 最后一个 val loss |
| `delta_train_loss` | float \| null | `train_loss_last - train_loss_first`（负值 = loss 下降） |
| `delta_val_loss` | float \| null | `val_loss_last - val_loss_min`（正值 = 从 min 反弹 = 可能过拟合） |
| `train_loss_sample_count` | int ≥ 0 | train 采样点数 |
| `val_loss_count` | int ≥ 0 | val 评估次数 |

## 配置建议

| 参数 | 推荐值 | 范围 | 理由 |
|---|---|---|---|
| `max_steps` | 5000 | 1000-10000 | OWT 1 epoch ≈ 1024 步；5000 步 ≈ 4.9 epoch，足以展示曲线 dynamics |
| `log_interval` | 50 | 10-100 | 5000 步 / 50 = 100 个采样点 |
| `validation_interval` | 200 | 100-500 | 5000 步 / 200 = 25 个 val 点；过密增加 val 耗时，过疏损失曲线细节 |
| `validation_batches` | 8 | 4-32 | validation cache 17M tokens，8 batches × 8 batch × 64 seq = 4096 tokens / 17M = 充分估计 |
| `batch_size` | 8 | 8-32 | 显存使用与 model size 相关；d_model=64 时 8/16/32 显存均在 0.1GB 量级 |
| `sequence_length` | 64 | 64 | 与 max_seq_len 匹配；64 是 cache 设计参数 |
| `warmup_steps` | 100 | 50-500 | max_steps 的 2-10%；warmup 后 loss 稳定下降 |
| `amp.dtype` | `bfloat16` | bfloat16/float16 | RTX 5070 Ti (12GB) 支持 bf16；fp16 需 GradScaler |

### 模型规模 sensitivity（RTX 5070 Ti 实测，2026-08-26）

| d_model | n_layers | d_ff | params | 单步 | 5000 步 + 25 val |
|---|---|---|---|---|---|
| 64 | 2 | 256 | 0.66M | 8.2ms | ~41s |
| 96 | 3 | 384 | 1.23M | 10.7ms | ~54s |
| 128 | 4 | 512 | 2.10M | 13.0ms | ~65s |
| 192 | 6 | 768 | 5.11M | 18.2ms | ~91s |

峰值显存均 < 0.2GB（12GB 上限的 1.5%）。

## 绘图协议

`scripts/plot_dense_curve.py`：

```bash
# 单一曲线
.venv/python.exe scripts/plot_dense_curve.py \
    --input artifacts/dense-owt-formal-curve-result.json \
    --output artifacts/dense-owt-formal-curve.png

# 双曲线 overlay
.venv/python.exe scripts/plot_dense_curve.py \
    --input artifacts/dense-owt-formal-curve-result.json \
    --input artifacts/dense-owt-formal-curve-medium-result.json \
    --output artifacts/dense-owt-formal-curve-overlay.png \
    --overlay
```

输出 PNG：
- 800×500，dpi=120
- 蓝线 = train_loss（左轴）；橙线 = val_loss（右轴），含 min marker 虚线
- 标题含 experiment_id、参数数量、last_train_loss、min_val_loss

### matplotlib 不可用时

`plot_dense_curve.py` 在 import 阶段检测 matplotlib；如未安装，退出码非 0 并打印：

```text
matplotlib is required for plot_dense_curve.py; install it via `pip install matplotlib`.
```

不内置 ASCII 字符图降级——matplotlib 是项目级 pyproject.toml 依赖，不应缺失。

## 数据规模 sanity check

跑 5000 步后：
- `delta_train_loss < 0`（loss 实际下降）；若 ≥ 0 则说明 learning rate / warmup / 数据绑定有 bug
- `val_loss_min` 应出现在 step ≥ `warmup_steps` 后
- `delta_val_loss` 在 5000 步内通常 ≥ 0（val 仍在下降，未过拟合）；> 0.5 需关注（可能过拟合信号）

## 不在 N4 范围

- 多 seed sweep → P1-03
- mean/std/CI → P1-03
- Perplexity 计算 → 后续独立协议
- 模型质量评估（人评 / LLM judge）
- 不同 d_model / n_layers 的消融（→ N5+ 计划）
- 引入 wandb / tensorboard 等外部可视化依赖

## 与前面阶段的边界

- N3 之前的 `dense-owt-formal-cache-result.json` 是 100 步 smoke；schema v1.1 不再接受，标记为 pre-N4 旧 artifact
- N2 不受影响（routing stats 与 dense 训练曲线协议无交叉）
- 不复用 `dense-owt-formal.example.yaml` 的 100-step 配置；N4 是独立 config