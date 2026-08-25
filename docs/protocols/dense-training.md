# Dense 训练闭环协议

## 训练输入

训练入口：

```bash
.venv/Scripts/python.exe scripts/train_dense.py \
  --config configs/dense_training.example.yaml \
  --output artifacts/dense-owt-mvp-result.json
```

配置中的 tokenizer artifact、train/validation token cache、原始 source 和 byte limit 都必须显式填写。训练启动时会校验：

- tokenizer artifact hash；
- train/validation cache 文件 hash；
- train/validation metadata hash；
- source hash、split 和 byte limit；
- tokenizer vocab size 与模型配置；
- `sequence_length <= model.max_seq_len`。

模型 `vocab_size` 由已加载的 tokenizer artifact 决定。若 YAML 显式填写 `model.vocab_size`，它必须与 artifact 一致；因此不会静默使用旧的 `256` 默认值。

## causal loss

batch sampler 返回：

```text
inputs  = tokens[s : s + T]
targets = tokens[s + 1 : s + T + 1]
```

Dense model 对 `inputs` 输出 `T` 个 logits，训练 loop 直接将全部 logits 与显式 targets 计算交叉熵：

```python
logits, _ = model(inputs)
loss = causal_loss(logits, targets)
```

不能将已右移的 `targets` 再传给 `DenseTransformer.forward(labels=...)`，因为旧接口会再次进行 shift。

## 优化和验证

当前 MVP 使用：

- AdamW；
- 可选 gradient clipping；
- 固定随机种子；
- 按 `validation_interval` 计算 validation loss；
- validation 使用独立 validation cache 和 batcher；
- validation 默认不产生梯度。

当前尚未实现学习率 scheduler、AMP、分布式 rank-aware sampler 或梯度累积。

## checkpoint

checkpoint 是原子写入的 PyTorch payload，包含：

- checkpoint type/version；
- Dense `TransformerConfig`；
- model state；
- optimizer state；
- step/epoch；
- tokenizer artifact SHA-256 和 vocab size；
- train/validation metadata SHA-256；
- 训练配置和创建时间。

resume 时会拒绝模型配置、tokenizer artifact 或任一 cache metadata 不匹配的 checkpoint。

当前 checkpoint 使用 PyTorch pickle payload，仅允许加载受信任的本地产物，不应加载未知来源 checkpoint。后续稳定化阶段应评估 `weights_only=True` 或 safetensors。

```bash
.venv/Scripts/python.exe scripts/train_dense.py \
  --config configs/dense_training.example.yaml \
  --resume artifacts/checkpoints/dense-owt-mvp.pt
```

## generation

配置中设置 `prompt` 和 `max_new_tokens` 后，训练结束会：

1. 使用同一 tokenizer encode prompt；
2. 调用 Dense KV-cache generation；
3. 使用同一 tokenizer decode；
4. 将结果写入 CLI JSON 输出。

## 当前实验边界

当前配置使用：

- OWT train/validation 各 1 MiB newline-aligned cache；
- tokenizer `owt-bpe/v0.2.0`，vocab size 8192；
- `d_model=64`、2 layers、64 sequence length；
- 100 training steps。

该设置用于验证训练、验证、checkpoint、resume 和 generation 闭环，不代表完整 OWT 训练结果、模型质量或正式性能结论。
