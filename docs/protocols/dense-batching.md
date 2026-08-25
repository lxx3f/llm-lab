# Dense token stream batch sampler 协议

## 目的

`TokenStreamBatcher` 将单个 split 的连续 `uint16` token cache 转换为 Dense causal LM 可直接消费的 `(input_ids, target_ids)` batch。

## causal shift

对每个窗口起点 `s` 和序列长度 `T`：

```text
input_ids  = tokens[s : s + T]
target_ids = tokens[s + 1 : s + T + 1]
```

因此训练 loop 应使用显式 target 计算一次 causal loss：

```python
logits, _ = model(input_ids)
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1),
)
```

当前 `DenseTransformer.forward(labels=...)` 会对传入 labels 再执行内部 shift，因此不要把已经右移的 `targets` 直接传给 `labels`。兼容旧接口时，`model(input_ids, labels=input_ids)` 仍然有效。

## 随机 batch

`TokenStreamBatcher.sample()`：

- 从当前 split 的合法起点均匀采样 `batch_size` 个窗口；
- 窗口不跨越 cache 末尾；
- 窗口之间允许重叠；
- `seed` 固定时，CPU generator 产生可复现起点；
- 输出会转换为模型 embedding 所需的 `torch.long`。

## epoch batch

`iter_epoch()` 使用连续、不重叠窗口进行确定性遍历：

- 每个 batch 包含 `batch_size * sequence_length` 个连续 token；
- 最后不足一个完整 batch 的 token 被丢弃；
- 输入和 target 只在各自窗口内做一步移位；
- train 和 validation 必须分别创建 batcher，不能混用 cache。

## cache 加载

使用 `load_token_cache()`：

```python
from architecture_lab.data import TokenStreamBatcher, load_token_cache

tokens, cache_info = load_token_cache(
    token_path="data/processed/owt-sample/train.tokens.uint16",
    metadata_path="data/processed/owt-sample/train.metadata.json",
    source_path="data/raw/owt-sample/owt_train.txt",
    tokenizer_path="artifacts/tokenizers/owt-bpe/v0.2.0/tokenizer.json",
    split="train",
    max_bytes=512 * 1024 * 1024,
)
batcher = TokenStreamBatcher(tokens, batch_size=4, sequence_length=128, seed=42)
inputs, targets = batcher.sample()
```

加载前会调用 `validate_token_cache()`，校验 cache、source 和 tokenizer 的 hash 绑定。默认使用 memory map，避免把正式 cache 全量复制到 Python heap；返回的 token tensor 仍是 CPU `torch.uint16`，batch 输出才搬到训练 device。

## 当前边界

- 当前 batch sampler 只负责连续 token stream，不负责 shuffle buffer、分布式 rank 切分或 packed document mask；
- `iter_epoch()` 是确定性 smoke/training MVP，正式多 GPU 训练前需要补充 rank-aware 采样；
- `sequence_length` 必须不超过 Dense `TransformerConfig.max_seq_len`；
- cache 仍然不提交 Git。
