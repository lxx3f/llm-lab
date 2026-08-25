# OWT token cache 协议

## 目的

将 OWT UTF-8 文本转换为可供 Dense causal LM 训练使用的连续 token ID 文件，同时绑定 tokenizer 和原始数据 hash，避免错误复用缓存。

## 本地布局

原始数据和处理后 cache 均不提交 Git：

```text
data/raw/owt-sample/
data/processed/owt-sample/
```

cache 文件布局：

```text
data/processed/owt-sample/
├── train.tokens.uint16
├── train.metadata.json
├── validation.tokens.uint16
└── validation.metadata.json
```

## 编码语义

- 按 UTF-8 行流式读取；
- `--max-bytes` 只在完整行边界截断；
- 每行内使用 tokenizer 的 special token 逻辑；
- 不把完整源文件载入内存；
- token ID 使用 little-endian `uint16`；
- 当前 `owt-bpe/v0.2.0` vocab size 为 8192，满足 uint16 范围；
- cache 文件是连续 token stream，后续 batch sampler 可按 sequence length 采样。

## metadata 必填绑定

每份 `*.metadata.json` 记录：

- split；
- cache 文件名、SHA-256、token count、dtype 和 endianness；
- source path、完整 source SHA-256、文件大小；
- 实际编码字节数；
- 请求的 byte limit；
- newline-aligned 标志；
- tokenizer artifact path 和 SHA-256；
- tokenizer vocab size、special tokens 和 special token IDs；
- data version；
- config hash 和创建时间。

## 生成命令

使用当前本地 OWT tokenizer：

```bash
.venv/python.exe scripts/encode_token_cache.py \
  --input data/raw/owt-sample/owt_train.txt \
  --tokenizer artifacts/tokenizers/owt-bpe/v0.2.0/tokenizer.json \
  --output-root data/processed/owt-sample \
  --split train \
  --max-bytes 536870912

.venv/python.exe scripts/encode_token_cache.py \
  --input data/raw/owt-sample/owt_valid.txt \
  --tokenizer artifacts/tokenizers/owt-bpe/v0.2.0/tokenizer.json \
  --output-root data/processed/owt-sample \
  --split validation \
  --max-bytes 67108864
```

覆盖已有 cache 时必须显式使用 `--force`。

## 校验

训练启动前应调用 `validate_token_cache()`，至少检查：

- metadata type/version；
- cache 文件 hash；
- token count 与文件大小；
- dtype 和 endianness；
- source hash、文件大小和 UTF-8 encoding；
- split、byte limit 和 newline alignment；
- tokenizer artifact hash、vocab size、special tokens 和 special token IDs。

如果任一项不匹配，应拒绝训练，而不是静默重新解释旧 token ID。

## 当前边界

- cache 当前使用 `uint16`，词表 ID 不能超过 65535；
- metadata 的 `data_version` 由 CLI 参数提供，默认 `OWT-SAMPLE-v1`；
- 当前已经实现随机/确定性 batch sampler、训练 loop、validation、checkpoint 和 resume；训练侧协议见 `docs/protocols/dense-batching.md` 与 `docs/protocols/dense-training.md`；
- cache 产物不进入 Git，复现依赖源文件 hash、tokenizer artifact hash 和命令参数。
