# OWT sample 数据记录

## 数据集定位

当前项目第一版真实语言模型训练数据改用 Stanford CS336 提供的 OWT sample：

```text
https://huggingface.co/datasets/stanford-cs336/owt-sample
```

下载文件：

```text
owt_train.txt.gz
owt_valid.txt.gz
```

官方数据文件地址：

```text
https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
```

这是 CS336 作业使用的 OpenWebText sample，不是完整 OpenWebText 数据集。

## 本地目录

数据保存在：

```text
data/raw/owt-sample/
├── owt_train.txt.gz
├── owt_train.txt
├── owt_valid.txt.gz
└── owt_valid.txt
```

该目录已被 `.gitignore` 忽略，原始数据不会提交到 Git。

## 文件信息

| 文件 | 压缩后 | 解压后 | SHA-256 |
|---|---:|---:|---|
| `owt_train.txt.gz` | 4,591,240,837 bytes | 约 12 GB | `b19ae88cfbc4016b304c348522455adcc7191a89e38ccf` |
| `owt_valid.txt.gz` | 111,785,382 bytes | 约 277 MB | `bc73db5da2f19c360836b9c8a88094e13346ec83798c2f40060be39135768c80` |

完整下载和解压后的本地文件 hash：

```text
owt_train.txt:
bbeb7f291a981ecfd5cf44b84d0f654b9e96c53dff99f2556b7d2cccaf8c1918

owt_valid.txt:
2406f278e71829d273b315e9b403285baea7022b26a96d2728dd8b776ea40660
```

下载时已执行：

```bash
gzip -t owt_train.txt.gz
gzip -t owt_valid.txt.gz
```

压缩文件完整性检查通过。

## 版本信息

- dataset name：`stanford-cs336/owt-sample`；
- dataset revision：当前下载对应 Hugging Face commit `488afadefc224407de5645a3c8c53b64ed923537`；
- 数据版本：`OWT-SAMPLE-v1`；
- train/validation：使用官方提供的 train/valid 文件；
- 编码：UTF-8 文本；
- 下载时间：2026-08-25；
- 原始数据许可证/使用条款：需要以数据集页面和上游 OpenWebText 条款为准，当前不将其标记为 CC0 或 MIT。

## 当前策略

- OWT 是第一版真实训练和验证数据；
- OWT 原始文件不提交 Git；
- tokenizer artifact 和编码缓存需要记录 OWT 文件 hash；
- train tokenizer 时只使用 `owt_train.txt`，不能使用 validation 文件，避免验证信息泄漏。

## 当前状态

Tokenizer CLI 已支持 newline-aligned 分块读取及 `--max-training-bytes` 固定前缀。它不会将 OWT train 全文拼接到内存；16 MiB OWT 前缀试跑已成功，metadata 也能记录实际纳入训练的字节范围。

当前尚未生成 32k vocab、完整 OWT train 或 1 GiB 前缀的正式 artifact。当前可供 Dense 数据管线使用的本地 artifact 是：

```text
artifacts/tokenizers/owt-bpe/v0.2.0/
```

其配置为 OWT train 的 newline-aligned 约 64 MiB 前缀和 8192 vocab；metadata 记录了实际训练字节数及完整 source hash。该 artifact 用于推进 token cache 和训练接口，不用于正式模型规模结论。

后续仍需：

```text
BPE merge 性能优化
→ 固定范围 OWT tokenizer artifact
→ OWT token cache
→ Dense training loop
```
