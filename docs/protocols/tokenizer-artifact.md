# BPE tokenizer artifact 协议

## 目录布局

Tokenizer artifact 固定保存为：

```text
artifacts/
└── tokenizers/
    └── <name>/
        └── <version>/
            ├── tokenizer.json
            └── metadata.json
```

例如：

```text
artifacts/tokenizers/tinystories-bpe/v0.1.0/
├── tokenizer.json
└── metadata.json
```

`artifacts/` 是项目级实验产物目录。小型、可复现且不含敏感信息的 tokenizer artifact 可以纳入版本控制；原始数据、模型权重和大文件不应放入该目录。

## tokenizer.json

由 `BPETokenizer.save()` 生成，包含：

- `artifact_type`：固定为 `llm-lab-bpe-tokenizer`；
- `version`：tokenizer 实现版本；
- `vocab`：token ID 和 token bytes 的十六进制表示；
- `merges`：BPE merge pair；
- `special_tokens`：保留的特殊 token；
- `artifact_sha256`：对前述内容 canonical JSON 的 SHA-256。

加载时会重新计算 hash；内容被篡改或手动修改时，`BPETokenizer.load()` 会拒绝加载。

## metadata.json

由 CLI 生成，当前 metadata 版本为 `1.0`：

```json
{
  "metadata_type": "llm-lab-tokenizer-metadata",
  "metadata_version": "1.0",
  "tokenizer": {
    "name": "tinystories-bpe",
    "version": "v0.1.0",
    "implementation": "llm-lab-bpe-v1",
    "algorithm": "byte-level-bpe",
    "pretokenization": "gpt2-regex-v1",
    "vocab_size_requested": 10000,
    "vocab_size_actual": 10000,
    "merge_count": 9743,
    "special_tokens": ["<|endoftext|>"]
  },
  "source": {
    "path": "data/tinystories/train.txt",
    "kind": "manually-authored",
    "license": "CC0-1.0",
    "data_version": "D0",
    "sha256": "...",
    "size_bytes": 123,
    "encoding": "utf-8"
  },
  "artifacts": {
    "tokenizer_file": "tokenizer.json",
    "tokenizer_sha256": "..."
  },
  "config_sha256": "...",
  "created_at": "2026-08-25T00:00:00Z"
}
```

metadata 中的 `source.path` 应使用仓库内可复现的相对路径；可以通过 CLI 的 `--source-path` 显式设置。真实接入外部数据时，还需要在数据协议中补充来源、许可证和处理版本。

## CLI

运行：

```bash
.venv/Scripts/python.exe scripts/train_bpe_tokenizer.py \
  --input data/toy/train.txt \
  --artifact-root artifacts/tokenizers \
  --name tinystories-bpe \
  --version v0.1.0 \
  --vocab-size 10000 \
  --special-token '<|endoftext|>' \
  --source-path data/toy/train.txt \
  --data-version D0 \
  --license CC0-1.0
```

如果目标 artifact 目录非空，CLI 默认拒绝覆盖。确认重新生成时显式使用：

```bash
.venv/Scripts/python.exe scripts/train_bpe_tokenizer.py ... --force
```

## 当前边界

- CLI 当前读取完整 UTF-8 文本到内存，适用于小型实验；
- BPE 训练暂未实现多进程或增量语料统计；
- metadata 当前记录源文件 hash、数据版本和许可证，但尚未自动关联数据集版本 registry；
- tokenizer artifact 还未接入 Dense Transformer 训练脚本。
