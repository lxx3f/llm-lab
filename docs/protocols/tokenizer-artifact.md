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
artifacts/tokenizers/owt-bpe/v0.1.0/
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
    "path": "data/raw/owt-sample/owt_train.txt",
    "kind": "external-dataset",
    "license": "upstream-terms",
    "data_version": "OWT-SAMPLE-v1",
    "sha256": "...",
    "size_bytes": 123,
    "encoding": "utf-8"
  },
  "training": {
    "input_path_is_used_only_for_metadata": false,
    "special_tokens_excluded_from_bpe_merges": true,
    "chunk_size_bytes": 8388608,
    "max_training_bytes": 1073741824,
    "newline_aligned_prefix": true
  },
  "artifacts": {
    "tokenizer_file": "tokenizer.json",
    "tokenizer_sha256": "..."
  },
  "config_sha256": "...",
  "created_at": "2026-08-25T00:00:00Z"
}
```

metadata 中的 `source.path` 应使用仓库内可复现的相对路径；可以通过 CLI 的 `--source-path` 显式设置。`source.kind`、`source.license` 和 `source.data_version` 用于记录数据来源。对大文件，`training` 还记录 chunk size、训练字节上限以及是否按 newline 对齐。

## CLI

当前 CLI 使用 newline-aligned 分块读取，不会将完整输入文件拼接到一个字符串中。对于 OWT，建议先使用固定前缀生成可复现的 tokenizer artifact：

```bash
.venv/python.exe scripts/train_bpe_tokenizer.py \
  --input data/raw/owt-sample/owt_train.txt \
  --artifact-root artifacts/tokenizers \
  --name owt-bpe \
  --version v0.1.0 \
  --vocab-size 32000 \
  --special-token '<|endoftext|>' \
  --source-path data/raw/owt-sample/owt_train.txt \
  --data-version OWT-SAMPLE-v1 \
  --license upstream-terms \
  --source-kind external-dataset \
  --chunk-size-bytes 8388608 \
  --max-training-bytes 1073741824
```

`--max-training-bytes` 会选择以完整 UTF-8 行结束的确定性前缀；训练范围会写入 `metadata.json`。不传该参数时会扫描整个文件，但 pair frequency 表仍可能占用大量内存。

如果目标 artifact 目录非空，CLI 默认拒绝覆盖。确认重新生成时显式使用：

```bash
.venv/python.exe scripts/train_bpe_tokenizer.py ... --force
```

## 当前边界

- 训练输入采用 newline-aligned 分块读取；
- `--max-training-bytes` 适合固定 OWT 前缀，避免一次性载入约 12GB 文本；
- BPE pair frequency 表仍保存在内存中，完整 OWT 训练仍可能需要更高效的统计/merge 实现；
- metadata 当前记录源文件 hash、数据版本、许可证和训练范围，但尚未自动关联数据集版本 registry；
- tokenizer artifact 还未接入 Dense Transformer 训练脚本。
