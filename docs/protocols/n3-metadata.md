# N3 统一 experiment metadata 协议

> 所有 N3 result（Dense 训练、MoE 训练、N2 latency benchmark、N2 routing stats）必须在顶层 `metadata` 块携带完全相同的 10 字段。这是为了在不开新连接配置的前提下，能从单一 result JSON 还原跑出该结果的全部上下文。

## 字段定义

| 字段 | 类型 | 来源 | 缺失行为 |
|---|---|---|---|
| `git_commit` | string | `git rev-parse HEAD` | subprocess 失败或非 git 仓库 → 字面字符串 `"unset"` |
| `config_sha256` | string | 当前 config 文件字节的 SHA256 hex | 文件缺失或不可读 → `"unset"` |
| `python_version` | string | `platform.python_version()` | 始终非空 |
| `pytorch_version` | string | `torch.__version__` | torch 不可用 → `"unset"` |
| `cuda_version` | string | `torch.version.cuda` | CUDA 不可用 → `"unset"` |
| `gpu_name` | string \| null | `torch.cuda.get_device_name(0)` | CUDA 不可用 → `null` |
| `gpu_compute_capability` | string \| null | 序列化为 `"major.minor"`（合同首选形式） | CUDA 不可用 → `null` |
| `tokenizer_revision` | string \| null | tokenizer artifact `<dir>/metadata.json` 的 `tokenizer.version` | 缺失或损坏 → `null` |
| `dataset_hash` | string | train cache `<dir>/metadata.json` 的 `source.sha256` | 缺失或损坏 → `"unset"` |
| `seed` | integer | settings 中的 seed | None / 负数 → `0` |

合同 contract-required 字段（`git_commit`/`config_sha256`/`python_version`/`pytorch_version`/`cuda_version`/`dataset_hash`）的 schema type 严格为 `string`，不接受 `null`。可选字段（`gpu_name`/`gpu_compute_capability`/`tokenizer_revision`）的 schema 接受 `string|null`，实现侧对 CPU 环境返回 `null`。`seed` schema type 严格为 `integer`。

## 收集入口

```python
from architecture_lab.experiment_metadata import collect_metadata

metadata = collect_metadata(
    config_path=Path("configs/dense_training.owt-formal.example.yaml"),
    tokenizer_artifact_dir=Path("artifacts/tokenizers/owt-bpe/v0.2.0"),
    train_cache_dir=Path("data/processed/owt-sample/train.metadata.json"),
    seed=42,
)
```

`train_cache_dir` 既可传 train cache 所在目录（函数自动读 `<dir>/metadata.json`），也可直接传 `metadata.json` 文件路径。

返回 dict 严格 10 字段（`METADATA_FIELDS`）。collector 不会初始化 CUDA，也不会 raise。

## CLI 集成

N3 把 metadata 注入的责任放在 CLI 层：

- `scripts/train_dense.py`：训练完成后用 settings 实际读取的 `data.tokenizer` 父目录与 `data.train_metadata` 路径调 `collect_metadata`，把返回 dict 写入 `result["metadata"]`，再交给 `write_training_result` 落盘。
- `scripts/train_moe.py`：同上。
- `scripts/run_n2_benchmark.py` 与 `scripts/run_n2_routing_stats.py`：构造 result dict 时直接调用 `collect_metadata` 注入 `"metadata"` 块。

CLI 不会把 metadata 注入委托给 result builder；这避免了 N3 合同第 4 条"允许文件列表"与"在落盘前注入 metadata"的内部矛盾。

## 调用点与 schema 注入

4 个 result schema（`dense_training_result`、`moe_training_result`、`n2_benchmark_result`、`n2_routing_stats`）的顶层 `required` 都包含 `metadata`，`$defs/metadata` 块定义字段集合与类型约束。每个 schema 上的 `Draft202012Validator` 在对应 representative artifact 上 `iter_errors` 必须为空。

## 测试分层

- `tests/test_experiment_metadata.py`：`collect_metadata` 单元测试（字段集固定、文件 hash、缺失降级、CUDA 缺失、负 seed 归零、`"unset"` 字面）。
- `tests/test_dense_result_schema.py`：合成 result + metadata 注入 + schema 校验。
- `tests/test_moe_training.py`：训练 result + metadata 注入 + schema 校验。
- `tests/test_n2_benchmark.py`：benchmark result 含 metadata 块。
- `tests/test_n2_result_schema.py`：artifact schema 校验。

## 验证命令

```bash
.venv/python.exe scripts/run_tests.py full
```

预期：86 tests passed；Stage 0 examples 5/5 PASS。