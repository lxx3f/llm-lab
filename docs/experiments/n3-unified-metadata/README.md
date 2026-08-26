# N3 统一实验元数据 — 实验记录

## 范围

完成 roadmap N3：在 Dense/MoE training result、N2 latency benchmark、N2 routing stats 四类结果 JSON 中加入 `metadata` 块，由 `architecture_lab/experiment_metadata.py::collect_metadata()` 自动收集；不引入新依赖，不动训练/benchmark 协议本身。metadata 注入责任放在 CLI 层（contract allowlist 友好的实现位置）。

## 实现清单

```text
architecture_lab/experiment_metadata.py      # 收集器（含 inline tokenizer metadata 读）
scripts/train_dense.py                        # 用 settings 实际路径注入 metadata
scripts/train_moe.py                        # 同上
scripts/run_n2_benchmark.py                 # 注入 metadata
scripts/run_n2_routing_stats.py             # 注入 metadata
schemas/dense_training_result.schema.json   # +metadata required
schemas/moe_training_result.schema.json     # +metadata required
schemas/n2_benchmark_result.schema.json     # +metadata required
schemas/n2_routing_stats.schema.json        # +metadata required
tests/test_experiment_metadata.py           # 新增单测（"unset"/0 语义）
tests/test_dense_result_schema.py           # 增强 metadata 断言（CLI 等价注入）
tests/test_moe_training.py                  # 增强 metadata 断言（CLI 等价注入）
tests/test_n2_benchmark.py                  # 增强 metadata 断言
scripts/run_tests.py                        # 注册 test_experiment_metadata
docs/protocols/n3-metadata.md               # 协议
docs/experiments/n3-unified-metadata/README.md  # 本文件
docs/plans/reviews/stage-n3-unified-metadata.md  # 阶段审查
docs/plans/open-issues.md                   # P1-01/P1-02 关闭
README.md                                   # 同步
```

## 字段示例（合同首选形式）

9 个本地 artifact 中每个 `metadata` 块示意：

```json
{
  "git_commit": "<sha256-hex>",
  "config_sha256": "<sha256-hex>",
  "python_version": "3.12.13",
  "pytorch_version": "2.10.0+cu128",
  "cuda_version": "12.8",
  "gpu_name": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
  "gpu_compute_capability": "12.0",
  "tokenizer_revision": "v0.2.0",
  "dataset_hash": "<sha256-hex>",
  "seed": 42
}
```

完整字段集固定为 `tests/test_experiment_metadata.py::CollectMetadataTests::test_field_set_is_frozen` 中断言的 10 个键。

## 缺失值与 sentinel

- contract-required 非空字符串字段（`git_commit`/`config_sha256`/`python_version`/`pytorch_version`/`cuda_version`/`dataset_hash`）缺失时返回字面字符串 `"unset"`；schema type 严格 `string`，不允许 `null`。
- 可选字段（`gpu_name`/`gpu_compute_capability`/`tokenizer_revision`）在 CUDA 不可用 / 文件缺失时返回 `null`；schema 接受 `string|null`。
- `seed` 缺失或为负数归一为 `0`；schema type 严格 `integer`，不允许 `null`。

## 验证

```text
.venv/python.exe scripts/run_tests.py full → 86 tests passed
Stage 0 examples → 5/5 PASS
N3 单元测试 → 14/14 PASS
```

artifact schema 校验（`Draft202012Validator.iter_errors` 在每个文件上为空）：

```text
artifacts/dense-owt-formal.json                   → 0 errors
artifacts/dense-owt-formal-cache-result.json      → 0 errors
artifacts/moe-owt-formal.json                     → 0 errors
artifacts/moe-owt-formal-cache-result.json        → 0 errors
artifacts/n2-a-dense.json                         → 0 errors
artifacts/n2-a-moe.json                           → 0 errors
artifacts/n2-b-dense.json                         → 0 errors
artifacts/n2-b-moe.json                           → 0 errors
artifacts/n2-moe-routing-stats.json               → 0 errors
```

`git_commit` 字段与 `git rev-parse HEAD` 输出在所有 9 个 artifact 中逐个匹配。

## 不包含

- 多 seed sweep、mean/std/CI、p50/p95 实跑 — 仍归 P1-03，N3 不动
- N4 Dense 正式长训练曲线
- N2 协议 A/B 的 alignment、capacity、collect_stats、cache binding 改动
- Dense/MoE 模型结构或训练超参
- 新外部依赖

## 已知边界

- `config_sha256` 是 YAML 文件字节 SHA256，不是 canonical settings dict hash
- `dataset_hash` 是 train cache `source.sha256`，不是 cache 文件 SHA256
- CPU 环境下 `cuda_version`/`gpu_name`/`gpu_compute_capability` 均为 `null`，属正常降级
- 预 N3 artifact 在 schema 升级后不再通过校验；必须重新生成
- `gpu_compute_capability` 合同首选形式为 `"major.minor"`（如 `"12.0"`），保留对历史 `"sm_X"` 形式的 schema 容错