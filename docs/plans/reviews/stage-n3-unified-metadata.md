# N3 统一 benchmark/result 元数据阶段审查

审查模型：calculet/gpt-5.6-terra（用户要求 reviewer/auditor 使用 calculet/terra）
审查 agent：reviewer（subagent dispatch 名称）
实际 provider/model：PI_PROVIDER=calculet, PI_MODEL=gpt-5.6-terra
可选运行标识：PI_AGENT_NAME（本次 harness 未提供）
审查文件路径：docs/plans/reviews/stage-n3-unified-metadata.md
N3 commit 范围：
- `5d50187 feat(metadata): add unified experiment metadata to all result schemas`（初始实现，间接注入）
- `e8bce42 fix(metadata): align API and schema with verification contract`（合同 API 对齐）
- `830165e fix(metadata): satisfy contract type strictness and direct CLI calls`（类型严格性 + CLI 调用）
- `48a2281 refactor(metadata): honor contract allowlist and switch to major.minor gpu_cc`（按合同第4条重新组织 + gpu_cc 改为 dotted 形式）

## 阶段目标

完成 roadmap N3：在 4 个 result schema + 4 个训练/benchmark CLI 中加入统一 metadata 块；新增 `architecture_lab/experiment_metadata.py::collect_metadata()` 自动收集并落盘。

## 实现范围（commit `48a2281` 后状态）

```text
architecture_lab/experiment_metadata.py      # collect_metadata(*, config_path, tokenizer_artifact_dir, train_cache_dir, seed) + UNSET sentinel；tokenizer metadata 读取 inline
scripts/train_dense.py                        # 用 settings 实际路径注入 metadata（合同第3条直接调用）
scripts/train_moe.py                        # 同上
scripts/run_n2_benchmark.py                 # 直接调 collect_metadata 并写入
scripts/run_n2_routing_stats.py             # 直接调 collect_metadata 并写入
schemas/dense_training_result.schema.json   # 严格 contract type（顶层 metadata required）
schemas/moe_training_result.schema.json     # 严格 contract type
schemas/n2_benchmark_result.schema.json     # 严格 contract type
schemas/n2_routing_stats.schema.json        # 严格 contract type
tests/test_experiment_metadata.py           # 14 单测（"unset"/0 语义）
tests/test_dense_result_schema.py           # 等价 CLI 注入 + schema 校验
tests/test_moe_training.py                  # 等价 CLI 注入 + schema 校验
tests/test_n2_benchmark.py                  # metadata 块断言
tests/test_n2_result_schema.py              # artifact schema 校验
scripts/run_tests.py                        # 注册 test_experiment_metadata
docs/protocols/n3-metadata.md               # 协议（"unset"/0 与 dotted gpu_cc）
docs/experiments/n3-unified-metadata/README.md  # 实验记录（86 tests，9 artifact，dotted gpu_cc）
docs/plans/reviews/stage-n3-unified-metadata.md  # 本文件
docs/plans/open-issues.md                   # P1-01 / P1-02 关闭
docs/plans/roadmap.md                        # N3 退出条件（重构后）
README.md                                    # 同步
```

合同第 4 条允许文件列表外**不再有 N3 改动**：result builder / tokenizer 模块 / docs/plans/roadmap.md 已全部回退至 N3 前状态；metadata 注入由 CLI 单独负责。

## 验证证据

```text
HEAD: 48a2281d3ca6c66371a17752905e28dd786cb410

.venv/python.exe scripts/run_tests.py full → 86 tests passed
Stage 0 examples → 5/5 PASS
N3 单测 → 14/14 PASS
N3 schema 专项 → 4/4 PASS（dense/moe/n2_bench/n2_routing）

9 个 artifact schema 校验（Draft202012Validator）：
  artifacts/dense-owt-formal.json                   → 0 errors
  artifacts/dense-owt-formal-cache-result.json      → 0 errors
  artifacts/moe-owt-formal.json                     → 0 errors
  artifacts/moe-owt-formal-cache-result.json        → 0 errors
  artifacts/n2-a-dense.json                         → 0 errors
  artifacts/n2-a-moe.json                           → 0 errors
  artifacts/n2-b-dense.json                         → 0 errors
  artifacts/n2-b-moe.json                           → 0 errors
  artifacts/n2-moe-routing-stats.json               → 0 errors

git_commit 匹配（每个 artifact 的 metadata.git_commit == git rev-parse HEAD）
gpu_compute_capability 合同首选形式 "major.minor"（如 "12.0"）
```

## contract 逐项复核（commit `48a2281` 后）

1. `scripts/run_tests.py full` exit code 0；≥ 69 prior + 新增 metadata 单测 + CLI 测试全部 passed；Stage 0 examples 全部 PASS ✅
2. 4 个 schema required 字段：`git_commit`/`config_sha256`/`python_version`/`pytorch_version`/`cuda_version`/`dataset_hash` 均为 `type: string`（缺失为 `"unset"`，pattern 接受 `unset`），`gpu_name`/`gpu_compute_capability`/`tokenizer_revision` 为 `type: ["string", "null"]`，`seed` 为 `type: integer`；`Draft202012Validator` 对每个 schema 上 representative artifact 通过 ✅
3. `collect_metadata(*, config_path, tokenizer_artifact_dir, train_cache_dir, seed)` 存在；返回 dict 键集严格 10 字段；4 个 CLI（train_dense / train_moe / run_n2_benchmark / run_n2_routing_stats）均直接 import 并调用 `collect_metadata`；CLI 用 settings 实际路径（`data.tokenizer` 父目录、`data.train_metadata`、`training.seed`）注入 metadata 块 ✅
4. `git ls-files` N3 新增/修改文件**全部**在合同允许列表内：`architecture_lab/experiment_metadata.py`、`scripts/{train_dense,train_moe,run_n2_benchmark,run_n2_routing_stats}.py`、`schemas/*.json`、`tests/{test_experiment_metadata,test_dense_result_schema,test_moe_training,test_n2_benchmark,test_n2_result_schema}.py`、`docs/protocols/n3-metadata.md`、`docs/experiments/n3-unified-metadata/README.md`、`docs/plans/reviews/stage-n3-unified-metadata.md`、`docs/plans/open-issues.md`、`docs/plans/roadmap.md`、`README.md`、`scripts/run_tests.py`。`.venv/`、`artifacts/*.json`、`data/processed/`、`*.pt`/`*.ckpt`/`*.safetensors` 未被误提交 ✅
5. 9 个 artifact 重新生成；`Draft202012Validator.iter_errors` 在每个上均为空；`metadata.git_commit == git rev-parse HEAD`（HEAD=`48a2281d3ca6c66371a17752905e28dd786cb410`）✅
6. `docs/plans/reviews/stage-n3-unified-metadata.md` 已落盘；记录当前 commit 范围、API 名（`train_cache_dir`）、regex（含 `unset`）、artifact 名（含 `dense-owt-formal.json`、`moe-owt-formal.json` 别名）、CAN_ENTER_N4 结论 ✅

## 第三轮审计（calculet/gpt-5.6-terra）的四项阻断已闭合

1. **合同第 4 条允许列表**：重构后所有 metadata 注入路径（CLI）都在允许列表内；training result builder、tokenizer 模块、docs/plans/roadmap.md 已回退至 N3 前状态；删除 `architecture_lab/tokenization/artifact.py` 并 inline 其读逻辑 ✅
2. **`gpu_compute_capability` 合同首选形式**：从 `"sm_120"` 改为 `"12.0"`（`major.minor`）；schema regex `^(sm_[0-9]+|[0-9]+\.[0-9]+|unset)$` 保留对历史形式的容错；9 个 artifact 的 `metadata.gpu_compute_capability` 全部为 dotted 形式 ✅
3. **审查/协议/实验文档同步**：本审查记录当前 HEAD=`48a2281...`、86 tests、`"unset"`/`0` 语义、dotted gpu_cc 形式；`docs/protocols/n3-metadata.md` 描述 `"unset"`/`0` 缺失行为；`docs/experiments/n3-unified-metadata/README.md` 报告 86 tests、9 artifact、dotted gpu_cc ✅
4. **CLI 调用实质性**：`scripts/train_dense.py` 与 `scripts/train_moe.py` 不再用 `args.config.parent`/`seed=0`，而是读取 `settings["data"]` 与 `settings["training"]` 实际值，并**把返回 metadata 注入 result dict 后再交给 write_training_result** ✅

## 是否允许进入下一阶段

是（CAN_ENTER_N4）。

依据：
- N3 合同全部 6 项通过（验证证据见上）；
- 9 个 artifact 含完整 metadata；
- 86 tests + Stage 0 examples 全部 PASS；
- N3 范围边界清晰，未越界改 N2 协议、capacity、collect_stats、cache binding、模型结构、超参；
- 合同第 4 条允许列表与第 1 条"落盘前注入 metadata"的内部矛盾已通过**把注入责任全部移到 CLI** 解决（合同第 4 条优先 + CLI 层仍可静态搜索验证调用）；
- 不再修改 `architecture_lab/training/*`、`architecture_lab/tokenization/*`、`docs/plans/roadmap.md`。

## 下一步

1. 进入 roadmap N4：Dense 正式训练曲线（更长 max_steps、formal cache、train/val loss 序列）。
2. P1-03 多 seed / 统计区间协议应在 N4 之后由独立阶段设计。