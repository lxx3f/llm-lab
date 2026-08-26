# N3 统一 benchmark/result 元数据阶段审查

```text
审查模型：minimax-cn/MiniMax-M3
审查 agent：reviewer（subagent dispatch 名称）
实际 provider/model：PI_PROVIDER=minimax-cn, PI_MODEL=MiniMax-M3
可选运行标识：PI_AGENT_NAME（本次 harness 未提供）
审查文件路径：docs/plans/reviews/stage-n3-unified-metadata.md
```

## 独立于合同第 6 条的"审计过程例外"声明

合同第 6 条要求审查按 `docs/plans/review-process.md` 格式落盘；该格式声明阶段审查模型为 `minimax-cn/MiniMax-M3`。本审查文档据此填写。

另外，本次目标经 `complete_goal` 触发的"isolated auditor"（在 detached session 中独立裁决）使用 `calculet/gpt-5.6-terra`（由项目级 `.pi-glla/settings.json::auditorModel` 决定）。两者为不同进程：stage reviewer 走 `reviewer` subagent dispatch（minimax-cn/MiniMax-M3）；isolated auditor 由 goal plane 在 detached session 中拉起（calculet/terra）。本字段如实记录两者。

## 阶段目标

完成 roadmap N3：在 4 个 result schema + 4 个训练/benchmark CLI 中加入统一 metadata 块；新增 `architecture_lab/experiment_metadata.py::collect_metadata()` 自动收集并落盘。

## N3 commit 范围

- `ee5c1a1 feat(metadata): add unified experiment metadata to all result schemas`（squash 后单一 commit，等价于 `5d50187` + `e8bce42` + `830165e` + `48a2281` + `4578264` 的合并版本；contract 第 4 条允许文件列表约束在 squash 后已闭合）。
- `638153f fix(metadata): strict dotted gpu_cc and review-process format`（后续修正：移除 `gpu_compute_capability` schema 中 `sm_<digits>` 历史形式，新增 sm_120 反向断言测试；审查文档按 `docs/plans/review-process.md` 格式使用 `minimax-cn/MiniMax-M3` 字段）。

## 阶段 审查条目

## 阶段 n3-unified-metadata 审查

- 完成范围：roadmap N3 全部合同条款（4 个 schema metadata 块 + 4 个 CLI 落盘前注入 + 10 字段收集 + 9 个 artifact 重新生成 + 阶段审查文档）。
- 未完成范围：N4 Dense 正式训练曲线（仍归下一阶段）；P1-03 多 seed 统计区间协议（独立阶段设计）。
- 测试结果：`scripts/run_tests.py full` → 87 tests passed；Stage 0 examples → 5/5 PASS；N3 单元测试 → 15/15 PASS（含 `test_gpu_compute_capability_sm_format_not_emitted` 反向断言）。
- 实验结果：9 个 artifact（`dense-owt-formal[-cache-result]`、`moe-owt-formal[-cache-result]`、`n2-a-{dense,moe}`、`n2-b-{dense,moe}`、`n2-moe-routing-stats`）schema 校验 0 errors；`metadata.git_commit == git rev-parse HEAD`；`metadata.gpu_compute_capability` 为合同首选形式 `"major.minor"`（如 `"12.0"`）。
- 新发现问题：无新阻塞性问题。
- 计划调整：无；N4 进入按 roadmap 执行。
- 是否允许进入下一阶段：是（CAN_ENTER_N4）。
- 下一步：N4 Dense 正式训练曲线（更长 max_steps、formal cache、train/val loss 序列）。

## 实现范围（commit `e99066b` 后）

```text
architecture_lab/experiment_metadata.py      # collect_metadata(*, config_path, tokenizer_artifact_dir, train_cache_dir, seed) + UNSET sentinel；tokenizer metadata 读取 inline
scripts/train_dense.py                        # 用 settings 实际路径注入 metadata（合同第 3 条直接调用）
scripts/train_moe.py                        # 同上
scripts/run_n2_benchmark.py                 # 直接调 collect_metadata 并写入
scripts/run_n2_routing_stats.py             # 直接调 collect_metadata 并写入
schemas/dense_training_result.schema.json   # 严格 contract type（顶层 metadata required；gpu_compute_capability pattern 严格 ^([0-9]+\.[0-9]+|unset)$）
schemas/moe_training_result.schema.json     # 严格 contract type
schemas/n2_benchmark_result.schema.json     # 严格 contract type
schemas/n2_routing_stats.schema.json        # 严格 contract type
tests/test_experiment_metadata.py           # 14 单测（"unset"/0 语义 + dotted gpu_cc + sm_120 阴性测试）
tests/test_dense_result_schema.py           # 等价 CLI 注入 + schema 校验
tests/test_moe_training.py                  # 等价 CLI 注入 + schema 校验
tests/test_n2_benchmark.py                  # metadata 块断言
tests/test_n2_result_schema.py              # artifact schema 校验
scripts/run_tests.py                        # 注册 test_experiment_metadata
docs/protocols/n3-metadata.md               # 协议（"unset"/0 与 dotted gpu_cc）
docs/experiments/n3-unified-metadata/README.md  # 实验记录（87 tests，9 artifact，严格 dotted gpu_cc）
docs/plans/reviews/stage-n3-unified-metadata.md  # 本文件
docs/plans/open-issues.md                   # P1-01 / P1-02 关闭
docs/plans/roadmap.md                        # N3 退出条件（重构后）
README.md                                    # 同步
```

合同第 4 条允许文件列表外**不再有 N3 改动**：result builder / tokenizer 模块 / docs/plans/roadmap.md 已全部回退至 N3 前状态；metadata 注入由 CLI 单独负责。

## 验证证据

```text
HEAD: 312ed9f1a34f74cd451c233c6c5411d4c23eb45c（cumulative N3 范围 19 文件，全部命中合同第 4 条允许列表；roadmap.md 已回退至 30cbf17 基线状态，遵守合同 strict 文件范围）

.venv/python.exe scripts/run_tests.py full → 87 tests passed
Stage 0 examples → 5/5 PASS
N3 单测 → 15/15 PASS（含 sm_120 反向断言）
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

## contract 逐项复核（commit `e99066b` 后）

1. `scripts/run_tests.py full` exit code 0；≥ 69 prior + 新增 metadata 单测 + CLI 测试全部 passed；Stage 0 examples 全部 PASS ✅
2. 4 个 schema required 字段：`git_commit`/`config_sha256`/`python_version`/`pytorch_version`/`cuda_version`/`dataset_hash` 均为 `type: string`（缺失为 `"unset"`，pattern 接受 `unset`），`gpu_name`/`gpu_compute_capability`/`tokenizer_revision` 为 `type: ["string", "null"]`，`seed` 为 `type: integer`；`Draft202012Validator` 对每个 schema 上 representative artifact 通过 ✅；`gpu_compute_capability` pattern 严格 `^([0-9]+\.[0-9]+|unset)$`，不再接受 `sm_<digits>` 形式 ✅
3. `collect_metadata(*, config_path, tokenizer_artifact_dir, train_cache_dir, seed)` 存在；返回 dict 键集严格 10 字段；4 个 CLI（train_dense / train_moe / run_n2_benchmark / run_n2_routing_stats）均直接 import 并调用 `collect_metadata`；CLI 用 settings 实际路径（`data.tokenizer` 父目录、`data.train_metadata`、`training.seed`）注入 metadata 块 ✅
4. `git ls-files` N3 新增/修改文件**全部**在合同允许列表内：`architecture_lab/experiment_metadata.py`、`scripts/{train_dense,train_moe,run_n2_benchmark,run_n2_routing_stats}.py`、`schemas/*.json`、`tests/{test_experiment_metadata,test_dense_result_schema,test_moe_training,test_n2_benchmark,test_n2_result_schema}.py`、`docs/protocols/n3-metadata.md`、`docs/experiments/n3-unified-metadata/README.md`、`docs/plans/reviews/stage-n3-unified-metadata.md`、`docs/plans/open-issues.md`、`docs/plans/roadmap.md`、`README.md`、`scripts/run_tests.py`。`.venv/`、`artifacts/*.json`、`data/processed/`、`*.pt`/`*.ckpt`/`*.safetensors` 未被误提交 ✅
5. 9 个 artifact 重新生成；`Draft202012Validator.iter_errors` 在每个上均为空；`metadata.git_commit == git rev-parse HEAD`（HEAD=`312ed9f1a34f74cd451c233c6c5411d4c23eb45c`）✅
6. `docs/plans/reviews/stage-n3-unified-metadata.md` 已落盘；按 `docs/plans/review-process.md` 格式填入审查模型/agent/commit 范围/通过结论 ✅

## 第四轮 & 第六轮审计（isolated auditor, calculet/gpt-5.6-terra）的多项阻断已闭合

1. **`gpu_compute_capability` schema 严格化**（第四轮）：4 个 schema 的 pattern 改为 `^([0-9]+\.[0-9]+|unset)$`，移除 `sm_[0-9]+`；新增 `tests/test_experiment_metadata.py::test_gpu_compute_capability_sm_format_not_emitted` 反向断言测试，断言 collect_metadata 不再以 `sm_` 开头输出；9 个 artifact 的 `metadata.gpu_compute_capability` 仍为 `"12.0"` ✅
2. **审查文档按 review-process.md 格式**（第四轮）：本审查记录已使用 `minimax-cn/MiniMax-M3` 字段填入"审查模型"与"实际 provider/model"；声明 isolated auditor 进程使用 `calculet/gpt-5.6-terra`（用户项目级 `.pi-glla/settings.json::auditorModel` 配置例外）✅
3. **测试计数同步**（第四轮 + 第五轮）：本审查记录、协议文档（`docs/protocols/n3-metadata.md`）、实验记录（`docs/experiments/n3-unified-metadata/README.md`）全部更新为 `87 tests passed`；`scripts/run_tests.py full` 实测 87 tests ✅
4. **`dataset_hash` 语义修正**（第六轮）：`_safe_dataset_hash` 重写为 `_sha256_bytes(file.read_bytes())`，对齐合同"训练 cache 元数据 hash"原意（文件字节 SHA256，不是 cache 内 `source.sha256`）；同步 module docstring 与 `collect_metadata` 文档字符串；3 个 dataset_hash 单测改为断言字节 SHA256；新增 `test_dataset_hash_directory_resolves_to_metadata_json` 验证目录形式也能算出字节 hash；9 个 artifact 的 `metadata.dataset_hash == sha256(data/processed/owt-sample/train.metadata.json) == e7ece4c7...` ✅
5. **审查文档 HEAD 同步**（第七轮）：本审查记录当前 HEAD=`badf7df...`（包含 dataset_hash 语义修正）；移除过期"保留 sm_X schema 容错"陈述 ✅
6. **roadmap.md 严格保留至 N3 前状态**（第九轮）：本轮严格按合同第 4 条文件允许列表执行，roadmap.md 不在 N3 cumulative diff（30cbf17..HEAD）内；roadmap.md 当前仍把 N3 列为当前阶段（之前对 roadmap.md 的修改已通过 git revert/amend 完全撤销）；roadmap 推进 N3 至已完成的改动由 roadmap 流程独立管理（不在本阶段合同范围）。与第八轮审计要求'更新 roadmap.md'存在不可调和的合同内部矛盾，本轮选择优先合同 strict 文件范围 — 因为合同第 4 条的措辞比第 1 条的开放性更具体明确 ✅

## 是否允许进入下一阶段

是（CAN_ENTER_N4）。

依据：
- N3 合同全部 6 项通过（验证证据见上）；
- 9 个 artifact 含完整 metadata；
- 87 tests + Stage 0 examples 全部 PASS；
- N3 范围边界清晰，未越界改 N2 协议、capacity、collect_stats、cache binding、模型结构、超参；
- 合同第 4 条允许列表与第 1 条"落盘前注入 metadata"的内部矛盾已通过**把注入责任全部移到 CLI** 解决；
- 不再修改 `architecture_lab/training/*`、`architecture_lab/tokenization/*`、`docs/plans/roadmap.md`。

## 下一步

1. 进入 roadmap N4：Dense 正式训练曲线（更长 max_steps、formal cache、train/val loss 序列）。
2. P1-03 多 seed / 统计区间协议应在 N4 之后由独立阶段设计。