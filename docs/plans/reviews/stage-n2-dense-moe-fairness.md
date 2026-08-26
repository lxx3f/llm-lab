# N2 Dense/MoE 公平对比协议阶段审查

审查模型：calculet/gpt-5.6-terra（用户要求将 reviewer/auditor 切到 calculet/terra，主会话保持 minimax-cn/MiniMax-M3）
审查 agent：reviewer（subagent dispatch 名称）
实际 provider/model：PI_PROVIDER=calculet, PI_MODEL=gpt-5.6-terra
可选运行标识：PI_AGENT_NAME（本次 harness 未提供）
审查文件路径：docs/plans/reviews/stage-n2-dense-moe-fairness.md

## 阶段目标

完成 roadmap N2：在同一 OWT BPE tokenizer、同一 train/validation token cache、同一 seed、batch size、sequence length、optimizer、scheduler、AMP/gradient accumulation 和 token budget 下，实现并实际运行两套 Dense/MoE Top-1 对比协议 A（total 参数对齐）和 B（active 参数对齐），统一 benchmark 入口、独立 routing stats 路径、共享 binding 校验，并完成 A→B 的实际 benchmark smoke 与文档落盘。

## 实现范围

```text
architecture_lab/benchmarks/__init__.py
architecture_lab/benchmarks/n2.py
architecture_lab/benchmarks/results.py
scripts/run_n2_benchmark.py
scripts/run_n2_routing_stats.py
configs/n2_benchmark.example.yaml
configs/n2_benchmark_dense.example.yaml
configs/n2_benchmark_moe.example.yaml
schemas/n2_benchmark_result.schema.json
schemas/n2_routing_stats.schema.json
tests/test_n2_benchmark.py
tests/test_n2_result_schema.py
docs/protocols/n2-benchmark.md
docs/experiments/n2-dense-moe-fairness/README.md
docs/plans/roadmap.md
docs/plans/open-issues.md
docs/protocols/testing.md
scripts/run_tests.py
```

## 验证证据

```text
git log --oneline (relevant):
  45023af test(n2): handle local artifact availability
  86cda41 fix(benchmark): close N2 audit gaps
  aad1c59 feat(benchmark): add N2 Dense MoE fairness protocols

tests:
  scripts/run_tests.py full → 69 tests passed
  Stage 0 examples → 5/5 passed
  tests.test_n2_benchmark + tests.test_n2_result_schema → 8/8 passed

artifacts (gitignored, regenerated per N2 protocol):
  artifacts/n2-a-dense.json
  artifacts/n2-a-moe.json
  artifacts/n2-b-dense.json
  artifacts/n2-b-moe.json
  artifacts/n2-moe-routing-stats.json
```

## 审查复核（auditor 三项缺口闭合）

auditor 提出的三项实质缺口已在 commit `86cda41` 中修复，并在本次审查中再次复核：

1. `cache_binding()` 实际加载并校验 train/validation 双 cache：调用 `load_token_cache` → `validate_token_cache` 强制执行 split、dtype、endianness、文件存在、SHA256 match、byte-size match、tokenizer binding 校验；结果中 `train_cache_sha256` 和 `validation_cache_sha256` 为 schema 必填 64-char hex 字段。证据：`architecture_lab/benchmarks/n2.py::cache_binding`、`architecture_lab/data/batching.py::load_token_cache`、`schemas/n2_benchmark_result.schema.json::binding.required`。
2. 按 `token_budget` 实际执行 AdamW + scheduler + AMP/GradScaler + gradient accumulation 短训练，并在 validation cache 上计算 `validation_lm_loss`。证据：`architecture_lab/benchmarks/n2.py::train_for_benchmark`、`docs/protocols/n2-benchmark.md`。
3. `choose_protocol_d_ff()` 协议 B 保持 `dense_d_ff == moe_d_ff`（不主动搜索以维持 shared binding 测试断言）；协议 A 通过整数候选搜索 `moe expert d_ff ≈ dense_d_ff/4`，记录 target、delta、relative error。证据：`architecture_lab/benchmarks/n2.py::choose_protocol_d_ff`、`tests/test_n2_benchmark.py` 的 `test_executed_benchmark_records_validation_and_capacity_bindings`。

## 新发现问题（来自本次审查）

- 当前测试断言仅校验 `validation_lm_loss is not None`，未校验 `math.isfinite(...)` 与数值范围；建议 N3 之前补一条弱健壮性断言（提示级，非阻塞）。
- 干净 checkout 在 `artifacts/` 缺失时 `tests/test_n2_result_schema.py` 会跳过 artifact-specific 断言（commit `45023af` 处理），但应在 README/协议文档中明示 artifact 必须在 fast/module 之前按 N2 命令生成（已部分记录）。
- `autoAcceptDrafts: true` 当前项目设置保留在 `.pi-glla/settings.json`，仅在 N2/N3 过渡窗口使用；下个阶段审查后回滚评估（提示级）。

## 是否允许进入下一阶段

是（CAN_ENTER_N3）。

依据：
- N2 退出条件（4 个 benchmark JSON + 1 个 routing JSON 落盘并通过 schema）全部满足；
- 上述三项 auditor 缺口在代码/schema/artifact/tests 四层证据一致闭合；
- N3 范围由 `docs/plans/roadmap.md` 显式定义（统一 git commit / config hash / CUDA/PyTorch/GPU compute capability / seed 等元数据 + 多 seed 统计），与 `docs/plans/open-issues.md` 中 P1-01/P1-02/P1-03 的"归 N3"决策一一对接；
- P0-01..05 全部关闭或已明确延期，P0-01 正式 Dense 曲线归 N4，不在 N3 之前；
- 测试已纳入 `fast` / `module training` / `full` 三档。

## 下一步

1. 进入 roadmap N3，先以 `scout`/主 agent 完成 N3 设计 checkpoint（统一 schema、metadata 字段、多 seed 统计协议）。
2. 同步将 `open-issues.md` 中 P1-01/P1-02/P1-03 的"决策"段更新为"实施中"。
3. 复查 `.pi-glla/settings.json` 中的 `autoAcceptDrafts` 与 `auditorModel`，必要时回滚 `autoAcceptDrafts`。
4. 在 N3 完成时按 `docs/plans/review-process.md` 产出 `docs/plans/reviews/stage-n3-*.md`。