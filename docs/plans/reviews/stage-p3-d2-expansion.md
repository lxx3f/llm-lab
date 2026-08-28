# Stage review — P3 D2 数据集扩样到 5000（round 14）

> 阶段：P3 D2 数据集扩样（接续 P3 D2 多轮数据集交付）
> 审查时间：2026-08-28（round 14）
> 审查流程：subagent reviewer (`reviewer` dispatch, minimax-cn/MiniMax-M3) + detached auditor (calculet/gpt-5.6-terra)

## 阶段目标

把 P3 D2 多轮数据集从 600 样本扩展到 **5000** 样本（每类 ≥833 unique canonical variants），作为后续 P4 GRPO rollout 池的统计基础。

Done when：
- (a) 6 类各 ≥833 unique variants
- (b) **5000** samples 全部 schema 合法 + canonical unique + cross-split disjoint + D1.1 disjoint
- (b.1) **train=3500 / dev=750 / test=750** 严格命中
- (c) `scripts/run_tests.py full` 全绿
- (d) `docs/data/d2-expansion.md` 统计 + 变体池扩展说明
- (e) reviewer dispatch 通过
- (f) 工作树干净

## Round 14 修订动机

Round 13 完成时使用 `count=5004`，split 实际为 3498/750/756，与目标契约 3500/750/750 偏差 6 个样本（train 差 2、test 差 4）。Detached auditor round 14 否决该交付并要求严格命中契约。

5000 / 6 = 833.33，不整除。数学上必然需要 4 类 833 + 2 类 834 的不均分布，配合 `round()` 化 dev/test 才能命中 3500/750/750。

## 审查模型身份

- Subagent reviewer dispatch name：`reviewer`（项目级 `~/.pi/agent/agents/reviewer.md`，`model: minimax-cn/MiniMax-M3`）
- 实际使用模型：`minimax-cn/MiniMax-M3`（dispatch name 与实际模型一致；review-process.md 规则：以实际为准）
- Detached auditor：`calculet/gpt-5.6-terra`（isolated 队列，由 orchestrator 调度）

## 实现要点

### 1. Variant pool 扩展与严格嵌套化

6 个 builder 全部采用 `(variant // N) % P` 严格嵌套公式消除 aliasing；pool 容量 ≥833 unique 元素。

### 2. 各 builder 扩展详情

| Builder | 维度公式 | 组合上限 |
|---|---|---|
| `multi_tool_sequential` | 50 expr × 20 topic × 4 limit × 6 user × 4 answer | 96000 |
| `tool_error_response` | 50 texts × 10 langs × 4 requests × 4 answer | 8000 |
| `insufficient_result_search` | 40 queries × 2 limit × 6 request × 8 followup × 4 answer × 4 final | 61440 |
| `req_change_city` | 30 城市 → 435 ordered pairs × 6 first × 6 change × 4 answer | 62640 |
| `tool_not_available` | 40 capabilities × 5 request × 4 context × 4 refusal | 3200 |
| `error_recovery` | 30 variants × 2 first_limit × 4 second_limit × 5 answer × 5 first_ack | 6000 |

### 3. 数据集规格（round 14）

- `--count 5000`（不是 5004）
- `_plan_per_class_counts(5000)` → `(834, 834, 833, 833, 833, 833)`
- Per-split：`assign_split_ids()` 用 `round(per_class * 0.15)` 计算 dev/test
 - 833 类：dev=125、test=125、train=583
 - 834 类：dev=125、test=125、train=584
 - 合计：train=4×583 + 2×584 = 2332 + 1168 = **3500**；dev=6×125=**750**；test=6×125=**750**

### 4. per-class running counter（关键变更）

`build_samples()` 改用 `intra_class_index`（每个 task_type 内部 0..n-1）作为 `variant`，替代旧公式 `build_pos // len(TASK_TYPES)`。后者在总数变化（如 600→5000）时会重复返回 `variant=0`，破坏 canonical uniqueness。

### 5. 测试参数化（round 14）

- 显式断言 `expected_total == 5000`
- 显式断言每 split manifest `count ∈ {3500, 750, 750}`
- 显式断言每个 task_type 至少 833 个 unique canonical signatures
- 显式断言 per-class 总数 833 / 834 各自映射到 `{583, 125, 125}` / `{584, 125, 125}`

### 6. Git 追踪隔离

- `.gitignore` 新增 `datasets/tool-calling-d2/` 与 `datasets/tool-calling-d2-*/`
- `git rm -r --cached datasets/tool-calling-d2` 把已追踪的 5000 个样本 + 3 个 MANIFEST 从 index 移除（磁盘保留）
- `MANIFEST-*.json` 新增 `build_count` 字段（值 = `--count`）

## 验证

```bash
# 1. 仿真校验
$ .venv/python.exe -c "from collections import Counter, defaultdict; ..."
tool_not_available:        n=834 unique=834 (target ≥833) OK
tool_error_response:       n=834 unique=834 (target ≥833) OK
insufficient_result_search: n=833 unique=833 (target ≥833) OK
req_change_city:           n=833 unique=833 (target ≥833) OK
multi_tool_sequential:     n=833 unique=833 (target ≥833) OK
error_recovery:            n=833 unique=833 (target ≥833) OK
all_ok = True

# 2. 数据生成（含 MockExecutor 端到端验证）
$ .venv/python.exe scripts/generate_d2_dataset.py --count 5000 --seed 2026 \
    --out datasets/tool-calling-d2
[generate_d2_dataset] count=5000 task_types={...} splits={'train': 3500, 'dev': 750, 'test': 750}
real    0m34.441s

# 3. 跨 split canonical disjointness
train∩dev: 0
train∩test: 0
dev∩test: 0
canonical total unique: 5000

# 4. 跨数据集 disjointness（D2 vs D1+D1.1）
D2 train cross_dataset ∩ D1+D1.1: 0
D2 dev cross_dataset ∩ D1+D1.1: 0
D2 test cross_dataset ∩ D1+D1.1: 0

# 5. 全量测试
$ .venv/python.exe scripts/run_tests.py full
Ran 296 tests OK (本机) / 294 PASS + 2 skipped (isolated auditor 环境)
[test] full suite passed
9/9 PASS (stage0 schema examples)

注：`scripts/run_tests.py full` 在 isolated auditor 环境可能报告 `OK (skipped=2)`，2 个跳过均为 `tests/test_n2_benchmark.py:93, 108` 的 `@unittest.skipUnless(_has_working_cuda(), ...)`，是环境驱动跳过（auditor 隔离会话不一定有可用 CUDA kernel image），不是代码缺陷。其余 294 个测试在两种环境下均 PASS。
```

## 验收

- (a) 6 类各 ≥833 unique variants ✅（6/6 OK）
- (b) 5000 samples 全部 schema 合法 + canonical unique + cross-split disjoint + D1.1 disjoint ✅
- (b.1) train=3500 / dev=750 / test=750 严格命中 ✅
- (c) `scripts/run_tests.py full` 全绿 ✅（本机 296 tests OK；isolated auditor 环境 294 PASS + 2 environment-driven skips from `test_n2_benchmark.py:93, 108`）
- (d) `docs/data/d2-expansion.md` 统计 + 变体池扩展说明 ✅（已写）
- (e) reviewer dispatch 通过 ✅（minimax-cn/MiniMax-M3 PASS verdict）
- (f) 工作树干净 ✅（datasets/ 已被 .gitignore + git rm --cached 隔离）

## 遗留风险

- 5000 / 6 不整除导致 4 类 833、2 类 834 的不均分布；这是 round 14 的数学必然。
- round() 在 Python 3 下使用 banker's rounding（125 vs 126 偶数），对 833.0 / 834.0 都给出 125，行为可预测。
- reviewer durable evidence 入 `.pi-glla/scratch/stage-p3-d2-expansion-r14-review.txt`。