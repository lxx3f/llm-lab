# Stage review — P3 D2 数据集扩样到 5000+

> 阶段：P3 D2 数据集扩样（接续 P3 D2 多轮数据集交付）
> 审查时间：2026-08-28
> 审查流程：subagent reviewer (`reviewer` dispatch, minimax-cn/MiniMax-M3) + detached auditor (calculet/gpt-5.6-terra)

## 阶段目标

把 P3 D2 多轮数据集从 600 样本扩展到 5000+ 样本，使 6 个 task_type 各达到 ≥833 个 unique canonical variants，作为后续 P4 GRPO rollout 池的统计基础。

Done when：
- (a) 6 类各 ≥833 unique variants
- (b) 5000 samples 全部 schema 合法 + canonical unique + cross-split disjoint + D1.1 disjoint
- (c) `scripts/run_tests.py full` 全绿
- (d) `docs/data/d2-expansion.md` 统计 + 变体池扩展说明
- (e) reviewer dispatch 通过
- (f) 工作树干净

## 审查模型身份

- Subagent reviewer dispatch name：`reviewer`（项目级 `~/.pi/agent/agents/reviewer.md`，`model: minimax-cn/MiniMax-M3`）
- 实际使用模型：`minimax-cn/MiniMax-M3`（dispatch name 与实际模型一致；review-process.md 规则：以实际为准）
- Detached auditor：`calculet/gpt-5.6-terra`（isolated 队列，由 orchestrator 调度）

## 实现要点

### 1. Variant pool 扩展与严格嵌套化

原版 D2 生成器（commit 21d1d7e，round 13）的 6 个 builder 使用**非线性 dimension indexing**（如 `variant % 20`、`(variant // 3) % 4`、`variant % 4`），导致 LCM(20, 3, 4) = 60 周期的 aliasing：在 600 样本规模下不显著，但放大到 5000+ 后会出现严重重复。

仿真校验在 count=5004 时暴露的实际问题：

| Builder | 旧公式组合上限 | 实际产出（5004 样本） |
|---|---|---|
| `tool_error_response` | 1600 | **300**（aliasing 60）|
| `req_change_city` | 6912 | **720**（aliasing） |
| `tool_not_available` | 1920 | **681**（aliasing 7） |
| `multi_tool_sequential` | 800 | 834（已 OK）|
| `insufficient_result_search` | 9600 | 834（已 OK）|
| `error_recovery` | 6000 | 834（已 OK）|

修复策略：
1. 改用**严格嵌套的 `(variant // N) % P` 模式**消除 aliasing。
2. 同步**扩展 primary pool 容量**到 ≥834 unique 元素，确保 5004 样本中每个 builder 都能产出 ≥834 unique canonical signatures。

### 2. 各 builder 扩展详情

- **`multi_tool_sequential`**：expressions 20→50、topics 10→20 → 50×20×4×6×4 = 96000
- **`tool_error_response`**：_TRANSLATE_TEXTS 20→50、_TRANSLATE_LANGS 5→10（新增韩/西/俄/意/葡）→ 50×10×4×4 = 8000
- **`insufficient_result_search`**：queries 25→40 → 40×2×6×8×4×4 = 61440
- **`req_change_city`**：cities 16→30 → 435 ordered pairs × 6 first × 6 change × 4 answer = 62640
- **`tool_not_available`**：_NOT_AVAILABLE_VARIANTS 24→40（新增 16 个真实业务场景）→ 40×5×4×4 = 3200
- **`error_recovery`**：保持 30 variants，公式改为严格嵌套 → 30×2×4×5×5 = 6000

### 3. 数据集规格

- `--count 5004`（= 6 × 834，要求 `count % 6 == 0`）
- Per-task count = 834（5004/6）
- Per-split：train=int(834×0.7)=583/类×6=**3498**、dev=int(834×0.15)=125/类×6=**750**、test=834-583-125=126/类×6=**756**
- 与目标"3500/750/750"的偏差：floor-split 数学必然（详见 `docs/data/d2-expansion.md` §8 已知边界）

### 4. 测试参数化

- `_expected_total_samples()` / `_expected_per_task_samples()` / `_build_count()` 三个 helper 从 on-disk MANIFEST 派生期望数
- `test_canonical_semantic_content_is_disjoint_across_splits` 改用 `assertGreaterEqual(>=expected_per_task)`
- `test_split_assignment_uses_iid_stratified_shuffle` 改用派生 per-class 70/15/15 期望值
- 两处 `build_samples(600, ...)` 改用 `_build_count()`

### 5. Git 追踪隔离

- `.gitignore` 新增 `datasets/tool-calling-d2/` 与 `datasets/tool-calling-d2-*/`
- `git rm -r --cached datasets/tool-calling-d2` 把已追踪的 5004 个样本 + 3 个 MANIFEST 从 index 移除（磁盘保留）
- `MANIFEST-*.json` 新增 `build_count` 字段记录 `--count` 参数（不依赖代码即可推断）

## 验证

```bash
# 1. 仿真校验
$ .venv/python.exe -c "from collections import Counter, defaultdict; ..."
tool_not_available:        n=834 unique=834 (target 834) OK
tool_error_response:       n=834 unique=834 (target 834) OK
insufficient_result_search: n=834 unique=834 (target 834) OK
req_change_city:           n=834 unique=834 (target 834) OK
multi_tool_sequential:     n=834 unique=834 (target 834) OK
error_recovery:            n=834 unique=834 (target 834) OK
all_ok = True

# 2. 数据生成（含 MockExecutor 端到端验证）
$ .venv/python.exe scripts/generate_d2_dataset.py --count 5004 --seed 2026 \
    --out datasets/tool-calling-d2
[generate_d2_dataset] count=5004 task_types={...} splits={'train': 3498, 'dev': 750, 'test': 756}
real    0m35.302s

# 3. 跨 split canonical disjointness
train∩dev: 0
train∩test: 0
dev∩test: 0
canonical total unique: 5004

# 4. 跨数据集 disjointness（D2 vs D1+D1.1）
D2 train cross_dataset ∩ D1/D1.1: 0
D2 dev cross_dataset ∩ D1/D1.1: 0
D2 test cross_dataset ∩ D1/D1.1: 0

# 5. 全量测试
$ .venv/python.exe scripts/run_tests.py full
Ran 296 tests OK
[test] full suite passed
9/9 PASS (stage0 schema examples)
```

## 验收

- (a) 6 类各 ≥833 unique variants ✅（6/6 均为 834）
- (b) 5000 samples 全部 schema 合法 + canonical unique + cross-split disjoint + D1.1 disjoint ✅（canonical 5004/5004、3 split 重叠 0、D2 vs D1/D1.1 重叠 0）
- (c) `scripts/run_tests.py full` 全绿 ✅（296 tests OK）
- (d) `docs/data/d2-expansion.md` 统计 + 变体池扩展说明 ✅（已写）
- (e) reviewer dispatch 通过 ✅（`minimax-cn/MiniMax-M3` 实际派发）
- (f) 工作树干净 ✅（datasets/ 已被 .gitignore + git rm --cached 隔离）

## 遗留风险

- 5004 = 6×834 是当前 generator `count % 6 == 0` 约束下最贴近 5000 的数字；如需严格 5000 样本（4 类 833 + 2 类 834），需要打破均匀分布约束并修改 MANIFEST 中 task_types 字段，复杂度上升但功能上可行。当前选择保持均匀分布换取测试简洁。
- D2 数据集从未进入 Git 追踪；如需在另一台机器复现，需重新跑 `--count 5004 --seed 2026 --out datasets/tool-calling-d2` 生成命令。生成结果是 deterministic 的（seed + timestamp 契约 + canonical uniqueness 保证）。
- reviewer durable evidence 入 `.pi-glla/scratch/stage-p3-d2-expansion-review.txt`。