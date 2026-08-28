# D2 数据集扩样记录（5000 样本，2026-08-28 round 14 修订）

> 阶段：P3 D2 数据集扩样（接续 P3 D2 多轮数据集交付）。
> 状态：已交付 5000 样本 + 3500/750/750 split 契约；测试已参数化；待 detached auditor 终审。
> 关键变更（round 14）：从 5004 样本修订为 5000 样本以严格命中目标 `train=3500 / dev=750 / test=750`。

## 1. 背景

P3 阶段初版 D2 多轮数据集交付 600 样本（6 类各 100），但**单类样本量不足以支撑正式 GRPO rollout 池**：5000+ 样本才能让训练池与 held-out 评测之间达到统计上可对比的差异。本扩样阶段把 6 个 builder 的 variant 池从 ~100 unique 扩到 ≥833 unique，重新生成 5000 样本（per-task 833 / 834），保持 canonical semantic uniqueness + IID stratified 70/15/15 split + 跨 split canonical disjointness + 跨 D1/D1.1 dataset signature disjointness。

## 2. 目标

- (a) 6 类各 ≥833 unique canonical variants
- (b) **5000** samples（**train=3500 / dev=750 / test=750**）schema 合法 + canonical unique + cross-split disjoint + D1.1 disjoint
- (c) `scripts/run_tests.py full` 全绿
- (d) `docs/data/d2-expansion.md` 统计 + 变体池扩展说明（本文件）
- (e) reviewer dispatch 通过
- (f) 工作树干净；数据集不进入 Git 追踪

## 3. 变体池扩展

每个 builder 把 variant 池扩到远高于 834 unique variants，并通过**严格嵌套的 `(variant // N) % P` 维度公式**消除之前的 aliasing。

### 3.1 各 builder 扩展后维度

| Builder | 扩展后维度公式 | 扩展后组合上限 |
|---|---|---|
| `multi_tool_sequential` | 50 expr × 20 topic × 4 limit × 6 user × 4 answer | 96000 |
| `tool_error_response` | 50 texts × 10 langs × 4 requests × 4 answer | 8000 |
| `insufficient_result_search` | 40 queries × 2 limit × 6 request × 8 followup × 4 answer × 4 final | 61440 |
| `req_change_city` | 30 城市 → 435 ordered pairs × 6 first × 6 change × 4 answer | 62640 |
| `tool_not_available` | 40 capabilities × 5 request × 4 context × 4 refusal | 3200 |
| `error_recovery` | 30 variants × 2 first_limit × 4 second_limit × 5 answer × 5 first_ack | 6000 |

### 3.2 严格嵌套 `(variant // N) % P` 模式

每个 builder 都改用此模式代替旧的非线性 `//3, //7, //36` 等公式：

```python
idx_0 = variant % P_0
idx_1 = (variant // P_0) % P_1
idx_2 = (variant // (P_0 * P_1)) % P_2
...
```

这种嵌套保证 `LCM(P_0, P_1, P_2, ...) = P_0 × P_1 × P_2 × ...`，无 aliasing。

### 3.3 新增的扩展池内容

- **`_TRANSLATE_TEXTS`**: 20 → 50（含中英文短句、ML 工程术语、运维短语等）
- **`_TRANSLATE_LANGS`**: 5 → 10（新增韩语、西班牙语、俄语、意大利语、葡萄牙文）
- **`queries`** (`insufficient_result_search`): 25 → 40（新增 "Hyperparameter sweep 2026" 等更多 LLM 工程术语）
- **`cities`** (`req_change_city`): 16 → 30（覆盖二三线城市）
- **`_NOT_AVAILABLE_VARIANTS`**: 24 → 40（新增 "列车到站查询" 等 16 个真实业务场景的不可用能力）
- **`expressions`** (`multi_tool_sequential`): 20 → 50（覆盖更大整数范围）
- **`topics`** (`multi_tool_sequential`): 10 → 20

## 4. 数据集规格

### 4.1 规模

- 总样本数: **5000**（`--count 5000`）
- 5000 / 6 = 833 remainder 2 → 4 个 builder 各 833、2 个 builder 各 834
- Per-task count:
 - `tool_not_available`: **834**
 - `tool_error_response`: **834**
 - `insufficient_result_search`: **833**
 - `req_change_city`: **833**
 - `multi_tool_sequential`: **833**
 - `error_recovery`: **833**
- Per-split count:
 - train: **3500**（4×583 + 2×584 = 2332 + 1168）
 - dev: **750**（6×125）
 - test: **750**（6×125）
- split 分配算法：`assign_split_ids()` 使用 `round(per_class × 0.15)` 计算 dev/test（不是 `int(floor)`），使 833 与 834 都 round 到 125；train = per_class - dev - test。这样 6 类 dev=test=125 全对齐，总数刚好 3500/750/750。
- `build_samples()` 新增 `_plan_per_class_counts()` helper：默认调用即可拿到 `[834, 834, 833, 833, 833, 833]` 分布，外部可通过 `per_class_counts=` 显式覆盖。

### 4.2 IID stratified split

`assign_split_ids()` 按 task_type 分组 seeded shuffle，per-class 严格 70/15/15。同一 seed 下每次重跑结果完全一致（测试覆盖 `test_split_assignment_is_deterministic_for_same_seed`）。

### 4.3 canonical_content_signature 维度

去除 `id` / `call_id` / `tool_call_id` / `depends_on` / `metadata.created_at` / `metadata.split` 等 bookkeeping 字段，保留 task_type、tools、消息内容、工具参数/结果与 expected_answer。

### 4.4 跨数据集语义互斥

`cross_dataset_signature()` 投影 8 字段（task_type、schema_version、user_turns、assistant_turns、tool_turns、tool_names、expected_tool_calls、expected_answer）跨 D1/D1.1/D2 形状完全一致。

## 5. 验证（已完成）

```bash
# 1. 仿真校验（生成前）
$ .venv/python.exe -c "from collections import Counter, defaultdict; ..."
tool_not_available:        n=834 unique=834 (target ≥833) OK
tool_error_response:       n=834 unique=834 (target ≥833) OK
insufficient_result_search: n=833 unique=833 (target ≥833) OK
req_change_city:           n=833 unique=833 (target ≥833) OK
multi_tool_sequential:     n=833 unique=833 (target ≥833) OK
error_recovery:            n=833 unique=833 (target ≥833) OK
all_ok = True

# 2. 数据生成（带 MockExecutor 端到端验证）
$ time .venv/python.exe scripts/generate_d2_dataset.py --count 5000 --seed 2026 \
    --out datasets/tool-calling-d2
[generate_d2_dataset] count=5000 task_types={...} splits={'train': 3500, 'dev': 750, 'test': 750}
real    0m34.441s

# 3. 跨 split canonical disjointness（实际数据集）
train∩dev: 0
train∩test: 0
dev∩test: 0
canonical total unique: 5000 (expect 5000)

# 4. 跨数据集 disjointness（D2 vs D1+D1.1）
D1+D1.1 cross_dataset signatures: 1626
D2 train cross_dataset ∩ D1/D1.1: 0
D2 dev cross_dataset ∩ D1/D1.1: 0
D2 test cross_dataset ∩ D1/D1.1: 0

# 5. 测试（含 3500/750/750 契约显式断言）
$ .venv/python.exe scripts/run_tests.py full
Ran 296 tests OK（本机环境）/ 294 PASS + 2 environment-driven skips（isolated auditor 环境）
[test] full suite passed
9/9 PASS (stage0 schema examples)

注：isolated auditor 环境可能报告 `OK (skipped=2)`；2 个跳过均为 `tests/test_n2_benchmark.py:93, 108` 的 `@unittest.skipUnless(_has_working_cuda(), ...)`，是环境驱动跳过（auditor 隔离会话不一定有可用 CUDA kernel image），不是代码缺陷。其余 294 个测试在两种环境下均 PASS。
```

## 6. 工程改动

### 6.1 scripts/generate_d2_dataset.py

- 6 个 builder 全部采用严格嵌套 `(variant // N) % P` 维度公式
- 新增 `_plan_per_class_counts(total)` helper：返回 `count // 6` + 余数分配
- `build_samples()` 改用 per-class running counter（`intra_class_index`）作为 `variant`，而不是旧的 `build_pos // len(TASK_TYPES)`（后者会因总数变化而误判唯一性）
- `_TRANSLATE_TEXTS`: 20→50; `_TRANSLATE_LANGS`: 5→10
- `_NOT_AVAILABLE_VARIANTS`: 24→40
- `queries` (insufficient_result_search): 25→40
- `cities` (req_change_city): 16→30
- `expressions` (multi_tool_sequential): 20→50
- `topics` (multi_tool_sequential): 10→20
- `assign_split_ids()` 用 `round(per_class * 0.15)` 计算 dev/test，train 吸收余数；从而在 833/834 类都得到 dev=test=125
- `_write_manifests()` 新增 `build_count` 字段

### 6.2 tests/test_d2_dataset.py

- 新增 3500/750/750 split 的显式断言（manifest `count` 字段）
- 新增 per-task ≥833 unique canonical signatures 的显式断言
- per-class 总数 833 / 834 各自映射到 `{833: 583 train}, {834: 584 train}` 的期望分布表
- `_build_count()` helper 从 MANIFEST 读取 `build_count` 字段
- `tests/test_d2_dataset.py` 49 tests 全绿

### 6.3 .gitignore

- 新增 `datasets/tool-calling-d2/` 与 `datasets/tool-calling-d2-*/`
- `git rm -r --cached datasets/tool-calling-d2` 把已追踪的 5000 个样本与 3 个 MANIFEST 从 index 移除（磁盘保留）

## 7. 复现命令

```bash
# 1. 生成 5000 样本数据集（不带 Git 追踪）
.venv/python.exe scripts/generate_d2_dataset.py --count 5000 --seed 2026 \
    --out datasets/tool-calling-d2

# 2. 校验 D2 schema 合法 + canonical unique + disjointness
.venv/python.exe scripts/run_tests.py fast
.venv/python.exe scripts/validate_stage0.py --examples

# 3. 复跑 D2 真实推理 reward（任意公开模型）
export HF_ENDPOINT=https://hf-mirror.com
.venv/python.exe scripts/eval_transformers.py \
    --model <model-id> \
    --samples-dir datasets/tool-calling-d2/dev \
    --output artifacts/<safe-name>-eval-d2dev.json
.venv/python.exe scripts/reward_offline.py \
    --transcripts artifacts/<safe-name>-eval-d2dev.json \
    --samples-dir datasets/tool-calling-d2/dev \
    --checkpoint <model-id> \
    --transcript-kind model_generated \
    --output artifacts/<safe-name>-eval-d2dev-reward.json
```

## 8. 已知边界

- 数据集未进入 Git 追踪；如需在另一台机器复现，需重新跑生成命令（依赖 deterministic seed `2026` 与 `1785000000 + seed + index` 时间戳契约）。
- `--count` 必须是 ≥6 的整数；5000 是目标样本数，generator 内部自动把 5000 按 `count // 6 + remainder` 分布到 6 个 task_types（4 类 833 + 2 类 834）。其他 N 值也会按此规则自动规划。
- 测试通过 `_build_count()` 从 MANIFEST 读取 build_count 参数以参数化断言（不再硬编码 600/100）；测试与生成结果解耦。
- D2 vs D1/D1.1 cross_dataset_signature 投影对 8 字段保持完全一致的形状（见 `docs/plans/open-issues.md` round 12 段）。

## 9. 下一步

1. P4 GRPO MVP：使用 D2 5000 样本作为 GRPO rollout pool
2. P5-03 vLLM backend：在 Linux/WSL/Docker 环境接入公开 instruction-tuned 模型
3. P4 后端基准对比：D2 5000 训练 + P5-02/P5-03 推理 + P2 reward offline 的全链路 review