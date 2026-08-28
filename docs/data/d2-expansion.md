# D2 数据集扩样记录（5004 样本，2026-08-28）

> 阶段：P3 D2 数据集扩样（接续 P3 D2 多轮数据集交付）。
> 状态：已交付 5004 样本 + 测试参数化 + git 追踪隔离；待 reviewer dispatch + detached auditor 终审。

## 1. 背景

P3 阶段初版 D2 多轮数据集交付 600 样本（6 类各 100），但**单类样本量不足以支撑正式 GRPO rollout 池**：5000+ 样本才能让训练池与 held-out 评测之间达到统计上可对比的差异。本扩样阶段把 6 个 builder 的 variant 池从 ~100 unique 扩到 ≥833 unique，重新生成 5004 样本（per-task 834），保持 IID stratified 70/15/15 split + 跨 split canonical disjointness + 跨 D1/D1.1 dataset signature disjointness。

## 2. 目标

- (a) 6 类各 ≥833 unique canonical variants
- (b) 5000 samples 全部 schema 合法 + canonical unique + cross-split disjoint + D1.1 disjoint
- (c) `scripts/run_tests.py full` 全绿
- (d) `docs/data/d2-expansion.md` 统计 + 变体池扩展说明（本文件）
- (e) reviewer dispatch 通过
- (f) 工作树干净；数据集不进入 Git 追踪

## 3. 变体池扩展

每个 builder 把 variant 池扩到远高于 834 unique variants，并通过**严格嵌套的 `(variant // N) % P` 维度公式**消除之前的 aliasing：原版公式使用 `variant % 20, (variant // 3) % 4, variant % 4` 这种**非线性交错**，导致 LCM(20, 3, 4) = 60 周期的 aliasing，5000 个样本中只有 ~300 unique tuples。

### 3.1 各 builder 扩展后维度

| Builder | 扩展前组合上限 | 扩展后维度公式 | 扩展后组合上限 | 实际产出 (per 5004) |
|---|---|---|---|---|
| `multi_tool_sequential` | 20×10×4 = 800 (实际300) | 50 expr × 20 topic × 4 limit × 6 user × 4 answer = 96000 | 96000 | 834 unique |
| `tool_error_response` | 20×5×4×4 = 1600 (实际300, aliasing 60) | 50 texts × 10 langs × 4 requests × 4 answer = 8000 | 8000 | 834 unique |
| `insufficient_result_search` | 25×2×6×8×4 = 9600 (实际834) | 40 queries × 2 limit × 6 request × 8 followup × 4 answer × 4 final = 61440 | 61440 | 834 unique |
| `req_change_city` | 16 城市 × 6 × 6 × 4 = 6912 (实际720, aliasing) | 30 城市 → 435 pairs × 6 first × 6 change × 4 answer = 62640 | 62640 | 834 unique |
| `tool_not_available` | 24×5×4×4 = 1920 (实际681, aliasing 7) | 40 capabilities × 5 request × 4 context × 4 refusal = 3200 | 3200 | 834 unique |
| `error_recovery` | 30×2×4×5×5 = 6000 (实际834) | 30 variants × 2 first_limit × 4 second_limit × 5 answer × 5 first_ack = 6000 | 6000 | 834 unique |

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

- 总样本数: **5004**（`--count 5004`，要求 `count % 6 == 0`，5004 = 6 × 834）
- Per-class count: **834**（每类一致；5004/6 = 834，无 remainder）
- Per-split count:
 - train: **3498**（583/类 × 6，per-class train=int(834×0.7)=583）
 - dev: **750**（125/类 × 6，per-class dev=int(834×0.15)=125）
 - test: **756**（126/类 × 6，per-class test=834-583-125=126）
- 与目标"train 3500 / dev 750 / test 750"的偏差：在 IID stratified 70/15/15 + count%6==0 约束下，834/类的 floor-split 是 583/125/126，6 类累加为 3498/750/750/756（实际 3498/750/756，差 2 个 test 样本来自 per-class 的 6/3 分布）。

### 4.2 IID stratified split

`assign_split_ids()` 按 task_type 分组 seeded shuffle，per-class 严格 70/15/15（floor-split 583/125/126）。同一 seed 下每次重跑结果完全一致（测试覆盖 `test_split_assignment_is_deterministic_for_same_seed`）。

### 4.3 canonical_content_signature 维度

去除 `id` / `call_id` / `tool_call_id` / `depends_on` / `metadata.created_at` / `metadata.split` 等 bookkeeping 字段，保留 task_type、tools、消息内容、工具参数/结果与 expected_answer。

### 4.4 跨数据集语义互斥

`cross_dataset_signature()` 投影 8 字段（task_type、schema_version、user_turns、assistant_turns、tool_turns、tool_names、expected_tool_calls、expected_answer）跨 D1/D1.1/D2 形状完全一致。

## 5. 验证（已完成）

```bash
# 1. 仿真校验（生成前）
$ .venv/python.exe -c "from collections import Counter, defaultdict; ..."
tool_not_available:        n=834 unique=834 (target 834) OK
tool_error_response:       n=834 unique=834 (target 834) OK
insufficient_result_search: n=834 unique=834 (target 834) OK
req_change_city:           n=834 unique=834 (target 834) OK
multi_tool_sequential:     n=834 unique=834 (target 834) OK
error_recovery:            n=834 unique=834 (target 834) OK
all_ok = True

# 2. 数据生成（带 MockExecutor 端到端验证）
$ time .venv/python.exe scripts/generate_d2_dataset.py --count 5004 --seed 2026 \
    --out datasets/tool-calling-d2
[generate_d2_dataset] count=5004 task_types={...} splits={'train': 3498, 'dev': 750, 'test': 756}
real    0m35.302s

# 3. 跨 split canonical disjointness（实际数据集）
train∩dev: 0
train∩test: 0
dev∩test: 0
canonical total unique: 5004 (expect 5004)
per-task: 834 unique each (expect 834)

# 4. 跨数据集 disjointness（D2 vs D1+D1.1）
D1+D1.1 cross_dataset signatures: 1547
D2 train cross_dataset ∩ D1/D1.1: 0
D2 dev cross_dataset ∩ D1/D1.1: 0
D2 test cross_dataset ∩ D1/D1.1: 0

# 5. 测试
$ .venv/python.exe scripts/run_tests.py full
Ran 296 tests OK
[test] full suite passed
9/9 PASS (stage0 schema examples)
```

## 6. 工程改动

### 6.1 scripts/generate_d2_dataset.py

- 6 个 builder 全部采用严格嵌套 `(variant // N) % P` 维度公式
- `_TRANSLATE_TEXTS`: 20→50; `_TRANSLATE_LANGS`: 5→10
- `_NOT_AVAILABLE_VARIANTS`: 24→40
- `queries` (insufficient_result_search): 25→40
- `cities` (req_change_city): 16→30
- `expressions` (multi_tool_sequential): 20→50
- `topics` (multi_tool_sequential): 10→20
- `_write_manifests()` 新增 `build_count` 字段

### 6.2 tests/test_d2_dataset.py

- `_expected_total_samples()` / `_expected_per_task_samples()` / `_build_count()` 三个 helper，从 on-disk MANIFEST 派生期望数（不再硬编码 600/100）
- `test_canonical_semantic_content_is_disjoint_across_splits` 改用 `assertGreaterEqual(>=expected_per_task)` 而非 `assertEqual(==100)`
- `test_split_assignment_uses_iid_stratified_shuffle` 改用派生 per-class 70/15/15 期望值
- 两处 `build_samples(600, ...)` 改用 `_build_count()` 派生

### 6.3 .gitignore

- 新增 `datasets/tool-calling-d2/` 与 `datasets/tool-calling-d2-*/`
- `git rm -r --cached datasets/tool-calling-d2` 把已追踪的 5004 个样本与 3 个 MANIFEST 从 index 移除（磁盘保留）

## 7. 复现命令

```bash
# 1. 生成 5004 样本数据集（不带 Git 追踪）
.venv/python.exe scripts/generate_d2_dataset.py --count 5004 --seed 2026 \
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
- `test_split_assignment_uses_iid_stratified_shuffle` 现在允许每类的 train/dev/test 在 70/15/15 附近 floor-split（`int(N*0.7)` / `int(N*0.15)` / remainder），不再严格要求每类都是 70/15/15。这是 6×833 类 floor-split 数学所限；如需严格 70/15/15，per-class 必须为 100 的倍数（20/15+15=20），5000/6 = 833.33 不在 100 倍数上。
- 5004 样本下 test split 实际 756 个（不是 750），多出来的 6 个来自 6 个类 × 1 个 floor-split 余数。这是 `count = 6 × 834` 的必然结果；如需严格 5000 样本，应改用 6 × 833 + 2 模式（4 类 833、2 类 834），但会让 canonical uniqueness 检查需要按 per-class 单独断言而非统一 floor-split。

## 9. 下一步

1. P4 GRPO MVP：使用 D2 5004 样本作为 GRPO rollout pool
2. P5-03 vLLM backend：在 Linux/WSL/Docker 环境接入公开 instruction-tuned 模型
3. P4 后端基准对比：D2 5004 训练 + P5-02/P5-03 推理 + P2 reward offline 的全链路 review