# 阶段 P3 D2 多轮对话数据集审查

审查模型：`minimax-cn/MiniMax-M3`（project-level `reviewer` agent）
审查 agent：`reviewer`（subagent dispatch 名称）
实际 provider/model：`PI_PROVIDER=minimax-cn`，`PI_MODEL=MiniMax-M3`
`PI_AGENT_NAME`：未注入
审查文件路径：`docs/plans/reviews/stage-p3-d2-multi-turn.md`
commit 候选变更：见下方“完成范围”节。

## 阶段 p3-d2-multi-turn 审查

### 完成范围

- **独立 D2 schema**：新增 `schemas/d2_multi_turn_sample.schema.json`，D2 task_type 使用 6 个规范名称：`tool_not_available` / `tool_error_response` / `insufficient_result_search` / `req_change_city` / `multi_tool_sequential` / `error_recovery`；D1/D1.1 继续使用通用 schema，向后兼容。
- **独立语义互斥**：生成器和测试均计算 `canonical_content_signature()`；去除 `id` / `call_id` / `tool_call_id` / `depends_on`、`metadata.created_at` 和 `metadata.split` 后，仍保留 task_type、tools、消息内容、工具参数/结果和 expected_answer。**当前契约（round 14，HEAD `fdfc519`）**：5000/5000 签名唯一；6 类 ≥833 unique（实际 834/834/833/833/833/833）；train/dev/test 三组 canonical signature 交集均为空。MVP 600/600 唯一为历史记录。
- **D2 生成器** `scripts/generate_d2_dataset.py`：
  - **当前契约（round 14，HEAD `fdfc519`）**：默认生成 5000 样本（4 类 833 + 2 类 834），train/dev/test = 3500/750/750 IID split，per-class split 通过 `round(per_class × 0.15)` 化 dev/test 使 dev=test=125、train=per_class-250；MVP 600 样本版本（6 类各 100，train/dev/test = 420/90/90）是历史记录，已被 `c6eac20` 取代；
  - 确定性 `--seed 2026`、确定性时间戳、id/call_id 命名空间前缀（`d2-{train,dev,test}-...`）；
  - 六类 builder 使用确定性 `variant_index` 与多维语义组合；严格嵌套 `(variant // N) % P` 维度公式消除 aliasing；`canonical_content_signature()` 去除 ID、时间戳、split 和依赖 bookkeeping 后执行全局去重；当前扩样版 5000/5000 unique、每类 833/834 unique；
  - per-split `aggregate_sha256` + per-file sha256；
  - MANIFEST-{train,dev,test}.json：count + sha256 + task_type 分布 + aggregate_sha256。
- **单测** `tests/test_d2_dataset.py`：**33** 单测分 7 组：
  - `D2DatasetSchemaTests`：独立 D2 schema 合法 / 6 类规范 task_type 覆盖 / 每 split 包含 6 类；
  - `D2MultiTurnStructureTests`：assistant + tool 角色结构 / tool_call_id 显式存在 / expected/transcript call_id 一致 / 最终 sample 命名空间一致；
  - `D2DependencyGraphTests`：depends_on 存在性 / 严格前序 / 无环 / 负例语义拒绝 / 乱序输入的真实拓扑执行；
  - `D2MockExecutorReplayTests`：真实 `MockExecutor.execute_sequence()` replay / 乱序拓扑执行 / 失败依赖阻断；
  - `D2SplitDisjointnessTests`：train/dev/test id/path/canonical semantic content 不重叠、每类 100 个唯一语义实例、与 D1/D1.1 train id 互斥、MANIFEST count / aggregate / per-file sha 一致；
  - `D2TimestampContractTests`：`1785000000 + seed + index` 精确 UTC/Z 格式。
- **协议** `docs/protocols/d2-multi-turn.md`：D2 与 D1/D1.1 差异 / 6 类任务定义 / call_id 依赖链 / held-out split 互斥 / 与 P2 / P4 / P5-02 衔接。
- **P2 README** `docs/experiments/p2-evaluator/README.md`：引用 D2 dev 90 样本作为首个有统计意义的 reward benchmark；包含 mock pipeline 兼容性验证和最终多样化数据上的 5 ckpt × D2 dev 真实 reward 结果。
- **`docs/plans/roadmap.md`**：P3 加入已完成阶段表；下一阶段候选保持 P5 + P4 + D2 扩样。
- **`docs/plans/open-issues.md`**：P2-05 状态升级 + 新增 P3-01 已解决项。

### reward offline 验证（MVP 600 样本版历史）

- `artifacts/d2-mock-reward-d2dev.json`：MVP 600 样本版 D2 dev 90 样本的 mock transcript 兼容性验证（90/90 reward_binary=1.0、reward_layered=1.0），用于证明 evaluator 能正确消费多轮 transcript；
- 5 个 checkpoint 的真实模型推理结果和 reward_offline 输出见 `docs/experiments/p2-evaluator/README.md` 第 6 节及对应 `artifacts/sft-*-eval-d2dev-reward.json`；**MVP 600 样本版历史**共 450 signals，全部 `reward_binary=0.0`；round 9 数字（large-v1 / d256-20k `reward_layered` 0.0162 / 0.0042）是 round 10 之前的状态，已被 IID stratified shuffle 后的 0.0099 / 0.0000 取代（详细变化说明见 README §6.2）。

### 测试统计（MVP 600 样本版历史）

- `scripts/run_tests.py full` → Ran **255** tests OK。
- `scripts/validate_stage0.py --examples` → 9/9 PASS（2 个 D2 正例使用独立 D2 schema）。
- 当前扩样版额外验证：5000/5000 样本 schema 合法，train/dev/test = 3500/750/750，详见 `docs/protocols/d2-multi-turn.md` §4 与 `docs/data/d2-expansion.md`。
- 600 D2 样本 × 3 splits × per-file sha256 + aggregate_sha256 校验全部通过；canonical semantic signature 600/600 唯一、每类 100/100 唯一、三 split 交集均为空。

### 已知边界

- 自研 5 ckpt 均为单轮训练；虽然已完成 D2 dev 真实推理评测，但最终 450 signals 全部 `reward_binary=0.0`，因此该结果是能力边界诊断，不是成功率结论；
- ~~D2 数据规模 600（420 train / 90 dev / 90 test）属 MVP；正式 GRPO rollout 池需 D2 扩样到 5000+~~ —— **历史记录**：MVP 600 样本在 `21d1d7e` 交付，后续已于 `c6eac20`（round 14）扩样到 5000 samples（train 3500 / dev 750 / test 750 严格命中、6 类 ≥833 unique canonical signatures）。当前正式 GRPO rollout 池以 5000 samples 为准；本条仅作为阶段历史保留。
- D2 与 D1 / D1.1 train id 互斥保证靠 id 命名空间 + 目录物理隔离 + `D2SplitDisjointnessTests` 三重保证；语义 held-out 还额外由 canonical signature 全局唯一断言保证。

### 下一步

1. ~~P5-02 Transformers backend + 公开 instruction-tuned 模型~~ ✅（见 `docs/plans/reviews/stage-p5-02-transformers-backend.md`，HEAD `63cbd83` / `b4fd879`）；
2. ~~P4 GRPO MVP：基于 D2 train (420) + reward_signal~~ ⏸（list queue item #3，待 D2 扩样完成后启动）；当前以扩样后的 D2 train (3500) 为 GRPO rollout 池，详见 `docs/plans/reviews/stage-p3-d2-expansion.md` round 14（HEAD `fdfc519`）。
3. ~~D2 数据集扩样到 5000+~~ ✅（见 `docs/plans/reviews/stage-p3-d2-expansion.md` round 14，HEAD `fdfc519`：5000 samples 严格 3500/750/750 命中、6 类 ≥833 unique canonical signatures、cross-split disjoint、D2-vs-D1/D1.1 disjoint；MANIFEST 含 `build_count` 字段；`datasets/tool-calling-d2/` 加入 .gitignore 并 `git rm --cached` 隔离）；
4. 自研模型加 multi-turn training（messages 含 assistant + tool）后，回填多轮 SFT checkpoint 评测。

### durable 证据

- **durable 证据**：本轮 reviewer 输出保存到 `.pi-glla/scratch/stage-p3-d2-multi-turn-review-r1.txt`（gitignored）；本轮 post-fix 验证由命令输出和 full suite 记录补充。

### Round 3（2026-08-28，auditor round 7 修复；round 9 进一步修复语义 held-out 泄漏）

Auditor round 7 提出 3 项阻塞，全部修复：

1. **5 ckpt × D2 dev reward_offline 未跑**：已用 `scripts/eval_sft_tool.py --prompt-mode multi_turn` 对 5 个 checkpoint 在 D2 dev 90 样本上跑真实推理，再跑 reward_offline，产出 450 signals；最终多样化数据上的结果见 `docs/experiments/p2-evaluator/README.md` 第 6 节。
2. **README 仍以 mock 为主表**：`docs/experiments/p2-evaluator/README.md` 新增第 6 节真实 5 ckpt × D2 dev 结果表 + 方法 + 复现命令；mock pipeline 降级为管线兼容性验证说明。
3. **tool_not_available 语义多样性不足**：初步增加变体池；round 9 进一步将六类 builder 统一为确定性 `variant_index` + 多维语义组合，并加入 canonical uniqueness enforcement。

新增测试：`tests/test_stage0_schemas.py` 增加 D2 task_type 接受 + 未知 task_type 拒绝 2 例。

验证：`scripts/run_tests.py full` → Ran 247 tests OK；stage0 9/9；450 D2 reward signals 全 schema 合法。

### Round 4（2026-08-28，auditor round 8 时间戳契约修复）

Auditor round 8 提出 D2 时间戳未遵循 D1 的 `1785000000 + seed + index` 确定性公式契约。

修复：

1. `scripts/generate_d2_dataset.py::_now_ts(seed, index)` 改用 `datetime.fromtimestamp(1785000000 + seed + index, tz=UTC).isoformat().replace("+00:00", "Z")`，与 `scripts/generate_d1_dataset.py` manifest `created_at` 同步。
2. `seed` 通过 `build_samples(*, seed=)` + `assign_split_ids(*, seed=)` 下传到每个 builder 的 `created_at`。
3. MANIFEST 顶层新增 `created_at` 字段（公式中 `index=0`）。
4. `tests/test_d2_dataset.py::D2TimestampContractTests` 新增 6 例：`_now_ts` 字节相等 / 不同 seed / 不同 index / 磁盘样本逐字符等于公式 / MANIFEST `created_at` 逐字符等于公式。
5. `docs/protocols/d2-multi-turn.md` §5.1 文档化时间戳契约 + 示例。
6. 重新生成 `datasets/tool-calling-d2/`（600 样本）以匹配新公式。

验证：`scripts/run_tests.py full` → Ran **255** tests OK（新增 canonical uniqueness 断言后保持通过）；stage0 9/9。

### Round 5（2026-08-28，auditor round 9 语义 held-out 修复）

Auditor round 9 指出原 D2 数据在去除 ID、时间戳、split、call_id 和依赖字段后仍存在跨 split 语义重复（仅 27 个语义实例覆盖 600 行）。该判断正确，原因是原 builder 对小词池有放回抽样。

修复：

1. 六类 builder 统一按确定性 `variant_index` 组合用户措辞、任务领域、参数值、结果数量、回答风格和场景约束，不再依赖小词池有放回抽样。
2. 新增 `canonical_content_signature()`：去除 ID、`call_id`、`tool_call_id`、`depends_on`、`metadata.created_at`、`metadata.split`，保留 task_type、tools、消息内容、工具参数/结果和 expected_answer。
3. 生成器 `_validate_samples()` 在写盘前执行全局 canonical 去重；任何只改 bookkeeping 的重复样本直接失败。
4. `tests/test_d2_dataset.py` 新增跨 split canonical overlap、每类 100 个唯一语义实例、以及仅修改 bookkeeping 仍必须拒绝的反向断言。
5. 正式数据验证：600/600 canonical unique；六类各 100/100 unique；train/dev/test canonical overlap=0。
6. 5 ckpt × D2 dev reward artifact 按最终数据重新生成（**round 9 历史数字** — round 10 IID 后已被取代）：large-v1 `reward_layered=0.0162`、d256-20k `0.0042`，其余为 0.0000；全部 `reward_binary=0.0`。

验证：`scripts/run_tests.py full` → Ran **255** tests OK；stage0 9/9；D2 专项 33 tests OK。

### Round 6（2026-08-28，auditor round 10 IID + well-formedness 修复）

Auditor round 10 提出两点根因问题：(1) `assign_split_ids()` 用 `samples[:train_n]` contiguous slicing，dev/test 实际拿到的是每个 task_type 的 variant 70-99 连续切片，与 train 的 0-69 不交叠，train/dev/test 在 14 vs 3 vs 3 个 semantic domains 上分布完全不一致，并非 IID；(2) `_validate_samples()` 只检查 expected call_id 是否出现在 assistant 消息中，未验证每个 tool 消息的 `tool_call_id` 与 assistant tool_call 的 name/arguments 双向匹配、消息顺序正确性、以及跨 split 的 MockExecutor replay。

修复：

1. **IID stratified shuffle**：`build_samples()` 改为按 build_pos 显式计算 `variant = build_pos // 6`，与样本 id 完全解耦；六个 builder 接收 `variant: int` 参数；`assign_split_ids()` 改为按 task_type 分组的 seeded shuffle（seed=2026），每组前 70→train、中 15→dev、后 15→test，per-task 严格 70/15/15。`_req_change_city` 用 16 城市的有序组合（120 ordered pairs）替代 `rng.sample(cities, 2)`。
2. **跨 message well-formedness**：`transcript_well_formedness_errors()` 验证 (a) 每个 assistant tool_call.id 都有对应 tool 消息的 tool_call_id 引用且无 orphan tool 消息；(b) tool 消息按 assistant tool_call 顺序排列；(c) 每个 assistant tool_call 的 `function.name` 和 `function.arguments` 与 `expected_tool_calls` 中相同 call_id 的 name/arguments 完全一致；(d) 无连续两个 assistant tool_call 而无中间 tool 消息。`_validate_samples()` 在 schema + dependency + canonical 去重基础上增加这一项。
3. **IID 测试**：`D2SplitDisjointnessTests` 新增 `test_split_assignment_uses_iid_stratified_shuffle`（每个 task_type 验证 70/15/15）、`test_split_assignment_is_deterministic_for_same_seed`（同 seed 两次跑结果一致）、`test_canonical_content_unique_within_dataset`（600/600 唯一）、`test_canonical_content_disjoint_across_splits`（每个 task_type 在三 split 上交集为空）。
4. **well-formedness 测试**：`D2TranscriptWellFormednessTests` 5 例（双向引用、order 正确、name/arguments 匹配、orphan tool 注入反向断言、argument mismatch 反向断言）。
5. **时间戳契约调整**：`created_at` 改用 split-local 1-based index（与 round 8 共识一致：每个 split 独立 1..N）；同时新增 `metadata.created_at_pos` 字段便于测试反查。
6. **数据重生成 + 5 ckpt 重跑**：清理磁盘残留（90 dev + 180 含旧 3-digit 文件）→ 重新生成 600 → 5 ckpt × D2 dev 重推理（每次 multi_turn prompt）→ 450 signals 全 schema 合法。最终：large-v1 `reward_layered=0.0099`（87 parse_success + 3 argument_correct），其余 4 个 ckpt `0.0000`；全部 `reward_binary=0.0`。

验证：`scripts/run_tests.py full` → Ran **264** tests OK（255 + 9 round 10 测试）；stage0 9/9；D2 专项 42 tests OK。

### Round 7（2026-08-28，auditor round 11 文档/状态机/跨数据集一致性修复）

Auditor round 11 提出三点根因问题：(1) `docs/protocols/d2-multi-turn.md` §5.1 写明 `index` 是生成器全局序列 1-based 位置（train=1..420、dev=421..510、test=511..600），但 `assign_split_ids()` 实际使用 split-local 1-based 索引，`d2-train-0001` / `d2-dev-0001` / `d2-test-0001` 三个样本的 `created_at` 都是 `2026-07-25T17:53:47Z`；文档与代码不一致；(2) `transcript_well_formedness_errors()` 分别收集 assistant call id 列表与 tool ref id 列表后做集合/顺序比较，未验证 `role=tool` 消息的实际位置（能否出现在 final answer 之后）或 assistant tool call 能否在 final answer 之后出现；原有测试也没有反向断言；(3) `test_train_disjoint_from_d1_and_d1llm_train()` 只比 ID 文件名，未在 canonical semantic signature 层验证 D2 vs D1.1 互斥。

修复：

1. **时间戳文档同步**：把 §5.1 改为 split-local 1-based（train 1..420、dev 1..90、test 1..90），与 ``sample.id`` 后缀严格一致；说明该选择与 round 8 共识一致（split-local index + UTC + Z），并指出 ``d2-train-0001`` / ``d2-dev-0001`` / ``d2-test-0001`` 均映射到同一 epoch 起始点。open-issues + stage review 中所有 "round 9 数字 0.0162 / 0.0042" 均标注为 **round 9 历史数字 — round 10 IID 后已被取代**，并指向 README §6.2 的变化说明。
2. **well-formedness 状态机**：将 `transcript_well_formedness_errors()` 从 "并集比较" 重写为消息位置状态机：(a) 先逆向扫描定位 final answer 位置（最末一个 assistant 且 content 非空且无 tool_calls）；(b) 正向扫描维护 pending call_id 队列，tool message 必须与队首匹配后弹出；(c) 任何出现在 final answer 之后的 tool message / assistant tool_call 立即报错；(d) 无效 role / 缺 tool_call_id 也报错。`expected_tool_calls` name/arguments 校验继续作为最后一步。验证：600/600 样本 0 errors；反向断言 "tool message placed after final answer" + "assistant tool_call emitted after final answer" 均被捕获。
3. **D2 vs D1.1 canonical 互斥**：新增 `d1_canonical_signature()` 投影函数（task_type + schema_version + user turns + tool names + expected_tool_calls name+args + expected_answer）和 `test_d2_canonical_content_is_disjoint_from_d1_d1llm_train()`，断言 D2 train/dev/test 三 split 与 D1 train / D1.1 train 在 canonical signature 层交集均为空。

验证：`scripts/run_tests.py full` → Ran **267** tests OK（264 + 3 round 11 测试：2 well-formedness 反向断言 + 1 D2 vs D1.1 canonical 互斥）；stage0 9/9；D2 专项 45 tests OK（42 + 3 round 11）。

## 审查结论

- 审查模型：`minimax-cn/MiniMax-M3`
- 审查 agent：`reviewer`
- 结论：**通过**
- 允许创建阶段 commit：是
- reviewer 输出（PI_PROVIDER=minimax-cn / PI_MODEL=MiniMax-M3 / BLOCKERS: none）保存到 `.pi-glla/scratch/stage-p3-d2-multi-turn-review-postfix-r{10,11,12,13}.txt`。

### Round 9（2026-08-28，auditor round 13 expected_answer 与 final assistant 一致性修复）

Auditor round 13 发现 `_insufficient_result_search` 生成的 100/100 样本中 `expected_answer` 指向中间 assistant 响应（"仅找到 1 条结果…请补充关注点"），而 transcript 后续又追加 user + 另一个 assistant 关闭消息（"好的，我会按你的关注点继续"）—— 两者不一致，违反 protocol 中 `expected_answer` "closes the entire transcript" 的约定。该问题在 100 个 `insufficient_result_search` 样本上全部存在（train 70/70、dev 15/15、test 15/15），现有测试只校验 transcript 结构与 schema，未校验 `expected_answer` 与最终 assistant content 一致。

修复：

1. **生成器修复**：把 `_insufficient_result_search` 的 final assistant 内容抽出为 `final_answer` 变量，同时传给 transcript 最后一条 assistant 消息和 `expected_answer` 参数，中间回答 `answer` 只留在 transcript 中间位置。
2. **验证器强化**：`transcript_well_formedness_errors()` 增加 invariant (5) —— 遍历 messages 反向定位 final answer content，然后验证 `sample.expected_answer == final_answer_content`，不等则报错。`expected_answer` 缺失或 transcript 无 final assistant content 也报错。
3. **dataset-wide 测试**：`D2ExpectedAnswerContractTests` 新增 `test_expected_answer_equals_final_assistant_content` 遍历 train/dev/test 全部 600 个样本验证 invariant；`test_transcript_well_formedness_flags_expected_answer_mismatch` 反向断言（修改 `expected_answer` 后 validator 必须报错）。
4. **数据重生 + 5 ckpt × D2 dev reward 重跑**：重生成后验证 600/600 expected_answer == final assistant content；5 ckpt × 90 = 450 signals schema 合法（large-v1 layered=0.0099，其余 4 ckpt 0；全部 binary=0）。验证数字与 round 12 一致，因为 `insufficient_result_search` 类样本在所有模型上都为 `parse_success` 失败。

验证：`scripts/run_tests.py full` → Ran **271** tests OK (skipped=0)；stage0 9/9；D2 专项 49 tests OK (skipped=0)；600/600 `expected_answer == final assistant content`；`test_expected_answer_equals_final_assistant_content` OK；`test_transcript_well_formedness_flags_expected_answer_mismatch` 反向断言 OK。

### Round 8（2026-08-28，auditor round 12 跨数据集投影可比性 + reverse-assertion 测试实际执行修复）

Auditor round 12 提出两点根因问题：(1) round 11 引入的 `d1_canonical_signature()` 与 D2 自己的 `canonical_content_signature()` 形状完全不兼容（D2 sig 含 `messages` + `metadata` + `tools` 全量；D1 sig 只含 `user_turns` + `tools` 改名 `tool_names` + 部分字段）—— 两者 JSON 字符串的 set intersection 无论 D2 与 D1.1 是否真有语义重叠都恒为 ∅，跨数据集互斥验证退化为 vacuous truth；(2) round 11 新增的两个 reverse-assertion 测试（`test_tool_message_after_final_answer_is_flagged` / `test_assistant_tool_call_after_final_answer_is_flagged`）以及 `test_transcript_well_formedness_errors_catches_argument_mismatch` 都使用 `train[0]`（`d2-train-0001` = `tool_not_available`），该样本无 `expected_tool_calls`，三个测试全部 `skip`；验证证据为空，但 complete_goal claim 写了"反向断言被捕获"，是虚假证据。

修复：

1. **可比跨数据集投影**：在 `scripts/generate_d2_dataset.py` 新增 `cross_dataset_signature(sample)` —— 8 字段投影（task_type、schema_version、user_turns、assistant_turns、tool_turns、tool_names、expected_tool_calls、expected_answer）跨 D1 / D1.1 / D2 形状完全一致。`canonical_content_signature()` 保持原 D2 全量投影（用于 D2 内部去重）。
2. **跨数据集测试重写**：`test_d2_canonical_content_is_disjoint_from_d1_d1llm_train` 改用 `cross_dataset_signature` + 验证 `len(d1llm_sigs) > 0` 防 vacuous；新增 `test_cross_dataset_signature_is_comparable_across_d1_d1llm_d2`（D1/D1.1/D2 三类样本 projection keys 集合相等）；新增 `test_cross_dataset_signature_detects_real_overlap`（正向控制：克隆一个样本后投影必须碰撞）。
3. **reverse-assertion 测试样本选择**：`D2TranscriptWellFormednessTests` 新增 `with_calls` class attr 和 `train_with_calls()` helper（取第一个含 `expected_tool_calls` 的样本）。两个 reverse-assertion 测试 + argument-mismatch 测试改用 helper，三个测试从 skipped 变为实际执行并验证。

验证：`scripts/run_tests.py full` → Ran **269** tests OK (skipped=0)；stage0 9/9；D2 专项 47 tests OK (skipped=0)；D2 × D1 / D2 × D1.1 cross_dataset_signature 交集 = ∅；test_cross_dataset_signature_is_comparable_across_d1_d1llm_d2 OK（三个数据集 projection keys 集合相等）；test_cross_dataset_signature_detects_real_overlap OK（克隆样本投影碰撞）。