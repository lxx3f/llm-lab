# 阶段 P3 D2 多轮对话数据集审查（当前契约：5000/3500/750/750）

> 本文档是 P3 阶段 D2 多轮对话数据集的 stage review。它描述当前活跃契约（5000 样本 / train 3500 / dev 750 / test 750，HEAD：见当前 main）。当前契约前的全部历史交付（早期 P3 MVP 交付 + 扩样阶段偏离契约尝试）均已迁出本文档，统一归档在 `docs/plans/open-issues.md` P3-01 段（line 769-841）以及同文件后段的扩样阶段拒绝归档段（line 919+）；P3 扩样阶段的实施记录与设计决策在 `docs/data/d2-expansion.md`。

审查模型：`minimax-cn/MiniMax-M3`（project-level `reviewer` agent）
审查 agent：`reviewer`（subagent dispatch 名称）
实际 provider/model：`PI_PROVIDER=minimax-cn`，`PI_MODEL=MiniMax-M3`
`PI_AGENT_NAME`：未注入
审查文件路径：`docs/plans/reviews/stage-p3-d2-multi-turn.md`
当前契约 commit：HEAD（当前 main；详见 git log -1）（5000/3500/750/750 严格命中，6 类 ≥833 unique canonical signatures，cross-split disjoint，D2-vs-D1/D1.1 disjoint，MANIFEST 含 `build_count` 字段，`datasets/tool-calling-d2/` 已 gitignore + git rm --cached 隔离）。

## 阶段 p3-d2-multi-turn 审查

### 完成范围（当前活跃契约）

- **独立 D2 schema**：`schemas/d2_multi_turn_sample.schema.json`，6 个规范 task_type：`tool_not_available` / `tool_error_response` / `insufficient_result_search` / `req_change_city` / `multi_tool_sequential` / `error_recovery`；D1/D1.1 继续使用通用 schema，向后兼容。
- **独立语义互斥**：生成器和测试均计算 `canonical_content_signature()`；去除 `id` / `call_id` / `tool_call_id` / `depends_on`、`metadata.created_at` 和 `metadata.split` 后，仍保留 task_type、tools、消息内容、工具参数/结果和 expected_answer。**当前契约（round 14；HEAD：当前 main）**：5000/5000 签名唯一；6 类 ≥833 unique（实际 834/834/833/833/833/833）；train/dev/test 三组 canonical signature 交集均为空。
- **D2 生成器** `scripts/generate_d2_dataset.py`：
  - **当前契约**：默认生成 5000 样本（4 类 833 + 2 类 834），train/dev/test = 3500/750/750 IID split，per-class split 通过 `round(per_class × 0.15)` 化 dev/test 使 dev=test=125、train=per_class-250；
  - 确定性 `--seed 2026`、确定性时间戳、id/call_id 命名空间前缀（`d2-{train,dev,test}-...`）；
  - 六类 builder 使用确定性 `variant_index` 与多维语义组合；严格嵌套 `(variant // N) % P` 维度公式消除 aliasing；`canonical_content_signature()` 去除 ID、时间戳、split 和依赖 bookkeeping 后执行全局去重；
  - per-split `aggregate_sha256` + per-file sha256；
  - MANIFEST-{train,dev,test}.json：count + sha256 + task_type 分布 + aggregate_sha256，并新增 `build_count` 字段（= 5000）。
- **单测** `tests/test_d2_dataset.py`：**49 + 2 = 51** 单测分 8 组：
  - `D2DatasetSchemaTests`：独立 D2 schema 合法 / 6 类规范 task_type 覆盖 / 每 split 包含 6 类；
  - `D2MultiTurnStructureTests`：assistant + tool 角色结构 / tool_call_id 显式存在 / expected/transcript call_id 一致 / 最终 sample 命名空间一致；
  - `D2DependencyGraphTests`：depends_on 存在性 / 严格前序 / 无环 / 负例语义拒绝 / 乱序输入的真实拓扑执行；
  - `D2MockExecutorReplayTests`：真实 `MockExecutor.execute_sequence()` replay / 乱序拓扑执行 / 失败依赖阻断；
  - `D2SplitDisjointnessTests`：train/dev/test id/path/canonical semantic content 不重叠、6 类各 833/834 unique 语义实例、与 D1/D1.1 train id 互斥、MANIFEST count / aggregate / per-file sha 一致、5000/3500/750/750 显式断言；
  - `D2TimestampContractTests`：`1785000000 + seed + index` 精确 UTC/Z 格式；
  - `D2ExpectedAnswerContractTests`：`expected_answer == final assistant content` 遍历全部 5000 样本验证 invariant；
  - `D2GeneratorDefaultContractTests`（round 14 新增）：`--count` 默认值 = 5000，且磁盘数据集满足 5000/3500/750/750。
- **协议** `docs/protocols/d2-multi-turn.md`：D2 与 D1/D1.1 差异 / 6 类任务定义 / call_id 依赖链 / held-out split 互斥 / 与 P2 / P4 / P5-02 衔接。
- **`docs/plans/roadmap.md`**：P3 行已更新为当前契约 5000/3500/750/750。
- **`docs/plans/open-issues.md`**：P3-01 段（line 769-841）记录当前契约前的初始 P3 交付，已加 banner 标注"被 round 14 取代"；同文件后段（line 919+）记录扩样阶段偏离契约尝试与被否决策。

### 验证（当前活跃契约）

- `scripts/run_tests.py full` → Ran **298** tests OK（49 D2 现有 + 2 默认契约新增 = 51 D2 tests；总测试数 298 包含全部子模块）。
- `scripts/validate_stage0.py --examples` → 9/9 PASS（含 D2 正例）。
- 5000/5000 样本 schema 合法，6 类各 833/834 unique canonical signatures，train/dev/test = 3500/750/750 严格命中；MANIFEST 含 `build_count=5000`。
- canonical semantic signature 跨 split disjoint（train∩dev = train∩test = dev∩test = ∅）；D2-vs-D1/D1.1 cross_dataset_signature disjoint。
- 详见 `docs/data/d2-expansion.md`（5000 样本扩样记录）、`docs/protocols/d2-multi-turn.md`（当前契约主协议）、`docs/plans/reviews/stage-p3-d2-expansion.md`（扩样 stage review）。

### 已知边界（当前活跃契约）

- 自研 5 ckpt 均为单轮训练；与当前扩样版 D2 dev 750 的真实推理评测归档于 `docs/experiments/p2-evaluator/README.md` §6 + `docs/plans/open-issues.md` P3-01 段 line 769-841；当前活跃公开模型评测见 §7 与 `docs/plans/reviews/stage-p5-02-transformers-backend.md`。
- D2 与 D1 / D1.1 train id 互斥保证靠 id 命名空间 + 目录物理隔离 + `D2SplitDisjointnessTests` 三重保证；语义 held-out 还额外由 canonical signature 全局唯一断言保证。

### 下一步（当前活跃契约）

1. P5-02 Transformers backend + 公开 instruction-tuned 模型 ✅（见 `docs/plans/reviews/stage-p5-02-transformers-backend.md`，HEAD：见当前 main）。
2. P4 GRPO MVP ✅（HEAD `1fae6f0` + `99646fa`；详见 `docs/plans/reviews/stage-p4-grpo-mvp.md` + `docs/plans/reviews/stage-p4-grpo-smoketest.md`；detached auditor + minimax-M3 subagent reviewer 联合复审均已 PASS）；policy 已基于当前扩样版 D2 train (3500) + reward_signal 验证 natural advantages + real CPU+GPU smoke。
3. 自研模型加 multi-turn training（messages 含 assistant + tool）后，回填多轮 SFT checkpoint 在当前扩样版 D2 dev 750 上评测。

### durable 证据

- 本轮 reviewer 输出保存到 `.pi-glla/scratch/stage-p3-d2-multi-turn-review-r{1,2}.txt` 及 `-postfix-r{7,8,9,10,11,12,13}.txt`（gitignored）。
- 当前扩样版 5000 样本 stage review：`docs/plans/reviews/stage-p3-d2-expansion.md`。
- 独立审计（list queue item #2）：`docs/plans/reviews/stage-p3-d2-expansion-independent-audit.md`。

### 历史归档

本文档早期版本描述的旧版 P3 交付 / 早期 split 比例 / Round 5–9 修复轨迹 / 历史 reward 数字等已不再属于"当前契约"范畴，全部归档至 `docs/plans/open-issues.md` P3-01 段（line 769-841，该段以独立 banner 明确标注其内容仅作历史记录）。扩样阶段的全部偏离契约尝试（含被否决策与失败原因）统一归档至 `docs/plans/open-issues.md` 后段的扩样阶段拒绝归档段（line 919+）。