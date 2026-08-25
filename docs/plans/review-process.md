# 阶段完成审查流程

## 目的

每完成一个阶段性开发目标，先进行一次独立审查，再进入下一阶段，避免功能持续堆叠而实验协议、正确性和文档滞后。

## 触发条件

以下任一情况完成后触发审查：

- 一个 README 阶段完成；
- 一个 MVP 完成；
- 一个模型模块或数据管线模块完成；
- 一组实验或 benchmark 完成；
- 计划顺序发生调整。

## 审查内容

### 1. 计划一致性

- 当前实现是否真的满足阶段目标；
- README 状态是否准确；
- 是否有功能被提前标记为完成；
- 下一阶段是否仍然合理；
- 是否需要拆分、延后或新增任务。

### 2. 正确性

- 单元测试和集成测试是否覆盖核心路径；
- 边界条件、异常路径和失败处理是否验证；
- 训练、推理、prefill/decode、cache 等语义是否一致；
- 是否存在只在 smoke test 下成立的假设。

### 3. 实验有效性

- 对比实验是否公平；
- 配置、数据、token budget 和 seed 是否固定；
- benchmark 是否包含额外统计或同步开销；
- 指标定义、分母和测量方法是否明确；
- 结果是否被误写成正式结论。

### 4. 可复现性

- 依赖版本、设备、运行命令是否记录；
- 配置、代码、数据和结果是否能关联；
- 是否需要保存 git commit、hash 或运行元数据；
- 新环境能否完成最小验证。

### 5. 工程和文档

- 文件是否放在约定目录；
- 文档链接和示例是否仍然有效；
- `.gitignore` 是否排除了本地环境和大文件；
- 是否产生未记录的中间产物；
- 是否需要更新 schema、配置或报告。

### 6. 风险和后续计划

- 记录新发现的 P0/P1/P2 问题；
- 判断是否阻塞下一阶段；
- 调整开发顺序；
- 明确下一步只处理哪些问题，避免范围蔓延。

## 审查执行方式

阶段审查由独立子 agent 执行，不由主 agent 自我审查。审查子 agent 的模型固定为：

```text
minimax-cn/MiniMax-M3
```

主 agent 的职责是：

1. 完成阶段实现；
2. 运行测试、校验和 benchmark；
3. 更新代码、配置、README、`docs/` 和 `open-issues.md`；
4. 将阶段目标、变更摘要、测试结果、风险和待审查文件提交给子 agent；
5. 根据子 agent 审查意见修复问题；
6. 只有在子 agent 明确判定“通过”或“有条件通过且无阻塞问题”后，才创建阶段 commit。

子 agent 审查结果必须记录在：

```text
docs/plans/reviews/stage-<kebab-case-name>.md
```

记录至少包含：

- 审查模型：`minimax-cn/MiniMax-M3`（对应用户指定的 MiniMax M3）；
- 审查 agent：实际调用的 subagent dispatch 名称；
- 如果 harness 提供 `PI_AGENT_NAME`，将其作为附加运行标识记录；
- 审查范围和 commit 候选变更；
- 发现的问题及严重级别；
- 通过/不通过结论；
- 主 agent 的修复或后续动作。

如果当前 agent harness 无法选择或确认 `minimax-cn/MiniMax-M3`，不得声称审查已完成；应暂停自动 commit 并报告阻塞原因。审查记录必须包含子 agent 返回的 `PI_PROVIDER`、`PI_MODEL` 和主 agent 调用时指定的审查 agent 名称；主 agent 在 commit 前核对这些字段。`PI_AGENT_NAME` 不是所有 harness 都会注入，因此仅在可用时记录，不作为阻塞条件。

## 审查产物

每次审查至少更新：

- `docs/plans/open-issues.md`：新增、关闭或调整问题；
- 对应阶段文档：记录实现范围、验证结果和局限性；
- `README.md`：同步阶段状态和下一步计划。

每份审查记录必须包含：

```text
审查模型：minimax-cn/MiniMax-M3
审查 agent：reviewer（subagent dispatch 名称）
实际 provider/model：PI_PROVIDER=minimax-cn, PI_MODEL=MiniMax-M3
可选运行标识：PI_AGENT_NAME（harness 提供时记录）
审查文件路径：docs/plans/reviews/stage-<kebab-case-name>.md
```

## 审查条目格式

```markdown
## 阶段 <kebab-case-name> 审查

- 完成范围：
- 未完成范围：
- 测试结果：
- 实验结果：
- 新发现问题：
- 计划调整：
- 是否允许进入下一阶段：是 / 否 / 有条件
- 下一步：
```

## Git 提交流程

阶段审查通过后自动创建一个 Git commit；阶段审查未通过时不提交阶段完成 commit。

执行顺序固定为：

```text
完成阶段开发
→ 运行测试和校验
→ 更新阶段文档、README 和 open-issues
→ 执行阶段审查
→ 审查通过
→ git add 相关变更
→ git commit
```

提交要求：

- 一个阶段对应一个主要 commit，避免把多个阶段混在同一个提交中；
- commit 只包含当前阶段相关的代码、配置、测试和 `docs/` 文档；
- `.venv/`、缓存、模型权重、密钥和本地 IDE 文件不得提交；
- commit message 的 Conventional Commit 标题使用清晰的英文格式；
- commit message 的详细说明部分统一使用中文，记录主要改动、设计决策、测试结果和已知限制；
- 示例：

```text
perf(tokenization): optimize incremental BPE merges

详细说明：
- 将 BPE merge 改为增量 pair count；
- 通过 tokenizer、CLI 和 Dense/MoE 测试；
- 记录性能边界和后续计划。
```
- 提交前必须检查 `git status`、测试结果和待提交文件；
- 如果 Git 用户信息、测试或审查不满足要求，应停止自动提交并报告原因；
- 不执行 `git push`，除非用户明确要求。

当前仓库尚无历史 commit。后续第一个真正完成并通过审查的阶段将创建对应的首个阶段 commit；在此之前不把未审查的中间状态标记为阶段完成。

## 通过标准

阶段只有在以下条件满足后，才可以标记为完成并进入下一阶段：

1. 核心目标和边界明确；
2. 关键正确性测试通过；
3. 已知限制和风险已记录；
4. 运行方式和依赖已记录；
5. README、配置、代码和报告一致；
6. 不存在未处理的阻塞性 P0 问题。

Smoke benchmark 通过不等于阶段通过；如果只有链路验证完成，应标记为 smoke/forward MVP，而不是完整阶段完成。
