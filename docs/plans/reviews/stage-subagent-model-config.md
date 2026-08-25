# subagent 审查模型配置验证

## 验证目的

确认阶段审查是否可以通过 subagent 插件固定使用用户指定的 MiniMax M3。

## 代码结论

subagent 扩展读取 agent Markdown frontmatter 中的 `model` 字段，并在启动子进程时追加：

```text
--model <agent.model>
```

因此可以通过用户级 agents 目录中的 `reviewer.md` 固定审查模型。Windows 默认路径为 `%USERPROFILE%\.pi\agent\agents\reviewer.md`，Linux/macOS 通常为 `~/.pi/agent/agents/reviewer.md`。

## 当前配置

```yaml
name: reviewer
model: minimax-cn/MiniMax-M3
tools: read, grep, find, ls, bash
```

使用 provider 前缀是必要的，因为本地存在多个同名或相近模型配置：

- `calculet/minimax-m3-mxfp8`：当前 distributor 没有可用 channel，实测返回 `model_not_found`；
- `minimax-cn/MiniMax-M3`：MiniMax CN provider 的可用模型。

## 实际验证

子 agent 返回的 harness 环境变量：

```text
PI_MODEL=MiniMax-M3
PI_PROVIDER=minimax-cn
PI_REASONING_LEVEL=high
```

其中 `PI_REASONING_LEVEL` 是本次验证的附加运行信息，不是所有阶段审查记录的强制字段。

结论：

```text
实际运行模型：minimax-cn/MiniMax-M3
状态：可用
```

## 变更范围

- 用户级 agents 目录中的 `reviewer.md`：固定 `minimax-cn/MiniMax-M3`；
- `AGENTS.md`：记录阶段审查必须由子 agent 使用 MiniMax M3；
- `docs/plans/review-process.md`：记录模型、agent、审查产物和无法确认时的阻塞规则。

```text
审查模型：minimax-cn/MiniMax-M3
审查 agent：reviewer（subagent dispatch 名称）
实际 provider/model：PI_PROVIDER=minimax-cn, PI_MODEL=MiniMax-M3
可选运行标识：PI_AGENT_NAME（本次 harness 未提供）
```

## 阶段审查结论

本次由 `reviewer` 子 agent 使用 `minimax-cn/MiniMax-M3` 完成只读审查。

- Critical：无；
- Warnings：无；
- P0-02 标题、现状、缺口和分隔：通过；
- P1-00 位置和优先级：通过；
- 审查文件路径约定：统一为 `docs/plans/reviews/stage-<kebab-case-name>.md`；
- 中文 commit body 规则：已记录；
- 无法确认模型身份时停止 commit：已记录；
- 审查结论：通过。
