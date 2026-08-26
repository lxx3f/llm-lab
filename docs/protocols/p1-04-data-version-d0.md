# P1-04 D0 数据版本协议

> 状态：已解决（2026-08-26，D0 定义）。本文档固定工具调用数据版本 D0/D1/D2 的定义，重点是 D0 的最小可复现集合。

## 背景

`examples/tool_calling/` 已有 3 个手工样例（sample-001 / sample-002-no-tool / sample-003-multi-tool），但缺少版本化、可复现的数据集定义。P1-04 固定数据版本的协议，避免后续评测结果无法溯源。

## 数据版本定义

### D0：最小手工样例集（当前）

- **用途**：schema 验证 + 评测链路 smoke；
- **规模**：≥3 个样例（no_tool / single_tool / multi_tool）；
- **来源**：手工构造（标注 `source: synthetic`）；
- **split**：单集合（不分 train/dev/test，D0 不用于训练）；
- **hash**：每个样例文件 sha256 + 集合清单 hash；
- **生成**：手工编辑 + `validate_stage0.py` schema 校验。

### D1：模型生成样例集（当前以模板 MVP 交付，D1.1 需 LLM API）

- **用途**：SFT 训练 + 评测数据；
- **规模**：≥100 样例（canonical 126）；
- **来源**：**当前 MVP（mock-only）**：`scripts/generate_d1_dataset.py` 确定性模板生成（标注 `source: d1-synthetic-template`），配合 `examples/d1_mocks.py` 产生 deterministic `expected_result`，并由 `MockExecutor` 端到端验证 108/108 expected calls 可执行；**D1.1**：需 LLM API 凭证，使用真实 LLM 生成 + 人工过滤（标注 `source: <model-name>@<version>`）；
- **split**：train/dev/test = 100/13/13；
- **hash**：每个样例 + 分片 + 全集合 hash；
- **质量检查**：schema valid 100%、parse_success 100%、tool_execution_valid ≥ 95%。

### D2：规模化生产集（后续）

- **用途**：正式训练 + GRPO；
- **规模**：≥1000 样例；
- **来源**：D1 扩展 + 工具执行回放；
- **split**：IID + compositional split（避免模板记忆）；
- **hash**：版本化数据管线输出，记录 pipeline_version + 输入模型 + 过滤规则版本。

## D0 固定定义

### 目录结构

```
examples/tool_calling/
├── sample-001.json            # multi_tool
├── sample-002-no-tool.json    # no_tool
├── sample-003-multi-tool.json # multi_tool
└── MANIFEST.json              # D0 清单（所有样例的 sha256 + 元数据）
```

### MANIFEST.json schema

```json
{
  "schema_version": "1.0",
  "data_version": "D0",
  "split": "none",
  "samples": [
    {"path": "sample-001.json", "sha256": "<hex64>", "task_type": "multi_tool"}
  ],
  "samples_sha256": "<hex64 of concatenated sample hashes>",
  "created_at": "<ISO8601>"
}
```

### 样例必须满足

1. `tool_calling_sample.schema.json` v1.0 校验通过；
2. `metadata.task_type` ∈ {no_tool, single_tool, multi_tool, tool_error, insufficient_result, requirement_change}；
3. `metadata.source` = `synthetic`（D0 全部手工）；
4. 至少覆盖 3 种 task_type（D0 覆盖 no_tool / single_tool / multi_tool）。

## 校验命令

```bash
# schema 校验（已有 Stage 0 协议）
.venv/python.exe scripts/validate_stage0.py

# D0 清单生成 + hash 校验（已实现）
.venv/python.exe scripts/build_d0_manifest.py          # 只校验
.venv/python.exe scripts/build_d0_manifest.py --write  # 生成 MANIFEST.json
```

## 自动化验证

- `tests/test_d0_manifest.py`（4 tests）：build_manifest 有效性 / MANIFEST hash 匹配文件 / source=synthetic / CLI 校验通过；
- `tests/test_artifact_provenance.py`（2 tests）：夜间跑 artifact 存在且 git_commit 属于已知夜间 run commit 集合；共享 medium control == HEAD。

## 与评测链路关系

```
D0 样例 → schema 校验 → mock executor（P1-01）→ 执行结果
```

- P1-01 mock executor 的 example 已用 D0 风格样例（`examples/mock_execution/sample-mock-001.json`）；
- 正式评测（P2）用 D0 样例集作为最小验证集。

## 遗留

- D0 样例 metadata.source 已统一为 `synthetic`（2026-08-26，auditor 复核后）；MANIFEST.json 已生成并提交；
- D1/D2 的触发条件：进入 SFT/GRPO 阶段（P4）前必须完成 D1；
- compositional split 设计留 D2。