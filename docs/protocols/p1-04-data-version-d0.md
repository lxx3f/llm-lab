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

### D1：模型生成样例集（模板版 + D1.1 LLM 生成版）

- **用途**：SFT 训练 + 评测数据；
- **规模**：模板版 126 样例（canonical，train/dev/test = 100/13/13）+ D1.1 LLM 生成版 **1500 个 train 样例**；D1.1 当前只有 train split，未生成独立 dev/test；
- **来源**：
  - **模板版**：`scripts/generate_d1_dataset.py` 确定性模板生成（标注 `source: d1-synthetic-template`），配合 `examples/d1_mocks.py` 产生 deterministic `expected_result`，并由 `MockExecutor` 端到端验证；
  - **D1.1（2026-08-27 交付并扩展）**：`scripts/generate_d1_llm.py` + MiniMax-M3 API 真实 LLM 生成；1500 个文件的 per-file `metadata.source` 分为 `MiniMax-M3@2026-08-27`（126）和 `MiniMax-M3@minimax-cn`（1374）；当前 manifest 的 `count=1500`，详见 `docs/experiments/d1-llm/README.md`；
- **task_type 分布**：D1.1 的 6 类各 250（no_tool / single_tool / multi_tool / tool_error / insufficient_result / requirement_change）；
- **split**：D1 模板版为 train/dev/test = 100/13/13；D1.1 当前仅为 `train`，没有独立 dev/test；
- **hash**：每个样例 + 分片 + 全集合 hash；MANIFEST 同时记录每个文件的 source；
- **质量检查**：D1.1 当前 1500/1500 schema valid，1555/1555 expected calls 带 deterministic `expected_result`；正确 transcript 的 P1-05 分类为 1500/1500 `first_failure=None`；这些是数据管线验证结果，不等同于模型泛化评测。

### D2：多轮对话集（已交付，详见 `docs/protocols/d2-multi-turn.md`）

- **用途**：正式训练 + GRPO + 多轮对话能力评测；P5-02 公开模型 benchmark；
- **规模**：**5000 样本**（4 类 833 + 2 类 834），train/dev/test = **3500/750/750** IID split；每类 ≥833 unique canonical semantic signatures；6 类 task_type（`tool_not_available` / `tool_error_response` / `insufficient_result_search` / `req_change_city` / `multi_tool_sequential` / `error_recovery`）。
- **来源**：D1 扩展 + 工具执行回放；6 类确定性 builder + canonical semantic uniqueness 校验；
- **split**：IID split（按 task_type 分层的 seeded shuffle；70/15/15，round 14 改用 `round()` 化 dev/test 严格命中 1125+125+125 per-class），不重叠 + 跨 D1/D1.1 train canonical disjoint；
- **hash**：每个样例 + 分片 + 全集合 hash；MANIFEST 同时记录每个文件的 source、aggregate hash 与 `build_count` 字段；本轮交付 path 为 `datasets/tool-calling-d2/`，已 gitignore + `git rm --cached` 隔离。

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
D0 样例 → schema 校验 → mock executor（P1-02；历史文档正文曾写 P1-01）→ 执行结果
```

- P1-02 mock executor 的 example 已用 D0 风格样例（`examples/mock_execution/sample-mock-001.json`）；历史阶段文档 `stage-p1-mock-executor.md` 的早期正文曾使用 P1-01 名称，路线图现统一以 P1-02 为规范编号；
- 正式评测（P2）用 D0 样例集作为最小验证集。

## 遗留

- D0 样例 metadata.source 已统一为 `synthetic`（2026-08-26，auditor 复核后）；MANIFEST.json 已生成并提交；
- D1 的触发条件：进入 SFT/GRPO 阶段（P4）前必须完成 D1；
- D2 多轮对话集已交付（D2 契约 = 5000/3500/750/750；详见 `docs/protocols/d2-multi-turn.md`）；D2 compositional split 设计后续评估。