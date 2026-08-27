# D1.1 真实 LLM 生成数据集

> 状态：阶段交付（2026-08-27）。
>
> 用真实 LLM（MiniMax-M3）生成工具调用样例，替代/补充模板版 D1 的确定性模板。D1.1 的 `metadata.source = "MiniMax-M3@2026-08-27"`，与模板版 `d1-synthetic-template` 完全可区分。

## 与 D1 模板版的区别

| 维度 | D1 模板版 | D1.1 LLM 生成版 |
|---|---|---|
| 来源 | `scripts/generate_d1_dataset.py` 确定性模板 | `scripts/generate_d1_llm.py` + MiniMax-M3 API |
| source 标注 | `d1-synthetic-template` | `MiniMax-M3@2026-08-27` |
| 用户请求 | 模板套话（"请计算 X"） | 真实 LLM 自然语言（含真实场景：房贷、汇率、天气规划） |
| 多样性 | 6 个模板反复套 | 每样例独立采样，自然语言变体丰富 |
| expected_result | MockExecutor 确定性计算 | MockExecutor 确定性计算（扩展支持浮点/幂） |

## 生成管线

```text
LLM 扮演用户 → 生成自然语言请求（中文）
    ↓
LLM 扮演标注器 → 输出 expected_tool_calls + expected_answer（JSON）
    ↓
MockExecutor 语义验证（每 call 执行，result == expected_result，否则重试）
    ↓
no_tool 语义守卫（no_tool 有调用 → 拒绝重试）
    ↓
schema 校验 → 写盘 + MANIFEST（sha256 每文件 + 聚合 hash）
```

- 每样例最多 **3 次重试**（LLM 噪声：浮点表达式、多余参数、JSON 解析失败均自动重试）；
- 部分失败时**成功样例仍写盘**（不丢已完成工作），exit code 非 0 标记部分完成；
- `--skip-existing` 支持断点续跑（跳过已存在文件 + MANIFEST 从磁盘全量重建）。

## 生成命令

```bash
# 126 样例（6 task_type × 21 轮循环）
D1_LLM_API_KEY=<key> .venv/python.exe -u scripts/generate_d1_llm.py \
    --count 126 --out datasets/tool-calling-d1-llm

# 断点续跑（跳过已生成）
D1_LLM_API_KEY=<key> .venv/python.exe -u scripts/generate_d1_llm.py \
    --count 126 --skip-existing --out datasets/tool-calling-d1-llm

# 管线测试（不调 API）
.venv/python.exe scripts/generate_d1_llm.py --count 6 --dry-run
```

环境变量：`D1_LLM_API_KEY`（必填）、`D1_LLM_API_BASE`（默认 MiniMax anthropic 端点）、`D1_LLM_MODEL`（默认 MiniMax-M3）、`D1_LLM_MAX_TOKENS`（默认 4096）。

## 数据集统计

- **126 样例**，6 task_type 各 **21**（no_tool / single_tool / multi_tool / tool_error / insufficient_result / requirement_change）；
- **0 个 no_tool 坏样例**（no_tool 有 expected_calls 的由生成器语义守卫拒绝）；
- **130/130 expected_calls 全部有 deterministic expected_result**（MockExecutor 端到端验证 result == expected_result）；
- 126/126 schema valid（`schemas/tool_calling_sample.schema.json`）；
- 8 级分类器对正确 transcript 126/126 `first_failure=None`。

## 文件索引

- 生成器：`scripts/generate_d1_llm.py`
- 数据集：`datasets/tool-calling-d1-llm/train/*.json`（126）+ `MANIFEST-train.json`（source/aggregate_sha256/每文件 sha256）
- 测试：`tests/test_d1_llm.py`（8 个只读验证测试，`skipUnless` 数据集存在时运行）
- 依赖 mock：`examples/d1_mocks.py`（`d1_calculate` 扩展支持浮点 + `^` 幂，兼容模板集整数路径）

## 复现说明

- LLM 生成有随机性（temperature 0.9 用户 / 0.3 标注），**无法逐字节复现**——数据集是生成产物，MANIFEST 记录每文件 sha256 + 聚合 hash 供校验；
- 模板版 D1（确定性可复现）保留在 `datasets/tool-calling-d1/`，作为对比基线；
- 真实推理后端接入后，D1.1 可用于 P2 评测（8 级分类器消费 transcript）。
