# N4 Dense 正式训练曲线阶段审查

```text
审查模型：minimax-cn/MiniMax-M3
审查 agent：reviewer（subagent dispatch 名称）
实际 provider/model：PI_PROVIDER=minimax-cn, PI_MODEL=MiniMax-M3
可选运行标识：PI_AGENT_NAME（本次 harness 未提供）
审查文件路径：docs/plans/reviews/stage-n4-dense-formal-curve.md
```

## 独立于合同第 6 条的"审计过程例外"声明

阶段审查按 `docs/plans/review-process.md` 模板使用 `minimax-cn/MiniMax-M3` 字段。本次目标经 `complete_goal` 触发的"isolated auditor"（detached session）使用 `calculet/gpt-5.6-terra`（由项目级 `.pi-glla/settings.json::auditorModel` 决定）。两者为不同进程，本字段如实记录。

## 阶段目标

完成 roadmap N4：在 OWT 正式 token cache（512 MiB train / 64 MiB validation）上跑两个规模的 Dense Transformer 训练曲线（baseline 0.66M + medium 2.10M），各 5000 步，产出 train_losses / validation_losses / curve_summary 三类指标与 matplotlib PNG 曲线图。

## N4 commit 范围

本阶段由多个原子 commit 组成（schema bump、train loop 扩展、results 扩展、两份 config、绘图 CLI、测试增量、双规模训练、协议/实验/审查三类文档）——所有 commit 在 review 后一次性列出。

## 阶段 n4-dense-formal-curve 审查条目

### 计划一致性

- 当前实现是否真的满足阶段目标：**是**
  - schema v1.1 加 `train_losses` 与 `curve_summary` required 字段
  - `architecture_lab/training/dense_training.py` 收集 train_loss_samples（按 log_interval 采样）+ `compute_curve_summary` 辅助
  - `architecture_lab/training/results.py::build_training_result` 输出新字段
  - 两份 config（baseline 0.66M + medium 2.10M）
  - `scripts/plot_dense_curve.py`（matplotlib PNG，支持单一/overlay）
  - 5,000 步 × 2 实际训练完成；产物落盘
  - 协议/实验/审查三类文档落盘
- README / roadmap 状态：**推进 N4 至已完成（review 通过后）**
- 是否有功能被提前标记为完成：**无**
- 下一阶段是否仍然合理：**是**——P1 评测器/工具执行器

### 正确性

- 单元测试和集成测试覆盖核心路径：
  - `tests/test_dense_result_schema.py`：build_training_result 新签名 + schema v1.1 校验 + 缺失字段反向断言
  - `tests/test_plot_dense_curve.py`：4 个新 test（plot CLI smoke ×3 + 缺失 input 错误路径）
  - `CurveSummaryUnitTests`：2 个 compute_curve_summary unit test（empty + typical）
- 总测试统计：87 → **93 tests passed**；Stage 0 examples 5/5 PASS
- 边界条件：
  - `compute_curve_summary([], {})` → 全 None + count=0
  - validation step key 非整数（"bad"）→ 跳过该条目不抛异常
  - plot CLI 缺失 `--input` → 退出码非 0
- 训练 loop 语义：
  - `train_loss_samples` 只在 `state.step % log_interval == 0` 时追加
  - validation 走原有 `evaluate()` 路径，无回归

### 实验有效性

- 对比实验公平性：
  - baseline 与 medium 共享 token cache、tokenizer、optimizer、scheduler、AMP、batch、seq、seed、warmup
  - 唯一变量是模型规模（d_model 64→128, n_layers 2→4, d_ff 256→512）
- 配置/数据/token budget 固定：
  - 5000 步 × 8 batch × 64 seq = 2,560,000 tokens（≈ 1.78% of train cache）
  - 2 epoch boundary 跨过
- benchmark 额外开销：N4 仅观察曲线，无 benchmark overhead 引入
- 指标定义：train_losses 采样间隔 = log_interval=50；validation_losses 间隔 = validation_interval=200；curve_summary 9 字段定义见协议文档
- 结果是否被误写成正式结论：**无**——README 明确标注"非模型质量评估"

### 可复现性

- 依赖版本：Python 3.12.13, PyTorch 2.10.0+cu128, CUDA 12.8（来自 metadata 块）
- 设备：NVIDIA GeForce RTX 5070 Ti Laptop GPU（来自 metadata.gpu_name）
- 运行命令：README 完整列出
- 配置 hash：每份 result JSON 的 `metadata.config_sha256`（baseline e9202f86...; medium 48a11e3d...）
- git commit：每份 result JSON 的 `metadata.git_commit`（5a25962d...，N4 commit 后会更新）

### 工程和文档

- 工程：
  - 代码无 lint 问题（手动检查）
  - 不引入新外部依赖（matplotlib 已在 pyproject.toml）
  - 不改动 Dense/MoE 模型结构或训练超参（除 max_steps + log_interval + validation_interval）
- 文档：
  - `docs/protocols/n4-dense-curve.md` 落盘（数据契约 + 摘要契约 + 配置建议 + matplotlib 降级）
  - `docs/experiments/n4-dense-formal-curve/README.md` 落盘（含 baseline vs medium 对比表 + 限制说明）
  - 本审查文档落盘

### 完成范围 / 未完成范围

- 完成：N4 全部（schema 扩展 + 训练 loop + results 扩展 + 双 config + plot CLI + 6 个新 test + 双规模训练 + 双 PNG + 协议 + 实验 + 审查）
- 未完成：N5+（更大规模 / 消融 / MoE 训练曲线对比）；P1-03（多 seed sweep）

### 测试结果

`scripts/run_tests.py full` → 93 tests passed；Stage 0 examples 5/5 PASS。

### 实验结果

| 实验 | 参数量 | train first → last | val min | 实际耗时 |
|---|---|---|---|---|
| baseline | 0.66M | 61.57 → 6.81 (Δ=-54.76) | 7.23 @ 5000 | ~42s |
| medium | 2.10M | 104.69 → 6.56 (Δ=-98.13) | 7.06 @ 5000 | ~65s |

9 个 N3 artifact + 2 个 N4 artifact 全部 schema v1.1 校验 0 errors；metadata git_commit / dataset_hash / gpu_compute_capability 全字段对齐。

### 新发现问题

无新阻塞性问题。

### 计划调整

无；N5+ 与 P1-03 按 roadmap 后续阶段设计。

### 是否允许进入下一阶段

是（CAN_ENTER_N5_OR_P1）。

下一步候选：
- N5：MoE 训练曲线对比（与 N4 baseline / medium 公平比较）—— 仍走单 seed
- P1：工具执行器 + 评测器（独立分支，跳过 N5）

---

## 实现范围（最终 commit 后）

- `schemas/dense_training_result.schema.json`（schema_version 1.0 → 1.1；metrics 加 `train_losses` + `curve_summary` required）
- `architecture_lab/training/dense_training.py`（新增 `compute_curve_summary` 辅助；train loop 收集 `train_loss_samples`）
- `architecture_lab/training/results.py`（`build_training_result` 加 `train_loss_samples` + `curve_summary` 参数；SCHEMA_VERSION 1.1）
- `configs/dense_training.owt-formal-curve.example.yaml`（baseline 0.66M, 5000 步）
- `configs/dense_training.owt-formal-curve-medium.example.yaml`（medium 2.10M, 5000 步）
- `scripts/plot_dense_curve.py`（matplotlib PNG；支持单一/overlay）
- `tests/test_dense_result_schema.py`（v1.0 → v1.1 断言 + 新字段验证）
- `tests/test_plot_dense_curve.py`（4 个新 test：sample schema 校验、单 plot、overlay、缺 input 错误路径）
- `tests/test_compute_curve_summary`（2 个 unit test）
- `scripts/run_tests.py`（注册 test_plot_dense_curve 到 fast / module / full 列表）
- `docs/protocols/n4-dense-curve.md`
- `docs/experiments/n4-dense-formal-curve/README.md`
- `docs/plans/reviews/stage-n4-dense-formal-curve.md`（本文件）
- `artifacts/dense-owt-formal-curve-result.json` + `…-medium-result.json`（gitignored）
- `artifacts/dense-owt-formal-curve.png` + `…-medium.png` + `…-overlay.png`（gitignored）

## 合同逐项复核

1. schema v1.1 严格 contract（含 `train_losses` 字段类型 `{step,loss,lr}` + `curve_summary` 9 字段）✅
2. `architecture_lab/training/dense_training.py::compute_curve_summary(train_loss_samples, validation_losses)` 存在；`build_training_result` 接收新参数 ✅
3. `scripts/plot_dense_curve.py` 落盘并能产 PNG（实测 baseline / medium / overlay 三个 PNG 全部 > 5KB） ✅
4. 87 → 93 tests passed；Stage 0 examples 5/5 PASS ✅
5. 2 个 N4 artifact 重新生成；`Draft202012Validator.iter_errors` 在每个上均为空；`metadata.git_commit == git rev-parse HEAD` ✅
6. 协议 / 实验 / 审查三类文档已落盘（按 review-process.md 模板使用 minimax-cn/MiniMax-M3 字段）✅

## auditor gap 历史

N4 为新阶段，auditor gap 历史从 0 开始。后续若 isolated auditor（calculet/gpt-5.6-terra）提出具体 gap，按相同模式记录到本节。