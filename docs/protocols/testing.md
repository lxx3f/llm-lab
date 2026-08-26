# 测试分层约定

## 目的

测试按反馈速度和覆盖范围分为 `fast`、`module`、`full` 三档。不要因为一个文档或局部函数改动，就默认运行所有 CLI 和训练闭环测试；但阶段审查和 commit 前必须运行 `full`。

统一入口：

```bash
.venv/python.exe scripts/run_tests.py <mode>
```

脚本从仓库根目录运行，任意失败均返回非零退出码。

## fast：日常快速反馈

```bash
.venv/python.exe scripts/run_tests.py fast
```

包含：

- data API：batching、token cache hash/边界；
- N2 benchmark API、shared train/validation binding、capacity 语义和 N2 result schema；
- Dense result schema 的正反例；
- Stage 0 JSON schema 测试；
- Dense forward/KV cache 正确性；
- BPE API 测试。

排除：

- subprocess CLI 集成；
- Dense 真正训练、checkpoint/resume；
- MoE 回归；
- 真实 OWT smoke 和 GPU benchmark。

适用：纯函数、schema、BPE、batching、Dense forward 和文档修改后的快速检查。

## module：按改动模块回归

```bash
.venv/python.exe scripts/run_tests.py module data
.venv/python.exe scripts/run_tests.py module training
.venv/python.exe scripts/run_tests.py module architecture
.venv/python.exe scripts/run_tests.py module tokenization
```

| 模块 | 覆盖内容 |
|---|---|
| `data` | batching、token cache API、token cache CLI、tokenizer artifact CLI |
| `training` | Dense 训练、scheduler、AMP、gradient accumulation、checkpoint/resume、结果 schema；N2 benchmark training/binding/capacity 回归 |
| `architecture` | Dense/MoE forward、backward、KV cache、generation 正确性 |
| `tokenization` | BPE API 和 tokenizer artifact CLI |

适用：完成一个独立子功能、修复模块 bug、准备请求代码审查前。

## full：阶段审查和 commit 门槛

```bash
.venv/python.exe scripts/run_tests.py full
```

包含所有当前 unittest 模块，外加：

```bash
.venv/python.exe scripts/validate_stage0.py --examples
```

适用：阶段审查前、阶段 commit 前、影响多个模块的重构后。

## 真实训练与 benchmark

不放入任一默认测试档。它们需要显式运行，并在实验记录中报告：

```bash
.venv/python.exe scripts/train_dense.py \
  --config configs/dense_training.example.yaml \
  --output artifacts/dense-owt-mvp-scheduler-result.json

.venv/python.exe scripts/train_dense.py \
  --config configs/dense_training.owt-formal.example.yaml \
  --output artifacts/dense-owt-formal-scheduler-amp-result.json
```

原因：它们是训练 smoke/实验，不是快速单元测试；正式范围 cache smoke 只覆盖数据流前缀，不代表全 cache 训练质量。

## 增加新测试的规则

- 纯函数、数据结构、schema：默认加入 `fast`；
- CLI subprocess：加入对应 `module` 和 `full`，除非能证明 API 测试已覆盖且无 CLI 合约；
- 训练、checkpoint/resume、GPU/AMP：加入 `module training` 和 `full`；
- 真实数据训练、长 benchmark：不加入默认档，写入实验/benchmark 命令和文档；N2 的短训练 API smoke 已作为 training regression 纳入默认 `module/full`，而四个 A/B benchmark CLI 与 routing CLI 仍通过实验命令显式运行；
- N2 schema 测试在本地 artifact 存在时校验四个 benchmark JSON 和 routing JSON；由于 `artifacts/*.json` 被 `.gitignore` 忽略，干净 checkout 缺少这些实验产物时测试会跳过 artifact-specific assertions。阶段审查/复现必须先按 N2 文档命令生成并显式校验 artifacts；artifact 本身不提交到 Git。
- 新模块时必须同步更新 `scripts/run_tests.py`、本协议和 README。
