# LLM Lab

面向大语言模型研究与工程实践的个人实验项目，围绕以下闭环展开：

```text
模型架构实现 → 训练数据构造 → 模型训练/微调 → vLLM 部署 → 统一评测 → 失败分析
```

项目不追求一次性实现完整的大模型训练平台，而是先构建一个规模可控、结果可复现、能够持续扩展的实验体系。

## 数字一览

- **4 种 attention 架构**：Dense MHA + MoE Top-1 + GQA + 简化 MLA 全部从零实现（PyTorch 2.13 / sm_120）
- **多规模训练**：0.66M / 2.10M / 12M Dense 与 MoE 在 OWT 正式 cache 上完整曲线（最多 50k 步）
- **跨模型 LM 评测**：5 个公开 instruction-tuned 模型（SmolLM2 360M/1.7B、Qwen2.5 0.5B/1.5B/3B）277MB held-out OWT per-token loss + PPL
- **推理后端对比**：Transformers vs vLLM × 5 模型 × 2 batch × 90 样本 = 20 组合 4 轴对比表
- **数据管线**：D1 (126) / D1.1 (1500) / D2 (5000) 三套工具调用数据集 + 8 级失败分类器
- **训练闭环**：SFT (Dense + MoE) + GRPO MVP + 跨模型公平 benchmark
- **Schema 体系**：11 个 JSON Schema + git commit / config sha256 / token cache sha256 全绑定

**详细见 [`docs/SHOWCASE.md`](docs/SHOWCASE.md)**：含 4 条可贴简历的 bullet + 5 张关键 plot + 同 base 硬性判定标准 + 6 问面试 talking points。

**文档导航：见 [`docs/INDEX.md`](docs/INDEX.md)**：所有 experiments / protocols / stage reviews / archive 一站式入口。

## 项目结构

```text
llm-lab/
├── architecture_lab/         # 自研模型实现（PyTorch 原生）
│   ├── models/               # Dense MHA / MoE Top-1 / GQA / 简化 MLA
│   ├── training/             # 自研训练循环（AMP bf16 + warmup_cosine）
│   ├── data/                 # Token cache + batching
│   ├── benchmarks/           # N2 benchmark
│   ├── tokenization/         # OWT BPE
│   ├── execution/            # Mock executor
│   └── tests/                # 单元测试
├── scripts/                  # 38 个 CLI 入口（train / eval / plot / audit）
├── tests/                    # 29 个 pytest 文件（421 tests）
├── configs/                  # 22 个 example 配置
├── schemas/                  # 11 个 JSON Schema（带 git commit / cache sha256 binding）
├── docs/                     # 文档（详见 docs/INDEX.md）
│   ├── SHOWCASE.md          # 简历项目摘要（核心入口）
│   ├── INDEX.md             # 文档导航
│   ├── experiments/          # 27 个实验 README + protocol + result
│   ├── plans/                # roadmap + open-issues + stage reviews
│   ├── protocols/            # 22 个协议文档
│   ├── data/                 # OWT 数据契约
│   ├── reports/              # final-audit / night-run 等总结
│   ├── environment.md        # 环境说明
│   ├── internal/             # 内部中文规划文档
│   └── archive/              # 已关闭 / 不再维护的交付物
├── examples/                 # 样本数据 + sample result JSON
├── artifacts/                # 训练产物（checkpoint / result JSON / plot PNG，gitignored）
└── AGENTS.md                 # 内部协作规则（中文）
```

## 当前状态

- [x] 4 种 attention 架构（Dense MHA + MoE Top-1 + GQA + 简化 MLA）从零实现 + 单元测试
- [x] N13 三方对比：Dense MHA val_min 7.058 / GQA 7.106 / MLA 7.122，GQA + MLA KV cache 缩为 MHA 1/4
- [x] 真实 OWT 评测：5 个公开 instruction-tuned 模型 × 277MB held-out
- [x] 后端对比：Transformers vs vLLM × 5 模型 × 2 batch × 90 样本 4 轴
- [x] 数据管线：D1 + D1.1 + D2（5000 样本）+ 8 级失败分类器
- [x] 训练闭环：SFT（5 ckpt）+ GRPO MVP + 多规模 / 多 seed / 5 个 ablation sweep
- [x] Schema 体系：11 个 JSON Schema + 全 metadata binding
- [x] 38 个 stage review + 27 个 experiment README

详细历史状态见 [`docs/internal/README-chinese-detail.md`](docs/internal/README-chinese-detail.md)。

## 快速命令

```bash
# 激活项目 venv
source .venv/bin/activate  # Linux/macOS / git-bash
# 或
.venv/python.exe scripts/run_tests.py fast  # Windows 直跑

# 三档测试
.venv/python.exe scripts/run_tests.py fast                                  # API/schema/forward
.venv/python.exe scripts/run_tests.py module <data|training|architecture|tokenization>
.venv/python.exe scripts/run_tests.py full                                  # 阶段审查 / commit 前

# N13 单测
.venv/python.exe -m pytest tests/test_gqa_mla_models.py -v                  # 13 tests

# 真实 OWT 评测 selftest（97 PASS / 0 FAIL）
python3 scripts/eval_owt_real.py --selftest

# N13 训练复现（5 分钟 / 模型）
python3 -u scripts/train_dense.py --config configs/gqa-owt-formal-curve-medium.example.yaml \
  --output artifacts/gqa-owt-formal-curve-medium-result.json
python3 -u scripts/train_dense.py --config configs/mla-owt-formal-curve-medium.example.yaml \
  --output artifacts/mla-owt-formal-curve-medium-result.json

# N13 三方对比 plot
python3 scripts/plot_n13_comparison.py
```

## 运行环境

Python 3.10 + PyTorch 2.13 + CUDA 13.0 + Transformers 5.15 + vLLM 0.27.1（TORCH_SDPA）。

RTX 5070 Ti (sm_120) 已验证支持。完整环境说明见 [`docs/environment.md`](docs/environment.md)。

OWT 原始数据位于 `data/raw/owt-sample/`，**不会提交到 Git**（`.gitignore` 覆盖）。契约路径是 `datasets/owt-sample/owt_valid.txt`（Windows symlink）。

## 原则

1. 先做小而完整的闭环，再扩展功能。
2. 自研架构和开源模型训练线解耦，避免互相阻塞。
3. 不把"使用开源框架"本身当作成果，重点记录改动、验证和实验结论。
4. 不把"使用 vLLM 部署"当作成果，重点记录延迟、吞吐、显存和输出一致性。
5. 所有模型效果结论都应有统一数据、固定配置和可复现实验支撑。
6. 数据项目必须记录来源、许可证、处理版本和验证结果。
7. 真实结果出来后再写入简历，不提前使用未验证的指标。

## 关联文档

| 入口 | 用途 |
|---|---|
| [`docs/SHOWCASE.md`](docs/SHOWCASE.md) | 简历项目摘要 + bullet + plot |
| [`docs/INDEX.md`](docs/INDEX.md) | 全文档导航 |
| [`docs/plans/roadmap.md`](docs/plans/roadmap.md) | 路线图 |
| [`docs/plans/open-issues.md`](docs/plans/open-issues.md) | 决策记录 |
| [`docs/experiments/`](docs/experiments/) | 27 个实验详情 |
| [`docs/plans/reviews/`](docs/plans/reviews/) | 38 个 stage review |
| [`docs/protocols/`](docs/protocols/) | 22 个协议文档 |
| [`docs/internal/`](docs/internal/) | 内部中文规划文档 |
| [`docs/archive/`](docs/archive/) | 已关闭的实验 / 阶段 |
