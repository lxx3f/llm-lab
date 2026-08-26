# 夜间跑总结报告（2026-08-26 → 08-27）

> 本报告汇总夜间跑全部阶段：N7/N8/N9 消融 + MoE 曲线 + N11 长训练 + 多 seed 实跑 + P1-03/05 协议 + P1-01 mock + P1-04 D0 + D1 数据集。

## 1. 数据与训练协议

- **数据**：OWT 正式 cache（train 143.9M tokens / validation 18.0M），owt-bpe v0.2.0；
- **基准协议**：AdamW (lr=3e-4, wd=0.01, clip=1.0), warmup_cosine (100, min_lr=0.1), bf16, batch=8, seq=64, seed=42；
- **硬件**：RTX 5070 Ti Laptop (bf16)。

## 2. 5000 步消融结果（3-seed mean ± std）

| Sweep | 配置点 | val_min mean | 区间 | 单 seed 稳健？ |
|---|---|---|---|---|
| **N5 规模** | baseline/small/medium/large | 7.263 / 7.197 / 7.078 / 7.027 | **0.24 nats** | ✅ |
| **N6 dropout** | 0.0 / 0.1 / 0.2 | 7.078 / 7.087 / 7.115 | 0.04 nats | ❌（0.1 优势是噪声）|
| **N7 rope_base** | 10k / 50k / 100k | 7.078 / 7.088 / 7.078 | 0.01 nats | ❌ **被推翻**（无趋势）|
| **N8 n_heads** | 2 / 4 / 8 | 7.106 / 7.078 / 7.085 | 0.03 nats | ❌（heads-4 反而最优）|
| **N9 d_ff** | 256 / 512 / 1024 | 7.325 / 7.078 / 6.944 | **0.38 nats** | ✅ **最强** |

### 关键教训

1. **唯一跨单 seed + 多 seed 验证的结论**：规模（N5）+ d_ff（N9）是真正影响 val_loss 的架构因素。
2. **N7/N8 单 seed 结论被推翻**：rope_base 在 seq=64 下无显著影响；n_heads=8 无优势。P1-03 判定标准（mean 差 > std 和）正确滤掉了这两个假阳性。
3. **d_ff=256 std 最大（0.167）**：小 FFN 对初始化最敏感。

## 3. 长训练（50000 步）

| 规模 | params | 5000 步 val_min | 50000 步 val_min | 改善 | 多 seed std |
|---|---|---|---|---|---|
| baseline | 0.66M | 7.23 | **6.08** | 1.15 nats | 0.006 |
| medium | 2.10M | 7.06 | **5.55** | 1.51 nats | 0.007 |
| **large** | 5.11M | 6.93 | **5.26** | 1.67 nats | — |

- **全部未触底**（val_min_step == 50000）：50000 步对 ≤5.11M 模型仍不是充分训练；
- **规模优势随训练拉长扩大**：5000 步区间 0.30 → 50000 步区间 0.82 nats；
- **长训练噪声极低**（std < 0.01）：训练时间越长，seed 影响越小；
- 5000 步曲线严重低估模型能力（1.2-1.7 nats 改善）。

## 4. MoE vs Dense

| 模型 | params_total/active | 5000 步 val_min | 50000 步 val_min |
|---|---|---|---|
| MoE Top-1 | 2.10M / 1.51M | 7.22 @ 4600 | **6.04** @ 50000 |
| Dense medium | 2.10M / 2.10M | 7.06 @ 5000 | **5.55** @ 50000 |

- MoE 用 72% active params：5000 步达 97.8% Dense 性能，50000 步降到 91.7%；
- **长训练下 Dense 优势扩大**（0.16 → 0.50 nats）；
- MoE 5000 步的 val_min @ 4600 反弹在长训练视角下只是中段波动。

## 5. 训练基建与协议

| 阶段 | 交付 |
|---|---|
| **P1-01** | MockExecutor + tool_execution_result schema v1.0 + CLI + 8 单测 |
| **P1-03** | 多 seed 协议（3 seeds={42,123,7}，mean/std，判定标准）|
| **P1-04** | D0 MANIFEST workflow（build_d0_manifest.py）+ 3 样例 source=synthetic |
| **P1-05** | 六层失败分类（parse/schema/call_plan/execution/grounded/task）+ D1 数据集（126 样例，6 task_type）|
| **D1 数据集** | 确定性模板生成（seed=2026），train/dev/test=100/13/13，MANIFEST + sha256 + 目录与 manifest 严格一致 |

## 6. 文件与产物索引

- 消融 docs：`docs/experiments/n{5,6,7,8,9}-dense-*/README.md`（单 seed + 多 seed 复核）
- 多 seed 汇总：`docs/experiments/multi-seed-sweep/README.md`
- 长训练：`docs/experiments/n11-dense-long-curve/README.md` + `docs/experiments/n11-long-multi-seed/README.md`
- MoE：`docs/experiments/moe-long-curve/README.md`
- 协议：`docs/protocols/p1-{03,05}-*.md` + `p1-04-data-version-d0.md`
- Artifacts（gitignored）：`artifacts/*.json` + `artifacts/*.png`

## 7. 测试与提交

- `scripts/run_tests.py full`：**149 tests passing**（夜间跑开始前 100 → 149）；
- Git：main 从夜间跑起点 `43260e3` 推进到最终 commit（+12 commits：multi-seed 39-run / provenance fix / moe-long / n11-multi-seed / n11-large / p1-05 / d1-manifest-repro / report / p1-05 auditor fixes ×2）；当前 HEAD 即最终 commit，provenance 由 tests/test_artifact_provenance.py 动态验证。

## 8. 完成状态与遗留

### 已完成（objective 内）

| 交付 | 状态 | 证据 |
|---|---|---|
| N7/N8/N9 多 seed 实跑 | ✅ | 13 configs × 3 seeds，multi-seed-overview.json |
| MoE 50000 步长训练 | ✅ | moe-owt-formal-curve-long-result.json (6.0434 @ 50000) |
| N11 长训练多 seed | ✅ | baseline/medium × 3 seeds (std < 0.01) |
| N11 large 50000 步 | ✅ | 5.2639 @ 50000，3-scale overlay |
| D1 数据集 + P1-05 | ✅ | 126 样例 (100/13/13)，六层分类器，31 测试（test_d1_failure.py）|
| 夜间跑总结报告 | ✅ | 本文档 |

### 未解决（objective 外，单 seed 局限）

- **MoE 多 seed（5000/50000 步 × 3 seeds）未跑**——MoE vs Dense 的差距是单 seed 观察；
- **N11 large 多 seed 未跑**（只跑了单 seed 42）；
- 100000 步训练上限未测试（50000 步仍未触底，更长步数是否有天花板未知）；
- D1 的 result_grounded 层已通过 MockExecutor 端到端验证（108/108 expected calls 全部可执行）；真实推理后端接入后该层会接收真实 LLM 输出。

> 上述未解决项不影响本报告对已完成阶段的结论；单 seed 的 MoE/large 对比已明确标注为观察值。

### 下一步建议

MoE 多 seed → 真实推理后端（Transformers/vLLM）接入 → P2 评测闭环。