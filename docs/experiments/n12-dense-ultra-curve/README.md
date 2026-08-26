# N12 Dense Ultra-Long Training Curve（100000 步）

> 验证更长训练步数下 val 是否触底/反弹。把 baseline 0.66M + medium 2.10M 都拉到 100000 步（50000 步的 2×），对比 50000 步曲线。

## 摘要

| 规模 | 50000 步 val_min | 100000 步 val_min | 改善 | val_min_step | Δ_val (98000→100000) |
|---|---|---|---|---|---|
| baseline 0.66M | 6.0796 | **5.7492** | **-0.33** | 98000 | +0.012 |
| medium 2.10M | 5.5453 | **5.2325** | **-0.31** | 98000 | +0.015 |

**关键观察**：

1. **100000 步 vs 50000 步**：两个规模都持续改善 ~0.3 nats，说明 50000 步确实没触底，模型还在学。
2. **饱和信号**：两个规模 val_min 都出现在 step 98000（最后 2%），且 val 在最后 2000 步有 +0.012 ~ +0.015 的轻微反弹——这是 saturation 的明确信号：val_loss_min_step 不再是 training end，说明进一步训练收益递减。
3. **规模优势依然成立**：100000 步下 medium 仍优于 baseline（5.2325 vs 5.7492，区间 0.52 nats）。
4. **训练时间**：baseline 0.66M 100000 步约 17 min；medium 2.10M 100000 步约 44 min（两个并发跑，实际总墙钟 ~45 min）。

## 运行命令

```bash
.venv/python.exe -u scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-ultra-baseline.example.yaml \
    > /tmp/ultra-baseline.log 2>&1 &

.venv/python.exe -u scripts/train_dense.py \
    --config configs/dense_training.owt-formal-curve-ultra-medium.example.yaml \
    > /tmp/ultra-medium.log 2>&1 &

# 等待完成后，从 log 提取 JSON（训练脚本默认 print 到 stdout，不直接写文件）：
python -c "import json; from pathlib import Path; log = Path('/tmp/ultra-baseline.log').read_text(); sv = log.find('schema_version'); s = log.rfind('{', 0, sv); e = log.rfind('}') + 1; obj = json.loads(log[s:e]); Path('artifacts/dense-owt-formal-curve-ultra-baseline-result.json').write_text(json.dumps(obj, ensure_ascii=False, indent=2))"
```

## 与 N11 long curve 的关系

| 阶段 | 步数 | 规模 | 主要发现 |
|---|---|---|---|
| N4 baseline | 5000 | 0.66M | dense baseline 曲线建立 |
| N11 long baseline | 50000 | 0.66M | val_min=6.08 @ 50000（未触底）|
| **N12 ultra baseline** | **100000** | **0.66M** | val_min=5.75 @ 98000（接近饱和，+0.012 反弹）|
| N11 long medium | 50000 | 2.10M | val_min=5.55 @ 50000（未触底）|
| **N12 ultra medium** | **100000** | **2.10M** | val_min=5.23 @ 98000（接近饱和，+0.015 反弹）|

## 数据契约

- 训练 schema：`dense_training_result.schema.json` v1.1
- 元数据：`metadata.git_commit = cb44a8bd7a97c22eecbeb4d099aeeeccaa4aed02`
- artifact 路径：
  - `artifacts/dense-owt-formal-curve-ultra-baseline-result.json`
  - `artifacts/dense-owt-formal-curve-ultra-medium-result.json`
- overlay：`artifacts/dense-owt-formal-curve-ultra-vs-long.png`（800×500，4 曲线：ultra-baseline / long-baseline / ultra-medium / long-medium）

## 测试

- `tests/test_artifact_provenance.py::ArtifactProvenanceTests::test_every_objective_artifact_has_known_commit_and_metadata` — 验证 ultra artifact 的 git_commit 在 KNOWN_NIGHT_RUN_COMMITS 中 + seed=42 + status=completed + optimizer_steps=100000
- `tests/test_artifact_provenance.py::ArtifactProvenanceTests::test_ultra_curve_saturation_signature` — 强约束饱和信号：val_min_step ≥ 95% × 100000，delta_val_loss ∈ (0, 0.1)，val_last ≈ val_min ± 0.05
