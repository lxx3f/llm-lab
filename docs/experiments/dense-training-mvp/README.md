# Dense OWT training MVP

## 目的

验证以下真实数据闭环：

```text
OWT text
→ OWT BPE tokenizer
→ uint16 token cache
→ contiguous batch sampler
→ Dense Transformer training
→ validation loss
→ checkpoint/resume
→ tokenizer generation
```

代码入口：

```text
architecture_lab/training/dense_training.py
scripts/train_dense.py
```

## 运行方式

```bash
.venv/Scripts/python.exe scripts/train_dense.py \
  --config configs/dense_training.example.yaml \
  --output artifacts/dense-owt-mvp-scheduler-result.json

# formal 512 MiB / 64 MiB cache smoke
.venv/Scripts/python.exe scripts/train_dense.py \
  --config configs/dense_training.owt-formal.example.yaml \
  --output artifacts/dense-owt-formal-scheduler-amp-result.json
```

配置使用本地、已忽略的产物：

```text
artifacts/tokenizers/owt-bpe/v0.2.0/tokenizer.json
data/processed/owt-sample-smoke/  # 1 MiB smoke
data/processed/owt-sample/       # formal 512 MiB / 64 MiB cache
```

## 已验证能力

- tokenizer artifact 动态决定模型 vocab size；
- train/validation cache hash 校验；
- 显式 next-token targets；
- AdamW 和 gradient clipping；
- validation loss；
- 原子 checkpoint；
- checkpoint resume；
- tokenizer prompt generation；
- 正式 512 MiB/64 MiB cache 已生成并通过 source/tokenizer/cache hash 校验；
- warmup + cosine scheduler、梯度累积和 AMP 配置；
- 正式结果 schema 校验；

## 本地 smoke 结果

配置：`configs/dense_training.example.yaml` 使用 1 MiB cache；另有 `configs/dense_training.owt-formal.example.yaml` 使用正式范围 cache。两者均启用 warmup + cosine scheduler，AMP 默认关闭。

1 MiB 配置的结果（启用 warmup + cosine 后）：

```text
train steps: 100
train cache: OWT train 1 MiB newline-aligned prefix
validation cache: OWT validation 1 MiB newline-aligned prefix
model vocab size: 8192
sequence length: 64

step 50 validation_loss: 57.813842
step 100 validation_loss: 51.347665
step 100 last_train_loss: 49.561340
```

正式范围 cache 已生成并完成同配置 100-step smoke（启用 warmup + cosine）：

```text
train cache: 143,918,122 tokens / 536,870,901 encoded bytes
validation cache: 17,999,093 tokens / 67,105,041 encoded bytes
step 50 validation_loss: 57.813842
step 100 validation_loss: 51.347665
step 100 last_train_loss: 49.561340
```

正式范围 cache 的 100-step smoke 只从连续 stream 前缀取样：train 每次 update 使用 256 tokens，100 steps 约消费 25,600 token（未启用梯度累积）；validation 每次只评估 4 个 batch，即 1,024 token。因此该 smoke 不是对完整 143,918,122 / 17,999,093 token cache 的覆盖性评估。

```text
The meaning of life is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is is
```

该重复输出符合极短训练和极小模型的 smoke 预期，不能外推到正式训练效果。

## 测试

```bash
.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py" -q
.venv/Scripts/python.exe -m unittest discover -s architecture_lab/tests -p "test_*.py" -q
.venv/Scripts/python.exe -m unittest discover -s architecture_lab/tokenization/tests -p "test_*.py" -q
```

训练 checkpoint 和结果文件位于被 `.gitignore` 排除的 `artifacts/` 下，不提交 Git。

## 未完成范围

- 尚未实现分布式 rank-aware sampler；
- 尚未形成完整 OWT 训练质量结论；
- 尚未进入 Dense/MoE 公平训练对比。
