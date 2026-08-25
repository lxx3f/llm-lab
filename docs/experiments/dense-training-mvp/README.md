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
  --output artifacts/dense-owt-mvp-result.json
```

配置使用本地、已忽略的产物：

```text
artifacts/tokenizers/owt-bpe/v0.2.0/tokenizer.json
data/processed/owt-sample/train.*
data/processed/owt-sample/validation.*
```

## 已验证能力

- tokenizer artifact 动态决定模型 vocab size；
- train/validation cache hash 校验；
- 显式 next-token targets；
- AdamW 和 gradient clipping；
- validation loss；
- 原子 checkpoint；
- checkpoint resume；
- tokenizer prompt generation。

## 本地 smoke 结果

配置：`configs/dense_training.example.yaml`。

```text
train steps: 100
train cache: OWT train 1 MiB newline-aligned prefix
validation cache: OWT validation 1 MiB newline-aligned prefix
model vocab size: 8192
sequence length: 64

step 50 validation_loss: 55.168423
step 100 validation_loss: 22.880113
step 100 last_train_loss: 20.823641
```

生成结果为一次短跑产物，不能作为语言质量结论：

```text
The meaning of life is Gaza Gaza Gaza ...
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

- 正式 512 MiB/64 MiB cache 尚未生成；
- 尚未实现学习率 scheduler、AMP、梯度累积和分布式采样；
- 尚未形成完整 OWT 训练质量结论；
- 尚未进入 Dense/MoE 公平训练对比。
