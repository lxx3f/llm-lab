# OWT 数据源切换审查

## 完成范围

- 确认第一版真实语言模型训练数据使用 Stanford CS336 OWT sample；
- 下载 `owt_train.txt.gz` 和 `owt_valid.txt.gz`；
- 完成 gzip 完整性检查；
- 解压并记录压缩/解压后文件 hash；
- 将原始数据放在 `data/raw/owt-sample/`；
- 更新 `.gitignore`，排除原始、处理后数据和 tokenizer artifact 本地产物；
- 增加 `docs/data/owt-sample.md`；
- 保留 toy corpus 作为 smoke 数据，不再作为正式语言模型训练数据。

## 未完成范围

- 尚未训练 OWT BPE tokenizer；
- 尚未实现大语料分块/流式 BPE；
- 尚未编码 OWT train/validation；
- 尚未实现 dataset cache；
- 尚未实现 Dense training loop；
- 尚未形成正式训练结果。

## 数据确认

```text
owt_train.txt.gz: 约 4.59 GB
owt_valid.txt.gz: 约 106.6 MB
owt_train.txt:    约 12 GB
owt_valid.txt:    约 277 MB
```

官方数据集 revision：

```text
488afadefc224407de5645a3c8c53b64ed923537
```

## 新发现问题

当前 tokenizer CLI 使用完整文本读入内存：

```python
text = path.read_text(encoding="utf-8")
```

不能直接用于约 12GB 的 OWT train 文件。下一步必须先实现：

```text
分块读取
→ 分块预分词
→ 持久化 token frequency 统计
→ BPE merge 训练
→ artifact metadata
```

## 计划调整

```text
OWT 数据登记
→ 大语料 BPE 训练方案
→ OWT tokenizer artifact
→ OWT token cache
→ Dense training loop
```

## 是否允许进入下一阶段

有条件允许。数据源决策和下载阶段通过；P0-01 仍未完成，当前阻塞点转为“大语料 tokenizer 实现”。

## Git 提交

审查通过后创建阶段 commit，未执行 `git push`。
