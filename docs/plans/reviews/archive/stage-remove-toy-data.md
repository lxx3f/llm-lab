# toy 数据删除审查

## 完成范围

- 删除 `data/toy/train.txt`；
- 删除 `data/toy/validation.txt`；
- 删除 `artifacts/tokenizers/toy-bpe/v0.1.0/`；
- 删除 `docs/data/toy-dataset.md`；
- 删除当前文档中将 toy 数据作为保留 smoke 数据的表述；
- 将 OWT 明确为第一版真实训练数据；
- 保留 BPE 实现、CLI、协议和测试代码；
- 更新 OWT tokenizer CLI 示例。

## 保留内容

```text
data/raw/owt-sample/
```

原始 OWT 数据仍保留在本地，但由 `.gitignore` 排除，不进入 Git。

## 未完成范围

- OWT BPE tokenizer；
- 大语料流式/分块 BPE；
- OWT token cache；
- Dense training loop；
- checkpoint 和 generation。

## 计划调整

后续不再维护 toy 数据分支，统一使用：

```text
OWT sample
→ OWT tokenizer
→ OWT token cache
→ Dense training
```

BPE 单元测试继续使用内存中的小型字符串 fixture，不新增仓库内 toy 数据文件。

## 是否允许进入下一阶段

通过。删除动作和文档调整完成；下一阶段聚焦大语料 BPE，不再回到 toy 数据方案。

## Git 提交

审查通过后创建阶段 commit，未执行 `git push`。
