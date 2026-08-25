# 第一版 toy 训练数据

## 当前决策

第一版 Dense 训练闭环使用仓库内人工构造的最小语料，仅用于验证：

- BPE tokenizer 训练和加载；
- 文本编码；
- train/validation 数据读取；
- Dense Transformer 训练 loss 是否下降；
- checkpoint 和 generation 链路。

它不用于支持模型能力、数据规模或泛化性能结论。

## 文件

```text
data/toy/train.txt
 data/toy/validation.txt
```

实际路径中的文件名没有前导空格；上面的缩进仅用于展示目录结构。

当前统计：

| 文件 | 行数 | 字节数 |
|---|---:|---:|
| `train.txt` | 50 | 4161 |
| `validation.txt` | 14 | 1037 |

## 来源和许可证

- 来源：`llm-lab-manual-example`；
- 内容类型：人工构造；
- 许可证：`CC0-1.0`；
- 数据版本：`D0`；
- 编码：UTF-8；
- 训练数据 SHA-256：`9b64d47b4f2f793c7fdeb36c43996355f88a765a572ec15ef89431a039f3f33b`；
- 验证数据 SHA-256：`2ec574d113e841bdd986e44788f3d807b61d87be5ef06b965013c2b270f9a872`。

数据按文档/行块预先分为 train 和 validation，没有在编码后随机切分 token stream。当前语料中的 `<|endoftext|>` 用于文档边界。

## Special token 决策

第一版只使用：

```text
<|endoftext|>
```

暂不增加 BOS、EOS、PAD：

- `<|endoftext|>` 同时作为文档边界和序列拼接分隔符；
- causal LM 不需要 PAD，因为 batch 采样暂时使用连续 token stream；
- BOS/EOS 的语义与现有数据格式尚未必要，避免增加第一版变量。

## Tokenizer artifact

使用：

```text
artifacts/tokenizers/toy-bpe/v0.1.0/
├── tokenizer.json
└── metadata.json
```

配置：

- algorithm：byte-level BPE；
- pretokenization：GPT-2 regex；
- requested vocab size：512；
- actual vocab size：512；
- special token ID：`<|endoftext|>` → `256`。

artifact metadata 中记录了 source path、source hash、license、data version、tokenizer hash 和 config hash。

## 当前边界

- toy corpus 很小，不能用于正式语言模型效果结论；
- 当前只含一份 train/validation split，尚未建立测试集；
- 尚未实现编码后的 dataset cache；
- 尚未开始 Dense Transformer 训练。
