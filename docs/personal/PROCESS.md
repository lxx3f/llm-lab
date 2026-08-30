# LLM Lab 项目过程与实现详档

> **文档用途**：这是给后续分析 Agent 读取的事实资料。目标是完整描述本项目做过什么、为什么做、如何实现、遇到什么问题、如何解决，以及哪些结论可以使用、哪些不能过度延伸。
>
> **写作原则**：本文优先记录真实代码、实验产物和已验证结果，不将规划内容当作已完成内容，不将负面结果包装成正面结果，不把不同 base 模型之间的差异归因于架构收益。
>
> **项目入口**：[`../../README.md`](../../README.md)  
> **文档导航**：[`../INDEX.md`](../INDEX.md)  
> **面试理解稿**：[`INTERVIEW_PREP.md`](INTERVIEW_PREP.md)

---

## 1. 项目定位和原始目标

项目名称为 `llm-lab`，定位是一个面向大语言模型研究与工程实践的个人实验仓库。原始目标不是训练一个大规模可用的基础模型，而是构建一个小规模、可复现、可持续扩展的实验闭环：

```text
模型架构实现
    → 数据构造与处理
    → 模型训练 / 微调
    → Transformers / vLLM 部署与推理
    → 统一评测
    → 失败案例分析
```

原始规划包含四条主线：

1. 在 CS336 风格的 Decoder-only Transformer 基础上实现 Dense、MoE、GQA 和简化 MLA，并比较模型 loss、参数量、激活参数量、KV Cache 等指标。
2. 建立工具调用和结构化指令数据管线，覆盖数据生成、归一化、校验、工具执行验证、数据版本和 held-out split。
3. 建立统一评测和推理后端，接入 Transformers 与 vLLM，记录 latency、throughput、结构化输出和工具调用 reward。
4. 将架构、数据、SFT、GRPO、部署和评测串成一个端到端实验体系。

实际实现遵循了一个重要的解耦决策：

- **自研架构线**使用原生 PyTorch 小模型，优先验证 forward、训练、解码和 KV Cache 的正确性。
- **公开模型训练/评测线**使用 Hugging Face 生态模型，优先完成数据、SFT/GRPO、Transformers/vLLM 和 evaluator。
- 不强求自研 MLA/MoE 模型立即适配 vLLM，避免架构实验阻塞公开模型后端和数据管线工作。

---

## 2. 项目结构和主要职责

```text
architecture_lab/       自研模型、训练、token cache、benchmark、tokenizer、mock executor
configs/                YAML 实验配置
scripts/                训练、评测、数据生成、缓存、聚合、绘图、审计 CLI
schemas/                JSON Schema 契约
examples/               schema-conformant 样例数据和结果
artifacts/              本地训练结果、图、模型和缓存；大部分被 gitignore

tests/                  项目级 pytest

docs/
  experiments/          每个实验的 README / protocol / result
  protocols/            跨实验协议
  plans/                roadmap / open issues / stage reviews
  data/                 数据来源与数据契约
  reports/               总结报告和外部审计记录
  internal/             原始中文规划细节
  archive/              已关闭或不作为当前主线的交付物
```

当前主线代码主要位于：

- `architecture_lab/models/dense_transformer.py`
- `architecture_lab/models/moe_transformer.py`
- `architecture_lab/models/gqa_transformer.py`
- `architecture_lab/models/mla_transformer.py`
- `architecture_lab/training/dense_training.py`
- `architecture_lab/training/moe_training.py`
- `architecture_lab/tokenization/`
- `architecture_lab/data/`
- `scripts/eval_transformers.py`
- `scripts/eval_backend_comparison.py`
- `scripts/eval_owt_real.py`
- `scripts/_hf_backend.py`

---

## 3. 实现过程总览

### 3.1 Stage 0：统一契约和最小样例

最开始先定义数据和结果的 schema，而不是直接把实验输出写成不稳定的 JSON。主要契约包括：

- tool calling sample
- model output
- tool execution result
- evaluation result
- reward signal
- GRPO step result
- Dense training result
- MoE training result
- N2 benchmark result
- N2 routing statistics
- D2 multi-turn sample

每个 schema 配有 examples 和测试。这样后续数据生成、评测、训练结果和聚合脚本可以共享字段约束。

### 3.2 Dense Transformer 基线

Dense baseline 是所有架构实验的基础。实现采用 decoder-only Transformer，主要组件包括：

- tied token embedding / LM head
- causal self-attention
- RoPE
- RMSNorm
- SwiGLU FFN
- causal language-model loss
- full forward
- incremental decode
- KV Cache
- checkpoint 保存与恢复
- warmup + cosine learning-rate scheduler
- AMP/bfloat16
- gradient accumulation
- validation loss 和训练曲线记录

训练循环通过 `TokenStreamBatcher` 从 token cache 取 batch，模型只负责 logits 和 loss，结果由 `build_training_result()` 按 schema 落盘。

Dense 实验从短 smoke 开始，逐步扩展到：

- 0.66M / small baseline
- 2.10M / medium baseline
- 多规模 sweep
- dropout sweep
- RoPE base sweep
- attention head 数 sweep
- FFN hidden size sweep
- 50k 长训练
- 200k ultra-curve 方向
- 多 seed 复现

### 3.3 MoE Top-1

MoE Top-1 在 Dense FFN 位置增加 router 和多个 SwiGLU experts：

```text
hidden state
    → router logits
    → top-1 expert index
    → capacity / overflow handling
    → expert FFN
    → combine
```

实现和记录了：

- router
- top-1 routing
- expert dispatch
- capacity factor
- dropped token / overflow
- auxiliary load-balancing loss
- expert token statistics
- active parameter count
- checkpoint / resume
- MoE result schema

早期版本只有 forward MVP，后续补上了训练循环、正式 OWT cache 训练、曲线和 Dense/MoE 公平比较。MoE 正式曲线配置为约 2.10M total params，4 experts，Top-1 active FFN，5000 步训练。

已验证结果之一：

- MoE 5000 步：train loss `117.90 → 6.79`
- `val_min = 7.22 @ step 4600`
- 配置记录 total params 和 top-1 active params

这些数值用于说明训练链路和架构实验结果，不代表大模型能力或 MoE 的普适收益。

### 3.4 N2 Dense/MoE 公平比较

直接比较 Dense 和 MoE 容易被参数总量、激活参数量、routing overhead 混淆。因此 N2 定义了两套协议：

- **Protocol A：相同 total parameters**
- **Protocol B：相同 active parameters**

结果中单独记录：

- total parameters
- active parameters
- train loss
- validation loss
- auxiliary loss
- routing statistics
- dropped token
- latency 相关字段

N2 的短训练 smoke 结果不作为正式模型质量结论，主要用于验证比较协议、统计字段和 schema 是否正确。

### 3.5 GQA 实现（N13）

GQA 的核心变化是：Q 头数量保持 `n_heads`，K/V 头数量改为更小的 `num_kv_heads`。

约束：

```text
n_heads % num_kv_heads == 0
```

例如 N13 使用：

```text
n_heads = 4
num_kv_heads = 1
```

实现细节：

- Q projection 输出 `d_model`
- K/V projection 输出 `num_kv_heads * head_dim`
- K/V cache 只保存较少的 KV heads
- attention 前使用 `repeat_interleave` 将 KV head 扩展到 query head 数
- 保持 RoPE、causal mask、SDPA、增量 decode 接口不变
- `num_kv_heads == n_heads` 时退化为 MHA 形态
- `num_kv_heads == 1` 是极端 MQA 形态

### 3.6 简化 MLA 实现（N13）

N13 的 MLA 是教学和小模型实验用的简化版本，不是 DeepSeek-V2 的完整 MLA。

当前实现：

```text
c_kv = W_DKV(x)             # d_model → latent_dim
K   = W_UK(c_kv)            # latent_dim → d_model
V   = W_UV(c_kv)            # latent_dim → d_model
Q   = W_Q(x)                # 直接投影，不压缩 Q
```

关键特点：

- K/V 共享一个 latent down-projection
- KV Cache 保存 `c_kv` 压缩 latent，而不是完整 K/V
- 每次 attention 前从 latent 重建 K/V
- Q 直接投影，未实现完整 MLA 的 Q 压缩
- RoPE 施加在重建后的 K 上
- 未实现完整 decoupled RoPE

N13 配置：

```text
n_heads = 4
head_dim = 32
latent_dim = 64
```

这个设计的优点是代码清楚、能够训练、能够增量 decode，并能直接展示 cache 压缩；缺点是每次需要重建 K/V，实际 decode FLOPs 不一定优于 GQA。

### 3.7 N13 三方架构实验

N13 使用相同的 OWT formal token cache、相同训练设置和相同 seed，对比：

| 架构 | 参数量 | train loss first → last | val_min |
|---|---:|---:|---:|
| Dense MHA | 2,098,304 | 104.69 → 6.56 | 7.058 |
| GQA，`num_kv_heads=1` | 2,000,000 | 103.84 → 6.71 | 7.106 |
| 简化 MLA，`latent_dim=64` | 2,065,536 | 111.72 → 6.68 | 7.122 |

在单 batch、单 token、float32、单 layer 的理论 cache 估算中：

- Dense MHA：`2 × 4 × 32 × 4 = 1024 bytes`
- GQA：`2 × 1 × 32 × 4 = 256 bytes`
- MLA：`64 × 4 = 256 bytes`

因此本配置下 GQA 和 MLA 的 cache 都是 Dense MHA 的 1/4。

重要限制：

- 只有一个 seed
- 只有一个 `num_kv_heads` 配置
- 只有一个 `latent_dim` 配置
- 模型很小，不能推导大模型收益
- 没有在 N13 中做真实 decode latency benchmark
- 简化 MLA 不等同于完整 MLA

---

## 4. 数据管线过程

### 4.1 数据类型和版本

项目围绕工具调用和结构化指令数据设计了 D0/D1/D1.1/D2 等版本：

- D0：基础或原始版本
- D1：模板化工具调用数据
- D1.1：LLM 生成的扩展版本
- D2：包含多轮、错误恢复和执行验证的扩展数据

已落盘或验证的规模包括：

- D1 模板数据：126 条
- D1.1 LLM 生成数据：约 1500 条
- D2 多轮数据：5000 条级别

D2 覆盖：

- 不需要工具
- 单工具调用
- 多工具串联
- 参数错误
- 工具执行失败
- 错误恢复
- 用户修改约束
- 多轮依赖
- dangling dependency
- cyclic dependency

### 4.2 数据生成和校验链路

最终采用的思路不是“调用模型后直接保存 JSON”，而是：

```text
任务模板
→ 参数采样
→ 模型生成
→ 规则检查
→ JSON Schema 校验
→ 工具执行
→ 最终答案检查
→ manifest / version / provenance
```

每条数据尽量绑定：

- source
- license
- task_type
- data_version
- pipeline_version
- created_at
- original sample id
- validation result
- execution result

### 4.3 Mock Executor

Mock Executor 用于在不依赖真实外部系统的情况下验证工具调用链路：

- 工具名称是否存在
- 参数是否符合 schema
- 工具是否可以执行
- 工具返回值是否可被后续步骤使用
- 多工具依赖是否满足
- 错误恢复是否符合预期
- 最终回答是否依赖真实工具结果

### 4.4 八级失败分类

P1-05 将工具调用失败拆成可定位的层级，而不是只记录最终 success/fail：

```text
parse_success
→ schema_valid
→ tool_name_correct
→ argument_correct
→ call_plan_correct
→ execution_success
→ result_grounded
→ final_answer_correct
```

这样可以区分：

- JSON 解析失败
- schema 不合法
- 工具选错
- 参数错误
- 调用顺序错误
- 执行失败
- 没有使用工具结果
- 最终答案仍然错误

这个分类器后续被 reward evaluator 和失败分析复用。

---

## 5. Tokenizer 和 OWT 数据过程

### 5.1 BPE tokenizer

项目实现了基于 OWT sample 的 BPE tokenizer 训练和 artifact 管理：

- tokenizer 训练 CLI
- tokenizer.json
- 版本化 artifact
- vocabulary size
- special token ids
- metadata
- artifact SHA256
- tokenizer CLI 测试

### 5.2 Token cache

为了避免每次训练都重新编码文本，项目实现了 token cache：

- 流式读取文本
- UTF-8 处理
- 指定 dtype 和 endian
- train / validation 分开
- cache metadata
- source path / source SHA256
- tokenizer artifact SHA256
- token count
- encoded byte count
- token file SHA256

正式 OWT cache 使用了固定 source contract。OWT sample 的 validation 文件约 289,998,753 bytes，记录了 SHA256：

```text
2406f278e71829d273b315e9b403285baea7022b26a96d2728dd8b776ea40660
```

训练 cache 使用 `uint16 little-endian`；由于 Qwen2.5 tokenizer vocabulary 超过 uint16 范围，真实 OWT 评测的 per-model cache 使用了 `int32 little-endian`。

### 5.3 实际遇到的问题：Windows symlink

原本尝试使用普通的 `ln -sfn` 建立数据契约路径，但在 WSL / Git Bash / Windows 文件系统交互下，符号链接行为不稳定，曾被表现为复制目录。

最后使用 Windows：

```text
mklink /D datasets/owt-sample data/raw/owt-sample
```

并在文档里固定 source path 和 hash，避免不同环境下读到不同文件。

---

## 6. 训练系统实现

训练系统基于原生 PyTorch，重点是小而可复现，而不是接入大型训练框架。

### 6.1 训练配置

YAML 配置管理：

- model architecture
- vocab size
- d_model
- n_heads
- n_layers
- d_ff
- attention-specific parameters
- tokenizer path
- train / validation cache
- batch size
- sequence length
- learning rate
- weight decay
- warmup steps
- scheduler
- AMP
- checkpoint
- validation interval
- max steps
- seed

### 6.2 训练结果

每个训练结果包含：

- schema_version
- result_type
- experiment_id
- timestamp
- status
- model config
- parameter count
- data/cache records
- seed
- device
- dtype
- optimizer config
- scheduler config
- train loss samples
- validation losses
- curve summary
- checkpoint path / SHA256
- generation output
- experiment metadata

### 6.3 结果记录遇到的问题

最初 `build_training_result()` 把 architecture 硬编码为 `DenseTransformer`。当 GQA/MLA 复用 Dense training loop 后，即使实际实例化了 GQA/MLA，结果 JSON 仍显示 Dense。

解决方式：

- 让 `build_training_result()` 从 settings 读取 architecture
- 扩展 `dense_training_result.schema.json` 的 architecture enum
- 对 Dense / GQA / MLA 三个 result 做 schema validation
- 保留 `result_type = dense_training` 作为兼容字段，因为已有 schema 和测试依赖该值

这是一个典型的“模型已经换了，但 metadata 没有同步”的问题。

---

## 7. 统一推理和评测过程

### 7.1 Transformers backend

P5-02 复用了共享 Hugging Face backend：

- `resolve_device`
- `resolve_dtype`
- `load_causal_lm_model`
- tokenizer / chat template
- greedy generation
- generation failure 记录
- 结果 JSON

评测覆盖了 5 个公开 instruction-tuned 模型：

- HuggingFaceTB/SmolLM2-360M-Instruct
- HuggingFaceTB/SmolLM2-1.7B-Instruct
- Qwen/Qwen2.5-0.5B-Instruct
- Qwen/Qwen2.5-1.5B-Instruct
- Qwen/Qwen2.5-3B-Instruct

### 7.2 vLLM backend

P5-03 在 WSL2 环境完成 vLLM smoke。P5-04 使用 vLLM 和 Transformers 做同模型、同样本、同 batch 的对比。

### 7.3 P5-04 双后端对比

实验规模：

```text
5 models × 2 backends × 2 batch sizes × 90 samples = 20 combinations
```

记录：

- per-sample latency
- throughput
- reward_binary
- reward_layered
- parse success
- generation failure
- backend delta

数据加载采用 manifest-driven 设计。固定的 90 样本 manifest 记录每条 sample 的 path 和 SHA256，CLI 必须显式传入 manifest，避免错误地加载父 split 或临时目录。

代表性结果：

- 所有 20 个组合无 generation failure
- reward_binary 全为 0，是诚实负结果
- reward_layered 大致在 0.36–0.42
- vLLM batch=4 在固定 benchmark 上进入 throughput Top-5
- 最快组合为 Qwen2.5-0.5B vLLM batch=4：约 8.18 samples/s、122.3 ms/sample
- vLLM 相对 Transformers 的 throughput 提升依模型和 batch 而变，部分组合约 +55% 到 +90%

### 7.4 遇到的问题：reward_layered 全为 0

早期 P5-04 结果中 reward_layered 全为 0。根因不是模型都没有任何结构化能力，而是：

- CLI 从 manifest 得到的是 sample 字典
- reward 计算路径却从 `args.samples_dir` 再次 glob
- manifest 父目录中没有直接 sample JSON
- evaluator 得到空 sample 集合
- 最终 reward 被计算为 0

解决方式：

- manifest loader 得到 `samples_by_id`
- 直接将样本字典传入 `run_one_combination()` 和 `compute_reward()`
- 重新运行并通过历史结果范围校验
- 增加 wiring 和结果一致性测试

### 7.5 遇到的问题：batch generation failure

某些模型或 batch 组合可能出现单批次 generation failure。最终实现了：

- batch-level failure 记录
- per-sample retry
- 失败样本不静默丢失
- result JSON 记录 failure type
- 聚合时区分真实失败和空结果

---

## 8. 真实 OWT 评测 E

P5-02/P5-04 主要评测工具调用数据。E 进一步在真实文本分布上对 5 个公开模型进行纯 LM/per-token loss 评测。

### 8.1 评测方式

- 使用完整 OWT validation 文件
- 不把 validation 放入训练
- 每个模型使用自己的 tokenizer
- 以 seq_len=1024 的 non-overlapping windows 计算 causal cross-entropy
- 每个窗口检查 logits 和 loss 是否 finite
- 记录 sum loss、evaluated tokens、mean loss、perplexity
- 额外记录 loss per evaluated byte

### 8.2 关键方法学问题

不同 tokenizer 下，perplexity 不能简单横向比较。SmolLM2 和 Qwen2.5 的 vocab size 差异明显，单 token 包含的信息量不同。

因此报告同时记录：

- `mean_loss_nats`
- `perplexity`
- `loss_nats_per_evaluated_byte`

并明确把 per-byte 也视作辅助指标，而不是完全消除 tokenizer 影响的万能指标。

### 8.3 实际遇到的问题：NaN 结果

第一版全量评测曾在末尾产生 NaN 结果。原因是：

- 没有对每个 window 的 logits/loss 做 finite check
- forward 默认 cache 行为可能增加显存和状态复杂度
- 旧结果在流程末尾才暴露错误

解决方式：

- 每个窗口显式检查 logits/loss finite
- 非 finite 立即报告窗口号并失败
- forward 显式使用 `use_cache=False`
- 删除无效 NaN 结果，不把它们纳入交付
- 重新验证 5 个模型结果

### 8.4 代表性结果

- SmolLM2-1.7B：PPL 11.05
- Qwen2.5-3B：PPL 13.04
- 总 validation source 约 277 MiB 级别
- 5 个模型顺序执行总计约 6 小时以上

以上数值只表示本数据、tokenizer、revision 和计算定义下的结果，不代表模型能力排名。

---

## 9. SFT、GRPO 和 evaluator

### 9.1 SFT 工具调用训练

项目完成了 Dense 和 MoE 的工具调用 SFT MVP：

- 使用 D1/D1.1 数据
- 多个 checkpoint
- D1 dev / D2 dev 评测
- reward_binary
- reward_layered
- failure classification
- Dense / MoE 训练结果

结果中保留了公开模型或自研模型无法达到高 reward 时的负结果，不人为修改评测定义来制造正向指标。

### 9.2 确定性 evaluator

P2 evaluator 主要目标是把模型输出转为稳定、可分解的 reward：

- binary reward
- layered reward
- reward type
- parse / schema / tool / execution / grounding / final answer 层级

优先使用确定性规则，而不是一开始引入不可控的 LLM judge。

### 9.3 GRPO MVP

P4 完成了小规模 GRPO MVP 和 smoke 实验，覆盖：

- rollout / response
- reward signal
- advantage
- policy update
- step result
- 多 seed smoke
- CPU / 小模型正确性检查

该部分的定位是验证训练闭环和算法组件，而不是声称实现了工业级 RLHF 系统。

---

## 10. 可复现性基础设施

项目把“实验结果”看作数据产品，不只保存一个数字。

### 10.1 Metadata

实验 metadata 记录或绑定：

- git commit
- config SHA256
- tokenizer revision / artifact SHA256
- dataset/cache SHA256
- seed
- Python version
- PyTorch version
- CUDA version
- GPU name
- compute capability
- dtype
- backend
- model revision

### 10.2 Manifest 和 hash

数据和 benchmark 使用：

- source SHA256
- token file SHA256
- metadata SHA256
- sample manifest aggregate SHA256
- checkpoint SHA256
- config SHA256

一个曾经出现的问题是 `metadata_sha256` 的计算契约不一致。最终固定为：

```text
metadata_sha256 = sha256(JSON(metadata without metadata_sha256 field))
```

并让测试按该契约重新计算。

### 10.3 测试分层

项目提供三档测试入口：

```bash
python scripts/run_tests.py fast
python scripts/run_tests.py module <data|training|architecture|tokenization>
python scripts/run_tests.py full
```

当前项目级 pytest 收集约 421 tests。N13 新增 13 个模型测试；E 自带 97 项 selftest；P5-04 和数据、训练、schema 也分别有回归测试。

---

## 11. 文档和审计过程

每个重要阶段都配套：

- README
- protocol
- result / artifact
- stage review
- tests
- metadata

F 项目（公开模型 GQA vs MHA 可行性搜索）期间暴露了多个文档一致性问题，因此后续形成了更严格的文档审计约束：

- 不能把 19 个候选全部写成 verified no-go
- 5 个 verified rows 和 14 个 feasibility leads 必须分开
- `head_at_review` 不能伪装成当前 HEAD
- reviewer evidence 必须是外部 artifact
- source/doc commit 和 reviewer-evidence follow-up commit 分离
- 不能写死容易变化的文件大小
- 跳过的 test 不能作为通过证据
- 跨文档 candidate count 必须一致
- schema evidence 必须包含真实 schema、auditor-runnable command 和 recorded output

这些审计经验虽然不是主要功能，但改变了项目后续的实验记录方式。

F 的 4 个文档已移至：

```text
docs/archive/gqa-vs-mha-no-go/
docs/plans/reviews/archive/stage-gqa-vs-mha-no-go.md
```

原因是 F 是一个 incomplete feasibility review，不是当前架构实验主线。

---

## 12. 重要结论和不可过度延伸的地方

### 可以明确说的

- 已实现 Dense MHA、MoE Top-1、GQA、简化 MLA 四种自研路径。
- 已完成多个小规模 OWT 训练和曲线实验。
- GQA 和简化 MLA 在 N13 配置下可以训练、验证和增量 decode。
- N13 配置下 GQA/MLA 理论 KV Cache 为 Dense MHA 的 1/4。
- 5 个公开模型完成了真实 OWT per-token loss 评测。
- Transformers/vLLM 完成固定 90 样本、20 组合的后端对比。
- 工具调用数据拥有 schema、执行验证和分层失败分析。
- 训练、评测、数据和结果都具备 hash/metadata/test 绑定。

### 不能说的

- 不能说简化 MLA 等同 DeepSeek-V2 MLA。
- 不能说 N13 证明 GQA/MLA 在大模型上更好。
- 不能把理论 cache bytes 直接说成真实 decode latency 提升。
- 不能把单 seed、单配置的 val_min 差距当成统计显著结论。
- 不能把不同 tokenizer 下 PPL 直接排出模型能力名次。
- 不能说已完成自研模型的完整 vLLM 适配。
- 不能把 P5-04 reward_binary=0 隐藏或改写成正向结果。
- 不能把 F incomplete feasibility review 说成 19 个候选全部排除。

---

## 13. 关键文件索引

### 模型和训练

- `architecture_lab/models/dense_transformer.py`
- `architecture_lab/models/moe_transformer.py`
- `architecture_lab/models/gqa_transformer.py`
- `architecture_lab/models/mla_transformer.py`
- `architecture_lab/training/dense_training.py`
- `architecture_lab/training/moe_training.py`
- `architecture_lab/training/results.py`
- `architecture_lab/training/moe_results.py`

### 数据和 tokenization

- `architecture_lab/data/token_cache.py`
- `architecture_lab/data/batching.py`
- `architecture_lab/tokenization/bpe.py`
- `architecture_lab/tokenization/artifact.py`
- `scripts/encode_token_cache.py`
- `scripts/generate_d1_dataset.py`
- `scripts/generate_d2_dataset.py`
- `docs/data/owt-sample.md`

### 推理和评测

- `scripts/_hf_backend.py`
- `scripts/eval_transformers.py`
- `scripts/eval_backend_comparison.py`
- `scripts/eval_owt_real.py`
- `scripts/reward_offline.py`
- `scripts/classify_tool_failure.py`
- `scripts/grpo_train.py`

### N13

- `configs/gqa-owt-formal-curve-medium.example.yaml`
- `configs/mla-owt-formal-curve-medium.example.yaml`
- `scripts/plot_n13_comparison.py`
- `tests/test_gqa_mla_models.py`
- `docs/experiments/n13-gqa-vs-mla-vs-mha/`
- `docs/plans/reviews/stage-n13-gqa-vs-mla-vs-mha.md`

### 结果与契约

- `schemas/`
- `examples/evaluation_results/`
- `artifacts/owt-real-eval/`
- `artifacts/p5-04-backend-comparison/`
- `artifacts/dense-owt-formal-curve-medium-result.json`
- `artifacts/gqa-owt-formal-curve-medium-result.json`
- `artifacts/mla-owt-formal-curve-medium-result.json`

---

## 14. 当前状态

当前项目已经具备一个完整的小规模 LLM 实验闭环：

```text
自研架构
Dense / MoE / GQA / simplified MLA
        ↓
OWT tokenizer + token cache
        ↓
原生 PyTorch training + checkpoint + validation
        ↓
公开模型 Transformers / vLLM backend
        ↓
工具调用 evaluator + reward + failure taxonomy
        ↓
JSON Schema + hash metadata + tests + docs
```

后续仍可以继续做的技术方向包括：

- GQA `num_kv_heads` sweep
- MLA `latent_dim` sweep
- 三种架构的真实 prefill/decode latency benchmark
- N13 多 seed 复现
- 自研架构的 Hugging Face / vLLM 适配
- 更大规模或更长上下文实验

这些属于后续扩展，不是当前已交付结果的一部分。
