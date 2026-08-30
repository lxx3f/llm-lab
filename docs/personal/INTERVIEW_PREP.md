# 个人项目面试准备稿：LLM Lab

> 这是一份可独立携带的个人准备资料，不依赖仓库中的其他文档。目标是帮助你理解项目做了什么、为什么这样做、哪些结论可信、哪些地方不能过度宣传，以及如何回答常见追问。
>
> 先记住：这个项目不是训练一个大规模生产模型，而是一个**小规模、可复现的 LLM 实验闭环**。它的价值在于把模型架构、数据、训练、推理、评测、失败分析和实验记录串起来。

---

## 1. 先用一句话说清楚项目

> 我实现了一个面向 LLM 研究与工程实践的实验闭环：一条线用原生 PyTorch 实现并训练 Dense、MoE、GQA 和简化 MLA 等 Transformer 架构，另一条线构建工具调用数据、SFT/GRPO 训练、Transformers/vLLM 后端和确定性评测器，所有实验通过统一配置、JSON Schema、hash 和测试记录保证可复现。

如果对方只给你 30 秒，可以说：

> 这个项目主要解决两个问题：一是如何在相同数据和训练条件下比较不同 Transformer attention/FFN 架构；二是如何把工具调用数据、模型训练、推理后端和评测指标组成一个可以重复运行的实验系统。最终我完成了 Dense/MoE/GQA/简化 MLA 自研模型、工具调用数据管线、SFT/GRPO MVP，以及 Transformers 和 vLLM 的公开模型对比。

---

## 2. 项目到底做了哪些事情

可以把项目分成四层：

```text
Layer 1: 自研模型架构
Dense MHA → MoE Top-1 → GQA → simplified MLA

Layer 2: 数据和训练
OWT tokenizer/cache → 原生 PyTorch training
工具调用 D1/D1.1/D2 → SFT / GRPO

Layer 3: 推理和评测
Transformers backend + vLLM backend
确定性 evaluator + reward + failure taxonomy

Layer 4: 可复现工程
YAML config + JSON Schema + git/config/data hash + tests + reports
```

### 2.1 自研架构线

- Dense MHA：作为基线，包含 RoPE、RMSNorm、SwiGLU、causal attention、KV Cache、增量 decode。
- MoE Top-1：加入 router、Top-1 expert dispatch、capacity、overflow/dropped token、auxiliary load-balancing loss。
- GQA：Q 头保持 4 个，K/V 头减少到 `num_kv_heads`，通过 head repeat 供多个 Q head 使用。
- 简化 MLA：先把 K/V 联合压缩到 latent，再通过上投影重建 K/V；cache 保存 latent，而不是完整 K/V。

### 2.2 数据和训练线

- 用 OWT sample 训练 BPE tokenizer，并构建带 source/tokenizer/hash 的 token cache。
- 构造 D1、D1.1、D2 工具调用数据，覆盖单工具、多工具、参数错误、执行失败、多轮依赖和错误恢复。
- 用 Mock Executor 验证工具调用是否真的可执行。
- 用八级失败分类把“模型失败”拆开：解析、schema、工具名、参数、计划、执行、结果 grounding、最终答案。
- 完成 Dense/MoE SFT MVP 和小规模 GRPO MVP。

### 2.3 推理和评测线

- 复用共享 Hugging Face backend，支持 Transformers 加载、tokenizer、chat template 和 greedy generation。
- 接入 vLLM，在 WSL2 环境完成 smoke，并做 20 组合的固定 benchmark。
- 评测既有工具调用 reward，也有真实 OWT 上的纯语言模型 per-token loss/PPL。

---

## 3. 最应该熟悉的数字

不要背所有数字，重点记下面这些：

| 内容 | 数字 | 如何解释 |
|---|---:|---|
| 自研架构 | 4 种 | Dense MHA、MoE Top-1、GQA、简化 MLA |
| N13 架构实验 | 3-way | Dense MHA / GQA / simplified MLA |
| N13 训练规模 | 约 2.10M params、5000 steps | 小规模、可重复架构验证，不是大模型结论 |
| N13 val_min | 7.058 / 7.106 / 7.122 | 同数据、同 seed、同超参下的三方结果 |
| N13 KV cache | GQA/MLA 256 vs MHA 1024 bytes | 理论 cache footprint 为 1/4，不等于 latency 也提升 4 倍 |
| OWT 真实评测 | 5 模型、约 277MB held-out | 纯 LM/per-token loss/PPL |
| P5-04 backend | 5 模型 × 2 backend × 2 batch × 90 samples | 共 20 个组合 |
| P5-04 generation | 0 generation failure | 固定 benchmark 的工程稳定性 |
| 数据集 | D1 126、D1.1 约 1500、D2 5000 | 工具调用和多轮数据 |
| 测试 | 项目级约 421 tests | 另有 E selftest 97 项、N13 13 项模型测试 |

注意：仓库统计数字会随着后续提交变化。面试时优先讲实验配置和结果，不要把“当前文件数/commit 数”当成核心技术成果。

---

## 4. 架构理解：Dense MHA

### Q：Dense baseline 的一个 Transformer block 是什么？

回答结构：

1. 输入先经过 attention RMSNorm。
2. Q/K/V 通过线性层生成，每个 head 有自己的 K/V。
3. 对 Q/K 应用 RoPE。
4. 用 causal mask 的 scaled dot-product attention，保证当前位置不能看未来。
5. 输出经过 projection 后 residual add。
6. 再经过 FFN RMSNorm 和 SwiGLU，再 residual add。

项目中的基本形态：

```text
x
 → RMSNorm
 → QKV projection
 → reshape to [B, heads, L, head_dim]
 → RoPE
 → causal SDPA
 → output projection
 → residual
 → RMSNorm + SwiGLU
 → residual
```

### Q：为什么 Dense MHA 要作为 baseline？

因为后续 MoE、GQA 和 MLA 都需要一个稳定参照。如果 baseline 自己的 loss、cache、checkpoint 或增量 decode 不可靠，后面的架构差异就无法解释。

### Q：如何验证 KV Cache 是对的？

用两条路径计算同一个序列：

- 一次性 full forward；
- 先输入 prefix，再逐步输入新 token，复用 cache。

比较最后位置 logits 或预测 token。项目对 GQA/MLA 测试了这一点；由于 SDPA 在不同输入形状下可能选择不同 kernel，测试同时看近似 logits 和 argmax/top-k 一致性。

---

## 5. 架构理解：MoE Top-1

### Q：MoE 和 Dense FFN 的区别？

Dense FFN 对每个 token 使用同一组 FFN 参数；MoE 有多个 expert，每个 token 由 router 选择一个或多个 expert。Top-1 表示每个 token 只进入一个 expert。

```text
x → router → expert index → capacity check → expert FFN → combine
```

MoE 的典型优势是：

- 总参数可以增加；
- 每个 token 激活的参数较少；
- 可以用稀疏激活获得容量。

代价是：

- routing 和 dispatch 有额外开销；
- expert 负载可能不均衡；
- capacity 不够时会 drop token；
- 不能只看 total params 比较公平性。

### Q：为什么做 same-total 和 same-active 两种比较？

因为如果 Dense 和 MoE 的 total params 不同，结果可能只是模型容量差异；如果 active params 不同，单 token 的计算量又不同。N2 因此定义：

- Protocol A：尽量相同 total params；
- Protocol B：尽量相同 active params。

这比只报告“MoE loss 更低”更严谨。

### Q：auxiliary loss 的作用是什么？

它鼓励 router 把 token 更均匀地分配给 experts，避免某几个 expert 过载、其他 expert 几乎不训练。它是 routing regularizer，不是主语言模型 loss。

### Q：项目是否证明了 MoE 一定更好？

没有。项目证明的是：MoE Top-1 的 router、capacity、auxiliary loss、训练、checkpoint 和统计链路可以工作，并建立了更公平的比较协议。小规模、短训练和单 seed 不能证明大模型场景的普适收益。

---

## 6. 架构理解：GQA

### Q：GQA 如何节省 KV Cache？

MHA 中 Q/K/V 都有 `n_heads` 个 head。GQA 保持多个 Q heads，但让多个 Q heads 共享一个 K/V head。

例如项目的 N13：

```text
n_heads = 4
num_kv_heads = 1
q_per_kv = 4
```

K/V cache 的 head 数从 4 降到 1，因此理论 cache 从：

```text
2 × 4 × head_dim
```

变成：

```text
2 × 1 × head_dim
```

### Q：为什么代码里要 repeat_interleave？

SDPA 接收的 Q、K、V head 维度需要匹配。项目先缓存较少的 K/V heads，然后在 attention 计算前将每个 KV head 重复给对应的 Q heads。这样 cache 仍然较小，计算接口也简单。

### Q：`num_kv_heads=1` 还是标准 GQA 吗？

严格说它是 GQA 的极端情况，也可以称为 MQA。标准 GQA 通常会使用多个 KV heads，例如 4 个 Q heads 配 2 个 KV heads。项目 N13 选择 1 是为了让 cache 差异清楚、实现和实验保持最小。

### Q：N13 是否证明 GQA 比 MHA 好？

没有。N13 只在一个小规模、单 seed、单 `num_kv_heads` 设置下显示：GQA 用更少参数和更小理论 cache 达到了接近 Dense MHA 的 validation loss。没有做多 seed、latency benchmark 或多分组 sweep，因此不能宣传为普遍性能收益。

---

## 7. 架构理解：简化 MLA

### Q：项目里的 MLA 和 DeepSeek-V2 MLA 一样吗？

不一样。项目实现的是用于小模型实验的 simplified MLA，保留“压缩 KV 表示、减少 cache footprint”这个核心想法，但省略了完整 MLA 的一些设计，尤其是 decoupled RoPE 和 Q 的完整压缩路径。

项目实现：

```text
c_kv = W_DKV(x)       # d_model → latent_dim
K = W_UK(c_kv)        # latent_dim → d_model
V = W_UV(c_kv)        # latent_dim → d_model
Q = W_Q(x)            # Q 直接投影
```

### Q：简化 MLA 的 trade-off 是什么？

优势：

- cache 存 latent，理论 footprint 小；
- 可以显式观察压缩比；
- 结构容易在原生 PyTorch 中验证。

代价：

- 每次需要从 latent 重建 K/V；
- 重建增加 FLOPs；
- 简化 RoPE 路径不等同于完整 MLA；
- latent_dim 太小可能损失信息。

因此“cache 更小”不能直接等价为“decode 更快”。必须实际测量 prefill、decode、kernel 和显存。

### Q：为什么测试 MLA cache consistency 时没有要求 bitwise equal？

full forward 和 incremental decode 可能触发不同的 SDPA kernel 或不同 reduction order，float32 logits 存在数值差异。项目测试用合理的绝对误差，同时检查 argmax 和 top-k 是否一致，验证行为层面的一致性。

---

## 8. 训练和数据问题

### Q：为什么要做 tokenizer 和 token cache？

如果每次训练都从原始文本重新 tokenize：

- 运行时间更长；
- tokenizer 版本变化会导致数据变化；
- 很难判断实验差异来自模型还是数据。

token cache 将 source、tokenizer、token count、dtype 和 hash 固定下来，使不同架构确实读取同一批 token。

### Q：为什么 OWT validation 要记录 source hash？

因为“同一个文件名”不代表内容相同。source SHA256 可以确认评测用的文本版本没有被替换；token file SHA256 可以确认编码后的 token 没有变化。

### Q：D1/D1.1/D2 有什么区别？

- D1：模板化、可控的工具调用数据。
- D1.1：引入 LLM 生成的扩展数据。
- D2：更完整的多轮数据，包含多工具依赖、错误恢复、用户修改约束和 held-out split。

### Q：为什么需要 Mock Executor？

因为只检查 JSON 格式不代表工具调用真的正确。Mock Executor 能验证工具名、参数、调用顺序、依赖、执行结果和最终回答是否使用了真实工具结果。

---

## 9. 评测问题

### Q：八级 failure taxonomy 解决什么问题？

只报告“任务成功率 0.2”无法知道模型失败在哪里。八级分类将 failure 分解成：

```text
parse
→ schema
→ tool name
→ arguments
→ call plan
→ execution
→ result grounded
→ final answer
```

这样可以判断是生成格式、工具选择、参数、执行还是最终回答的问题。

### Q：为什么 P5-04 的 reward_binary 是 0？

固定 tool-calling benchmark 中，公开 instruction-tuned 模型没有 gold answer 提示，最终严格 binary success 全为 0。这个结果需要诚实保留，不能把它改成“模型没有任何能力”。因此同时看 layered reward：它能反映部分步骤是否正确，数值大约在 0.36–0.42。

### Q：为什么 PPL 不能直接比较 SmolLM2 和 Qwen2.5？

因为两个模型的 tokenizer 不同，vocabulary size 差别很大。per-token loss 的 token 粒度不同，同一个文本可能被切成不同数量的 token，单个 token 承载的信息量也不同。

项目因此同时保存：

- mean loss per token；
- perplexity；
- loss per evaluated byte。

但 per-byte 也不是完全消除 tokenizer 影响的万能指标，结论必须带 caveat。

### Q：P5-04 的后端对比测了什么？

在相同模型、相同样本、相同 batch 设定下比较：

- per-sample latency
- throughput
- reward_binary
- reward_layered
- parse success
- generation failure
- vLLM 相对 Transformers 的 delta

结果中 vLLM batch=4 的 throughput 进入 Top-5，最快组合是 Qwen2.5-0.5B vLLM batch=4，约 8.18 samples/s、122.3 ms/sample。这里应同时关注实验的样本范围、硬件、batch 设置和指标定义，不能脱离上下文引用单个数字。

---

## 10. 工程和可复现性问题

### Q：为什么使用 JSON Schema？

Schema 解决的是结构稳定性：字段是否存在、类型是否正确、枚举值是否合法。它不能替代语义校验，所以项目还需要：

- evaluator 规则；
- manifest/hash；
- 数据执行验证；
- 结果一致性测试；
- 人工检查和 caveat。

### Q：metadata 里为什么要绑定 git commit 和 config hash？

同一个脚本在不同 commit、不同 config 下可能产生完全不同结果。把 git commit、config SHA256、tokenizer/cache SHA256 和 seed 放进结果，才能在之后追溯“这个数字是怎么来的”。

### Q：项目遇到过哪些工程坑？

至少熟悉这几个：

1. **Windows/WSL symlink 行为不一致**：使用 `mklink /D` 固定数据契约路径，并记录 source hash。
2. **metadata hash 自包含问题**：`metadata_sha256` 不能把自身字段算进去，最终固定为对去掉该字段的 JSON 计算 SHA256。
3. **P5-04 reward 全 0**：manifest 样本没有从目录 glob 重新读取，改为把 `samples_by_id` 直接传到 reward 计算。
4. **batch generation failure**：加入 batch-level 失败记录和 per-sample retry，避免静默丢样本。
5. **OWT 评测 NaN**：逐 window 检查 finite，显式关闭不必要的 cache，删除无效结果后重跑。
6. **GQA/MLA 结果 metadata 仍写 Dense**：training result builder 曾硬编码 architecture，改为从 config 读取并扩展 schema enum。
7. **文档和测试数量 stale**：加入跨文档一致性检查，不再把跳过的测试当成通过证据。

---

## 11. 如何回答“项目最难的地方”

建议不要回答“我写了很多代码”。选择一个具体问题，用下面结构：

```text
背景：为什么这个问题重要
问题：原来的实现哪里不可靠
定位：通过什么日志/测试/对比找到了根因
方案：做了什么结构性修改
验证：用什么测试和实际结果确认修复
限制：这个方案还有什么边界
```

### 示例：P5-04 reward 全 0

> P5-04 初版所有 layered reward 都是 0，我没有直接认为模型完全失败，而是检查 evaluator 输入。发现 benchmark 使用 manifest 选择样本，但 reward 函数仍然从 manifest 父目录 glob JSON，因此拿到的是空样本集合。最终把 manifest loader 生成的 `samples_by_id` 显式传进每个组合的 reward 计算，并增加 wiring/结果一致性测试。重跑后 reward_layered 回到 0.36–0.42，说明问题在数据传递链路而不是模型输出本身。

### 示例：N13 MLA 设计取舍

> 我没有一开始实现完整 DeepSeek-V2 MLA，而是先实现一个简化版本，保留 KV latent compression 和 latent cache，省略 decoupled RoPE 和完整 Q compression。这样可以先验证 cache shape、增量 decode、训练 loss 和 schema。代价是每步需要重建 K/V，因此 cache footprint 减小不等于 decode latency 一定改善。这个版本适合架构教育实验，不应包装成完整 MLA 实现。

---

## 12. “项目有什么不足”应该怎么答

推荐主动承认以下边界：

1. 自研模型只有百万级参数，不能直接推断到 7B/70B。
2. N13 只有一个 seed、一个 `num_kv_heads`、一个 `latent_dim`。
3. N13 没有真实 prefill/decode latency benchmark。
4. MLA 是简化版，不是完整 DeepSeek-V2 MLA。
5. 公开模型 PPL 跨 tokenizer 不可直接比较。
6. 工具调用的 reward_binary 不是高分结果，说明 benchmark 和模型之间还有能力/提示/任务设计问题。
7. 自研架构没有完整接入 Hugging Face/vLLM 生产接口。
8. 统一 evaluator 主要是确定性规则，复杂开放任务的 LLM judge 尚未系统化。

承认这些限制不会削弱项目，反而能说明你知道实验结论的边界。

---

## 13. 不同岗位的讲法

### 偏模型/算法岗位

重点顺序：

1. Dense baseline 正确性和训练曲线；
2. MoE routing、capacity、auxiliary loss、active params；
3. GQA KV head sharing；
4. simplified MLA latent compression；
5. fair comparison 和 limitations。

### 偏 LLM 应用/后训练岗位

重点顺序：

1. D1/D1.1/D2 工具调用数据管线；
2. Mock Executor 和数据验证；
3. 八级 failure taxonomy；
4. SFT + GRPO；
5. evaluator、reward 和 held-out evaluation。

### 偏推理/平台工程岗位

重点顺序：

1. Transformers/vLLM backend；
2. manifest-driven benchmark；
3. latency/throughput/reward 四维结果；
4. batch retry 和 failure tracking；
5. metadata/hash/schema/reproducibility。

### 偏研究工程综合岗位

用完整闭环：

```text
架构 → 数据 → 训练 → 推理 → 评测 → 失败分析 → 重新设计
```

---

## 14. 独立复习顺序

这份准备稿本身包含项目介绍、关键数字、技术问答、难点、限制和复习清单，不要求打开仓库中的其他文档。复习时建议按以下顺序：

1. 先背熟一句话介绍和 30 秒版本。
2. 再理解 Dense、MoE、GQA、简化 MLA 的数据流和 trade-off。
3. 然后掌握 OWT、tokenizer、token cache、SFT/GRPO、评测器和后端对比的关系。
4. 最后准备一个真实工程问题、一个负结果和一个主动承认的局限性。

---

## 15. 最后检查清单

面试前确保自己能不看稿回答：

- [ ] 项目不是“大模型训练”，而是小规模可复现实验闭环。
- [ ] Dense baseline 每个模块是什么。
- [ ] MoE 为什么要区分 total params 和 active params。
- [ ] GQA 如何减少 KV cache。
- [ ] N13 的 MLA 为什么是 simplified，代价是什么。
- [ ] 为什么 cache 小不等于 latency 快。
- [ ] 为什么跨 tokenizer PPL 不能直接比较。
- [ ] P5-04 reward_binary 为什么全 0。
- [ ] P5-04 reward 全 0 的根因和修复过程。
- [ ] token cache、manifest、hash、schema 分别解决什么问题。
- [ ] 至少讲清楚一个真实 bug 的定位和验证过程。
- [ ] 能主动说出项目局限，而不夸大结论。
