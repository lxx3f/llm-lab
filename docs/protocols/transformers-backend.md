# P5-02 Transformers 推理后端协议

> 状态：阶段交付（2026-08-28）。
>
> 本协议固定 `scripts/eval_transformers.py` 的输入 / 输出契约、模型加载策略、chat template 行为、与 `scripts/reward_offline.py` 的衔接方式以及与自研 `eval_sft_tool.py` 的边界。

## 1. 目标与定位

- 提供公开 instruction-tuned 模型（Qwen2.5 / SmolLM2 / LLaMA 等）在 D2 多轮 held-out split 上的标准推理入口；
- 输出 schema 与 `scripts/eval_sft_tool.py` 严格一致（`{"summary", "rows": [...]}`），以便 `scripts/reward_offline.py --transcripts` 直接消费，无需额外适配；
- 与自研 SFT 模型走同一 reward 评估管线（reward_signal schema + P1-05 八级分类器），让公开模型与自研 ckpt 在 D2 dev 上可横向对比。

不包含的内容（明确边界）：

- 不训练 LoRA / 全参数微调；本协议只覆盖推理评测（eval-time only）；
- 不替代 `eval_sft_tool.py`；自研 ckpt（含 Dense / MoE 权重）继续走原有路径；
- 不接入 vLLM serving；vLLM 走 `scripts/run_n2_benchmark.py` 等独立路径（见 P5-03 后续工作）。

## 2. CLI 接口

```text
.venv/python.exe scripts/eval_transformers.py \\
    --model <HF model id or local path> \\
    --samples-dir datasets/tool-calling-d2/dev \\
    --output artifacts/<model>-eval-d2dev.json \\
    [--max-new-tokens 256] [--limit N] \\
    [--device auto|cuda|cpu] [--dtype bf16|fp16|fp32] \\
    [--revision main] [--trust-remote-code]
```

| Flag | 默认 | 含义 |
|---|---|---|
| `--model` | 必填 | Hugging Face model id（如 `Qwen/Qwen2.5-0.5B-Instruct`）或本地路径 |
| `--samples-dir` | `datasets/tool-calling-d2/dev` | D2 多轮样本目录；只接受 `*.json` 文件 |
| `--output` | 必填 | 写出 `{summary, rows}` JSON；父目录会自动创建 |
| `--max-new-tokens` | `256` | Greedy 生成 budget；与 `eval_sft_tool.py` 一致 |
| `--limit` | `0`（=全部）| 评测样本上限（用于 smoke test） |
| `--device` | `auto` | `auto` 优先 cuda，回退 cpu |
| `--dtype` | `bf16` | CPU 强制 fp32；CUDA 接受 bf16/fp16/fp32 |
| `--revision` | `main` | Hugging Face 模型 revision / tag / commit |
| `--trust-remote-code` | off | 透传给 `AutoModel/AutoTokenizer.from_pretrained` |

## 3. 输入数据契约

读 `datasets/tool-calling-d2/<split>/d2-{split}-NNNN.json`，每行至少含：

- `messages`：按时间排序的多轮消息数组，至少含 `system` + `user` + 终态 `assistant`；
- `tools`：函数描述数组（OpenAI tool schema 形式）；chat template 通过 `tokenizer.apply_chat_template(..., tools=sample['tools'])` 注入；
- `expected_tool_calls`：可选；不存在时表示该样本不期望工具调用（`tool_not_available` task_type）；
- `expected_answer`：可选；与 transcript 终态 assistant content 字节相同（见 `docs/protocols/d2-multi-turn.md` §3）。

`--samples-dir` 默认指向当前扩样版 D2 dev（750 样本）；其他 split（train / test）可通过修改 flag 切换。P5-02 的已交付横向评测使用该 dev split 中的固定 90 样本子集；MVP 600 样本版历史也使用过 90 样本。

## 4. Chat template 与 prompt 渲染

`_apply_chat_template()` 按以下优先级渲染：

1. 优先调用 `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=sample['tools'])`；
2. tokenizer 无 `chat_template` 或 `apply_chat_template()` 抛异常时回退到 `<role>: <content>` 拼接（仅 user/assistant/system）。

`_greedy_generate()` 在生成时：

- `do_sample=False`、`num_beams=1`；
- `pad_token_id` 优先 `tokenizer.pad_token_id`，缺失则回退 `tokenizer.eos_token_id`；
- 在 CUDA bf16 模式下需要 `pad_token_id` 已被设置，否则 padding 出错。

## 5. 输出契约

`args.output` 写出的 JSON schema：

```json
{
  "summary": {
    "checkpoint": "<model id>",
    "backend": "transformers",
    "revision": "<rev>",
    "device": "cuda|cpu",
    "dtype": "torch.bfloat16|...",
    "transformers_version": "<x.y.z>",
    "torch_version": "<x.y.z>",
    "samples_dir": "<path>",
    "total": <int>,
    "parse_success_count": <int>,
    "first_failure_distribution": {"<failure>": <count>, ...},
    "no_failure": <int>,
    "parse_success_rate": <float 0..1>
  },
  "rows": [
    {
      "sample_id": "d2-dev-NNNN",
      "task_type": "<task_type>",
      "user_turn": "<prompt preview, <=200 chars>",
      "generated": "<full generated text, NOT truncated>",
      "generated_preview": "<first 200 chars>",
      "extracted_calls": [...],
      "layers": {<P1-05 8 layers>},
      "first_failure": "<failure or None>"
    },
    ...
  ]
}
```

要点：

- `generated` 字段保留完整生成文本（不截断），便于离线复现 `extract_tool_calls`；
- `extracted_calls` 总是 `list`（即便空）；`None` 会被规范化为 `[]`，避免 no-tool 样本被错误判为 `parse_success=False`；
- `rows[*].layers` 与 `scripts/eval_sft_tool.py` 输出一致，直接喂给 `scripts/reward_offline.py` 的 `_to_transcript` → `classify()` → `_dominant_reward()` 流水线。

## 6. reward_offline 衔接

`scripts/reward_offline.py --transcripts <artifacts/...-eval-d2dev.json>` 已有的 `load_transcripts()` 直接消费 `data["rows"]`（见 `scripts/reward_offline.py` L188 附近），因此无需修改。`compute_reward()` 走原有 8 层分类 + `_dominant_reward()` 映射：

- `parse_success`：JSON call 解析失败（无 JSON 块）；
- `argument_correct`：tool name + arguments 通过但 execution / grounding / answer 失败；
- `final_answer_correct`：前 7 层通过但 final answer 与 `expected_answer` 不一致；
- `execution_correct`：所有适用层通过（`first_failure=None`）。

`reward_binary = (first_failure is None)`、`reward_layered = sum(layer is True for layer in 8 layers) / 8`，两者均与 `eval_sft_tool.py` 完全一致。

## 7. 已知边界

- **target-answer 泄漏修复（2026-08-28 下午）**：D2 样本 `messages` 末尾的 assistant content 消息是 `expected_answer`；必须在 `_apply_chat_template()` 中先调用 `_strip_terminal_assistant()` 删除它，再传给 `apply_chat_template(..., add_generation_prompt=True)`，否则模型在 prompt 里看到目标答案会直接复述。前向 tool_calls 与 tool 结果消息保留作为上下文。回归测试见 `tests/test_transformers_backend.py::GoldAnswerLeakageTests`；
- **chat template 适配**：不同模型对 `tools` 参数的处理不一致；当前实测的 5 个模型（Qwen2.5-0.5B/1.5B/3B-Instruct、SmolLM2-360M/1.7B-Instruct）均接受 `tools` 参数并按自家格式注入工具描述；如新增 LLaMA / Gemma 等需单独验证 chat template 输出；
- **D2 工具名 vs 公开模型训练分布**：D2 使用 `d1_calculate / d1_get_weather / d1_web_search` 等工具名，公开 instruction-tuned 模型通常未在 D2 数据集上微调，因此 `tool_name_correct` 大量失败是预期行为。`reward_layered > 0.33` 表明模型能进入后续层；
- **GPU 显存**：`Qwen2.5-3B-Instruct` bf16 约 6 GB，CUDA 12 GB 单卡可运行；4 GB 以上的模型（如 Phi-3.5-mini）可能 OOM，需要 `dtype=fp32` 不可行（精度 + 显存均不可）；
- **下载通道**：默认走 `https://hf-mirror.com`（通过 `HF_ENDPOINT` 环境变量），避免官方 HF 限速；离线场景下需要把模型预先缓存到 `artifacts/huggingface/`。

## 8. 可复现实验记录（2026-08-28 凌晨，2026-08-28 下午修正）

> **2026-08-28 下午修正记录**：原版 5 模型评测存在 **target-answer 泄漏** bug —— `_apply_chat_template()` 把 D2 样本的整条 `messages`（含 gold terminal assistant `content` = `expected_answer`）透传给 `apply_chat_template(..., add_generation_prompt=True)`，模型在 prompt 中看到目标答案后才生成，导致 `SmolLM2-360M reward_binary=0.0111` 的"首个 D2 dev 非退化 reward"实为 prompt 复述伪结果。修复后重跑 5 模型：所有 5 个模型 `reward_binary=0`（诚实负结果），`reward_layered` 仍在 0.33–0.43，明显高于自研 5 ckpt 的 0.0；`reward_layered > 0.05` 仍远超达成。

### 8.1 修复后结果（5 × 90 = 450 reward_signal；terminal assistant 已 strip）

| Model | reward_binary | reward_layered | reward_type 分布 | first_failure 分布 |
|---|---|---|---|---|
| SmolLM2-360M-Instruct | 0.0000 | 0.4236 | final_answer_correct × 15, argument_correct × 75 | tool_name_correct × 75, final_answer_correct × 15 |
| Qwen2.5-0.5B-Instruct | 0.0000 | 0.3634 | argument_correct × 85, final_answer_correct × 5 | tool_name_correct × 71, schema_valid × 8, argument_value_correct × 6, final_answer_correct × 5 |
| Qwen2.5-1.5B-Instruct | 0.0000 | 0.3690 | argument_correct × 84, final_answer_correct × 6 | tool_name_correct × 79, schema_valid × 5, final_answer_correct × 6 |
| Qwen2.5-3B-Instruct | 0.0000 | 0.3333 | argument_correct × 88, final_answer_correct × 2 | tool_name_correct × 67, schema_valid × 19, call_plan_matches × 1, argument_value_correct × 1, final_answer_correct × 2 |
| SmolLM2-1.7B-Instruct | 0.0000 | 0.4236 | final_answer_correct × 15, argument_correct × 75 | tool_name_correct × 75, final_answer_correct × 15 |

### 8.2 解读

- **`reward_binary` 全部为 0**：5 个模型没有任何一个完整过 P1-05 8 层分类器；这是诚实负结果。修复前 `SmolLM2-360M reward_binary=0.0111`（1 个 `execution_correct`）是因为 target-answer 泄漏，模型仅复述了 prompt 里出现的 expected_answer。
- **`reward_layered` 仍在 0.33–0.43**：5 模型全部显著高于自研 5 ckpt 的 0.0；说明 instruction-tuned 模型在多轮工具调用结构化生成上仍优于自研 SFT 模型。
- **`tool_name_correct` 仍是主要失败层**：D2 使用 `d1_*` 工具名，公开 instruction-tuned 模型未在 D2 数据集上微调，这是诚实预期的行为。
- **与自研 5 ckpt 横向对比**（详见 `docs/experiments/p2-evaluator/README.md` §7）：自研 5 ckpt 全部 90/90 `parse_success` 失败、`reward_layered=0`；公开模型即使不做 D2 微调，仍能在多层分类器上获得非退化 reward 分。

复现命令（任选模型）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
.venv/python.exe scripts/eval_transformers.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --samples-dir datasets/tool-calling-d2/dev \
  --output artifacts/qwen2.5-0.5b-instruct-eval-d2dev.json
.venv/python.exe scripts/reward_offline.py \
  --transcripts artifacts/qwen2.5-0.5b-instruct-eval-d2dev.json \
  --samples-dir datasets/tool-calling-d2/dev \
  --checkpoint Qwen/Qwen2.5-0.5B-Instruct \
  --transcript-kind model_generated \
  --output artifacts/qwen2.5-0.5b-instruct-eval-d2dev-reward.json
```