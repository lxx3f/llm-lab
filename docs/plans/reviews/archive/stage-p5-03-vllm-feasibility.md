# Stage Review — P5-03 vLLM 公开 instruction-tuned 模型后端可行性

- **List item**: 队列 item #5 "P5-03 vLLM 公开 instruction-tuned 模型后端可行性"
- **HEAD**: 当前 main（round-11 P4 GRPO smoketest + P5-03 vLLM smoke）
- **Date**: 2026-08-29
- **Reviewer model**: Minimax M3 (per `docs/plans/review-process.md`)

## 目标

> 开展 P5-03 vLLM 公开 instruction-tuned 模型后端可行性工作：检查 WSL/Linux/Docker、
> vLLM、CUDA 和显存兼容性；完成最小 smoke 或记录明确阻塞，严格不做自研模型完整
> vLLM 适配。

## 交付物

| 路径 | 状态 | 说明 |
|---|---|---|
| `docs/experiments/p5-03-vllm-feasibility/protocol.md` | ✅ (tracked) | 实验设计 / 验证契约 / 不做什么 |
| `docs/experiments/p5-03-vllm-feasibility/README.md` | ✅ (tracked) | 真实结果 + 3 个必需 workarounds |
| `docs/plans/reviews/stage-p5-03-vllm-feasibility.md` | ✅ (tracked, 本文件) | 阶段评审 |
| `scripts/vllm_smoke/smoke.py` | ✅ (tracked) | smoke 脚本（WSL 内 Python 3.10）|
| `artifacts/vllm-smoke/summary.json` | ✅ (gitignored) | 实测结果 |

## 验证项 vs 真实证据（直接来自 `summary.json`）

| 验证项 | 通过条件 | 真实结果 |
|---|---|---|
| vLLM installed | `import vllm` returns version | ✅ vllm 0.27.1 |
| WSL GPU passthrough | nvidia-smi 在 WSL 内可见 | ✅ RTX 5070 Ti visible (driver 610.47) |
| PyTorch supports sm_120 | CUDA compute OK | ✅ torch 2.13.0+cu130 |
| Model load | HF model loaded into vLLM | ✅ SmolLM2-360M loaded in 19.3s warm / 77.3s cold |
| Generation runs | at least 1 prompt produces output | ✅ 4 prompts, 157 tokens total, all finish_reason=stop |
| Peak GPU memory | < GPU VRAM | ✅ 6.10 GB / 12.82 GB |

**Smoke verdict**: ✅ **PASS** — vLLM 公开 instruction-tuned 模型后端在
WSL2 + RTX 5070 Ti sm_120 + Python 3.10 + vLLM 0.27.1 上可行（带 3 个 workarounds）。

## 三个必需 workarounds

1. **`VLLM_WSL2_ENABLE_PIN_MEMORY=1`**: WSL2 默认 `is_pin_memory_available() = False`
   导致 `UvaBuffer.__init__` 失败（"UVA is not available"）。在 import vllm 之前设置。
2. **卸载 `flashinfer-python`**: 0.6.16.post3 在 Python 3.10 上 module-load-time 故障
   （`array.array[int]` PEP-585 语法运行时错误）。vLLM 内部 try/except 无法捕获。
3. **`VLLM_USE_FLASHINFER_SAMPLER=0` + `VLLM_ATTENTION_BACKEND=TORCH_SDPA`**: 强制
   vLLM 用 PyTorch native SDPA attention（flashinfer sampler 不可用）。

详见 `docs/experiments/p5-03-vllm-feasibility/README.md`。

## 严格不做（已明确）

- ❌ 自研 Dense Transformer / MoE 等 vLLM 适配
- ❌ vLLM serving / FastAPI / Triton integration
- ❌ vLLM 与项目内 evaluator (P5-02) 集成
- ❌ LoRA / quantization / speculative decoding
- ❌ vLLM performance benchmark（throughput, latency p50/p99）
- ❌ Multi-GPU scaling
- ❌ Concurrent request handling

## 失败案例与限制

### 限制 1: flashinfer 卸载是硬要求

不卸载 flashinfer 会让 vLLM EngineCore 启动失败。可能的长期方案：
- 升级 Python 到 3.11+（如果 vLLM 0.27.1 兼容）
- 等待 flashinfer-python 修复
- 用更早的 vLLM 版本（不验证）

### 限制 2: Docker daemon 未运行

本机 Docker 29.5.3 已安装但 daemon `failed to connect to the docker API at
npipe:////./pipe/dockerDesktopLinuxEngine`。绕过 Docker 直接走 WSL 内的 vLLM
Python。如果以后需要 Docker-based vLLM serving，需要先启动 daemon。

### 限制 3: throughput 数字不完整

vLLM 自己 report 的 throughput 是 `input: 141 toks/s, output: 22.59 toks/s`（1 prompt）、
`output: 282 toks/s`（4 prompts batch）。这些数字仅作为 smoke reference，**不**
作为 benchmark。

## 与其他阶段的关联

- **P5-02 Transformers backend** (HEAD `b4fd879`)：本实验**不**集成到 P5-02；只是
  独立的可行性测试。
- **P4 GRPO 小规模正确性实验** (HEAD `74d3432` + round-11 at HEAD `99646fa`)：
  本实验**不**用 vLLM；P4 GRPO 用 HuggingFace Transformers + 自研 _policy_update。
- **架构项目** (`architecture-lab/`)：vLLM 适配明确 out-of-scope。

## 复现命令

```bash
# 1. WSL 内预条件（一次性）
wsl -d Ubuntu-22.04 -- bash -c "pip3 uninstall -y flashinfer-python"

# 2. 跑 smoke（artifact 写到 /mnt/c/.../artifacts/vllm-smoke/）
wsl -d Ubuntu-22.04 -- bash -c "python3 /mnt/c/Users/23236/repositories/llm-lab/scripts/vllm_smoke/smoke.py"
```

## 结论

P5-03 vLLM 公开 instruction-tuned 模型后端可行性 smoke 通过。3 个必需 workarounds
已识别、文档化、并固化在 smoke 脚本的 `os.environ` 设置中。

**后续建议**（非本次任务）:
1. 评估升级 Python 到 3.11+ 以消除 flashinfer workaround
2. 启动 Docker daemon 并测试 Docker-based vLLM serving（如果需要 production）
3. 自研模型 vLLM 适配属于架构项目的独立工作（out-of-scope for P5-03）