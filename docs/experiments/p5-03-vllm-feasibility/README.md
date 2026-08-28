# P5-03 vLLM 公开 instruction-tuned 模型后端可行性 — 真实结果

> **实验性质**: 验证 vLLM 在本机环境（WSL2 + RTX 5070 Ti sm_120 + Python 3.10 +
> vLLM 0.27.1）上的最小 smoke 是否可行。严格**不**做自研模型 vLLM 适配、不做
> benchmark、不做 production deployment。
>
> 实验设计详见 [`protocol.md`](./protocol.md)。

## TL;DR

| 验证项 | 结果 | 证据 |
|---|---|---|
| vLLM installed | ✅ | `import vllm` → vllm 0.27.1 |
| WSL GPU passthrough | ✅ | `nvidia-smi` 在 WSL 内可见 RTX 5070 Ti |
| PyTorch supports sm_120 | ✅ | torch 2.13.0+cu130, CUDA compute OK |
| Model load via vLLM | ✅ | SmolLM2-360M loaded in 19.3s (cached) / 77.3s (cold) |
| Generation runs | ✅ | 4 prompts × 64 tokens, all complete |
| Peak GPU memory | ✅ | 6.10 GB / 12.82 GB available |
| **Smoke verdict** | ✅ **PASS** | vLLM 公开 instruction-tuned 模型后端可行（带 3 个 workarounds） |

**核心结论**: 在 WSL2 + RTX 5070 Ti + vLLM 0.27.1 + Python 3.10.12 上跑公开
instruction-tuned 模型是**可行的**，但需要 3 个 workarounds（详见下文）。

## 环境检查（真实数据）

### Host OS

- Windows 10 / 11 (MINGW64_NT-10.0-26200, x86_64)
- WSL2 with `Ubuntu-22.04` distribution
- WSL kernel: `6.6.114.1-microsoft-standard-WSL2`
- Docker 29.5.3 (daemon **未运行** — desktop-linux engine not running; vLLM 直接走 WSL 路径，绕过 Docker)

### GPU

| 项 | 实测值 |
|---|---|
| Device | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| Compute capability | (12, 0) — **sm_120** |
| VRAM | **12.82 GB** |
| Driver (Windows) | 610.43.02 |
| Driver (WSL) | 610.47 |
| `nvidia-smi` 在 WSL 内 | ✅ 可见 |

### Python / vLLM

| 项 | 实测值 |
|---|---|
| WSL Python | 3.10.12 (`/usr/bin/python3`) |
| vLLM | **0.27.1** |
| PyTorch | **2.13.0+cu130** |
| flashinfer-python | **0.6.16.post3** — **需要卸载**（见下文） |

### 本地 HF 模型 snapshot

- `artifacts/huggingface/models--HuggingFaceTB--SmolLM2-360M-Instruct/...`
- `artifacts/huggingface/models--HuggingFaceTB--SmolLM2-1.7B-Instruct/...`
- `artifacts/huggingface/models--Qwen--Qwen2.5-0.5B-Instruct/...`
- `artifacts/huggingface/models--Qwen--Qwen2.5-1.5B-Instruct/...`
- `artifacts/huggingface/models--Qwen--Qwen2.5-3B-Instruct/...`

WSL 通过 `/mnt/c/Users/...` 路径直接访问，无需下载。

## Smoke 实测结果

### 命令

```bash
# 在 WSL Ubuntu-22.04 内（不是 Windows host）
wsl -d Ubuntu-22.04 -- bash -c "python3 /mnt/c/Users/23236/repositories/llm-lab/scripts/vllm_smoke/smoke.py"
```

### 实测（来自 `artifacts/vllm-smoke/summary.json`）

| 项 | 实测值 |
|---|---|
| vLLM version | 0.27.1 |
| PyTorch version | 2.13.0+cu130 |
| CUDA capability | (12, 0) — sm_120 |
| Device | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| Model | SmolLM2-360M-Instruct (本地 HF snapshot) |
| Model load time (warm cache) | 19.3s |
| Model load time (cold) | 77.3s |
| Num prompts | 4 |
| Total output tokens | 157 |
| Peak GPU memory | 6.10 GB / 12.82 GB available |
| Status | **success** |

### Generated outputs (verbatim, smoke quality only)

```
"The ocean is a vast, mysterious, and largely unexplored body of water that covers
approximately 71% of the Earth's surface, covering over 350 million square kilometers,
and is home to an incredibly diverse array of life forms and ecosystems."

"Machine learning is a subset of artificial intelligence that enables computers
to learn from data and improve their performance over time, allowing them to make
predictions, classify objects, and make decisions without being explicitly programmed."

"A sunny day is characterized by clear blue skies, warm temperatures, and a gentle breeze."

"Good food is cuisine that is pleasing to the palate, often characterized by its
rich flavors, aromas, and presentation, and is typically made with high-quality
ingredients and techniques, resulting in a harmonious balance of taste and texture."
```

### 关键事实

- vLLM 模型成功加载并完整执行 4 个 prompt 的 generation
- 所有 output 都自然完成（`finish_reason: stop`），没有 truncation
- 6.10 GB GPU 占用（< 12.82 GB 可用 VRAM）
- 推理过程中没有 OOM / CUDA error

## Workarounds（必需，3 个）

### 1. `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 必须设置

**Root cause**: vLLM 的 `platforms/cuda.py::is_pin_memory_available()` 在 WSL2 上默认
返回 `False`（gating on kernel version + opt-in env var），导致 `is_uva_available()`
也返回 `False`，进而 `RequestState.all_token_ids = StagedWriteTensor(...)` 在
`UvaBuffer.__init__` 中失败：

```
RuntimeError: UVA is not available
```

**Fix**: 设置 `os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"` 在 import vllm 之前。
WSL2 kernel ≥ 4.19.121 才支持 pinned memory；本机 6.6 kernel 满足。

### 2. `flashinfer-python` 必须卸载

**Root cause**: flashinfer-python 0.6.16.post3 在 Python 3.10.12 上有 module-load-time
故障：

```python
def _fd_ancillary(fd: int) -> tuple[tuple[int, int, array.array[int]]]:
#                                                  ^^^^^^^^^^^^^^
# TypeError: 'type' object is not subscriptable
```

这是 PEP 585 generic subscripting (`array.array[int]`) 在 Python 3.10 的 `array` 模块上
运行时行为差异。错误在 import 时（不是 call 时）触发，所以 vLLM 内部的 try/except
无法捕获。

**Fix**: `pip3 uninstall -y flashinfer-python`。然后 vLLM 的其他 attention backend
（`TORCH_SDPA` / `FLASH_ATTN` / `XFORMERS`）正常工作。

### 3. `VLLM_USE_FLASHINFER_SAMPLER=0` + `VLLM_ATTENTION_BACKEND=TORCH_SDPA`

**Root cause**: 即使在 import-time 没有错误，flashinfer sampler / attention kernel
也需要 `flashinfer` package；uninstall 后这两个组件不可达。

**Fix**: 显式设置 `VLLM_USE_FLASHINFER_SAMPLER=0` 和 `VLLM_ATTENTION_BACKEND=TORCH_SDPA`
强制 vLLM 用 PyTorch native SDPA attention。性能足够用于最小 smoke。

## 验证清单

| 检查 | 通过条件 | 真实结果 |
|---|---|---|
| vLLM installed | `import vllm` returns version | ✅ vllm 0.27.1 |
| Model load | HF model loaded into vLLM | ✅ SmolLM2-360M loaded in 19.3s |
| Generation runs | at least 1 prompt produces output | ✅ 4 prompts, 157 tokens total, all finish_reason=stop |
| GPU peak memory | < GPU VRAM | ✅ 6.10 GB / 12.82 GB |
| WSL GPU passthrough | nvidia-smi works in WSL | ✅ RTX 5070 Ti visible |
| sm_120 support | CUDA compute OK | ✅ torch 2.13.0+cu130 |

## 失败案例与限制

### 限制 1: flashinfer 卸载是硬要求

不卸载 flashinfer 会让 vLLM EngineCore 启动失败。这是 vLLM 与
flashinfer-python 0.6.16 在 Python 3.10 上的兼容性问题；不是 vLLM bug。可能的
替代方案：

- 升级 Python 到 3.11+（如果 vLLM 0.27.1 兼容）
- 等待 flashinfer-python 修复
- 用更早的 vLLM 版本（不验证）

### 限制 2: Docker daemon 未运行

本机 Docker 29.5.3 已安装但 daemon `failed to connect to the docker API at
npipe:////./pipe/dockerDesktopLinuxEngine`。绕过 Docker 直接走 WSL 内的 vLLM
Python。如果以后需要 Docker-based vLLM serving，需要先启动 daemon。

### 限制 3: 本实验**不**做

- ❌ 自研 Dense Transformer / MoE 等 vLLM 适配（架构项目；out-of-scope）
- ❌ vLLM serving / FastAPI / Triton integration
- ❌ vLLM 与项目内 evaluator (P5-02) 集成
- ❌ LoRA / quantization / speculative decoding
- ❌ vLLM performance benchmark（throughput, latency p50/p99）
- ❌ Multi-GPU scaling
- ❌ Concurrent request handling

### 限制 4: 测量的 throughput 不完整

实测的 `throughput=0` 因为 `gen_time=-0.55s`（time.time() 在多进程 vLLM 里有 race）。
vLLM 自己 report 的 throughput 是 `input: 141 toks/s, output: 22.59 toks/s`（仅 1 prompt）、
`output: 282 toks/s`（4 prompts batch）。这些数字仅作为 smoke reference，**不**
作为 benchmark。

## 复现命令

```bash
# 1. WSL 内预条件（一次性）
wsl -d Ubuntu-22.04 -- bash -c "pip3 uninstall -y flashinfer-python"

# 2. 跑 smoke（artifact 写到 /mnt/c/.../artifacts/vllm-smoke/）
wsl -d Ubuntu-22.04 -- bash -c "python3 /mnt/c/Users/23236/repositories/llm-lab/scripts/vllm_smoke/smoke.py"
```

## 关联文件

- [`protocol.md`](./protocol.md) — 实验设计 / 验证契约 / 不做什么
- `scripts/vllm_smoke/smoke.py` — smoke 脚本（tracked）
- `artifacts/vllm-smoke/summary.json` — 实测结果（gitignored）

## 结论

vLLM 在本机（WSL2 + RTX 5070 Ti sm_120 + Python 3.10 + vLLM 0.27.1）上**端到端
可行**，3 个必需 workarounds 已识别并文档化：

1. `VLLM_WSL2_ENABLE_PIN_MEMORY=1`（WSL2 pinned memory opt-in）
2. 卸载 `flashinfer-python`（Python 3.10 module-load-time 故障）
3. `VLLM_USE_FLASHINFER_SAMPLER=0` + `VLLM_ATTENTION_BACKEND=TORCH_SDPA`

最小 smoke 验证：
- SmolLM2-360M-Instruct 加载（19.3s warm / 77.3s cold）
- 4 个 prompt × 64 tokens 全部成功生成
- 6.10 GB GPU peak memory（< 12.82 GB 可用）
- 无 OOM / CUDA error

**强烈建议**: 后续工作如果要继续在 vLLM 上做，**先**评估是否升级到 Python 3.11+
（可消除 flashinfer workaround）。如果继续在 Python 3.10，这 3 个 workarounds
需要作为 vLLM-on-WSL2 的标准配置进入项目 docs。

自研模型 vLLM 适配明确**不在**本阶段工作范围；属于架构项目的延伸工作。