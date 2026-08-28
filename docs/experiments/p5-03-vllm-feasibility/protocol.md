# P5-03 vLLM 公开 instruction-tuned 模型后端可行性 — Protocol

## 目标

> 开展 P5-03 vLLM 公开 instruction-tuned 模型后端可行性工作：检查 WSL/Linux/Docker、
> vLLM、CUDA 和显存兼容性；完成最小 smoke 或记录明确阻塞，严格不做自研模型完整
> vLLM 适配。

## 实验范围（明确）

**包含**:
- 检查 host 环境（OS / WSL / Docker / CUDA / vLLM）的兼容性
- 在**公开 instruction-tuned 模型**（HuggingFaceTB SmolLM2-360M-Instruct +
  Alibaba Qwen2.5-0.5B-Instruct 等本地 snapshot）上跑最小 smoke
- 记录所有 workarounds 和环境配置
- 验证 vLLM 在 WSL2 + sm_120 GPU 上的可行性

**不包含**:
- ❌ 自研 Dense Transformer / MoE 等 vLLM 适配（架构项目；明确 out-of-scope）
- ❌ vLLM 服务化部署（FastAPI / Triton / 生产集群）
- ❌ vLLM 与本项目 P5-02 Transformers backend 集成
- ❌ LoRA / quantization / speculative decoding 等高级特性
- ❌ vLLM 性能基准测试（吞吐、延迟 p50/p99 等）

## 环境检查清单

- [ ] Host OS（Windows / Linux / WSL）
- [ ] WSL distribution + kernel 版本
- [ ] Docker daemon（运行状态 + GPU passthrough）
- [ ] NVIDIA driver + CUDA toolkit
- [ ] GPU compute capability（sm_XX）
- [ ] vLLM 版本（pip install / source build）
- [ ] PyTorch 版本 + CUDA capabilities 支持
- [ ] 本地 HF model snapshot 路径

## Smoke 任务

最小 smoke：load 一个公开 instruction-tuned model → generate 一个 prompt → 记录
throughput 和 peak GPU memory。**不**做 quality evaluation / benchmark。

## 验证契约

| 检查 | 通过条件 |
|---|---|
| vLLM installed | `import vllm` returns valid version |
| vLLM imports LLM/SamplingParams | no exception |
| Model load | HF model loaded into vLLM in < 300s |
| Generation runs | at least 1 prompt produces output |
| GPU peak memory | < GPU VRAM（不 OOM）|

## 不测量的项（明确不做）

- ❌ Throughput vs batch size（benchmark）
- ❌ Latency p50/p99（benchmark）
- ❌ Multi-GPU scaling（out of scope）
- ❌ Concurrent request handling（benchmark）
- ❌ vLLM 与项目内 evaluator 集成（out of scope）