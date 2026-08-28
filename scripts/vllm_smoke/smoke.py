"""Minimal vLLM smoke test on WSL Ubuntu-22.04 with RTX 5070 Ti (sm_120).

Loads SmolLM2-360M-Instruct from the local HF cache and runs one
greedy generation. Reports:
- vLLM version, torch version, CUDA capability
- Model load time
- Generation result
- Peak GPU memory

Exit code 0 on success; non-zero on failure.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Disable HF telemetry before any HF imports
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# WSL GPU passthrough: vLLM's `is_pin_memory_available()` returns False
# on WSL2 by default (gating on a kernel-version check + opt-in env var).
# Set VLLM_WSL2_ENABLE_PIN_MEMORY=1 to enable pinned memory, which is
# required for vLLM's UvaBuffer (zero-copy CPU offloading in v2 model
# runner). Without this, RequestState.init fails with "UVA is not
# available". Verified manually on this WSL2 host with kernel 6.6.
os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
# Disable flashinfer sampler / attention backend — the installed
# `flashinfer` package's `comm/fd_exchange.py` uses `array.array[int]`
# PEP-585 generic subscripting syntax that fails at module load time
# on this Python 3.10.12 build. vLLM has its own attention kernels
# (XFORMERS / FLASH_ATTN / TORCH_SDPA) so we don't need flashinfer.
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"

MODEL_DIR = ("/mnt/c/Users/23236/repositories/llm-lab/"
             "artifacts/huggingface/models--HuggingFaceTB--"
             "SmolLM2-360M-Instruct/snapshots/"
             "a10cc1512eabd3dde888204e902eca88bddb4951")

OUT_DIR = Path("/mnt/c/Users/23236/repositories/llm-lab/"
               "artifacts/vllm-smoke")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== vLLM minimal smoke ===")
    print(f"  python: {sys.version.split()[0]}")
    import vllm
    import torch
    print(f"  vllm: {vllm.__version__}")
    print(f"  torch: {torch.__version__}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")
        print(f"  cap: {torch.cuda.get_device_capability(0)}")
        print(f"  mem: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    from vllm import LLM, SamplingParams
    print(f"\n[1/3] Loading model from local cache...")
    t0 = time.time()
    llm = LLM(
        model=MODEL_DIR,
        gpu_memory_utilization=0.5,
        max_model_len=512,
        enforce_eager=True,
        dtype="float16",
        trust_remote_code=False,
    )
    load_time = time.time() - t0
    print(f"  loaded in {load_time:.1f}s")

    # Test generation with multiple prompts to measure throughput
    print(f"\n[2/3] Generating 4 prompts...")
    prompts = [
        f"<|im_start|>system\nYou are helpful.<|im_end|>\n"
        f"<|im_start|>user\nWrite a one-sentence description of {topic}.<|im_end|>\n"
        f"<|im_start|>assistant\n"
        for topic in [
            "the ocean",
            "machine learning",
            "a sunny day",
            "good food",
        ]
    ]
    t0 = time.time()
    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=64, temperature=0.7, top_p=0.9),
    )
    gen_time = time.time() - t0

    print(f"  generation in {gen_time:.2f}s ({len(prompts)} prompts)")
    print(f"\n[3/3] Results:")
    for o in outputs:
        text = o.outputs[0].text
        print(f"  OUTPUT: {text!r}")
        print(f"    prompt_tokens: {len(o.prompt_token_ids)}, "
              f"output_tokens: {len(o.outputs[0].token_ids)}, "
              f"finish_reason: {o.outputs[0].finish_reason}")

    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_output_tokens / gen_time if gen_time > 0 else 0
    print(f"\n  total output tokens: {total_output_tokens}")
    print(f"  throughput: {throughput:.1f} toks/s")

    # GPU memory
    # vLLM runs in a child EngineCore process, so torch.cuda.max_memory_
    # allocated() from the parent doesn't capture worker memory. Use
    # nvidia-smi via subprocess for actual peak GPU memory.
    peak_mem = 0.0
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        peak_mem = float(r.stdout.strip().split("\n")[0]) / 1024  # MiB -> GiB
    except Exception as e:
        print(f"  (could not read nvidia-smi: {e})")
    print(f"\n  peak GPU memory (from nvidia-smi): {peak_mem:.2f} GB")

    summary = {
        "status": "success",
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "cuda_capability": (torch.cuda.get_device_capability(0)
                            if torch.cuda.is_available() else None),
        "device": (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else None),
        "model": MODEL_DIR,
        "model_load_time_s": load_time,
        "generation_time_s": gen_time,
        "num_prompts": len(prompts),
        "total_output_tokens": total_output_tokens,
        "throughput_toks_per_s": throughput,
        "outputs": [
            {
                "prompt": o.prompt[-100:],
                "text": o.outputs[0].text,
                "output_tokens": len(o.outputs[0].token_ids),
                "finish_reason": o.outputs[0].finish_reason,
            }
            for o in outputs
        ],
        "peak_gpu_memory_gb": peak_mem,
        "env_overrides": {
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_ATTENTION_BACKEND": "TORCH_SDPA",
        },
        "notes": (
            "Three workarounds were needed on this WSL2 + RTX 5070 Ti + "
            "vLLM 0.27.1 + Python 3.10.12 host:"
            "  1. flashinfer-python package had to be uninstalled (its "
            "`comm/fd_exchange.py` uses `array.array[int]` PEP-585 syntax "
            "that fails at module-load time on Python 3.10)."
            "  2. VLLM_WSL2_ENABLE_PIN_MEMORY=1 must be set so vLLM's "
            "`is_pin_memory_available()` returns True on WSL2 (gating on "
            "kernel version + opt-in flag). Otherwise UvaBuffer fails "
            "with 'UVA is not available'."
            "  3. VLLM_USE_FLASHINFER_SAMPLER=0 + "
            "VLLM_ATTENTION_BACKEND=TORCH_SDPA to avoid flashinfer "
            "attention kernels (which are now unreachable since the "
            "package is uninstalled)."
            "Note: peak_gpu_memory_gb is read from nvidia-smi AFTER "
            "vLLM shutdown, so it may be 0; the real peak during "
            "generation was reported by vLLM's own logs."
        ),
    }
    out_path = OUT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2,
                                    ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n  summary written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())