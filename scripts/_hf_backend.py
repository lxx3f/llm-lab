"""Shared Hugging Face backend primitives for inference scripts.

Both ``scripts/eval_transformers.py`` (P5-02 Transformers backend for
tool-calling eval) and ``scripts/eval_owt_real.py`` (per-token CE loss on
public model OWT validation) load causal-LM models the same way: from a
local snapshot directory or Hugging Face model id with a dtype/device
contract. Keeping the contract in one module ensures:

- ``resolve_device`` / ``resolve_dtype`` semantics are identical across
  scripts (CPU forces fp32; bf16 default on CUDA).
- Both scripts use the same ``local_files_only=True`` + ``trust_remote_code``
  + ``cache_dir`` policy so loading behaviour is auditable in one place.
- ``load_causal_lm_model`` returns a model already moved to ``device`` and
  set to ``eval()`` mode; callers can immediately run inference.

This module is intentionally CLI-free so it can be imported from both
scripts without side effects.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def resolve_device(arg: str) -> str:
    """Resolve the device string to ``"cuda"`` / ``"cpu"``."""
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def resolve_dtype(name: str, device: str):
    """Resolve the dtype string to a torch dtype, forcing fp32 on CPU."""
    if name == "bf16":
        resolved = torch.bfloat16
    elif name == "fp16":
        resolved = torch.float16
    elif name == "fp32":
        resolved = torch.float32
    else:
        raise ValueError(f"unsupported dtype {name!r}")
    if device == "cpu" and resolved is not torch.float32:
        return torch.float32
    return resolved


def load_causal_lm_model(
    *,
    model_path_or_id: str | Path,
    dtype_name: str,
    device: str,
    cache_dir: str | Path | None = None,
    trust_remote_code: bool = False,
    local_files_only: bool = True,
) -> Any:
    """Load a causal-LM model from a local snapshot or HF model id.

    Returns the model moved to ``device`` in ``eval()`` mode. The matching
    tokenizer is loaded separately by callers via ``AutoTokenizer``.

    ``local_files_only=True`` is the default to keep inference reproducible
    (the scripts always pass an explicit ``model_dir`` derived from a
    pre-downloaded ModelScope/HF snapshot, never an HF model id).
    """
    from transformers import AutoModelForCausalLM

    device = resolve_device(device)
    dtype = resolve_dtype(dtype_name, device)
    resolved_cache = (
        str(cache_dir)
        if cache_dir is not None
        else os.environ.get("HF_HOME") or str(Path("artifacts") / "huggingface")
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path_or_id),
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        cache_dir=resolved_cache,
    )
    model.to(device)
    model.eval()
    return model