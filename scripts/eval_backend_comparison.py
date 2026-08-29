"""P5-04 双后端基准对比 (Transformers vs vLLM) — 统一推理接口.

Pipeline:
1. For each combination of (model, backend, batch_size), run the same
   D2 multi-turn held-out split through the selected backend.
2. Reuse the P5-02 `{summary, rows}` output schema so the same
   reward_offline pipeline can be applied.
3. Aggregate into a per-combination timing + reward summary.
4. Write per-run artifacts + aggregate comparison.json + comparison.csv.

Per the project's review-process.md the canonical reviewer for stage
reviews is `minimax-cn/MiniMax-M3`. The four axes are:

    latency (ms / sample)
    throughput (samples / s)
    reward_binary  (parse+plan+execute+ground+answer all pass)
    reward_layered (mean of 8 P1-05 layers)

Usage:
    .venv/python.exe scripts/eval_backend_comparison.py \\
        --models HuggingFaceTB/SmolLM2-360M-Instruct \\
                HuggingFaceTB/SmolLM2-1.7B-Instruct \\
                Qwen/Qwen2.5-0.5B-Instruct \\
                Qwen/Qwen2.5-1.5B-Instruct \\
                Qwen/Qwen2.5-3B-Instruct \\
        --backends transformers vllm \\
        --samples-dir datasets/tool-calling-d2/dev \\
        --output-dir artifacts/p5-04-backend-comparison/ \\
        --batch-sizes 1 \\
        --limit 20
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Default backend kwargs resolved per backend below; both backends read
# these from os.environ when not set on the CLI.
_DEFAULT_DTYPE = "bf16"
_DEFAULT_MAX_NEW_TOKENS = 256

# Disable HF telemetry + force offline model loading by default. We are
# running on a host with a fully-populated HF cache; network round-trips
# are slow and unreliable, and the only model artifacts we need are
# already in ``artifacts/huggingface/``. Users can override either via
# environment variables.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Point HF cache at our local snapshot directory. The repo stores HF
# artifacts under ``artifacts/huggingface/`` using the legacy v0 layout
# (``models--{org}--{name}/snapshots/{commit}/``); modern HF clients
# expect a ``hub/`` subdirectory. Setting both ``HF_HOME`` and
# ``HF_HUB_CACHE`` to the same path lets both clients find the cache.
_hf_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
if not _hf_cache:
    _hf_cache = str(ROOT / "artifacts" / "huggingface")
    os.environ["HF_HOME"] = _hf_cache
os.environ.setdefault("HF_HUB_CACHE", _hf_cache)

# Five canonical public models for P5-04 (matching P5-02 §8).
# ``revision`` values are the exact cached commit hashes for each
# model so the script can run with ``HF_HUB_OFFLINE=1`` (default).
# Update via ``huggingface-cli download`` if any of these commits
# become unreachable on the host.
DEFAULT_MODELS: tuple[tuple[str, str], ...] = (
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951"),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"),
)
DEFAULT_BACKENDS: tuple[str, ...] = ("transformers", "vllm")
DEFAULT_BATCH_SIZES: tuple[int, ...] = (1,)


class Backend(Protocol):
    """Minimal interface every backend must implement.

    The two real backends in this script (``TransformersBackend`` and
    ``VLLMBackend``) implement this. Tests use the mock backend in
    ``tests/test_eval_backend_comparison.py``.
    """

    name: str

    def setup(self, model_id: str, **kwargs: Any) -> None:
        """Load model + tokenizer. Backend holds them as instance state."""

    def chat_generate(
        self,
        messages_batch: list[list[dict[str, Any]]],
        tools_batch: list[list[dict[str, Any]] | None],
    ) -> tuple[list[str], list[str]]:
        """Run chat completion for each sample. Returns ``(generations, prompt_previews)``.

        ``tools_batch[i]`` is the tool schema list for sample ``i`` (or
        ``None`` if no tools should be injected).

        ``prompt_previews[i]`` is the rendered prompt's first 200 chars for
        sample ``i``; the comparison CLI uses this to populate the
        ``user_turn`` field of the persisted row so artifacts match
        ``eval_transformers._build_eval_row`` exactly.
        """

    def teardown(self) -> None:
        """Release GPU memory / close handles."""

    def metadata(self) -> dict[str, Any]:
        """Backend-specific metadata (version, dtype, device) for the summary."""


def _slugify(model_id: str) -> str:
    """Stable filesystem-safe slug from a HF model id."""
    return model_id.replace("/", "__").replace(" ", "_")


def _load_samples(samples_dir: Path, limit: int) -> list[dict[str, Any]]:
    """Load D2 samples from ``samples_dir/*.json`` (sorted by id)."""
    paths = sorted(samples_dir.glob("*.json"))
    if limit:
        paths = paths[:limit]
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def _strip_terminal_assistant(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove the final assistant message so the model isn't fed the gold answer.

    Mirrors ``eval_transformers._strip_terminal_assistant`` (P5-02 round-2
    target-answer-leakage fix). The last message is dropped if its
    ``role == "assistant"``.
    """
    if messages and messages[-1].get("role") == "assistant":
        return messages[:-1]
    return messages


def _apply_chat_template_transformers(
    tokenizer: Any,
    sample: dict[str, Any],
) -> str:
    """Render ``sample['messages']`` (terminal-assistant stripped) using the
    HF tokenizer's chat template. Falls back to ``<role>: <content>``
    joining if no template is available — same fallback order as
    ``eval_transformers._apply_chat_template``.
    """
    messages = _strip_terminal_assistant(sample["messages"])
    tools = sample.get("tools")
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
        )
    except Exception:
        # Fallback: simple "<role>: <content>" join for user / assistant / system
        parts: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts) + "\nassistant:"


class TransformersBackend:
    """Thin wrapper around ``AutoModelForCausalLM`` + greedy generation.

    Mirrors the relevant logic in ``eval_transformers._greedy_generate``.
    """

    name = "transformers"

    def __init__(self, dtype: str, max_new_tokens: int, device: str = "auto") -> None:
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.tokenizer: Any = None
        self.model: Any = None

    def setup(self, model_id: str, **kwargs: Any) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # CPU must be fp32 (matches eval_transformers._resolve_dtype)
        # ``torch.bf16`` doesn't exist — the canonical attribute is ``bfloat16``.
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = torch.float32 if device == "cpu" else dtype_map[self.dtype]
        cache_dir = os.environ.get("HF_HOME") or str(ROOT / "artifacts" / "huggingface")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, cache_dir=cache_dir,
        ).to(device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def chat_generate(
        self,
        messages_batch: list[list[dict[str, Any]]],
        tools_batch: list[list[dict[str, Any]] | None],
    ) -> tuple[list[str], list[str]]:
        import torch

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("TransformersBackend.setup() must be called first")

        prompts: list[str] = []
        for messages, tools in zip(messages_batch, tools_batch):
            stripped = _strip_terminal_assistant(messages)
            render_kwargs: dict[str, Any] = dict(
                tokenize=False, add_generation_prompt=True,
            )
            if tools is not None:
                render_kwargs["tools"] = tools
            try:
                prompt = self.tokenizer.apply_chat_template(
                    stripped, **render_kwargs,
                )
            except Exception:
                # Fallback join — same as setup()'s fallback path.
                parts = [
                    f"{m.get('role', 'user')}: {m.get('content', '')}" for m in stripped
                ]
                prompt = "\n".join(parts) + "\nassistant:"
            prompts.append(prompt)

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False, num_beams=1,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Decode only the newly generated tokens (skip prompt prefix).
        results: list[str] = []
        for i, ids in enumerate(output_ids):
            prompt_len = inputs["input_ids"][i].shape[0]
            new_ids = ids[prompt_len:]
            results.append(self.tokenizer.decode(new_ids, skip_special_tokens=True))
        prompt_previews = [p[:200] for p in prompts]
        return results, prompt_previews

    def teardown(self) -> None:
        import torch
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def metadata(self) -> dict[str, Any]:
        import torch
        import transformers
        return {
            "backend": "transformers",
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "dtype": self.dtype,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
        }


class VLLMBackend:
    """vLLM-based backend.

    Expects the standard P5-03 workarounds to be applied via env vars
    *before* importing vllm: ``VLLM_WSL2_ENABLE_PIN_MEMORY=1``,
    ``VLLM_USE_FLASHINFER_SAMPLER=0``, ``VLLM_ATTENTION_BACKEND=TORCH_SDPA``.
    See ``docs/experiments/p5-03-vllm-feasibility/README.md``.
    """

    name = "vllm"

    def __init__(self, dtype: str, max_new_tokens: int, gpu_memory_utilization: float) -> None:
        # vLLM's ``dtype`` argument uses long names ("bfloat16", "float16",
        # "float32"). Map the short CLI aliases to the canonical forms.
        vllm_dtype_map = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
        self.dtype = vllm_dtype_map.get(dtype, dtype)
        self.max_new_tokens = max_new_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm: Any = None
        self.tokenizer: Any = None
        self.model_id: str = ""

    def setup(self, model_id: str, **kwargs: Any) -> None:
        # Apply WSL2 workarounds defensively in case the user forgot.
        os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")
        # Force offline loading: ``HF_HUB_OFFLINE=1`` makes ``from_pretrained``
        # + vLLM's snapshot lookup read the local cache only.
        os.environ["HF_HUB_OFFLINE"] = "1"

        from vllm import LLM
        from transformers import AutoTokenizer

        cache_dir = os.environ.get("HF_HOME") or str(ROOT / "artifacts" / "huggingface")
        # ``revision`` is required when running offline: vLLM needs to find
        # the exact snapshot folder by commit hash. ``run_one_combination``
        # passes ``revision`` via ``kwargs`` when present.
        revision = kwargs.get("revision", "main")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir,
        )
        self.llm = LLM(
            model=model_id,
            revision=revision,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            download_dir=cache_dir,
            enforce_eager=True,  # cheaper on small models; no cuda graphs
            max_model_len=4096,
        )
        self.model_id = model_id

    def chat_generate(
        self,
        messages_batch: list[list[dict[str, Any]]],
        tools_batch: list[list[dict[str, Any]] | None],
    ) -> tuple[list[str], list[str]]:
        from vllm import SamplingParams

        if self.llm is None:
            raise RuntimeError("VLLMBackend.setup() must be called first")

        # Use the HF tokenizer's chat template to render prompts (consistent
        # with the Transformers backend so per-row outputs are comparable).
        prompts: list[str] = []
        for messages, tools in zip(messages_batch, tools_batch):
            stripped = _strip_terminal_assistant(messages)
            render_kwargs: dict[str, Any] = dict(
                tokenize=False, add_generation_prompt=True,
            )
            if tools is not None:
                render_kwargs["tools"] = tools
            try:
                prompt = self.tokenizer.apply_chat_template(
                    stripped, **render_kwargs,
                )
            except Exception:
                parts = [
                    f"{m.get('role', 'user')}: {m.get('content', '')}" for m in stripped
                ]
                prompt = "\n".join(parts) + "\nassistant:"
            prompts.append(prompt)

        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_new_tokens,
        )
        outputs = self.llm.generate(prompts, sampling, use_tqdm=False)
        generations = [o.outputs[0].text for o in outputs]
        prompt_previews = [p[:200] for p in prompts]
        return generations, prompt_previews

    def teardown(self) -> None:
        # vLLM holds CUDA memory in worker subprocesses; explicit
        # ``del`` plus ``empty_cache`` releases as much as possible.
        try:
            del self.llm
        except Exception:
            pass
        self.llm = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def metadata(self) -> dict[str, Any]:
        import vllm
        return {
            "backend": "vllm",
            "vllm_version": vllm.__version__,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_new_tokens": self.max_new_tokens,
            "model_id": self.model_id,
        }


def build_backend(backend: str, args: argparse.Namespace) -> Backend:
    if backend == "transformers":
        return TransformersBackend(
            dtype=args.dtype, max_new_tokens=args.max_new_tokens, device=args.device,
        )
    if backend == "vllm":
        return VLLMBackend(
            dtype=args.dtype, max_new_tokens=args.max_new_tokens,
            gpu_memory_utilization=args.vllm_gpu_mem_util,
        )
    raise ValueError(f"Unknown backend: {backend!r}")


def _row_from_sample(
    sample: dict[str, Any],
    prompt_text: str,
    generated: str,
    layer_counts: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the P5-02-compatible row + run P1-05 classifier on it.

    Mirrors ``eval_transformers._build_eval_row`` schema exactly so the
    output is consumable by ``scripts/reward_offline.py`` without
    any glue code.
    """
    from scripts.eval_transformers import _build_eval_row  # type: ignore[attr-defined]
    return _build_eval_row(sample, prompt_text, generated)


def _aggregate_timing(
    *,
    samples: list[dict[str, Any]],
    prompts: list[str],
    outputs: list[str],
    elapsed_s: float,
) -> dict[str, Any]:
    """Compute latency + throughput metrics from a single timed run."""
    total = len(outputs)
    per_sample_ms = (elapsed_s * 1000.0) / total if total else 0.0
    throughput_sps = total / elapsed_s if elapsed_s > 0 else 0.0
    return {
        "samples": total,
        "elapsed_s": elapsed_s,
        "per_sample_latency_ms": per_sample_ms,
        "throughput_samples_per_s": throughput_sps,
    }


def run_one_combination(
    *,
    model_id: str,
    backend_name: str,
    batch_size: int,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    backend_factory: type[Backend] | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Run a single (model, backend, batch) combination and return a summary.

    Returns a dict that the caller writes to disk as ``run_*.json`` plus
    adds to the aggregate comparison table.

    ``backend_factory`` is a test hook — production code uses
    ``build_backend`` derived from ``args``; tests can pass a mock factory
    to avoid importing transformers / vLLM.

    ``revision`` is the cached HF revision (commit hash) for ``model_id``;
    backends pass it through to ``from_pretrained`` / vLLM ``LLM`` so the
    runs can be offline (no network round-trip).
    """
    if backend_factory is None:
        backend = build_backend(backend_name, args)
    else:
        backend = backend_factory(
            dtype=args.dtype, max_new_tokens=args.max_new_tokens,
        )

    print(
        f"[p5-04] {model_id} backend={backend_name} batch={batch_size} "
        f"({len(samples)} samples)",
        flush=True,
    )
    backend.setup(model_id, revision=revision or "main")
    try:
        layer_counts: dict[str, int] = {}
        rows: list[dict[str, Any]] = []
        # Chunk samples into batch-sized groups for backend.chat_generate.
        start = time.perf_counter()
        for chunk_start in range(0, len(samples), batch_size):
            chunk = samples[chunk_start: chunk_start + batch_size]
            messages_batch = [s["messages"] for s in chunk]
            tools_batch = [s.get("tools") for s in chunk]
            try:
                # Backend returns (generated, prompt_preview) per sample so the
                # row's ``user_turn`` field can carry the actual rendered prompt
                # (matches P5-02 ``eval_transformers._build_eval_row``).
                generated, prompt_previews = backend.chat_generate(messages_batch, tools_batch)
            except Exception as exc:
                # If the entire batch fails, fall back to empty generations
                # so the run still produces a row per sample (matches
                # eval_transformers._greedy_generate error semantics).
                print(f"[p5-04] generation error: {exc}", flush=True)
                generated = [""] * len(chunk)
                prompt_previews = [s["messages"][-1].get("content", "")[:200] if s["messages"] else "" for s in chunk]
            # Build per-sample rows + apply P1-05 classifier.
            for sample, gen, prompt_prev in zip(chunk, generated, prompt_previews):
                row, result = _row_from_sample(sample, prompt_prev, gen, layer_counts)
                first = result["first_failure"]
                key = first if first is not None else "none"
                layer_counts[key] = layer_counts.get(key, 0) + 1
                rows.append(row)
        elapsed = time.perf_counter() - start
        timing = _aggregate_timing(
            samples=samples, prompts=[r["user_turn"] for r in rows],
            outputs=[r["generated"] for r in rows], elapsed_s=elapsed,
        )
        parse_success_count = sum(
            1 for r in rows if r["layers"].get("parse_success") is True
        )
        # Compute reward via reward_offline on the produced rows so the
        # numbers in this script's summary match what the comparison
        # table reports.
        try:
            from scripts.reward_offline import (
                load_samples as _load_samples_for_reward,  # type: ignore[attr-defined]
                compute_reward, _to_transcript,  # type: ignore[attr-defined]
            )
            samples_dir_for_reward = Path(args.samples_dir).resolve()
            samples_by_id = _load_samples_for_reward(samples_dir_for_reward)
            transcript_by_id = {r["sample_id"]: r for r in rows if r["sample_id"]}
            signals = []
            for sid, sample in samples_by_id.items():
                row = transcript_by_id.get(sid)
                if row is None:
                    continue
                signals.append(compute_reward(
                    sample, row,
                    transcript_kind="model_generated",
                    checkpoint=model_id,
                ))
            reward_binary = sum(1 for s in signals if s["reward_binary"]) / max(len(signals), 1)
            reward_layered = sum(s["reward_layered"] for s in signals) / max(len(signals), 1)
        except Exception as exc:
            # If reward_offline can't be run (e.g. test env without vllm),
            # fall back to a layer-derived approximation.
            print(f"[p5-04] reward_offline fallback (exc: {exc})", flush=True)
            n = max(len(rows), 1)
            reward_binary = sum(1 for r in rows if r["first_failure"] is None) / n
            reward_layered = (
                sum(
                    sum(1 for v in r["layers"].values() if v is True)
                    for r in rows
                )
                / (n * 8)
            )
        summary = {
            "model": model_id,
            "backend": backend_name,
            "batch_size": batch_size,
            "samples": timing["samples"],
            "elapsed_s": timing["elapsed_s"],
            "per_sample_latency_ms": timing["per_sample_latency_ms"],
            "throughput_samples_per_s": timing["throughput_samples_per_s"],
            "reward_binary": reward_binary,
            "reward_layered": reward_layered,
            "parse_success_count": parse_success_count,
            "parse_success_rate": parse_success_count / max(len(rows), 1),
            "first_failure_distribution": layer_counts,
            "backend_metadata": backend.metadata(),
            "samples_dir": str(args.samples_dir),
        }
    finally:
        backend.teardown()
    return {"summary": summary, "rows": rows}


def write_run_artifact(out_dir: Path, model_slug: str, backend: str, batch: int, payload: dict[str, Any]) -> Path:
    """Write a single run's payload to ``out_dir/run_*.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"run_{model_slug}__{backend}__b{batch}.json"
    out_path = out_dir / fname
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path


def write_aggregate_comparison(out_dir: Path, runs: list[dict[str, Any]]) -> tuple[Path, Path]:
    """Write ``comparison.json`` (full payloads) + ``comparison.csv`` (flat)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmp_json = out_dir / "comparison.json"
    cmp_json.write_text(
        json.dumps({"runs": [r["summary"] for r in runs]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    cmp_csv = out_dir / "comparison.csv"
    fieldnames = [
        "model", "backend", "batch_size", "samples", "elapsed_s",
        "per_sample_latency_ms", "throughput_samples_per_s",
        "reward_binary", "reward_layered", "parse_success_rate",
    ]
    with cmp_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in runs:
            row = {k: r["summary"].get(k) for k in fieldnames}
            writer.writerow(row)
    return cmp_json, cmp_csv


def compute_delta_percentages(runs: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    """Compute same-model backend Δ% for each (model, batch_size) pair.

    For each (model, batch_size), finds the transformers run + the vllm run
    (if both present) and computes vLLM − Transformers as a percentage of
    the transformers baseline. The convention is:

    - latency: lower is better, so a positive Δ% means vLLM is slower.
    - throughput: higher is better, so a positive Δ% means vLLM is faster.
    - reward_binary / reward_layered / parse_success_rate: higher is better;
      a positive Δ% means vLLM produced better-quality generations.

    Missing combinations return ``None`` for that field (preserves the
    schema even when one backend failed or was skipped).

    Returns ``{(model, batch_size) -> {field_name: delta_percent_or_None}}``.
    The keys are JSON-serializable strings ``"<model>__b<batch>"``.
    """
    metric_fields = (
        "per_sample_latency_ms", "throughput_samples_per_s",
        "reward_binary", "reward_layered", "parse_success_rate",
    )
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for r in runs:
        s = r["summary"]
        key = (s["model"], s["batch_size"])
        grouped.setdefault(key, {})[s["backend"]] = s

    out: dict[str, dict[str, float | None]] = {}
    for (model, batch_size), backends in grouped.items():
        tr = backends.get("transformers")
        vl = backends.get("vllm")
        per_key: dict[str, float | None] = {}
        for field in metric_fields:
            if tr is None or vl is None:
                per_key[field] = None
                continue
            tr_val = tr.get(field)
            vl_val = vl.get(field)
            if tr_val in (None, 0) or vl_val is None:
                per_key[field] = None
                continue
            base = float(tr_val)
            if base == 0.0:
                per_key[field] = None
                continue
            per_key[field] = (float(vl_val) - base) / abs(base) * 100.0
        out[f"{model}__b{batch_size}"] = per_key
    return out


def write_delta_csv(out_dir: Path, deltas: dict[str, dict[str, float | None]]) -> Path:
    """Write same-model Δ% table to ``comparison_delta.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison_delta.csv"
    fieldnames = [
        "model_batch",
        "delta_per_sample_latency_ms_pct",
        "delta_throughput_samples_per_s_pct",
        "delta_reward_binary_pct",
        "delta_reward_layered_pct",
        "delta_parse_success_rate_pct",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for model_batch, per in deltas.items():
            writer.writerow({
                "model_batch": model_batch,
                "delta_per_sample_latency_ms_pct":
                    "" if per["per_sample_latency_ms"] is None
                    else f"{per['per_sample_latency_ms']:.2f}",
                "delta_throughput_samples_per_s_pct":
                    "" if per["throughput_samples_per_s"] is None
                    else f"{per['throughput_samples_per_s']:.2f}",
                "delta_reward_binary_pct":
                    "" if per["reward_binary"] is None
                    else f"{per['reward_binary']:.2f}",
                "delta_reward_layered_pct":
                    "" if per["reward_layered"] is None
                    else f"{per['reward_layered']:.2f}",
                "delta_parse_success_rate_pct":
                    "" if per["parse_success_rate"] is None
                    else f"{per['parse_success_rate']:.2f}",
            })
    return out_path


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models", nargs="+", default=None,
        help="Hugging Face model ids to evaluate (default: 5 canonical public models). "
             "Pair each id with ``=`` and an explicit revision to run offline; e.g. "
             "``HuggingFaceTB/SmolLM2-360M-Instruct=a10cc1512eabd3dde888204e902eca88bddb4951``.",
    )
    p.add_argument(
        "--backends", nargs="+", choices=("transformers", "vllm"),
        default=list(DEFAULT_BACKENDS),
        help="Backends to compare (default: both transformers and vllm)",
    )
    p.add_argument(
        "--batch-sizes", nargs="+", type=int, default=list(DEFAULT_BATCH_SIZES),
        help="Per-call batch sizes to evaluate (default: 1)",
    )
    p.add_argument(
        "--samples-dir", type=Path,
        default=Path("datasets/tool-calling-d2/dev"),
        help="Directory of D2 multi-turn sample JSON files",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/p5-04-backend-comparison"),
        help="Output directory for per-run artifacts and comparison table",
    )
    p.add_argument("--limit", type=int, default=0, help="Sample cap (0 = all)")
    p.add_argument("--max-new-tokens", type=int, default=_DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default=_DEFAULT_DTYPE)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--vllm-gpu-mem-util", type=float, default=0.85)
    p.add_argument(
        "--parallel", action="store_true",
        help="Run different model+backend combinations in parallel threads",
    )
    return p


def main() -> int:
    args = _build_argparser().parse_args()
    # Resolve ``--models``: each item is either a bare id (use the canonical
    # revision that matches the local cache) or ``<id>=<revision>``.
    if args.models is None:
        model_ids: list[tuple[str, str]] = list(DEFAULT_MODELS)
    else:
        model_ids = []
        for item in args.models:
            if "=" in item:
                mid, rev = item.split("=", 1)
                model_ids.append((mid, rev))
            else:
                # Find the matching canonical revision by id, else default to "main".
                rev = next(
                    (r for m, r in DEFAULT_MODELS if m == item), "main",
                )
                model_ids.append((item, rev))
    samples = _load_samples(args.samples_dir, args.limit)
    if not samples:
        print(f"[p5-04] no samples found under {args.samples_dir}", flush=True)
        return 1
    print(f"[p5-04] {len(samples)} samples loaded from {args.samples_dir}", flush=True)

    combinations = [
        (m, r, b, bs)
        for m, r in model_ids
        for b in args.backends
        for bs in args.batch_sizes
    ]
    print(f"[p5-04] {len(combinations)} combinations to run: "
          f"{[(m, b, bs) for m, _, b, bs in combinations]}", flush=True)

    runs: list[dict[str, Any]] = []
    if args.parallel:
        # Run different backends/models in parallel only if the user opts in.
        # Within a single (model, backend, batch) combination we always run
        # serially — GPU can only host one backend process at a time anyway.
        with ThreadPoolExecutor(max_workers=len(combinations)) as pool:
            futures = {
                pool.submit(
                    run_one_combination,
                    model_id=m, backend_name=b, batch_size=bs,
                    samples=samples, args=args, revision=r,
                ): (m, b, bs)
                for m, r, b, bs in combinations
            }
            for fut, key in futures.items():
                try:
                    runs.append(fut.result())
                except Exception as exc:
                    print(f"[p5-04] combination {key} failed: {exc}", flush=True)
    else:
        for m, r, b, bs in combinations:
            try:
                runs.append(run_one_combination(
                    model_id=m, backend_name=b, batch_size=bs,
                    samples=samples, args=args, revision=r,
                ))
            except Exception as exc:
                print(f"[p5-04] combination ({m}, {b}, b={bs}) failed: {exc}", flush=True)

    # Write per-run artifacts and aggregate
    for run_payload in runs:
        s = run_payload["summary"]
        write_run_artifact(
            args.output_dir, _slugify(s["model"]), s["backend"],
            s["batch_size"], run_payload,
        )
    cmp_json, cmp_csv = write_aggregate_comparison(args.output_dir, runs)
    deltas = compute_delta_percentages(runs)
    cmp_delta_json = args.output_dir / "comparison_delta.json"
    cmp_delta_json.write_text(
        json.dumps(deltas, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    cmp_delta_csv = write_delta_csv(args.output_dir, deltas)
    print(
        f"[p5-04] wrote {len(runs)} runs to {args.output_dir}",
        flush=True,
    )
    print(f"[p5-04] aggregate JSON:    {cmp_json}", flush=True)
    print(f"[p5-04] aggregate CSV:     {cmp_csv}", flush=True)
    print(f"[p5-04] delta JSON:        {cmp_delta_json}", flush=True)
    print(f"[p5-04] delta CSV:         {cmp_delta_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
