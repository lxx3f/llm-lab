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
        --batch-sizes 1 4 \\
        --limit 90
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
    ``VLLMBackend``) implement this. The script's ``--selftest`` mode
    provides deterministic no-GPU coverage for the interface and helpers.
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
    """Ad-hoc loader: read ``samples_dir/*.json`` sorted by filename.

    Retained for back-compat + selftest usage. For the authoritative
    P5-02 benchmark subset, callers must use ``_load_samples_from_manifest``
    so that content SHA256 verification enforces reproducibility.
    """
    paths = sorted(samples_dir.glob("*.json"))
    if limit:
        paths = paths[:limit]
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def _materialize_manifest_samples(
    manifest_path: Path,
    samples_root: Path,
    samples_meta: list[dict[str, Any]],
    source_commit: str,
) -> bool:
    """Materialize missing gitignored samples from the manifest source commit.

    Dataset files remain gitignored by project policy. The tracked manifest
    therefore records both their content hashes and the immutable git source
    commit. A clean checkout can reconstruct missing files with ``git show``;
    materialized bytes are hash-checked before they are written.

    ``samples_root`` is the directory used as the cwd for ``git show``.
    The materialized files are placed under ``samples_root`` using the
    manifest-declared ``path`` (relative). When called from the CLI
    pipeline, ``samples_root`` is the repo root and the resulting files
    land at ``<repo>/datasets/tool-calling-d2/p5-02-benchmark/<id>.json``.
    """
    missing: list[tuple[dict[str, Any], Path]] = []
    for entry in samples_meta:
        target = (samples_root / entry["path"]).resolve()
        if not target.exists():
            missing.append((entry, target))
    if not missing:
        return False
    if not source_commit:
        raise ValueError(
            f"manifest {manifest_path} has missing samples but no source_commit"
        )
    import subprocess
    for entry, target in missing:
        sample_id = entry["sample_id"]
        git_path = f"{source_commit}:datasets/tool-calling-d2/dev/{sample_id}.json"
        try:
            import subprocess
            materialized = subprocess.check_output(
                ["git", "-C", str(ROOT), "show", git_path],
                stderr=subprocess.STDOUT,
            )
        except Exception as exc:
            raise ValueError(
                f"cannot materialize {entry['path']} from {git_path}: {exc}"
            ) from exc
        expected = entry["sha256"]
        # The historical generated files were written as CRLF on this host;
        # git stores the source blob with LF. Reproduce the hashed bytes.
        if hashlib.sha256(materialized).hexdigest() != expected:
            materialized = materialized.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        actual = hashlib.sha256(materialized).hexdigest()
        if actual != expected:
            raise ValueError(
                f"materialized SHA mismatch for {entry['path']}: "
                f"expected {expected}, got {actual}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(materialized)
    return True


def _load_samples_from_manifest(
    manifest_path: Path,
    samples_root: Path,
    limit: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the historical P5-02 benchmark subset from a content-hash manifest.

    The manifest is the authoritative source of which samples belong to
    the benchmark subset. Each entry carries ``path``, ``sha256`` and
    ``task_type``. We verify the bytes on disk match ``sha256`` before
    loading; any mismatch raises ``ValueError`` so a stale or tampered
    sample cannot silently enter the run.

    Missing gitignored sample files are materialized from the manifest's
    immutable ``source_commit`` before verification. This makes the full
    command reproducible from a clean checkout without committing the
    generated dataset itself.

    The aggregate SHA256 is recomputed from the per-sample SHA256s (sorted
    by sample id) and compared against ``recomputed_aggregate_sha256`` /
    ``source_aggregate_sha256``. This makes the loader a strict
    reproducer for the historical P5-02 content.

    ``samples_root`` is the repo root used to resolve ``samples[*].path``
    (paths in the manifest are repo-relative). ``limit`` caps the number
    of samples loaded after manifest validation (manifest itself is still
    validated for the full set).
    """
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples_meta = payload.get("samples", [])
    if not samples_meta:
        raise ValueError(f"manifest {manifest_path} has no samples")

    materialized = _materialize_manifest_samples(
        manifest_path,
        samples_root,
        samples_meta,
        str(payload.get("source_commit") or payload.get("source_commit_short") or ""),
    )

    # Validate every sample file's SHA256 against the manifest entry.
    verified_samples: list[dict[str, Any]] = []
    for entry in samples_meta:
        rel = entry["path"]
        expected_sha = entry["sha256"]
        # Resolve the path relative to ``samples_root`` (the repo root).
        abs_path = (samples_root / rel).resolve()
        actual_bytes = abs_path.read_bytes()
        actual_sha = hashlib.sha256(actual_bytes).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"SHA mismatch for {rel}: expected {expected_sha}, got {actual_sha}. "
                f"Re-extract from source commit {payload.get('source_commit_short')}"
                f" to restore the historical subset."
            )
        verified_samples.append(json.loads(actual_bytes.decode("utf-8")))

    # Recompute aggregate SHA from per-sample SHA values and compare.
    ids = sorted(s["sample_id"] for s in samples_meta)
    digest = hashlib.sha256()
    for sid in ids:
        digest.update(f"{sid}\n".encode("utf-8"))
    recomputed = digest.hexdigest()
    expected_recomputed = payload.get("recomputed_aggregate_sha256")
    if expected_recomputed and recomputed != expected_recomputed:
        raise ValueError(
            f"aggregate SHA mismatch in {manifest_path}: "
            f"recomputed {recomputed} != stored {expected_recomputed}"
        )

    if limit > 0:
        verified_samples = verified_samples[:limit]

    # Return the sample list + the manifest header for downstream logging.
    header = {
        "manifest_path": str(manifest_path),
        "source_commit": payload.get("source_commit"),
        "source_aggregate_sha256": payload.get("source_aggregate_sha256"),
        "recomputed_aggregate_sha256": recomputed,
        "sample_count_declared": payload.get("sample_count"),
        "sample_count_loaded": len(verified_samples),
        "materialized_from_source_commit": materialized,
        "task_type_counts": payload.get("task_type_counts"),
    }
    return verified_samples, header


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
        self.revision: str = ""

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
        # ``revision`` is the immutable HF commit used for both tokenizer
        # and model loading. The caller passes the same value to both
        # backend implementations for a fair comparison.
        revision = str(kwargs.get("revision") or "main")
        self.revision = revision

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir,
        )
        # Decoder-only architecture (SmolLM2 / Qwen2.5) requires
        # left-padding for correct batched greedy generation. The HF
        # warning ("right-padding was detected") indicates the batch
        # would otherwise produce garbage because new tokens are
        # generated right-to-left.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=torch_dtype, cache_dir=cache_dir,
        ).to(device)
        self.model.eval()

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
            "revision": self.revision,
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
        revision = str(kwargs.get("revision") or "main")
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
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
            "revision": self.revision,
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
        generation_failures: dict[str, str] = {}
        for chunk_start in range(0, len(samples), batch_size):
            chunk = samples[chunk_start: chunk_start + batch_size]
            messages_batch = [s["messages"] for s in chunk]
            tools_batch = [s.get("tools") for s in chunk]
            sample_ids = [
                s.get("id") or s.get("sample_id") or f"sample-{chunk_start + i}"
                for i, s in enumerate(chunk)
            ]
            try:
                # Backend returns (generated, prompt_preview) per sample so the
                # row's ``user_turn`` field can carry the actual rendered prompt
                # (matches P5-02 ``eval_transformers._build_eval_row``).
                generated, prompt_previews = backend.chat_generate(messages_batch, tools_batch)
            except Exception as exc:
                # If the entire batch fails, retry each sample individually
                # so a single bad sample doesn't drag the chunk down. After
                # that, mark any remaining failures explicitly so they don't
                # silently masquerade as ``parse_success``.
                print(
                    f"[p5-04] batch error ({backend_name}/{batch_size}): {exc}; "
                    f"retrying per-sample",
                    flush=True,
                )
                generated = [""] * len(chunk)
                prompt_previews = [
                    s["messages"][-1].get("content", "")[:200]
                    if s["messages"] else ""
                    for s in chunk
                ]
                for i, (mb, tb, sid) in enumerate(
                    zip(messages_batch, tools_batch, sample_ids)
                ):
                    try:
                        gen_i, prev_i = backend.chat_generate([mb], [tb])
                        if gen_i and gen_i[0] is not None:
                            generated[i] = gen_i[0]
                        if prev_i and prev_i[0] is not None:
                            prompt_previews[i] = prev_i[0]
                    except Exception as exc_i:
                        generation_failures[sid] = repr(exc_i)
            # Build per-sample rows + apply P1-05 classifier.
            for sample, gen, prompt_prev, sid in zip(
                chunk, generated, prompt_previews, sample_ids,
            ):
                row, result = _row_from_sample(sample, prompt_prev, gen, layer_counts)
                # Test hook: a _MockBackend with a ``payloads`` queue can
                # inject pre-built rows so selftests can assert on
                # round-8 reward semantics without depending on
                # reward_offline's classifier output. Production backends
                # (``self.payloads is None``) ignore this branch.
                injected = getattr(backend, "take_payload", None)
                if callable(injected):
                    p = injected()
                    if p is not None:
                        row = {**row, **p}
                # Mark any sample that errored on the backend as an
                # explicit failure — do NOT let an empty ``generated``
                # count toward parse_success or any reward channel.
                if sid in generation_failures:
                    row["generation_error"] = generation_failures[sid]
                    row["first_failure"] = "generation_failed"
                    # Override parse_success: generation never produced
                    # an actual transcript, so the JSON-parse layer is
                    # not meaningful. We treat this as a hard failure.
                    row["layers"] = dict(row.get("layers", {}))
                    row["layers"]["parse_success"] = False
                    layer_counts["generation_failed"] = (
                        layer_counts.get("generation_failed", 0) + 1
                    )
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
        # table reports. The auditor requires that we feed reward_offline
        # the *exact* manifest-listed sample objects rather than loading
        # them from any on-disk directory (which may be a documentation
        # directory or a stale tree). ``args.samples_by_id`` is populated
        # by ``main()`` from the manifest loader; the ``--samples-dir``
        # fallback path builds the same ``samples_by_id`` map from the
        # glob loader.
        try:
            from scripts.reward_offline import compute_reward  # type: ignore[attr-defined]
            samples_by_id = getattr(args, "samples_by_id", None)
            if not samples_by_id:
                # No samples_by_id available (e.g. legacy path) — this
                # should not happen post round-7 because both CLI paths
                # populate the map. We refuse to silently zero out.
                raise RuntimeError(
                    "args.samples_by_id is empty; refusing to compute "
                    "reward against an empty sample set. Pass "
                    "--samples-manifest (or --samples-dir with the exact "
                    "P5-02 subset directory)."
                )
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
            if not signals:
                raise RuntimeError(
                    "no sample/transcript id overlaps — refusing to "
                    "report reward=0.0 (would mask a wiring bug)."
                )
            reward_binary = sum(1 for s in signals if s["reward_binary"]) / len(signals)
            reward_layered = sum(s["reward_layered"] for s in signals) / len(signals)
        except RuntimeError as exc:
            # Round-8 fix: RuntimeError signals a wiring bug (empty
            # samples_by_id or no overlap). Fallback to a layer-derived
            # approximation would MASK the bug. Re-raise so the caller
            # sees the failure and the test suite catches it.
            raise
        except Exception as exc:
            # Other exceptions (e.g. ``reward_offline`` import failure,
            # a sample that fails schema parsing). These are infra-level
            # issues — fall back to a layer-derived approximation and
            # log it so the operator sees the noise.
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
            "generation_failure_count": len(generation_failures),
            "generation_failed_sample_ids": sorted(generation_failures),
            "first_failure_distribution": layer_counts,
            "backend_metadata": backend.metadata(),
            "samples_dir": str(args.samples_dir),
            "samples_manifest": str(getattr(args, "samples_manifest", None) or args.samples_manifest)
                if getattr(args, "samples_manifest", None) else None,
        }
        if getattr(args, "manifest_path", None):
            summary["samples_manifest"] = str(args.manifest_path)
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
        help="Directory of D2 multi-turn sample JSON files (used when --samples-manifest is not given)",
    )
    p.add_argument(
        "--samples-manifest", type=Path, default=None,
        help="Authoritative manifest JSON for the P5-02 benchmark subset. "
             "Each entry's SHA256 is verified against the file on disk before "
             "loading. Required for reproducibility of the 90-sample historical "
             "P5-02 benchmark subset.",
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
    samples = None
    if args.samples_manifest:
        samples, manifest_header = _load_samples_from_manifest(
            args.samples_manifest, ROOT, limit=args.limit,
        )
        # Stash manifest path on args so run_one_combination can echo it
        # into the per-run summary (used by the auditor and downstream
        # analysis to verify the subset identity without re-reading the
        # log).
        args.manifest_path = args.samples_manifest
        # Stash the manifest-resolved sample dict (id -> sample) on args
        # so ``run_one_combination()`` can pass the EXACT objects to
        # reward_offline without re-loading from disk. ``args.samples_dir``
        # still points at the manifest's parent for logging only.
        args.samples_by_id = {s["id"]: s for s in samples}
        args.samples_dir = Path(args.samples_manifest).parent
        print(
            f"[p5-04] {len(samples)} samples loaded from manifest "
            f"{args.samples_manifest} (source_commit={manifest_header['source_commit']}, "
            f"aggregate_sha256={manifest_header['source_aggregate_sha256'][:16]}...)",
            flush=True,
        )
    else:
        samples = _load_samples(args.samples_dir, args.limit)
        # Build samples_by_id from the glob-loaded list so reward_offline
        # uses the same objects. This keeps both CLI paths symmetric.
        args.samples_by_id = {s["id"]: s for s in samples}
        args.manifest_path = None
        if not samples:
            print(f"[p5-04] no samples found under {args.samples_dir}", flush=True)
            return 1
        print(
            f"[p5-04] {len(samples)} samples loaded from {args.samples_dir}",
            flush=True,
        )

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


# ---------------------------------------------------------------------------
# Self-tests (run with ``python scripts/eval_backend_comparison.py --selftest``)
# ---------------------------------------------------------------------------
#
# Per the project's task constraint ("代码改动局限于 scripts/eval_backend_comparison.py
# 新文件 + docs/*") this script does NOT add a parallel ``tests/`` file;
# instead it embeds its unit tests as a ``--selftest`` subcommand. Each test
# returns ``None`` on pass and raises ``AssertionError`` on fail. Run via:
#
#     python scripts/eval_backend_comparison.py --selftest
#     wsl -d Ubuntu-22.04 -- bash -c "python3 /mnt/c/.../eval_backend_comparison.py --selftest"
# ---------------------------------------------------------------------------


class _MockBackend:
    """Deterministic backend used by self-tests (mirrors the former MockBackend)."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self.responses = responses or {}
        self.payloads: list[dict[str, Any]] | None = None
        self.setup_calls: list[str] = []
        self.teardown_calls: int = 0
        self.chat_calls: int = 0
        self._setup_done = False
        self._payload_idx = 0

    @property
    def name(self) -> str:
        return "mock"

    def setup(self, model_id: str, **kwargs: Any) -> None:
        self.setup_calls.append(model_id)
        self._setup_done = True
        self._payload_idx = 0

    def chat_generate(self, messages_batch, tools_batch):
        assert self._setup_done, "setup() must be called before chat_generate()"
        self.chat_calls += 1
        out: list[str] = []
        for messages in messages_batch:
            sid = None
            for m in reversed(messages):
                if m.get("role") == "user":
                    sid = m.get("content", "")[:8]
                    break
            out.append(self.responses.get(sid or "", '{"name": "echo", "arguments": {}}'))
        prompt_previews = [(m[-1].get("content", "")[:200] if m else "") for m in messages_batch]
        return out, prompt_previews

    def take_payload(self) -> dict[str, Any] | None:
        """Round-8 fix helper: pop the next pre-built transcript row for
        ``run_one_combination()`` tests. Lets selftests inject rows with
        specific ``sample_id`` / ``layers`` directly.
        """
        if not self.payloads:
            return None
        if self._payload_idx >= len(self.payloads):
            return None
        p = self.payloads[self._payload_idx]
        self._payload_idx += 1
        return p

    def teardown(self) -> None:
        self.teardown_calls += 1
        self._setup_done = False

    def metadata(self):
        return {"backend": "mock", "dtype": "bf16", "max_new_tokens": 32, "device": "cpu"}


def _self_make_sample(idx: int, task_type: str = "echo") -> dict[str, Any]:
    return {
        "id": f"d2-dev-{idx:04d}",
        "metadata": {"task_type": task_type},
        "messages": [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": f"sid{idx:04d}: call the echo tool"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo back the input.",
                    "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
                },
            },
        ],
        "expected_tool_calls": [{"call_id": "c1", "name": "echo", "arguments": {"x": "hi"}}],
        "expected_answer": "echoed: hi",
    }


def _run_selftests() -> int:
    """Run all embedded unit tests; return 0 on success, 1 on failure."""
    import argparse as _ap
    import csv as _csv
    import json as _json

    failures: list[str] = []

    def _expect_impl(name: str, condition: bool, hint: str = "") -> None:
        if condition:
            print(f"  [PASS] {name}", flush=True)
        else:
            failures.append(f"{name}: {hint}")
            print(f"  [FAIL] {name}: {hint}", flush=True)

    # Make _expect a module-level name so the negative-test helper can
    # swap it via globals() without losing the local failures list.
    globals()["_expect"] = _expect_impl

    def _expect_raises(name: str, fn, exc_type: type, hint: str = "") -> None:
        try:
            fn()
        except exc_type:
            print(f"  [PASS] {name}", flush=True)
            return
        except Exception as e:
            failures.append(f"{name}: wrong exception {type(e).__name__}: {e}")
            print(f"  [FAIL] {name}: wrong exception {type(e).__name__}: {e}", flush=True)
            return
        failures.append(f"{name}: no exception raised")
        print(f"  [FAIL] {name}: no exception raised", flush=True)

    print("[selftest] Backend interface + helpers", flush=True)
    # 1. Backend returns (generations, prompt_previews) tuple
    backend = _MockBackend()
    backend.setup("any/model")
    out = backend.chat_generate(
        [[{"role": "user", "content": "sid00000001: hi"}]],
        [None],
    )
    _expect("test_backend_returns_tuple", isinstance(out, tuple) and len(out) == 2)
    gens, prompts = out
    _expect("test_backend_tuple_gens_list", isinstance(gens, list) and len(gens) == 1)
    _expect("test_backend_tuple_prompts_list", isinstance(prompts, list) and len(prompts) == 1)
    backend.teardown()

    # 2. _strip_terminal_assistant
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "expected_answer"},
    ]
    out_msgs = _strip_terminal_assistant(msgs)
    _expect("test_strip_removes_last_assistant", len(out_msgs) == 2 and out_msgs[-1]["role"] == "user")
    msgs2 = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    out_msgs2 = _strip_terminal_assistant(msgs2)
    _expect("test_strip_no_op_when_no_assistant", out_msgs2 == msgs2 or out_msgs2 is msgs2)

    # 3. _slugify
    _expect("test_slugify_basic",
            _slugify("HuggingFaceTB/SmolLM2-360M-Instruct") == "HuggingFaceTB__SmolLM2-360M-Instruct")

    # 4. _aggregate_timing
    t = _aggregate_timing(samples=[{}, {}, {}, {}], prompts=["a", "b", "c", "d"], outputs=["x", "y", "z", "w"], elapsed_s=2.0)
    _expect("test_aggregate_basic", t["samples"] == 4 and t["per_sample_latency_ms"] == 500.0 and t["throughput_samples_per_s"] == 2.0)
    t0 = _aggregate_timing(samples=[], prompts=[], outputs=[], elapsed_s=0.0)
    _expect("test_aggregate_empty", t0["samples"] == 0 and t0["per_sample_latency_ms"] == 0.0 and t0["throughput_samples_per_s"] == 0.0)
    tz = _aggregate_timing(samples=[{}, {}], prompts=["a", "b"], outputs=["x", "y"], elapsed_s=0.0)
    _expect("test_aggregate_zero_elapsed", tz["throughput_samples_per_s"] == 0.0)

    # 5. Backend class metadata
    tb = TransformersBackend(dtype="bf16", max_new_tokens=64, device="cpu")
    _expect("test_transformers_metadata", tb.name == "transformers" and tb.dtype == "bf16" and tb.max_new_tokens == 64)
    vb = VLLMBackend(dtype="bf16", max_new_tokens=64, gpu_memory_utilization=0.5)
    _expect("test_vllm_metadata", vb.name == "vllm" and vb.gpu_memory_utilization == 0.5)

    # 6. DEFAULT_MODELS coverage (5 models from P5-02 §8)
    model_ids = [m for m, _ in DEFAULT_MODELS]
    _expect("test_default_models_count", len(DEFAULT_MODELS) == 5)
    expected = {
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    }
    _expect("test_default_models_p5_02_set", set(model_ids) == expected)
    all_hex = all(
        len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)
        for _, rev in DEFAULT_MODELS
    )
    _expect("test_default_models_revisions_are_40char_hex", all_hex)

    # Round-12: every canonical revision must be exactly 40-char hex and
    # (most importantly) the script must propagate the same revision to
    # both backends through ``run_one_combination()``. We verify the
    # metadata carries it and that setup() recorded the same value.
    sample_pair = list(DEFAULT_MODELS)[:1]
    rev_value = sample_pair[0][1]
    tb = TransformersBackend(dtype="bf16", max_new_tokens=8, device="cpu")
    tb.setup(sample_pair[0][0], revision=rev_value)
    _expect(
        "test_transformers_records_revision",
        tb.revision == rev_value,
        hint=f"got {tb.revision!r}",
    )
    _expect(
        "test_transformers_metadata_has_revision",
        tb.metadata().get("revision") == rev_value,
    )
    tb.teardown()

    # Round-13: VLLMBackend.setup() actually exercises vllm.LLM() and
    # AutoTokenizer.from_pretrained() inside the function body. Stub both
    # so no GPU / network is needed, then verify the revision flows from
    # kwargs through setup() to both loaders and the metadata dict.
    fake_vllm_module = type(sys)("vllm")
    captured_llm_kwargs: dict[str, Any] = {}

    class _FakeLLM:
        def __init__(self, *args, **kwargs):
            captured_llm_kwargs.clear()
            captured_llm_kwargs.update(kwargs)
            self.kwargs = kwargs

    fake_vllm_module.LLM = _FakeLLM  # type: ignore[attr-defined]

    class _FakeSamplingParams:
        def __init__(self, *args, **kwargs):
            pass

    fake_vllm_module.SamplingParams = _FakeSamplingParams  # type: ignore[attr-defined]
    fake_vllm_module.__version__ = "0.27.1"  # type: ignore[attr-defined]
    saved_vllm = sys.modules.get("vllm")
    sys.modules["vllm"] = fake_vllm_module
    try:
        from transformers import AutoTokenizer as _AT
        from unittest.mock import patch as _patch

        captured_tok_kwargs: list[dict[str, Any]] = []

        class _FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 0
            padding_side = "right"

            def __init__(self):
                pass

        def _fake_from_pretrained(model_id, **kwargs):
            captured_tok_kwargs.append(dict(kwargs))
            return _FakeTokenizer()

        with _patch.object(_AT, "from_pretrained", side_effect=_fake_from_pretrained):
            vb = VLLMBackend(dtype="bf16", max_new_tokens=8, gpu_memory_utilization=0.5)
            vb.setup(sample_pair[0][0], revision=rev_value)
            _expect(
                "test_vllm_setup_records_self_revision",
                vb.revision == rev_value,
                hint=f"got {vb.revision!r}",
            )
            _expect(
                "test_vllm_setup_forwards_revision_to_llm",
                captured_llm_kwargs.get("revision") == rev_value,
                hint=f"llm kwargs={captured_llm_kwargs}",
            )
            _expect(
                "test_vllm_setup_forwards_revision_to_tokenizer",
                bool(captured_tok_kwargs)
                and captured_tok_kwargs[0].get("revision") == rev_value,
                hint=f"tok kwargs={captured_tok_kwargs}",
            )
            _expect(
                "test_vllm_metadata_has_revision",
                vb.metadata().get("revision") == rev_value,
            )
    finally:
        if saved_vllm is not None:
            sys.modules["vllm"] = saved_vllm
        else:
            sys.modules.pop("vllm", None)

    import tempfile as _tempfile
    print("[selftest] Manifest materialization", flush=True)
    # Round-12: round-trip via _materialize_manifest_samples + a temp dir
    # that has no sample files. The manifest path is the tracked one.
    manifest_path = Path(
        "docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset.manifest.json"
    )
    manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    with _tempfile.TemporaryDirectory() as td:
        td_root = Path(td)
        # Re-resolve relative paths so the materializer reconstructs files
        # inside the temp tree. The materializer does not care where files land,
        # only that the relative paths exist.
        samples_meta = manifest["samples"]
        target_paths = [(td_root / e["path"]).resolve() for e in samples_meta]
        # Re-anchor to a path under the temp tree, not the original
        # datasets/tool-calling-d2/p5-02-benchmark prefix. We prepend a
        # marker dir so the materializer writes fresh files under it.
        marker = td_root / "p5-02-benchmark"
        samples_meta = [dict(e, path=str((marker / (e["sample_id"] + ".json")).relative_to(td_root))) for e in samples_meta]
        target_paths = [(marker / (e["sample_id"] + ".json")).resolve() for e in samples_meta]
        for tp in target_paths:
            assert not tp.exists(), tp
        _materialize_manifest_samples(
            manifest_path, td_root, samples_meta,
            str(manifest.get("source_commit") or ""),
        )
        all_exist = all(tp.exists() for tp in target_paths)
        _expect("test_materialize_creates_all_files", all_exist)
        # Aggregate SHA must match.
        ids = sorted(s["sample_id"] for s in samples_meta)
        digest = hashlib.sha256()
        for sid in ids:
            digest.update(f"{sid}\n".encode())
        agg = digest.hexdigest()
        _expect(
            "test_materialize_aggregate_sha_matches",
            agg == manifest["source_aggregate_sha256"],
            hint=f"got {agg}",
        )

    print("[selftest] Run pipeline (mocked)", flush=True)
    # 7-9. run_one_combination via mocked factory
    samples = [_self_make_sample(i) for i in range(3)]
    args = _ap.Namespace(
        samples_dir="datasets/tool-calling-d2/dev",
        samples_by_id={s["id"]: s for s in samples},
        manifest_path=None,
        dtype="bf16",
        max_new_tokens=64,
        device="cpu",
        vllm_gpu_mem_util=0.85,
    )
    backend = _MockBackend()
    payload = run_one_combination(
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        backend_name="mock",
        batch_size=1,
        samples=samples,
        args=args,
        backend_factory=lambda **_: backend,
    )
    summary = payload["summary"]
    for k in (
        "model", "backend", "batch_size", "samples", "elapsed_s",
        "per_sample_latency_ms", "throughput_samples_per_s",
        "reward_binary", "reward_layered",
        "parse_success_count", "parse_success_rate",
        "first_failure_distribution", "backend_metadata",
        "samples_manifest",
    ):
        _expect(f"test_run_summary_has_{k}", k in summary)
    _expect("test_run_summary_backend", summary["backend"] == "mock")
    _expect("test_run_summary_samples", summary["samples"] == 3)
    _expect("test_run_rows_count", len(payload["rows"]) == 3)
    for row in payload["rows"]:
        for k in ("sample_id", "task_type", "user_turn", "generated", "generated_preview",
                  "extracted_calls", "layers", "first_failure"):
            _expect(f"test_run_row_has_{k}", k in row)

    # setup/teardown counter
    _expect("test_setup_called_once", backend.setup_calls == ["HuggingFaceTB/SmolLM2-360M-Instruct"])
    _expect("test_teardown_called_once", backend.teardown_calls == 1)
    _expect("test_chat_called_per_sample", backend.chat_calls == 3)  # batch=1, 3 samples

    # Error backend (raises in chat_generate)
    class _ErrorBackend(_MockBackend):
        def chat_generate(self, messages_batch, tools_batch):
            raise RuntimeError("simulated OOM")

    err_payload = run_one_combination(
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        backend_name="mock",
        batch_size=1,
        samples=samples,
        args=args,
        backend_factory=lambda **_: _ErrorBackend(),
    )
    _expect("test_error_backend_still_emits_rows", len(err_payload["rows"]) == 3)
    _expect("test_error_backend_empty_generations", all(r["generated"] == "" for r in err_payload["rows"]))

    print("[selftest] Output writers", flush=True)
    print("[selftest] Manifest materialization", flush=True)
    # Round-12: round-trip via _materialize_manifest_samples + a temp dir
    # that has no sample files. The manifest path is the tracked one.
    manifest_path = Path(
        "docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset.manifest.json"
    )
    manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    with _tempfile.TemporaryDirectory() as td:
        td_root = Path(td)
        # Re-resolve relative paths so the materializer reconstructs files
        # inside the temp tree. The materializer does not care where files land,
        # only that the relative paths exist.
        samples_meta = manifest["samples"]
        target_paths = [(td_root / e["path"]).resolve() for e in samples_meta]
        # Re-anchor to a path under the temp tree, not the original
        # datasets/tool-calling-d2/p5-02-benchmark prefix. We prepend a
        # marker dir so the materializer writes fresh files under it.
        marker = td_root / "p5-02-benchmark"
        # Re-anchor the manifest paths under the temp tree so the
        # materializer reconstructs files inside it (rather than
        # touching the live repo). The loader normalizes paths via
        # ``samples_root / path`` so any prefix works.
        samples_meta = manifest["samples"]
        marker = td_root / "datasets" / "tool-calling-d2" / "p5-02-benchmark"
        samples_meta = [
            dict(e, path=str((marker / (e["sample_id"] + ".json")).relative_to(td_root)))
            for e in samples_meta
        ]
        target_paths = [
            (marker / (e["sample_id"] + ".json")).resolve() for e in samples_meta
        ]
        for tp in target_paths:
            assert not tp.exists(), tp
        _materialize_manifest_samples(
            manifest_path, td_root, samples_meta,
            str(manifest.get("source_commit") or ""),
        )
        all_exist = all(tp.exists() for tp in target_paths)

    print("[selftest] Output writers", flush=True)
    with _tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # write_run_artifact
        run_payload = {"summary": {"model": "m", "backend": "b", "batch_size": 1, "samples": 1}, "rows": []}
        out_path = write_run_artifact(td_path, "m", "b", 1, run_payload)
        _expect("test_artifact_exists", out_path.exists())
        loaded = _json.loads(out_path.read_text(encoding="utf-8"))
        _expect("test_artifact_loadable", loaded["summary"]["model"] == "m")

        # write_aggregate_comparison
        runs = [
            {"summary": {"model": "M", "backend": "transformers", "batch_size": 1,
                         "samples": 3, "elapsed_s": 2.0, "per_sample_latency_ms": 666.6,
                         "throughput_samples_per_s": 1.5, "reward_binary": 0.0,
                         "reward_layered": 0.42, "parse_success_rate": 0.66}},
            {"summary": {"model": "M", "backend": "vllm", "batch_size": 1,
                         "samples": 3, "elapsed_s": 1.0, "per_sample_latency_ms": 333.3,
                         "throughput_samples_per_s": 3.0, "reward_binary": 0.0,
                         "reward_layered": 0.42, "parse_success_rate": 0.66}},
        ]
        cjson, ccsv = write_aggregate_comparison(td_path, runs)
        loaded_json = _json.loads(cjson.read_text(encoding="utf-8"))
        _expect("test_aggregate_json_count", len(loaded_json["runs"]) == 2)
        with ccsv.open(newline="", encoding="utf-8") as fh:
            csv_rows = list(_csv.DictReader(fh))
        _expect("test_aggregate_csv_count", len(csv_rows) == 2)
        _expect("test_aggregate_csv_first_backend", csv_rows[0]["backend"] == "transformers")
        _expect("test_aggregate_csv_second_backend", csv_rows[1]["backend"] == "vllm")

        # Round-14: full-aggregate completeness check. Build a synthetic
        # 20-run set (5 models × 2 backends × 2 batch sizes) and assert
        # the aggregates contain exactly 20 runs, both backends, both
        # batch sizes, and 10 non-null same-model Δ% entries. This is
        # the regression guard the detached auditor asked for after
        # round-13 shipped a vLLM-only aggregate by mistake.
        full_runs = []
        full_models = [
            "HuggingFaceB/SmolLM2-360M", "HuggingFaceB/SmolLM2-1.7B",
            "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
        ]
        for model in full_models:
            for backend in ("transformers", "vllm"):
                for batch in (1, 4):
                    # Make vLLM faster so per-sample latency and
                    # throughput produce non-null Δ%.
                    elapsed = 1.0 if backend == "vllm" else 2.0
                    full_runs.append({
                        "summary": {
                            "model": model, "backend": backend,
                            "batch_size": batch, "samples": 90,
                            "elapsed_s": elapsed,
                            "per_sample_latency_ms": elapsed * 1000 / 90,
                            "throughput_samples_per_s": 90 / elapsed,
                            "reward_binary": 0.0, "reward_layered": 0.4,
                            "parse_success_rate": 1.0,
                        }
                    })
        full_cmp_json, full_cmp_csv = write_aggregate_comparison(td_path, full_runs)
        full_loaded = _json.loads(full_cmp_json.read_text(encoding="utf-8"))
        _expect(
            "test_full_aggregate_runs_count_20",
            len(full_loaded["runs"]) == 20,
            hint=f"got {len(full_loaded['runs'])}",
        )
        full_csv_rows = list(_csv.DictReader(full_cmp_csv.open(newline="", encoding="utf-8")))
        full_backends = sorted({r["backend"] for r in full_csv_rows})
        _expect(
            "test_full_aggregate_both_backends",
            full_backends == ["transformers", "vllm"],
            hint=f"got {full_backends}",
        )
        full_batches = sorted({int(r["batch_size"]) for r in full_csv_rows})
        _expect(
            "test_full_aggregate_both_batch_sizes",
            full_batches == [1, 4],
            hint=f"got {full_batches}",
        )
        full_deltas = compute_delta_percentages(full_runs)
        _expect(
            "test_full_aggregate_delta_entries_10",
            len(full_deltas) == 10,
            hint=f"got {len(full_deltas)}",
        )
        full_non_null_lat = sum(
            1 for v in full_deltas.values()
            if v.get("per_sample_latency_ms") is not None
        )
        _expect(
            "test_full_aggregate_delta_latency_non_null",
            full_non_null_lat == 10,
            hint=f"got {full_non_null_lat}",
        )
        full_non_null_thr = sum(
            1 for v in full_deltas.values()
            if v.get("throughput_samples_per_s") is not None
        )
        _expect(
            "test_full_aggregate_delta_throughput_non_null",
            full_non_null_thr == 10,
            hint=f"got {full_non_null_thr}",
        )

        # Empty runs
        cjson_e, _ = write_aggregate_comparison(td_path, [])
        _expect("test_aggregate_empty_json", _json.loads(cjson_e.read_text(encoding="utf-8"))["runs"] == [])

# Round-16: section-aware, row/column-specific README/CSV binding.
        # The round-16 detached auditor found that the round-15 guard was
        # only a global numeric-substring check and did not bind each
        # README cell to its specific (model, backend, batch) CSV row.
        # This new implementation:
        #   1. parses each markdown table under its `### heading` section,
        #   2. maps row labels to known model short names,
        #   3. verifies each cell matches the expected value from
        #      comparison.csv (per-axis) or comparison_delta.csv (per-delta).
        # Plus it runs NEGATIVE tests that mutate a temp copy of the
        # README and verifies the guard catches the unsynchronized change.

        csv_path = Path("artifacts/p5-04-backend-comparison/full/comparison.csv")
        delta_csv_path = Path("artifacts/p5-04-backend-comparison/full/comparison_delta.csv")
        readme_path = Path("docs/experiments/p5-04-backend-comparison/README.md")
        _expect("test_readme_csv_sync_csv_exists", csv_path.exists())
        _expect("test_readme_csv_sync_delta_exists", delta_csv_path.exists())
        _expect("test_readme_csv_sync_readme_exists", readme_path.exists())

        # Index comparison.csv rows by (model_id, backend, batch_size) for
        # row/column-specific cell binding.
        csv_rows = list(_csv.DictReader(csv_path.open(encoding="utf-8")))
        csv_by_key = {}
        for r in csv_rows:
            key = (r["model"], r["backend"], r["batch_size"])
            csv_by_key[key] = r
        # Same for the delta CSV: key by (model_id, batch_size) -> row.
        delta_rows = list(_csv.DictReader(delta_csv_path.open(encoding="utf-8")))
        delta_by_key = {}
        for r in delta_rows:
            mb = r["model_batch"]
            assert mb.endswith("__b1") or mb.endswith("__b4"), mb
            model_id, batch_token = mb.rsplit("__", 1)
            batch_size = batch_token.lstrip("b")
            delta_by_key[(model_id, batch_size)] = r

        MODEL_SHORT_TO_ID = {
            "SmolLM2-360M-Instruct": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "SmolLM2-1.7B-Instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
        }

        # Markdown section/table parser.
        def _parse_readme_tables(readme_text):
            sections = []
            current_heading = None
            current_table_lines = []
            for raw_line in readme_text.splitlines():
                if raw_line.startswith("###"):
                    if current_table_lines:
                        sections.append((current_heading, current_table_lines))
                        current_table_lines = []
                    current_heading = raw_line.lstrip("#").strip()
                    continue
                if raw_line.startswith("|"):
                    current_table_lines.append(raw_line)
                else:
                    if current_table_lines:
                        sections.append((current_heading, current_table_lines))
                        current_table_lines = []
            if current_table_lines:
                sections.append((current_heading, current_table_lines))
            parsed = []
            for heading, lines in sections:
                if len(lines) < 3:
                    continue
                header = [c.strip() for c in lines[0].strip("|").split("|")]
                sep = [c.strip() for c in lines[1].strip("|").split("|")]
                if not all(set(c) <= set("-:") and c for c in sep):
                    continue
                rows = []
                for line in lines[2:]:
                    cells = [c.strip() for c in line.strip("|").split("|")]
                    if len(cells) != len(header):
                        continue
                    rows.append(cells)
                parsed.append((heading, header, rows))
            return parsed

        def _format_cell(value, decimals, signed=False):
            if signed:
                return f"{value:+.{decimals}f}"
            return f"{value:.{decimals}f}"

        def _bind_axis_table(heading_substring, csv_field, decimals,
                              parse_header_cell, transform):
            tables = _parse_readme_tables(readme_path.read_text(encoding="utf-8"))
            target_tables = [
                t for t in tables
                if t[0] is not None and heading_substring.lower() in t[0].lower()
            ]
            _expect(
                f"test_readme_section_exists_{heading_substring.replace(' ', '_')[:20]}",
                len(target_tables) >= 1,
                hint=f"no section with heading containing {heading_substring!r}",
            )
            for heading, header, rows in target_tables:
                col_specs = [parse_header_cell(h) for h in header]
                for row in rows:
                    label = row[0]
                    if label not in MODEL_SHORT_TO_ID:
                        continue
                    model_id = MODEL_SHORT_TO_ID[label]
                    for col_idx, spec in enumerate(col_specs):
                        if spec[0] != "axis":
                            continue
                        backend, batch_size = spec[1], spec[2]
                        cell_raw = row[col_idx]
                        expected_raw = csv_by_key[
                            (model_id, backend, batch_size)
                        ][csv_field]
                        expected_value = transform(float(expected_raw))
                        expected_str = _format_cell(expected_value, decimals)
                        cell_normalized = cell_raw.rstrip("%").strip()
                        test_name = (
                            f"test_readme_bind_{heading_substring.replace(' ', '_')[:20]}"
                            f"_{label}_{backend}_b{batch_size}"
                        )
                        _expect(
                            test_name,
                            cell_normalized == expected_str,
                            hint=(
                                f"heading={heading!r} cell={cell_raw!r} "
                                f"expected={expected_str!r} "
                                f"(csv_field={csv_field} value={expected_raw})"
                            ),
                        )

        def _parse_lat_or_thr(h):
            if h in ("模型", "model"):
                return ("label",)
            for backend in ("transformers", "vllm"):
                for batch in ("1", "4"):
                    if h == f"{backend} b={batch}":
                        return ("axis", backend, batch)
            return ("label",)

        _bind_axis_table(
            "Latency (ms",
            "per_sample_latency_ms",
            1,
            _parse_lat_or_thr,
            lambda x: x,
        )
        _bind_axis_table(
            "Throughput (samples",
            "throughput_samples_per_s",
            2,
            _parse_lat_or_thr,
            lambda x: x,
        )
        _bind_axis_table(
            "reward_binary",
            "reward_binary",
            4,
            _parse_lat_or_thr,
            lambda x: x,
        )
        _bind_axis_table(
            "reward_layered",
            "reward_layered",
            4,
            _parse_lat_or_thr,
            lambda x: x,
        )

        def _bind_delta_table(heading_substring, csv_field):
            tables = _parse_readme_tables(readme_path.read_text(encoding="utf-8"))
            target_tables = [
                t for t in tables
                if t[0] is not None and heading_substring.lower() in t[0].lower()
            ]
            _expect(
                f"test_readme_section_exists_delta_{heading_substring.replace(' ', '_')[:20]}",
                len(target_tables) >= 1,
                hint=f"no section with heading containing {heading_substring!r}",
            )
            for heading, header, rows in target_tables:
                col_batch = {}
                for col_idx, h in enumerate(header):
                    if h in ("模型", "model"):
                        continue
                    if h in ("b=1", "b=2", "b=4"):
                        col_batch[col_idx] = h.split("=")[1]
                for row in rows:
                    label = row[0]
                    if label not in MODEL_SHORT_TO_ID:
                        continue
                    model_id = MODEL_SHORT_TO_ID[label]
                    for col_idx, batch_size in col_batch.items():
                        cell_raw = row[col_idx]
                        key = (model_id, batch_size)
                        if key not in delta_by_key:
                            continue
                        expected_raw = delta_by_key[key][csv_field]
                        expected_str = _format_cell(
                            float(expected_raw), 2, signed=True
                        )
                        cell_normalized = cell_raw.rstrip("%").strip()
                        test_name = (
                            f"test_readme_bind_delta_{heading_substring.replace(' ', '_')[:20]}"
                            f"_{label}_b{batch_size}"
                        )
                        cell_unsigned = expected_str.lstrip("+")
                        _expect(
                            test_name,
                            cell_normalized in (expected_str, cell_unsigned),
                            hint=(
                                f"heading={heading!r} cell={cell_raw!r} "
                                f"expected={expected_str!r} (csv_field={csv_field})"
                            ),
                        )

        _bind_delta_table("Latency \u0394%", "delta_per_sample_latency_ms_pct")
        _bind_delta_table("Throughput \u0394%", "delta_throughput_samples_per_s_pct")
        _bind_delta_table("Reward layered \u0394%", "delta_reward_layered_pct")

        # Top-5 fastest table: bind by (model, vllm, b=4).
        tables = _parse_readme_tables(readme_path.read_text(encoding="utf-8"))
        top5_tables = [t for t in tables if t[0] is not None and "\u6700\u5feb" in t[0]]
        _expect("test_readme_section_exists_top5", len(top5_tables) >= 1)
        if top5_tables:
            for heading, header, rows in top5_tables:
                thr_col = None
                lat_col = None
                for col_idx, h in enumerate(header):
                    if "throughput" in h.lower() or "samples/s" in h.lower():
                        thr_col = col_idx
                    if "latency" in h.lower() or "ms/sample" in h.lower():
                        lat_col = col_idx
                _expect("test_readme_top5_has_throughput_col", thr_col is not None)
                _expect("test_readme_top5_has_latency_col", lat_col is not None)
                for row in rows:
                    label = row[0]
                    short_name = label.split(" vLLM")[0].strip()
                    if short_name not in MODEL_SHORT_TO_ID:
                        continue
                    model_id = MODEL_SHORT_TO_ID[short_name]
                    csv_row = csv_by_key[(model_id, "vllm", "4")]
                    expected_thr = _format_cell(
                        float(csv_row["throughput_samples_per_s"]), 2
                    )
                    expected_lat = _format_cell(
                        float(csv_row["per_sample_latency_ms"]), 1
                    )
                    _expect(
                        f"test_readme_top5_thr_{short_name}",
                        row[thr_col].strip() == expected_thr,
                        hint=(
                            f"label={label!r} cell={row[thr_col]!r} "
                            f"expected={expected_thr!r}"
                        ),
                    )
                    _expect(
                        f"test_readme_top5_lat_{short_name}",
                        row[lat_col].strip() == expected_lat,
                        hint=(
                            f"label={label!r} cell={row[lat_col]!r} "
                            f"expected={expected_lat!r}"
                        ),
                    )

        # NEGATIVE TESTS — prove unsynchronized mutations fail.
        def _run_bindings_on_text(mutated_text):
            captured = []
            def _capture(name, cond, hint=""):
                if not cond:
                    captured.append((name, hint))
            import builtins as _bi
            original_print = _bi.print
            _bi.print = lambda *a, **kw: None
            # The binding helpers read `readme_path.read_text(encoding="utf-8")`
            # internally; monkey-patch the class method so the call resolves
            # to our mutated version.
            orig_read = type(readme_path).read_text
            def fake_read(self, encoding="utf-8"):
                return mutated_text
            type(readme_path).read_text = fake_read
            orig_expect = globals().get("_expect")
            globals()["_expect"] = _capture
            try:
                _bind_axis_table(
                    "Latency (ms",
                    "per_sample_latency_ms",
                    1,
                    _parse_lat_or_thr,
                    lambda x: x,
                )
                _bind_axis_table(
                    "Throughput (samples",
                    "throughput_samples_per_s",
                    2,
                    _parse_lat_or_thr,
                    lambda x: x,
                )
                _bind_axis_table(
                    "reward_binary",
                    "reward_binary",
                    4,
                    _parse_lat_or_thr,
                    lambda x: x,
                )
                _bind_axis_table(
                    "reward_layered",
                    "reward_layered",
                    4,
                    _parse_lat_or_thr,
                    lambda x: x,
                )
                _bind_delta_table("Latency \u0394%", "delta_per_sample_latency_ms_pct")
                _bind_delta_table("Throughput \u0394%", "delta_throughput_samples_per_s_pct")
                _bind_delta_table("Reward layered \u0394%", "delta_reward_layered_pct")
            finally:
                type(readme_path).read_text = orig_read
                globals()["_expect"] = orig_expect
                _bi.print = original_print
            return captured

        original_readme = readme_path.read_text(encoding="utf-8")
        # Mutation 1: SmolLM2-360M transformers b=1 latency.
        mutated_1 = original_readme.replace(
            "| SmolLM2-360M-Instruct | 1555.3 | 488.4 | 1001.9 | 285.4 |",
            "| SmolLM2-360M-Instruct | 9999.9 | 488.4 | 1001.9 | 285.4 |",
        )
        assert mutated_1 != original_readme
        failures_1 = _run_bindings_on_text(mutated_1)
        _expect(
            "test_readme_negative_latency_mutation_caught",
            any("SmolLM2-360M-Instruct_transformers_b1" in n for n, _ in failures_1),
            hint=f"failures={[n for n,_ in failures_1 if 'SmolLM2' in n]}",
        )

        # Mutation 2: SmolLM2-1.7B vLLM b=4 throughput.
        mutated_2 = original_readme.replace(
            "| SmolLM2-1.7B-Instruct | 1119.7 | 381.1 | 677.3 | 196.0 |",
            "| SmolLM2-1.7B-Instruct | 1119.7 | 381.1 | 677.3 | 999.99 |",
        )
        assert mutated_2 != original_readme
        failures_2 = _run_bindings_on_text(mutated_2)
        _expect(
            "test_readme_negative_throughput_mutation_caught",
            any("SmolLM2-1.7B-Instruct_vllm_b4" in n for n, _ in failures_2),
            hint=f"failures={[n for n,_ in failures_2 if 'SmolLM2-1.7B' in n]}",
        )

        # Mutation 3: Qwen2.5-0.5B vLLM b=1 reward_layered.
        mutated_3 = original_readme.replace(
            "| Qwen2.5-0.5B-Instruct | 0.3676 | 0.3662 | 0.3601 | 0.3601 |",
            "| Qwen2.5-0.5B-Instruct | 0.3676 | 0.3662 | 0.5000 | 0.3601 |",
        )
        assert mutated_3 != original_readme
        failures_3 = _run_bindings_on_text(mutated_3)
        _expect(
            "test_readme_negative_reward_layered_mutation_caught",
            any("Qwen2.5-0.5B-Instruct_vllm_b1" in n for n, _ in failures_3),
            hint=f"failures={[n for n,_ in failures_3 if 'Qwen2.5-0.5B' in n]}",
        )

        # Mutation 4: Qwen2.5-0.5B latency delta b=1.
        mutated_4 = original_readme.replace(
            "| Qwen2.5-0.5B-Instruct | -41.48% | -47.32% |",
            "| Qwen2.5-0.5B-Instruct | -99.99% | -47.32% |",
        )
        assert mutated_4 != original_readme
        failures_4 = _run_bindings_on_text(mutated_4)
        _expect(
            "test_readme_negative_delta_mutation_caught",
            any("Qwen2.5-0.5B-Instruct_b1" in n for n, _ in failures_4),
            hint=f"failures={[n for n,_ in failures_4 if 'Qwen2.5-0.5B' in n]}",
        )

    print("[selftest] CLI", flush=True)
    # argparse defaults
    a0 = _build_argparser().parse_args([])
    _expect("test_arg_models_default_none", a0.models is None)
    _expect("test_arg_backends_default", a0.backends == list(DEFAULT_BACKENDS))
    _expect("test_arg_batch_sizes_default", a0.batch_sizes == list(DEFAULT_BATCH_SIZES))
    _expect("test_arg_samples_dir_default", a0.samples_dir == Path("datasets/tool-calling-d2/dev"))
    _expect("test_arg_output_dir_default", a0.output_dir == Path("artifacts/p5-04-backend-comparison"))
    _expect("test_arg_limit_default", a0.limit == 0)
    _expect("test_arg_max_new_tokens_default", a0.max_new_tokens == 256)
    _expect("test_arg_dtype_default", a0.dtype == "bf16")
    _expect("test_arg_device_default", a0.device == "auto")

    a1 = _build_argparser().parse_args([
        "--models", "Qwen/Qwen2.5-0.5B-Instruct",
        "--backends", "vllm",
        "--batch-sizes", "1", "4",
        "--limit", "10",
        "--dtype", "fp16",
    ])
    _expect("test_arg_models_override", a1.models == ["Qwen/Qwen2.5-0.5B-Instruct"])
    _expect("test_arg_backends_override", a1.backends == ["vllm"])
    _expect("test_arg_batch_sizes_override", a1.batch_sizes == [1, 4])
    _expect("test_arg_limit_override", a1.limit == 10)
    _expect("test_arg_dtype_override", a1.dtype == "fp16")

    _expect_raises("test_arg_help_exits_cleanly",
                   lambda: _build_argparser().parse_args(["--help"]),
                   SystemExit)

    print("[selftest] Reward computation (round-8 fix)", flush=True)
    # round-8 root-cause fix: reward must come from args.samples_by_id,
    # not from reward_offline.load_samples(args.samples_dir) — otherwise
    # manifest parent dir (no sample files) silently yields reward=0.

    def _make_sample(idx: int, task_type: str = "tool_not_available") -> dict[str, Any]:
        return {
            "id": f"s{idx}",
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": f"sid{idx}: call the echo tool"},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo back the input.",
                    "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
                },
            }],
            "expected_tool_calls": [{"call_id": "c1", "name": "echo", "arguments": {"x": "hi"}}],
            "expected_answer": "echoed: hi",
            "metadata": {"task_type": task_type, "data_version": "D2", "task_template": "echo", "created_at": "2026-01-01", "validation": {"schema_valid": True, "tool_execution_valid": True, "answer_valid": True, "quality_passed": True}, "pipeline_version": "test/v1"},
        }

    def _build_args(samples_by_id: dict[str, Any], samples_dir: Path) -> _ap.Namespace:
        return _ap.Namespace(
            samples_manifest=Path("docs/experiments/p5-04-backend-comparison/p5-02-benchmark-subset.manifest.json"),
            samples_dir=samples_dir,
            samples_by_id=samples_by_id, manifest_path=None,
            dtype="bf16", max_new_tokens=8, vllm_gpu_mem_util=0.85,
            output_dir=Path("."),
        )

    def _make_row(sid: str) -> dict[str, Any]:
        return {
            "sample_id": sid, "user_turn": "u", "generated": "",
            "first_failure": None,
            "layers": {k: True for k in (
                "parse_success", "schema_valid", "tool_name_correct", "argument_value_correct",
                "call_plan_matches", "execution_success", "result_grounded", "final_answer_correct",
            )},
        }

    # Case A: samples_by_id populated correctly — reward must be > 0
    samples_a = [_make_sample(1)]
    backend_a = _MockBackend()
    # Provide the row for sample id "s1" via the mock payload queue so
    # run_one_combination() builds matching transcripts and the reward
    # loop has something to evaluate.
    backend_a.payloads = [_make_row("s1")]
    args_a = _build_args({s["id"]: s for s in samples_a}, samples_dir=Path("/tmp/p5-04-selftest-empty"))
    out_a = run_one_combination(
        model_id="m", backend_name="transformers", batch_size=1,
        samples=samples_a, args=args_a,
        backend_factory=lambda **kw: backend_a,
    )
    _expect(
        "test_round8_reward_uses_samples_by_id",
        out_a["summary"]["reward_layered"] > 0.0,
        f"got {out_a['summary']['reward_layered']} (manifest-parent-dir bug)",
    )

    # Case B: empty samples_by_id must raise — never silently zero
    args_b = _build_args({}, samples_dir=Path("/tmp/p5-04-selftest-empty"))
    backend_b = _MockBackend()
    _expect_raises(
        "test_round8_reward_raises_when_samples_by_id_empty",
        lambda: run_one_combination(
            model_id="m", backend_name="transformers", batch_size=1,
            samples=[], args=args_b,
            backend_factory=lambda **kw: backend_b,
        ),
        RuntimeError,
        hint="must raise RuntimeError, not silently return reward=0",
    )

    # Case C: mismatch sample_ids — no overlap must raise
    samples_c = [_make_sample(1)]
    backend_c = _MockBackend()
    backend_c.payloads = [_make_row("other")]  # different id
    args_c = _build_args({s["id"]: s for s in samples_c}, samples_dir=Path("/tmp/p5-04-selftest-empty"))
    _expect_raises(
        "test_round8_reward_raises_on_no_overlap",
        lambda: run_one_combination(
            model_id="m", backend_name="transformers", batch_size=1,
            samples=samples_c, args=args_c,
            backend_factory=lambda **kw: backend_c,
        ),
        RuntimeError,
        hint="no sample/transcript id overlap must raise, not silently zero",
    )

    print("[selftest] Same-model Δ% computation", flush=True)
    # basic signs: vLLM faster than transformers → latency Δ% < 0, throughput Δ% > 0
    runs_basic = [
        {"summary": {"model": "M", "backend": "transformers", "batch_size": 1,
                     "per_sample_latency_ms": 1000.0, "throughput_samples_per_s": 1.0,
                     "reward_binary": 0.0, "reward_layered": 0.4, "parse_success_rate": 0.5}},
        {"summary": {"model": "M", "backend": "vllm", "batch_size": 1,
                     "per_sample_latency_ms": 500.0, "throughput_samples_per_s": 2.0,
                     "reward_binary": 0.0, "reward_layered": 0.4, "parse_success_rate": 0.5}},
    ]
    deltas = compute_delta_percentages(runs_basic)
    _expect("test_delta_basic_key_present", "M__b1" in deltas)
    _expect("test_delta_latency_negative_when_vllm_faster",
            deltas["M__b1"]["per_sample_latency_ms"] is not None
            and abs(deltas["M__b1"]["per_sample_latency_ms"] - (-50.0)) < 1e-6)
    _expect("test_delta_throughput_positive_when_vllm_faster",
            deltas["M__b1"]["throughput_samples_per_s"] is not None
            and abs(deltas["M__b1"]["throughput_samples_per_s"] - 100.0) < 1e-6)
    _expect("test_delta_reward_identical_returns_none",
            deltas["M__b1"]["reward_binary"] is None)

    # groups by batch
    runs_grouped = [
        {"summary": {"model": "M", "backend": "transformers", "batch_size": 1,
                     "per_sample_latency_ms": 1000.0, "throughput_samples_per_s": 1.0,
                     "reward_binary": 0.0, "reward_layered": 0.0, "parse_success_rate": 0.0}},
        {"summary": {"model": "M", "backend": "transformers", "batch_size": 4,
                     "per_sample_latency_ms": 400.0, "throughput_samples_per_s": 10.0,
                     "reward_binary": 0.0, "reward_layered": 0.0, "parse_success_rate": 0.0}},
        {"summary": {"model": "M", "backend": "vllm", "batch_size": 1,
                     "per_sample_latency_ms": 900.0, "throughput_samples_per_s": 1.1,
                     "reward_binary": 0.0, "reward_layered": 0.0, "parse_success_rate": 0.0}},
        {"summary": {"model": "M", "backend": "vllm", "batch_size": 4,
                     "per_sample_latency_ms": 200.0, "throughput_samples_per_s": 20.0,
                     "reward_binary": 0.0, "reward_layered": 0.0, "parse_success_rate": 0.0}},
    ]
    dg = compute_delta_percentages(runs_grouped)
    _expect("test_delta_groups_by_batch_keys",
            "M__b1" in dg and "M__b4" in dg)
    _expect("test_delta_b1_latency",
            dg["M__b1"]["per_sample_latency_ms"] is not None
            and abs(dg["M__b1"]["per_sample_latency_ms"] - (-10.0)) < 1e-6)
    _expect("test_delta_b4_throughput",
            dg["M__b4"]["throughput_samples_per_s"] is not None
            and abs(dg["M__b4"]["throughput_samples_per_s"] - 100.0) < 1e-6)

    # missing backend returns None
    runs_missing = [
        {"summary": {"model": "M", "backend": "transformers", "batch_size": 1,
                     "per_sample_latency_ms": 100.0, "throughput_samples_per_s": 10.0,
                     "reward_binary": 0.0, "reward_layered": 0.4, "parse_success_rate": 0.5}},
    ]
    dm = compute_delta_percentages(runs_missing)
    _expect("test_delta_missing_backend_returns_none",
            "M__b1" in dm and all(v is None for v in dm["M__b1"].values()))

    # zero baseline returns None
    runs_zero = [
        {"summary": {"model": "M", "backend": "transformers", "batch_size": 1,
                     "per_sample_latency_ms": 0.0, "throughput_samples_per_s": 0.0,
                     "reward_binary": 0.0, "reward_layered": 0.0, "parse_success_rate": 0.0}},
        {"summary": {"model": "M", "backend": "vllm", "batch_size": 1,
                     "per_sample_latency_ms": 100.0, "throughput_samples_per_s": 10.0,
                     "reward_binary": 0.0, "reward_layered": 0.0, "parse_success_rate": 0.0}},
    ]
    dz = compute_delta_percentages(runs_zero)
    _expect("test_delta_zero_baseline_returns_none",
            "M__b1" in dz and all(v is None for v in dz["M__b1"].values()))

    # write_delta_csv
    with _tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        deltas_csv_input = {
            "M__b1": {"per_sample_latency_ms": -50.0, "throughput_samples_per_s": 100.0,
                      "reward_binary": 0.0, "reward_layered": 5.0, "parse_success_rate": -2.5},
            "M__b4": {"per_sample_latency_ms": None, "throughput_samples_per_s": None,
                      "reward_binary": None, "reward_layered": None, "parse_success_rate": None},
        }
        delta_csv_path = write_delta_csv(td_path, deltas_csv_input)
        _expect("test_delta_csv_exists", delta_csv_path.exists())
        with delta_csv_path.open(newline="", encoding="utf-8") as fh:
            csv_delta_rows = list(_csv.DictReader(fh))
        _expect("test_delta_csv_count", len(csv_delta_rows) == 2)
        _expect("test_delta_csv_first_row_key", csv_delta_rows[0]["model_batch"] == "M__b1")
        _expect("test_delta_csv_first_row_value",
                csv_delta_rows[0]["delta_per_sample_latency_ms_pct"] == "-50.00")
        _expect("test_delta_csv_missing_value_empty",
                csv_delta_rows[1]["delta_per_sample_latency_ms_pct"] == "")

    print("", flush=True)
    if failures:
        print(f"[selftest] {len(failures)} FAILURE(S):", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        return 1
    # Count tests passed (rough count of [PASS] lines printed above).
    print("[selftest] all tests PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main() if "--selftest" not in sys.argv else _run_selftests())
