"""Unit tests for ``scripts/eval_backend_comparison.py``.

These tests cover the comparison CLI without invoking transformers / vLLM
or any GPU. A ``MockBackend`` provides deterministic, fast generations
so the tests run in seconds and need no model downloads.

Tests are grouped into four areas:
1. Backend interface + per-backend smoke (mocked)
2. Run pipeline + reward computation (mocked; reward_offline fall-back path)
3. Aggregate comparison writer (CSV / JSON output shape)
4. CLI smoke (--help, defaults)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from scripts.eval_backend_comparison import (  # noqa: E402
    Backend,
    DEFAULT_BACKENDS,
    DEFAULT_BATCH_SIZES,
    DEFAULT_MODELS,
    TransformersBackend,
    VLLMBackend,
    _aggregate_timing,
    _build_argparser,
    _slugify,
    _strip_terminal_assistant,
    write_aggregate_comparison,
    write_run_artifact,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockBackend:
    """Deterministic backend used by tests.

    The ``responses`` list maps a sample ``id`` to the generation string
    the backend should produce. If the id isn't in the map, a fixed
    placeholder is returned.
    """

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self.responses = responses or {}
        self.setup_calls: list[str] = []
        self.teardown_calls: int = 0
        self.chat_calls: int = 0
        self._setup_done = False

    @property
    def name(self) -> str:
        return "mock"

    def setup(self, model_id: str, **kwargs: Any) -> None:
        self.setup_calls.append(model_id)
        self._setup_done = True

    def chat_generate(
        self,
        messages_batch: list[list[dict[str, Any]]],
        tools_batch: list[list[dict[str, Any]] | None],
    ) -> list[str]:
        assert self._setup_done, "setup() must be called before chat_generate()"
        self.chat_calls += 1
        out: list[str] = []
        for messages in messages_batch:
            sid = None
            # Use the last user message's content as the synthetic id.
            for m in reversed(messages):
                if m.get("role") == "user":
                    sid = m.get("content", "")[:8]
                    break
            out.append(self.responses.get(sid or "", '{"name": "echo", "arguments": {}}'))
        return out

    def teardown(self) -> None:
        self.teardown_calls += 1
        self._setup_done = False

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "mock",
            "dtype": "bf16",
            "max_new_tokens": 32,
            "device": "cpu",
        }


def _make_sample(idx: int, task_type: str = "echo") -> dict[str, Any]:
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


def _make_samples(n: int) -> list[dict[str, Any]]:
    return [_make_sample(i) for i in range(n)]


# ---------------------------------------------------------------------------
# 1. Strip terminal assistant + slugify
# ---------------------------------------------------------------------------


def test_strip_terminal_assistant_removes_only_assistant() -> None:
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "expected_answer"},
    ]
    out = _strip_terminal_assistant(msgs)
    assert len(out) == 2
    assert out[-1]["role"] == "user"


def test_strip_terminal_assistant_no_op_when_last_is_user() -> None:
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    out = _strip_terminal_assistant(msgs)
    assert out is msgs or out == msgs


def test_slugify_basic() -> None:
    assert _slugify("HuggingFaceTB/SmolLM2-360M-Instruct") == "HuggingFaceTB__SmolLM2-360M-Instruct"
    assert _slugify("Qwen/Qwen2.5-3B-Instruct") == "Qwen__Qwen2.5-3B-Instruct"


# ---------------------------------------------------------------------------
# 2. Timing aggregation
# ---------------------------------------------------------------------------


def test_aggregate_timing_basic() -> None:
    timings = _aggregate_timing(
        samples=[{}, {}, {}, {}],
        prompts=["a", "b", "c", "d"],
        outputs=["x", "y", "z", "w"],
        elapsed_s=2.0,
    )
    assert timings["samples"] == 4
    assert timings["elapsed_s"] == 2.0
    assert timings["per_sample_latency_ms"] == 500.0
    assert timings["throughput_samples_per_s"] == 2.0


def test_aggregate_timing_empty_input() -> None:
    timings = _aggregate_timing(samples=[], prompts=[], outputs=[], elapsed_s=0.0)
    assert timings["samples"] == 0
    assert timings["per_sample_latency_ms"] == 0.0
    assert timings["throughput_samples_per_s"] == 0.0


def test_aggregate_timing_zero_elapsed() -> None:
    timings = _aggregate_timing(
        samples=[{}, {}], prompts=["a", "b"], outputs=["x", "y"], elapsed_s=0.0,
    )
    assert timings["throughput_samples_per_s"] == 0.0


# ---------------------------------------------------------------------------
# 3. Backend interface + smoke (mocked; no GPU)
# ---------------------------------------------------------------------------


def test_backend_interface_is_typed() -> None:
    backend: Backend = MockBackend()
    assert backend.name == "mock"
    backend.setup("any/model")
    out = backend.chat_generate(
        [[{"role": "user", "content": "sid00000001: hi"}]],
        [None],
    )
    assert isinstance(out, list) and len(out) == 1
    backend.teardown()


def test_transformers_backend_class_metadata() -> None:
    b = TransformersBackend(dtype="bf16", max_new_tokens=64, device="cpu")
    assert b.name == "transformers"
    assert b.dtype == "bf16"
    assert b.max_new_tokens == 64


def test_vllm_backend_class_metadata() -> None:
    b = VLLMBackend(dtype="bf16", max_new_tokens=64, gpu_memory_utilization=0.5)
    assert b.name == "vllm"
    assert b.gpu_memory_utilization == 0.5


def test_default_models_and_backends_cover_p5_02_set() -> None:
    assert len(DEFAULT_MODELS) == 5
    assert set(DEFAULT_BACKENDS) == {"transformers", "vllm"}
    # The 5 canonical models from P5-02 §8:
    assert "HuggingFaceTB/SmolLM2-360M-Instruct" in DEFAULT_MODELS
    assert "HuggingFaceTB/SmolLM2-1.7B-Instruct" in DEFAULT_MODELS
    assert "Qwen/Qwen2.5-0.5B-Instruct" in DEFAULT_MODELS
    assert "Qwen/Qwen2.5-1.5B-Instruct" in DEFAULT_MODELS
    assert "Qwen/Qwen2.5-3B-Instruct" in DEFAULT_MODELS


# ---------------------------------------------------------------------------
# 4. Per-run payload schema (mocked, no GPU)
# ---------------------------------------------------------------------------


def test_run_one_combination_uses_factory_hook(tmp_path: Path) -> None:
    """Verify the backend_factory hook + reward-offline fallback path."""
    import argparse

    from scripts.eval_backend_comparison import run_one_combination

    samples = _make_samples(3)
    args = argparse.Namespace(
        samples_dir=str(ROOT / "datasets/tool-calling-d2/dev"),
        dtype="bf16",
        max_new_tokens=64,
        device="cpu",
        vllm_gpu_mem_util=0.85,
    )
    backend = MockBackend()
    payload = run_one_combination(
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        backend_name="mock",
        batch_size=1,
        samples=samples,
        args=args,
        backend_factory=lambda **_: backend,
    )
    summary = payload["summary"]
    # Summary fields the comparison CSV writer relies on
    for k in (
        "model", "backend", "batch_size", "samples", "elapsed_s",
        "per_sample_latency_ms", "throughput_samples_per_s",
        "reward_binary", "reward_layered",
        "parse_success_count", "parse_success_rate",
        "first_failure_distribution", "backend_metadata",
    ):
        assert k in summary, f"missing summary field {k}"
    assert summary["backend"] == "mock"
    assert summary["samples"] == 3
    # Rows must have P5-02 / eval_sft_tool compatible schema
    assert len(payload["rows"]) == 3
    for row in payload["rows"]:
        for k in (
            "sample_id", "task_type", "user_turn", "generated",
            "generated_preview", "extracted_calls", "layers", "first_failure",
        ):
            assert k in row, f"missing row field {k}"


def test_run_one_combination_records_setup_and_teardown(tmp_path: Path) -> None:
    """Verify setup is called once and teardown is called once per run."""
    import argparse

    from scripts.eval_backend_comparison import run_one_combination

    samples = _make_samples(2)
    args = argparse.Namespace(
        samples_dir=str(ROOT / "datasets/tool-calling-d2/dev"),
        dtype="bf16",
        max_new_tokens=64,
        device="cpu",
        vllm_gpu_mem_util=0.85,
    )
    backend = MockBackend()
    run_one_combination(
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        backend_name="mock",
        batch_size=1,
        samples=samples,
        args=args,
        backend_factory=lambda **_: backend,
    )
    assert backend.setup_calls == ["HuggingFaceTB/SmolLM2-360M-Instruct"]
    assert backend.teardown_calls == 1
    assert backend.chat_calls == 2  # one call per sample (batch_size=1)


def test_run_one_combination_handles_generation_error(tmp_path: Path) -> None:
    """If a batch raises, we still emit one row per sample with empty generation."""
    import argparse

    from scripts.eval_backend_comparison import run_one_combination

    class ErrorBackend(MockBackend):
        def chat_generate(self, messages_batch, tools_batch):  # type: ignore[override]
            raise RuntimeError("simulated OOM")

    samples = _make_samples(2)
    args = argparse.Namespace(
        samples_dir=str(ROOT / "datasets/tool-calling-d2/dev"),
        dtype="bf16",
        max_new_tokens=64,
        device="cpu",
        vllm_gpu_mem_util=0.85,
    )
    payload = run_one_combination(
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        backend_name="mock",
        batch_size=1,
        samples=samples,
        args=args,
        backend_factory=lambda **_: ErrorBackend(),
    )
    assert len(payload["rows"]) == 2
    assert all(r["generated"] == "" for r in payload["rows"])


# ---------------------------------------------------------------------------
# 5. Aggregate comparison writer
# ---------------------------------------------------------------------------


def test_write_run_artifact_writes_to_disk(tmp_path: Path) -> None:
    payload = {
        "summary": {"model": "m", "backend": "b", "batch_size": 1, "samples": 1},
        "rows": [],
    }
    out = write_run_artifact(tmp_path, "m", "b", 1, payload)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["summary"]["model"] == "m"
    # Filesystem-safe slug
    assert "run_m__b__b1.json" in out.name


def test_write_aggregate_comparison_emits_csv_and_json(tmp_path: Path) -> None:
    runs = [
        {
            "summary": {
                "model": "M", "backend": "transformers", "batch_size": 1,
                "samples": 3, "elapsed_s": 2.0,
                "per_sample_latency_ms": 666.6,
                "throughput_samples_per_s": 1.5,
                "reward_binary": 0.0, "reward_layered": 0.42,
                "parse_success_rate": 0.66,
            },
            "rows": [],
        },
        {
            "summary": {
                "model": "M", "backend": "vllm", "batch_size": 1,
                "samples": 3, "elapsed_s": 1.0,
                "per_sample_latency_ms": 333.3,
                "throughput_samples_per_s": 3.0,
                "reward_binary": 0.0, "reward_layered": 0.42,
                "parse_success_rate": 0.66,
            },
            "rows": [],
        },
    ]
    cmp_json, cmp_csv = write_aggregate_comparison(tmp_path, runs)
    loaded = json.loads(cmp_json.read_text(encoding="utf-8"))
    assert len(loaded["runs"]) == 2
    with cmp_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[0]["backend"] == "transformers"
    assert rows[1]["backend"] == "vllm"


def test_write_aggregate_comparison_handles_empty(tmp_path: Path) -> None:
    cmp_json, cmp_csv = write_aggregate_comparison(tmp_path, [])
    loaded = json.loads(cmp_json.read_text(encoding="utf-8"))
    assert loaded["runs"] == []
    with cmp_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows == []  # only header


# ---------------------------------------------------------------------------
# 6. CLI behaviour
# ---------------------------------------------------------------------------


def test_argparser_defaults() -> None:
    args = _build_argparser().parse_args([])
    assert args.models == list(DEFAULT_MODELS)
    assert args.backends == list(DEFAULT_BACKENDS)
    assert args.batch_sizes == list(DEFAULT_BATCH_SIZES)
    assert args.samples_dir == Path("datasets/tool-calling-d2/dev")
    assert args.output_dir == Path("artifacts/p5-04-backend-comparison")
    assert args.limit == 0
    assert args.max_new_tokens == 256
    assert args.dtype == "bf16"
    assert args.device == "auto"


def test_argparser_overrides() -> None:
    args = _build_argparser().parse_args([
        "--models", "Qwen/Qwen2.5-0.5B-Instruct",
        "--backends", "vllm",
        "--batch-sizes", "1", "4",
        "--limit", "10",
        "--dtype", "fp16",
    ])
    assert args.models == ["Qwen/Qwen2.5-0.5B-Instruct"]
    assert args.backends == ["vllm"]
    assert args.batch_sizes == [1, 4]
    assert args.limit == 10
    assert args.dtype == "fp16"


def test_argparser_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ei:
        _build_argparser().parse_args(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "P5-04" in out or "backend comparison" in out.lower()
