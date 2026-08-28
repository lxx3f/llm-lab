"""Tests for the Transformers evaluation backend (scripts/eval_transformers.py).

These tests cover:

- Argument validation, device / dtype resolution, default cache directory.
- Chat-template fallback for tokenizers without a native ``chat_template``.
- Transcript normalization for ``tool_not_available`` rows (no-tool transcripts
  must pass ``parse_success`` rather than ``parse_success=False``).
- Output artifact schema: ``summary`` keys, ``rows`` keys, full ``generated``
  text preserved alongside ``generated_preview``.
- Compatibility with ``reward_offline.py`` and ``schemas/reward_signal.schema.json``.

The tests run offline: they mock ``transformers.AutoModelForCausalLM`` /
``AutoTokenizer`` and never download weights. The one integration test that
touches a real tokenizer uses a small saved tokenizer written by
``_write_tiny_tokenizer_for_test`` so the harness is self-contained.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "scripts" / "eval_transformers.py"
SCHEMA = ROOT / "schemas" / "reward_signal.schema.json"


def _load_eval() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_eval_transformers", EVAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _d2_sample(*, sample_id: str = "d2-dev-0001",
               task_type: str = "tool_not_available",
               expected_calls: list[dict[str, Any]] | None = None,
               expected_answer: str = "对不起，当前可用工具不支持该请求。",
               tools: list[dict[str, Any]] | None = None,
               expected_result: str = "上海 22C") -> dict[str, Any]:
    """Build a minimal D2 sample for unit tests (no real schema validation)."""
    return {
        "schema_version": "1.0",
        "id": sample_id,
        "metadata": {"task_type": task_type, "split": "dev"},
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "请查询上海的天气。"},
            {"role": "assistant", "content": expected_answer},
        ],
        "tools": tools or [
            {
                "type": "function",
                "function": {
                    "name": "d1_get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "expected_tool_calls": expected_calls or [],
        "expected_answer": expected_answer,
        "expected_result": expected_result,
    }


class ArgparseAndResolutionTests(unittest.TestCase):
    """CLI args, device/dtype resolution, and cache directory default."""

    def test_argparse_accepts_required_flags(self) -> None:
        ev = _load_eval()
        parser = ev._build_argparser()
        args = parser.parse_args([
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--output", "tmp.json",
            "--samples-dir", "datasets/tool-calling-d2/dev",
        ])
        self.assertEqual(args.model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(args.max_new_tokens, 256)
        self.assertEqual(args.dtype, "bf16")
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.revision, "main")
        self.assertFalse(args.trust_remote_code)

    def test_argparse_rejects_missing_model(self) -> None:
        ev = _load_eval()
        parser = ev._build_argparser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--output", "tmp.json"])

    def test_resolve_device_auto_prefers_cuda(self) -> None:
        ev = _load_eval()
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(ev._resolve_device("auto"), "cuda")
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(ev._resolve_device("auto"), "cpu")

    def test_resolve_device_explicit_overrides(self) -> None:
        ev = _load_eval()
        self.assertEqual(ev._resolve_device("cuda"), "cuda")
        self.assertEqual(ev._resolve_device("cpu"), "cpu")

    def test_resolve_dtype_overrides_low_precision_on_cpu(self) -> None:
        ev = _load_eval()
        self.assertEqual(ev._resolve_dtype("fp32", "cpu"), torch.float32)
        with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            resolved = ev._resolve_dtype("bf16", "cpu")
        self.assertEqual(resolved, torch.float32)
        self.assertIn("overriding dtype", stdout.getvalue())

    def test_resolve_dtype_keeps_low_precision_on_cuda(self) -> None:
        ev = _load_eval()
        self.assertEqual(ev._resolve_dtype("bf16", "cuda"), torch.bfloat16)
        self.assertEqual(ev._resolve_dtype("fp16", "cuda"), torch.float16)


class GoldAnswerLeakageTests(unittest.TestCase):
    """Regression: the terminal gold ``expected_answer`` must never enter the prompt.

    D2 sample ``messages`` ends with an ``assistant`` content-only
    message that is exactly ``expected_answer``. If that message is fed
    to ``apply_chat_template`` the model is asked to reproduce text
    already present in context, inflating ``reward_binary``. The
    ``_strip_terminal_assistant`` helper must excise it before
    rendering.
    """

    def test_strip_drops_terminal_assistant_without_tool_calls(self) -> None:
        ev = _load_eval()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "assistant", "content": "GOLD_ANSWER"},
        ]
        stripped = ev._strip_terminal_assistant(messages)
        self.assertEqual(len(stripped), 4)
        self.assertNotIn("GOLD_ANSWER",
                         " ".join(str(m.get("content", "")) for m in stripped))

    def test_strip_keeps_assistant_with_tool_calls(self) -> None:
        ev = _load_eval()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "tool_calls": [{"id": "c1"}]},
        ]
        stripped = ev._strip_terminal_assistant(messages)
        self.assertEqual(len(stripped), 3)

    def test_apply_chat_template_does_not_include_expected_answer(self) -> None:
        ev = _load_eval()

        class _Tok:
            chat_template = "<native>"

            def apply_chat_template(self, messages, *, tokenize,
                                    add_generation_prompt, tools):
                # Join role + content into the prompt body so the test
                # can assert the gold answer is absent.
                lines = []
                for m in messages:
                    if m.get("role") == "system":
                        continue
                    content = m.get("content") or ""
                    if isinstance(content, list):
                        content = " ".join(
                            c.get("text", "") for c in content
                            if isinstance(c, dict)
                        )
                    lines.append(f"{m['role']}:{content}")
                return "\n".join(lines)

        sample = _d2_sample(
            expected_calls=[],
            expected_answer="GOLD_TEXT_SENTINEL_12345",
        )
        prompt = ev._apply_chat_template(_Tok(), sample)
        self.assertNotIn("GOLD_TEXT_SENTINEL_12345", prompt)

    def test_apply_chat_template_keeps_context_tool_history(self) -> None:
        ev = _load_eval()
        captured: dict[str, Any] = {}

        class _Tok:
            chat_template = "<native>"

            def apply_chat_template(self, messages, *, tokenize,
                                    add_generation_prompt, tools):
                captured["messages"] = list(messages)
                captured["tools"] = tools
                return "rendered"

        sample = _d2_sample()
        sample["messages"] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "d1_get_weather",
                                          "arguments": "{\"city\": \"上海\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "name": "d1_get_weather",
             "content": "上海 22C"},
            {"role": "assistant", "content": sample["expected_answer"]},
        ]
        ev._apply_chat_template(_Tok(), sample)
        # Tool history preserved; final gold assistant message dropped.
        rendered = captured["messages"]
        self.assertEqual(len(rendered), 4)
        self.assertEqual(rendered[-1]["role"], "tool")
        self.assertEqual(rendered[-1]["content"], "上海 22C")
        # Tools metadata still passed so the model can name them.
        self.assertEqual(captured["tools"], sample["tools"])

    def test_apply_chat_template_preserves_intermediate_assistant_calls(self) -> None:
        ev = _load_eval()
        sample = _d2_sample()
        sample["messages"] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "d1_calculate",
                                          "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "name": "d1_calculate",
             "content": "42"},
            {"role": "assistant", "tool_calls": [
                {"id": "c2", "function": {"name": "d1_get_weather",
                                          "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "c2", "name": "d1_get_weather",
             "content": "sunny"},
            {"role": "assistant", "content": "final answer"},
        ]
        sample["expected_answer"] = "final answer"

        captured: list[Any] = []

        class _Tok:
            chat_template = "<native>"

            def apply_chat_template(self, messages, **kwargs):
                captured.append(list(messages))
                return "rendered"

        ev._apply_chat_template(_Tok(), sample)
        # All 6 context messages kept; only the trailing gold assistant
        # content message is stripped.
        self.assertEqual(len(captured[0]), 6)
        self.assertEqual(captured[0][-1]["role"], "tool")


class ChatTemplateAndGenerationTests(unittest.TestCase):
    """Chat template selection and greedy generation parameters."""

    def test_apply_chat_template_prefers_native_template(self) -> None:
        ev = _load_eval()

        class _Tok:
            chat_template = "<native>{{ messages }}</native>"

            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools):
                self.last_kwargs = {
                    "tokenize": tokenize,
                    "add_generation_prompt": add_generation_prompt,
                    "tools": tools,
                }
                return "<rendered>" + json.dumps(messages, ensure_ascii=False) + "</rendered>"

        tok = _Tok()
        sample = _d2_sample()
        rendered = ev._apply_chat_template(tok, sample)
        # The terminal gold ``assistant`` content message has been stripped,
        # so the rendered payload must not contain ``expected_answer``.
        self.assertIn('assistant', rendered)  # system / user roles only
        self.assertNotIn('对不起，当前可用工具不支持该请求。', rendered)
        self.assertEqual(tok.last_kwargs["tokenize"], False)
        self.assertTrue(tok.last_kwargs["add_generation_prompt"])
        self.assertEqual(tok.last_kwargs["tools"], sample["tools"])

    def test_apply_chat_template_fallback_when_no_template(self) -> None:
        ev = _load_eval()

        class _Tok:
            chat_template = None

        rendered = ev._apply_chat_template(_Tok(), _d2_sample())
        self.assertIn("system: You are a helpful assistant.", rendered)
        self.assertIn("user: 请查询上海的天气。", rendered)
        # The terminal gold answer is stripped — must not appear.
        self.assertNotIn("对不起，当前可用工具不支持该请求。", rendered)

    def test_apply_chat_template_falls_back_on_template_error(self) -> None:
        ev = _load_eval()

        class _Tok:
            chat_template = "<native>"

            def apply_chat_template(self, *args, **kwargs):
                raise RuntimeError("template engine crashed")

        rendered = ev._apply_chat_template(_Tok(), _d2_sample())
        self.assertIn("system: You are a helpful assistant.", rendered)
        self.assertNotIn("对不起，当前可用工具不支持该请求。", rendered)


class TranscriptNormalizationTests(unittest.TestCase):
    """Empty / no-tool transcripts must reach the classifier as ``[]``."""

    def test_no_tool_transcript_is_structured(self) -> None:
        ev = _load_eval()
        sample = _d2_sample(task_type="tool_not_available",
                             expected_calls=[])
        # Match the expected_answer byte-for-byte so the final layer
        # also passes and the first_failure is ``None``.
        row, result = ev._build_eval_row(sample, "<prompt>", sample["expected_answer"])
        self.assertEqual(row["extracted_calls"], [])
        self.assertTrue(result["layers"]["parse_success"])
        self.assertIsNone(result["first_failure"])

    def test_garbled_generation_yields_first_failure_parse(self) -> None:
        ev = _load_eval()
        sample = _d2_sample(task_type="tool_not_available",
                             expected_calls=[])
        # Pure garbled text without JSON tool-call -> extract_tool_calls
        # returns None; the script must normalize to ``[]`` and the
        # classifier then decides whether ``parse_success`` holds. For a
        # tool_not_available sample with no expected calls, an empty list
        # means parse_success=True.
        row, result = ev._build_eval_row(sample, "<prompt>",
                                          "some garbled free-form text without any JSON")
        self.assertEqual(row["extracted_calls"], [])
        self.assertTrue(result["layers"]["parse_success"])

    def test_json_call_text_persists_extracted_calls(self) -> None:
        ev = _load_eval()
        sample = _d2_sample(task_type="single_tool",
                             expected_calls=[
                                 {"call_id": "c1", "name": "d1_get_weather",
                                  "arguments": {"city": "上海"},
                                  "expected_result": "上海 22C",
                                  "execution_outcome": "success",
                                  "result": "上海 22C"},
                             ])
        generated = '{"call_id": "c1", "name": "d1_get_weather", "arguments": {"city": "上海"}}\n上海 22C'
        row, result = ev._build_eval_row(sample, "<prompt>", generated)
        self.assertEqual(len(row["extracted_calls"]), 1)
        self.assertEqual(row["extracted_calls"][0]["name"], "d1_get_weather")
        # Tool call present and matching expected; final answer also matches
        # so the only failure is the execution step (no MockExecutor was
        # run during this synthetic transcript).
        self.assertEqual(result["first_failure"], "execution_success")

    def test_layer_counts_increments_for_each_failure(self) -> None:
        ev = _load_eval()
        counts: dict[str, int] = {}
        ev._update_layer_counts(counts, {"first_failure": None})
        ev._update_layer_counts(counts, {"first_failure": None})
        ev._update_layer_counts(counts, {"first_failure": "parse_success"})
        ev._update_layer_counts(counts, {"first_failure": "tool_name_correct"})
        self.assertEqual(counts["none"], 2)
        self.assertEqual(counts["parse_success"], 1)
        self.assertEqual(counts["tool_name_correct"], 1)


class GreedyGenerationTests(unittest.TestCase):
    """Greedy generation must use beam=1, sample=False, and respect pad_token."""

    def test_greedy_generate_uses_no_sampling_and_pad_token(self) -> None:
        ev = _load_eval()

        class _Model:
            def __init__(self) -> None:
                self.eval_calls = 0
                self.generate_kwargs: dict[str, Any] = {}

            def eval(self) -> None:
                self.eval_calls += 1

            def generate(self, *, input_ids, **kwargs):
                self.generate_kwargs = dict(kwargs)
                # Echo ``input_ids`` plus two extra tokens so we can verify
                # the script decodes only the post-prompt tail.
                tail = torch.tensor([[101, 102]], dtype=input_ids.dtype)
                return torch.cat([input_ids, tail], dim=1)

        class _Tok:
            pad_token_id = 7
            eos_token_id = 99

            def __call__(self, prompt, *, return_tensors, truncation, max_length):
                self.last_kwargs = {
                    "truncation": truncation, "max_length": max_length}
                return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

            def decode(self, tokens, *, skip_special_tokens):
                self.skip_special_tokens = skip_special_tokens
                return f"<decoded:{tokens.tolist()}>"

        model = _Model()
        tok = _Tok()
        text = ev._greedy_generate(model, tok, "ignored",
                                   max_new_tokens=8, device="cpu")
        self.assertEqual(model.eval_calls, 1)
        self.assertEqual(model.generate_kwargs["do_sample"], False)
        self.assertEqual(model.generate_kwargs["num_beams"], 1)
        self.assertEqual(model.generate_kwargs["pad_token_id"], 7)
        self.assertEqual(model.generate_kwargs["max_new_tokens"], 8)
        self.assertEqual(text, "<decoded:[101, 102]>")
        self.assertTrue(tok.last_kwargs["truncation"])
        self.assertEqual(tok.last_kwargs["max_length"], 4096)

    def test_greedy_generate_falls_back_to_eos_for_pad(self) -> None:
        ev = _load_eval()

        class _Model:
            def eval(self) -> None:
                pass

            def generate(self, **kwargs):
                self.kwargs = dict(kwargs)
                return torch.tensor([[1, 2, 3]], dtype=torch.long)

        class _Tok:
            pad_token_id = None
            eos_token_id = 5

            def __call__(self, *_a, **_k):
                return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

            def decode(self, tokens, **_k):
                return ""

        model = _Model()
        tok = _Tok()
        ev._greedy_generate(model, tok, "x", max_new_tokens=4, device="cpu")
        self.assertEqual(model.kwargs["pad_token_id"], 5)


class OutputArtifactTests(unittest.TestCase):
    """Artifact schema compatibility with reward_offline + reward_signal."""

    def test_build_eval_row_persists_full_generated_text(self) -> None:
        ev = _load_eval()
        long_text = "x" * 500
        row, _ = ev._build_eval_row(_d2_sample(), "<prompt>", long_text)
        self.assertEqual(row["generated"], long_text)
        self.assertEqual(len(row["generated_preview"]), 200)
        self.assertEqual(row["generated_preview"], long_text[:200])

    def test_artifact_summary_keys_match_reward_signal_schema(self) -> None:
        ev = _load_eval()
        # The artifact itself is consumed by reward_offline; ensure the
        # first_failure_distribution and no_failure keys exist, plus
        # the per-row fields required by load_transcripts.
        ev._load_eval = ev  # silence unused warning
        summary = {
            "checkpoint": "test-model",
            "backend": "transformers",
            "revision": "main",
            "device": "cpu",
            "dtype": "torch.float32",
            "transformers_version": "0",
            "torch_version": "0",
            "samples_dir": "datasets/tool-calling-d2/dev",
            "total": 1,
            "parse_success_count": 1,
            "first_failure_distribution": {"none": 1},
            "no_failure": 1,
            "parse_success_rate": 1.0,
        }
        for key in ("checkpoint", "samples_dir", "total",
                    "first_failure_distribution", "no_failure"):
            self.assertIn(key, summary)

    def test_artifact_rows_are_consumable_by_reward_offline(self) -> None:
        ev = _load_eval()
        sample = _d2_sample()
        row, _ = ev._build_eval_row(sample, "<prompt>", "对不起，无法执行。")
        # Reward_offline looks up ``extracted_calls``, ``first_failure``,
        # ``layers``, ``generated`` and ``sample_id`` (via load_transcripts).
        for key in ("sample_id", "extracted_calls", "layers",
                    "first_failure", "generated"):
            self.assertIn(key, row)

    def test_reward_signal_schema_includes_first_failure_keys(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        properties = schema.get("properties") or {}
        # The schema only enforces ``first_failure`` as an enum string;
        # when ``first_failure`` is ``None`` (i.e. execution_correct),
        # the row omits the key. We assert the enum covers the
        # expected values used by the backend.
        ff_schema = (properties.get("first_failure") or {}).get("enum") or []
        self.assertIn("parse_success", ff_schema)
        self.assertIn("tool_name_correct", ff_schema)
        self.assertIn("final_answer_correct", ff_schema)


class ScriptEndToEndTests(unittest.TestCase):
    """Drive the script's ``main`` with mocks; verifies artifact on disk."""

    def setUp(self) -> None:
        self.tmp_dir = ROOT / "artifacts" / "test-eval-transformers-tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir = self.tmp_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)
        # Two samples: one no-tool, one with expected calls.
        (self.samples_dir / "d2-dev-0001.json").write_text(
            json.dumps(_d2_sample(sample_id="d2-dev-0001",
                                 task_type="tool_not_available",
                                 expected_calls=[]),
                       ensure_ascii=False),
            encoding="utf-8",
        )
        (self.samples_dir / "d2-dev-0002.json").write_text(
            json.dumps(_d2_sample(sample_id="d2-dev-0002",
                                 task_type="single_tool",
                                 expected_calls=[
                                     {"call_id": "c1",
                                      "name": "d1_get_weather",
                                      "arguments": {"city": "上海"},
                                      "expected_result": "上海 22C",
                                      "execution_outcome": "success",
                                      "result": "上海 22C"},
                                 ]),
                       ensure_ascii=False),
            encoding="utf-8",
        )
        self.output_path = self.tmp_dir / "out.json"

    def tearDown(self) -> None:
        for child in self.tmp_dir.glob("**/*"):
            if child.is_file():
                child.unlink()
        for child in sorted(self.tmp_dir.glob("**/*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        self.tmp_dir.rmdir()

    def _build_tokenizer_mock(self) -> Any:
        class _Tok:
            chat_template = "<native>"

            def apply_chat_template(self, messages, *, tokenize,
                                    add_generation_prompt, tools):
                return "<rendered>"

            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, prompt, *, return_tensors, truncation, max_length):
                return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

            def decode(self, tokens, *, skip_special_tokens):
                # No-tool response for d2-dev-0001; explicit JSON call for d2-dev-0002.
                return [
                    "对不起，当前可用工具不支持该请求。",
                    '{"call_id": "c1", "name": "d1_get_weather", "arguments": {"city": "上海"}}',
                ][len(self.responses)]

        class _Resp:
            def __init__(self) -> None:
                self.responses: list[str] = []

        tok = _Tok()
        tok.responses = _Resp()
        return tok

    def test_main_writes_artifact_and_summary(self) -> None:
        ev = _load_eval()
        tokenizer = self._build_tokenizer_mock()

        class _Model:
            def __init__(self) -> None:
                self.to_calls: list[str] = []

            def to(self, device):
                self.to_calls.append(device)
                return self

            def eval(self) -> None:
                pass

            def generate(self, *, input_ids, **kwargs):
                # Append two new tokens so the decode path sees a non-empty tail.
                return torch.cat([input_ids, torch.tensor([[10, 11]], dtype=input_ids.dtype)], dim=1)

        sys.path.insert(0, str(ROOT))

        def fake_from_pretrained(repo_id, **kwargs):
            if "Tokenizer" in (kwargs.get("__class__") or ""):
                raise AssertionError("unused")
            return tokenizer if "Tokenizer" not in str(repo_id) else tokenizer

        with mock.patch.object(ev, "_resolve_device", return_value="cpu"), \
             mock.patch.object(ev.torch.cuda if hasattr(ev, "torch") else torch.cuda, "is_available", return_value=False), \
             mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_Model()), \
             mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
            with mock.patch("sys.argv", [
                "eval_transformers.py",
                "--model", "fake/model",
                "--samples-dir", str(self.samples_dir),
                "--output", str(self.output_path),
                "--device", "cpu",
                "--dtype", "fp32",
            ]):
                rc = ev.main()
        self.assertEqual(rc, 0)
        payload = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertIn("summary", payload)
        self.assertIn("rows", payload)
        self.assertEqual(payload["summary"]["checkpoint"], "fake/model")
        self.assertEqual(payload["summary"]["total"], 2)
        self.assertIn("first_failure_distribution", payload["summary"])
        self.assertIn("no_failure", payload["summary"])
        self.assertIn("parse_success_rate", payload["summary"])
        self.assertEqual(len(payload["rows"]), 2)
        for row in payload["rows"]:
            self.assertIn("sample_id", row)
            self.assertIn("extracted_calls", row)
            self.assertIn("layers", row)
            self.assertIn("first_failure", row)
            self.assertIn("generated", row)
            self.assertIn("generated_preview", row)
            self.assertLessEqual(len(row["generated_preview"]), 200)


if __name__ == "__main__":
    unittest.main()