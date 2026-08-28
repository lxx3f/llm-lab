"""Tests for the D2 multi-turn dataset generator and dataset layout.

Covers::

- schema validity (Draft 2020-12) for all six multi-turn ``task_type``s;
- multi-turn ``messages`` structure (assistant role carries ``tool_calls``,
  tool role carries ``tool_call_id`` + ``name`` + ``content``);
- ``expected_tool_calls`` carries an explicit ``depends_on`` chain that
  matches the messages transcript;
- MockExecutor replay returns the declared ``expected_result`` for every
  expected call;
- held-out ``train / dev / test`` partitions are disjoint at the sample-id
  layer, path layer, and canonical semantic-content layer; train is disjoint
  from D1 / D1.1 train ids;
- ``MANIFEST-{train,dev,test}.json`` ``count`` and ``aggregate_sha256``
  match the on-disk files;
- canonical semantic content is unique across all 600 rows after removing
  bookkeeping identifiers, dependency references, split metadata, and timestamps.

The D2 directory is treated as a fixture: the tests skip themselves if
the dataset has not been generated yet (``--count 600`` via the
generator).
"""

from __future__ import annotations

import hashlib
import json
import unittest
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
D2 = ROOT / "datasets" / "tool-calling-d2"
SCHEMA = ROOT / "schemas" / "d2_multi_turn_sample.schema.json"
GENERATOR = ROOT / "scripts" / "generate_d2_dataset.py"

EXPECTED_SPLITS = {"train", "dev", "test"}
EXPECTED_TASK_TYPES = {
    "tool_not_available",
    "tool_error_response",
    "insufficient_result_search",
    "req_change_city",
    "multi_tool_sequential",
    "error_recovery",
}


def _dataset_present() -> bool:
    return (D2 / "MANIFEST-train.json").exists() and (
        D2 / "MANIFEST-dev.json").exists() and (D2 / "MANIFEST-test.json").exists()


def _expected_total_samples() -> int:
    """Total samples across the three D2 splits, derived from on-disk MANIFESTs.

    The D2 dataset is parameterizable by ``--count``; tests read the actual
    manifest counts instead of hard-coding so a 5004-sample scale-up does
    not require test edits.
    """
    total = 0
    for split in EXPECTED_SPLITS:
        manifest_path = D2 / f"MANIFEST-{split}.json"
        if not manifest_path.exists():
            return 0
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        total += manifest["count"]
    return total


def _expected_per_task_samples() -> int:
    """Number of samples per task_type in the on-disk dataset.

    The generator splits ``count`` evenly across the 6 task types via
    ``count // len(TASK_TYPES)``; any remainder rows go to the first few
    builders in TASK_TYPES order but tests treat the dataset as 6 equal
    slices for clarity.
    """
    total = _expected_total_samples()
    return total // len(EXPECTED_TASK_TYPES)


def _build_count() -> int:
    """Return the build_count used to generate the on-disk dataset.

    The generator echoes ``--count`` into ``MANIFEST-*.json`` metadata as
    ``build_count``; older manifests (D2 v2.0 round 9) may not have it,
    so we fall back to the sum of split counts in that case.
    """
    for split in EXPECTED_SPLITS:
        manifest_path = D2 / f"MANIFEST-{split}.json"
        if not manifest_path.exists():
            return 0
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "build_count" in manifest:
            return int(manifest["build_count"])
    return _expected_total_samples()


def _load_validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _load_split(split: str) -> list[dict]:
    manifest = json.loads((D2 / f"MANIFEST-{split}.json").read_text(encoding="utf-8"))
    samples: list[dict] = []
    for entry in manifest["samples"]:
        path = D2 / entry["path"]
        samples.append(json.loads(path.read_text(encoding="utf-8")))
    return samples


def _aggregate_sha(samples: list[dict]) -> str:
    h = hashlib.sha256()
    for sample in sorted(samples, key=lambda s: s["id"]):
        h.update(f"{sample['id']}\n".encode("utf-8"))
    return h.hexdigest()


@unittest.skipUnless(_dataset_present(), "D2 dataset not generated yet")
class D2DatasetSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validator = _load_validator()
        cls.train = _load_split("train")
        cls.dev = _load_split("dev")
        cls.test = _load_split("test")

    def test_all_samples_schema_valid(self) -> None:
        for sample in self.train + self.dev + self.test:
            with self.subTest(sample_id=sample["id"]):
                errors = list(self.validator.iter_errors(sample))
                self.assertEqual(errors, [],
                                 f"{sample['id']} schema errors: {errors}")

    def test_task_types_cover_all_six_multi_turn_flavours(self) -> None:
        seen = {sample["metadata"]["task_type"]
                for sample in self.train + self.dev + self.test}
        self.assertTrue(EXPECTED_TASK_TYPES.issubset(seen),
                        f"missing flavours: {EXPECTED_TASK_TYPES - seen}")

    def test_all_six_flavours_present_in_each_split(self) -> None:
        """Every generated split contains each canonical D2 task type."""
        for split, samples in (("train", self.train),
                               ("dev", self.dev),
                               ("test", self.test)):
            with self.subTest(split=split):
                seen = {sample["metadata"]["task_type"] for sample in samples}
                self.assertEqual(
                    seen,
                    EXPECTED_TASK_TYPES,
                    f"split {split!r} task types differ: "
                    f"missing={EXPECTED_TASK_TYPES - seen}, extra={seen - EXPECTED_TASK_TYPES}",
                )


@unittest.skipUnless(_dataset_present(), "D2 dataset not generated yet")
class D2MultiTurnStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = _load_split("train")

    def _has_multi_turn(self, sample: dict) -> bool:
        """Heuristic: multi-turn samples have a ``tool`` role or an
        ``assistant`` role with ``tool_calls`` in ``messages``."""
        return any(
            msg["role"] in ("assistant", "tool")
            for msg in sample["messages"]
        )

    def test_multi_turn_messages_carry_assistant_and_tool_roles(self) -> None:
        multi_turn = [s for s in self.train if self._has_multi_turn(s)]
        self.assertGreater(len(multi_turn), 0,
                           "expected at least one multi-turn sample")
        for sample in multi_turn:
            roles = {msg["role"] for msg in sample["messages"]}
            self.assertIn("user", roles)
            # Every multi-turn sample has at least one tool-call round.
            self.assertTrue("assistant" in roles or "tool" in roles,
                            f"{sample['id']} missing assistant/tool roles")

    def test_assistant_tool_calls_carry_call_id(self) -> None:
        for sample in self.train:
            for msg in sample["messages"]:
                if msg["role"] != "assistant":
                    continue
                for call in msg.get("tool_calls", []):
                    self.assertIn("id", call)
                    self.assertGreater(len(call["id"]), 0)

    def test_tool_messages_carry_tool_call_id(self) -> None:
        for sample in self.train:
            for msg in sample["messages"]:
                if msg["role"] != "tool":
                    continue
                self.assertIn("tool_call_id", msg)
                self.assertIn("name", msg)
                self.assertIsNotNone(msg.get("content"))

    def test_expected_tool_calls_match_transcript_call_ids(self) -> None:
        """Every assistant tool call belongs to the canonical expected plan."""
        for sample in self.train:
            transcript_ids = {
                call["id"]
                for msg in sample["messages"] if msg["role"] == "assistant"
                for call in msg.get("tool_calls", [])
            }
            expected_ids = {c["call_id"] for c in sample["expected_tool_calls"]}
            self.assertEqual(
                transcript_ids,
                expected_ids,
                f"{sample['id']} transcript ids {transcript_ids} "
                f"!= expected ids {expected_ids}",
            )

    def test_call_ids_use_final_sample_namespace(self) -> None:
        for sample in self.train:
            prefix = f"call-{sample['id']}-"
            for message in sample["messages"]:
                for call in message.get("tool_calls", []):
                    self.assertTrue(call["id"].startswith(prefix))
                if message.get("role") == "tool":
                    self.assertTrue(message["tool_call_id"].startswith(prefix))
            for call in sample["expected_tool_calls"]:
                self.assertTrue(call["call_id"].startswith(prefix))
                for dependency in call.get("depends_on", []) or []:
                    self.assertTrue(dependency.startswith(prefix))


@unittest.skipUnless(_dataset_present(), "D2 dataset not generated yet")
class D2MockExecutorReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = _load_split("train")

    def test_mock_executor_execute_sequence_returns_expected_results(self) -> None:
        """Replay every expected plan through the real MockExecutor API."""
        from architecture_lab.execution.mock_executor import MockExecutor
        from examples.d1_mocks import (
            d1_calculate, d1_get_weather, d1_web_search, d1_translate,
        )
        mock_functions = {
            "d1_calculate": d1_calculate,
            "d1_get_weather": d1_get_weather,
            "d1_web_search": d1_web_search,
            "d1_translate": d1_translate,
        }
        for sample in self.train:
            executor = MockExecutor()
            for tool in sample["tools"]:
                function = tool["function"]
                name = function["name"]
                executor.register_mock(name, mock_functions[name], function["parameters"])
            calls = [
                {
                    "tool_name": call["name"],
                    "call_id": call["call_id"],
                    "arguments": call["arguments"],
                    "depends_on": call.get("depends_on", []),
                }
                for call in sample["expected_tool_calls"]
            ]
            results = executor.execute_sequence(calls)
            self.assertEqual(len(results), len(calls), sample["id"])
            for expected, actual in zip(sample["expected_tool_calls"], results):
                with self.subTest(sample_id=sample["id"], call_id=expected["call_id"]):
                    self.assertEqual(actual["call_id"], expected["call_id"])
                    self.assertEqual(actual["outcome"], "success")
                    self.assertEqual(str(actual["result"]), str(expected["expected_result"]))

    def test_execute_sequence_uses_topological_order_for_shuffled_input(self) -> None:
        """A dependent call submitted first is held until its predecessor
        completes; returned results are in actual execution order."""
        from architecture_lab.execution.mock_executor import MockExecutor
        from examples.d1_mocks import d1_calculate, d1_web_search

        executor = MockExecutor()
        executor.register_mock(
            "d1_calculate", d1_calculate,
            {"type": "object", "properties": {"expression": {"type": "string"}},
             "required": ["expression"]},
        )
        executor.register_mock(
            "d1_web_search", d1_web_search,
            {"type": "object", "properties": {
                "query": {"type": "string"}, "limit": {"type": "integer"}},
             "required": ["query"]},
        )
        results = executor.execute_sequence([
            {"tool_name": "d1_web_search", "call_id": "c2",
             "arguments": {"query": "result 4", "limit": 3},
             "depends_on": ["c1"]},
            {"tool_name": "d1_calculate", "call_id": "c1",
             "arguments": {"expression": "2 + 2"}},
        ])
        self.assertEqual([result["call_id"] for result in results], ["c1", "c2"])
        self.assertEqual([result["outcome"] for result in results], ["success", "success"])
        self.assertEqual(results[1]["result"], "关于「result 4」找到 3 条结果。")

    def test_execute_sequence_rejects_failed_dependency_step(self) -> None:
        """A failed first step prevents its dependent step from executing."""
        from architecture_lab.execution.mock_executor import MockExecutor
        from examples.d1_mocks import d1_calculate

        executor = MockExecutor()
        executor.register_mock(
            "d1_calculate", d1_calculate,
            {"type": "object", "properties": {"expression": {"type": "string"}},
             "required": ["expression"]},
        )
        results = executor.execute_sequence([
            {"tool_name": "d1_calculate", "call_id": "c1",
             "arguments": {"expression": "not valid"}},
            {"tool_name": "d1_calculate", "call_id": "c2",
             "arguments": {"expression": "2 + 2"}, "depends_on": ["c1"]},
        ])
        self.assertEqual([result["outcome"] for result in results],
                         ["mock_exception", "mock_not_found"])
        self.assertIn("did not complete", results[1]["error"])


@unittest.skipUnless(_dataset_present(), "D2 dataset not generated yet")
class D2DependencyGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = _load_split("train") + _load_split("dev") + _load_split("test")

    def test_depends_on_references_exist(self) -> None:
        for sample in self.samples:
            ids = {call["call_id"] for call in sample["expected_tool_calls"]}
            for call in sample["expected_tool_calls"]:
                for dep in call.get("depends_on", []) or []:
                    self.assertIn(dep, ids, f"{sample['id']}: dangling {dep}")

    def test_depends_on_references_are_strictly_earlier(self) -> None:
        for sample in self.samples:
            positions = {call["call_id"]: index
                         for index, call in enumerate(sample["expected_tool_calls"])}
            for index, call in enumerate(sample["expected_tool_calls"]):
                for dep in call.get("depends_on", []) or []:
                    self.assertLess(positions[dep], index,
                                    f"{sample['id']}: dependency is not earlier")

    def test_depends_on_graph_is_acyclic(self) -> None:
        for sample in self.samples:
            calls = {call["call_id"]: call for call in sample["expected_tool_calls"]}
            remaining = set(calls)
            while remaining:
                ready = {
                    call_id for call_id in remaining
                    if all(dep not in remaining
                           for dep in calls[call_id].get("depends_on", []) or [])
                }
                self.assertTrue(ready, f"{sample['id']}: dependency cycle")
                remaining -= ready

    def test_invalid_examples_fail_dependency_semantics(self) -> None:
        """The negative fixtures are schema-shaped but fail D2 graph rules."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("generate_d2_for_test", GENERATOR)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for path in sorted((ROOT / "examples" / "d2_multi_turn").glob("sample-negative-*.json")):
            with self.subTest(path=path.name):
                sample = json.loads(path.read_text(encoding="utf-8"))
                errors = module.validate_semantics(sample)
                self.assertTrue(errors, f"negative fixture unexpectedly valid: {path}")

    def test_dependency_chain_has_semantic_stepwise_order(self) -> None:
        """The generated sequential examples have real dependency edges."""
        chained = [sample for sample in self.samples
                    if any(call.get("depends_on") for call in sample["expected_tool_calls"])]
        self.assertGreater(len(chained), 0)
        for sample in chained:
            positions = {call["call_id"]: index
                         for index, call in enumerate(sample["expected_tool_calls"])}
            for call in sample["expected_tool_calls"]:
                for dep in call.get("depends_on", []) or []:
                    self.assertLess(positions[dep], positions[call["call_id"]])


@unittest.skipUnless(_dataset_present(), "D2 dataset not generated yet")
class D2TranscriptWellFormednessTests(unittest.TestCase):
    """Round 10 cross-message invariants beyond JSON Schema's reach.

    These checks run on every split (not only train) so a regression in
    the IID split cannot hide split-specific transcript corruption.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.train = _load_split("train")
        cls.dev = _load_split("dev")
        cls.test = _load_split("test")
        cls.all_samples = cls.train + cls.dev + cls.test
        cls.with_calls = [s for s in cls.all_samples
                          if s.get("expected_tool_calls")]

    def train_with_calls(self) -> dict:
        if not self.with_calls:
            self.skipTest("no D2 sample contains expected_tool_calls")
        return deepcopy(self.with_calls[0])

    def _call_validator(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_well_formedness_module", GENERATOR)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_every_assistant_tool_call_has_corresponding_tool_message(self) -> None:
        """Every assistant tool_call.id must be referenced by a tool message
        in the same sample; no orphan tool messages either."""
        module = self._call_validator()
        for sample in self.all_samples:
            errors = module.transcript_well_formedness_errors(sample)
            self.assertEqual(
                [], errors,
                f"{sample['id']} transcript well-formedness: {errors}",
            )

    def test_tool_messages_appear_in_assistant_call_order(self) -> None:
        """The tool messages must follow the assistant tool calls in order,
        with no two consecutive assistant tool calls without an
        intervening tool response.
        """
        for sample in self.all_samples:
            asst_ids: list[str] = []
            tool_ids: list[str] = []
            for message in sample["messages"]:
                if message["role"] == "assistant":
                    asst_ids.extend(
                        call["id"] for call in message.get("tool_calls", [])
                    )
                elif message["role"] == "tool":
                    tool_ids.append(message["tool_call_id"])
            if not asst_ids:
                continue
            self.assertEqual(
                asst_ids, tool_ids,
                f"{sample['id']} tool order {tool_ids} != asst order {asst_ids}",
            )

    def test_assistant_tool_call_arguments_match_expected(self) -> None:
        """For each assistant tool call, the parsed function.arguments
        must equal the expected_tool_calls entry's ``arguments``.
        """
        import json
        expected_by_id = {}
        for sample in self.all_samples:
            for call in sample.get("expected_tool_calls", []):
                expected_by_id.setdefault(sample["id"], {})[call["call_id"]] = call
        for sample in self.all_samples:
            expected = expected_by_id.get(sample["id"], {})
            for message in sample["messages"]:
                if message["role"] != "assistant":
                    continue
                for call in message.get("tool_calls", []):
                    cid = call["id"]
                    self.assertIn(cid, expected,
                                  f"{sample['id']} unknown asst call {cid}")
                    exp = expected[cid]
                    self.assertEqual(
                        call["function"]["name"], exp["name"],
                        f"{sample['id']} {cid} name mismatch",
                    )
                    parsed = json.loads(call["function"]["arguments"])
                    self.assertEqual(
                        parsed, exp["arguments"],
                        f"{sample['id']} {cid} arguments mismatch",
                    )

    def test_transcript_well_formedness_errors_catches_orphan_tool_message(self) -> None:
        """Injecting an orphan tool message must be reported."""
        module = self._call_validator()
        sample = deepcopy(self.train[0])
        sample["messages"].append({
            "role": "tool",
            "content": "orphan",
            "tool_call_id": "call-d2-train-9999-0001",
            "name": "d1_calculate",
        })
        errors = module.transcript_well_formedness_errors(sample)
        self.assertTrue(
            any("tool message references unknown tool_call_id" in e for e in errors),
            f"orphan tool message not flagged: {errors}",
        )

    def test_transcript_well_formedness_errors_catches_argument_mismatch(self) -> None:
        """An assistant tool call whose arguments differ from expected must be reported."""
        module = self._call_validator()
        sample = self.train_with_calls()
        asst_calls = [
            call for message in sample["messages"]
            if message["role"] == "assistant"
            for call in message.get("tool_calls", [])
        ]
        original = asst_calls[0]["function"]["arguments"]
        asst_calls[0]["function"]["arguments"] = '{"expression": "999 + 1"}'
        try:
            errors = module.transcript_well_formedness_errors(sample)
            self.assertTrue(
                any("arguments mismatch" in e for e in errors),
                f"argument mismatch not flagged: {errors}",
            )
        finally:
            asst_calls[0]["function"]["arguments"] = original

    def test_tool_message_after_final_answer_is_flagged(self) -> None:
        """Position: tool message placed after the final answer is invalid."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_d2_for_after_final_test", GENERATOR)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        sample = self.train_with_calls()
        final_idx = None
        for idx, message in enumerate(sample["messages"]):
            if (message.get("role") == "assistant"
                    and not (message.get("tool_calls") or [])):
                final_idx = idx
        self.assertIsNotNone(final_idx,
                             "no assistant final-answer message found")
        orphan_id = sample["expected_tool_calls"][0]["call_id"]
        sample["messages"].append({
            "role": "tool",
            "tool_call_id": orphan_id,
            "name": "d1_calculate",
            "content": "orphan",
        })
        errors = module.transcript_well_formedness_errors(sample)
        self.assertTrue(
            any("after the final answer" in e for e in errors),
            f"tool-after-final-answer not flagged: {errors}",
        )

    def test_assistant_tool_call_after_final_answer_is_flagged(self) -> None:
        """Position: assistant tool call emitted after the final answer is invalid."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_d2_for_asst_after_final_test", GENERATOR)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        sample = self.train_with_calls()
        final_idx = None
        for idx, message in enumerate(sample["messages"]):
            if (message.get("role") == "assistant"
                    and not (message.get("tool_calls") or [])):
                final_idx = idx
        self.assertIsNotNone(final_idx)
        sample["messages"].append({
            "role": "assistant",
            "tool_calls": [{
                "id": "call-orphan-after-final",
                "type": "function",
                "function": {"name": "d1_calculate",
                             "arguments": '{"expression": "1+1"}'},
            }],
            "content": None,
        })
        errors = module.transcript_well_formedness_errors(sample)
        self.assertTrue(
            any("after the final answer" in e for e in errors),
            f"assistant-tool-call-after-final-answer not flagged: {errors}",
        )


@unittest.skipUnless(_dataset_present(), "D2 dataset not generated yet")
class D2SplitDisjointnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = _load_split("train")
        cls.dev = _load_split("dev")
        cls.test = _load_split("test")

    def test_split_ids_are_disjoint(self) -> None:
        train_ids = {s["id"] for s in self.train}
        dev_ids = {s["id"] for s in self.dev}
        test_ids = {s["id"] for s in self.test}
        self.assertEqual(train_ids & dev_ids, set(),
                         "train ∩ dev is non-empty")
        self.assertEqual(train_ids & test_ids, set(),
                         "train ∩ test is non-empty")
        self.assertEqual(dev_ids & test_ids, set(),
                         "dev ∩ test is non-empty")

    def test_split_paths_are_disjoint(self) -> None:
        """Files live under separate directories (datasets/tool-calling-d2/{train,dev,test}/)."""
        train_paths = {s["id"] + ".json" for s in self.train}
        dev_paths = {s["id"] + ".json" for s in self.dev}
        test_paths = {s["id"] + ".json" for s in self.test}
        self.assertEqual(train_paths & dev_paths, set())
        self.assertEqual(train_paths & test_paths, set())
        self.assertEqual(dev_paths & test_paths, set())

    def test_train_disjoint_from_d1_and_d1llm_train(self) -> None:
        d1_train = ROOT / "datasets" / "tool-calling-d1" / "train"
        d1llm_train = ROOT / "datasets" / "tool-calling-d1-llm" / "train"
        d1_ids: set[str] = set()
        if d1_train.exists():
            d1_ids |= {p.stem for p in d1_train.glob("*.json")}
        if d1llm_train.exists():
            d1_ids |= {p.stem for p in d1llm_train.glob("*.json")}
        d2_train_ids = {s["id"] for s in self.train}
        self.assertEqual(d2_train_ids & d1_ids, set(),
                         "D2 train ids overlap with D1/D1.1 train ids")

    def test_canonical_semantic_content_is_disjoint_across_splits(self) -> None:
        """ID disjointness alone is insufficient for a held-out split."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "generate_d2_for_signature_test", GENERATOR)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        signatures = {
            split: {module.canonical_content_signature(sample)
                    for sample in samples}
            for split, samples in (("train", self.train),
                                   ("dev", self.dev),
                                   ("test", self.test))
        }
        self.assertEqual(set(), signatures["train"] & signatures["dev"])
        self.assertEqual(set(), signatures["train"] & signatures["test"])
        self.assertEqual(set(), signatures["dev"] & signatures["test"])
        # Exact 5000 / 3500 / 750 / 750 contract (auditor round 14).
        expected_total = _expected_total_samples()
        self.assertEqual(
            5000, expected_total,
            f"expected exactly 5000 samples total, got {expected_total}",
        )
        self.assertEqual(
            5000, len(set().union(*signatures.values())),
            f"expected 5000 unique canonical signatures",
        )
        # Per-split counts must hit the 3500 / 750 / 750 contract.
        for split, expected in (("train", 3500), ("dev", 750), ("test", 750)):
            manifest = json.loads(
                (D2 / f"MANIFEST-{split}.json").read_text(encoding="utf-8"))
            self.assertEqual(
                expected, manifest["count"],
                f"{split} manifest count {manifest['count']} != {expected}",
            )
        all_samples = self.train + self.dev + self.test
        # Each task_type must have at least 833 unique canonical variants.
        # The on-disk distribution may differ per-class (some classes at
        # 833, others at 834, depending on the per_class_counts plan) so
        # we assert the minimum, not equality.
        for task_type in EXPECTED_TASK_TYPES:
            task_signatures = {
                module.canonical_content_signature(sample)
                for sample in all_samples
                if sample["metadata"]["task_type"] == task_type
            }
            self.assertGreaterEqual(
                len(task_signatures), 833,
                f"{task_type} must contain >= 833 unique semantic instances "
                f"(got {len(task_signatures)})",
            )

    def test_d2_canonical_content_is_disjoint_from_d1_d1llm_train(self) -> None:
        """D2 held-out split must be semantically disjoint from D1 / D1.1 train.

        The objective requires D2's held-out split to be a fully non-overlapping
        held-out benchmark. ID disjointness alone is insufficient: D2 samples
        may share their user-turn content with a D1.1 train sample even when
        the sample IDs do not collide. This test projects both D2 and the
        neighbouring D1 / D1.1 train samples into a single comparable
        ``cross_dataset_signature`` (8 fields: task_type, schema_version,
        user_turns, assistant_turns, tool_turns, tool_names, expected_tool
        calls, expected_answer) and asserts the sets are pairwise disjoint.
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "generate_d2_for_d1_disjoint_test", GENERATOR)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def load(path: Path) -> list[dict]:
            if not path.exists():
                return []
            return [json.loads(p.read_text(encoding="utf-8"))
                    for p in sorted(path.glob("*.json"))]

        def cross_sig(sample: dict) -> str:
            return module.cross_dataset_signature(sample)

        d1_train = load(ROOT / "datasets" / "tool-calling-d1" / "train")
        d1llm_train = load(ROOT / "datasets" / "tool-calling-d1-llm" / "train")
        d1_sigs: set[str] = {cross_sig(s) for s in d1_train}
        d1llm_sigs: set[str] = {cross_sig(s) for s in d1llm_train}

        for split, samples in (("train", self.train),
                               ("dev", self.dev),
                               ("test", self.test)):
            split_sigs = {cross_sig(s) for s in samples}
            overlap_d1 = split_sigs & d1_sigs
            overlap_d1llm = split_sigs & d1llm_sigs
            self.assertEqual(
                set(), overlap_d1,
                f"D2 {split} has {len(overlap_d1)} cross-dataset signatures "
                f"that also appear in D1 train",
            )
            self.assertEqual(
                set(), overlap_d1llm,
                f"D2 {split} has {len(overlap_d1llm)} cross-dataset signatures "
                f"that also appear in D1.1 train",
            )

        self.assertGreater(
            len(d1llm_sigs), 0,
            "D1.1 train is expected to exist; otherwise this check is "
            "vacuously satisfied",
        )

    def test_cross_dataset_signature_is_comparable_across_d1_d1llm_d2(self) -> None:
        """Both D1 / D1.1 / D2 samples must yield the SAME signature shape.

        This guards against a round 11-style regression where two distinct
        projections produce structurally-incompatible JSON whose set
        intersection is always empty regardless of the underlying data.
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "generate_d2_for_shape_test", GENERATOR)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def keys_of(sample_path: Path) -> set[str]:
            return set(json.loads(
                module.cross_dataset_signature(
                    json.loads(sample_path.read_text(encoding="utf-8"))
                )
            ).keys())

        d2_keys = keys_of(sorted((ROOT / "datasets" / "tool-calling-d2" / "train").glob("*.json"))[0])
        d1_path = ROOT / "datasets" / "tool-calling-d1" / "train"
        d1llm_path = ROOT / "datasets" / "tool-calling-d1-llm" / "train"
        if d1_path.exists():
            self.assertEqual(d2_keys,
                             keys_of(sorted(d1_path.glob("*.json"))[0]))
        if d1llm_path.exists():
            self.assertEqual(d2_keys,
                             keys_of(sorted(d1llm_path.glob("*.json"))[0]))

    def test_cross_dataset_signature_detects_real_overlap(self) -> None:
        """A fabricated duplicate must actually collide in cross_dataset_signature.

        If two samples project to the same user turn, task_type, expected
        tool calls and expected answer, their ``cross_dataset_signature``
        must be byte-identical. This is the positive control that proves
        the projection can detect overlap rather than structurally cannot.
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "generate_d2_for_collision_test", GENERATOR)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        candidates = [s for s in self.train + self.dev + self.test
                      if s.get("expected_tool_calls")]
        if not candidates:
            self.skipTest("no D2 sample contains expected_tool_calls")
        sample = deepcopy(candidates[0])
        duplicate = deepcopy(sample)
        duplicate["id"] = "d2-dev-9999"
        duplicate["metadata"] = dict(sample["metadata"])
        duplicate["metadata"]["split"] = "dev"
        self.assertEqual(
            module.cross_dataset_signature(sample),
            module.cross_dataset_signature(duplicate),
            "cross_dataset_signature must collide when two samples share the "
            "same semantic content (positive control)",
        )

    def test_canonical_duplicate_is_rejected_by_generator_validation(self) -> None:
        """Changing only bookkeeping fields must not bypass duplicate checks."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "generate_d2_for_duplicate_test", GENERATOR)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original = deepcopy(self.train[0])
        duplicate = deepcopy(original)
        duplicate["id"] = "d2-dev-999"
        duplicate["metadata"]["split"] = "dev"
        duplicate["metadata"]["created_at"] = "2026-07-25T18:00:00Z"
        errors = module._validate_samples([original, duplicate])
        self.assertTrue(
            any("duplicate canonical semantic content" in error for error in errors),
            f"duplicate content was not rejected: {errors}",
        )

    def test_split_assignment_uses_iid_stratified_shuffle(self) -> None:
        """Round 10: per-task variants must be evenly distributed 70/15/15.

        Under the previous contiguous slicing implementation, every
        task_type assigned the first 70 % of its variants to train, the
        next 15 % to dev, and the final 15 % to test. The new
        ``assign_split_ids`` does a seeded stratified shuffle per
        task_type so the dev/test slices sample the same per-task
        variant pool as train.
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "generate_d2_for_iid_test", GENERATOR)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Replay the build path with the default seed.
        import random
        build_count = _build_count() or 600
        samples = module.assign_split_ids(
            module.build_samples(build_count, random.Random(2026), seed=2026),
            seed=2026,
        )
        from collections import Counter
        per_type = {
            task: {"train": 0, "dev": 0, "test": 0}
            for task in EXPECTED_TASK_TYPES
        }
        for s in samples:
            per_type[s["metadata"]["task_type"]][s["metadata"]["split"]] += 1
        # Round 14: with 5000 total + round()-based dev/test allocation,
        # all 6 classes hit dev = test = 125, and train = per_class - 250.
        # Per-class totals are 833 or 834 depending on the per_class_counts
        # plan; expected per-task is therefore {833: 583 train}, {834: 584 train}.
        expected_per_task = {
            833: {"train": 583, "dev": 125, "test": 125},
            834: {"train": 584, "dev": 125, "test": 125},
        }
        for task, splits in per_type.items():
            # Find the per-class total by summing splits.
            per_class_n = sum(splits.values())
            self.assertIn(
                per_class_n, expected_per_task,
                f"{task} unexpected per-class total {per_class_n}: {splits}",
            )
            self.assertEqual(
                expected_per_task[per_class_n], splits,
                f"{task} split distribution not 70/15/15: {splits}",
            )

    def test_split_assignment_is_deterministic_for_same_seed(self) -> None:
        """Round 10: two invocations with the same seed must produce the
        same per-(task_type, variant) split assignment.
        """
        import importlib.util
        import random

        def _assignment():
            spec = importlib.util.spec_from_file_location(
                "generate_d2_for_determinism", GENERATOR)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            build_count = _build_count() or 600
            samples = module.assign_split_ids(
                module.build_samples(build_count, random.Random(2026), seed=2026),
                seed=2026,
            )
            # Bucket each sample by its (task_type, first user message
            # content) so we compare content groups, not ids.
            return sorted(
                (s["metadata"]["task_type"],
                 s["messages"][1]["content"] if len(s["messages"]) > 1 else "",
                 s["metadata"]["split"])
                for s in samples
            )

        first = _assignment()
        second = _assignment()
        self.assertEqual(first, second)

    def test_canonical_content_unique_within_dataset(self) -> None:
        """Round 10: every row's canonical semantic signature must be unique."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "generate_d2_for_unique_test", GENERATOR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        all_samples = self.train + self.dev + self.test
        seen: dict[str, str] = {}
        for s in all_samples:
            sig = module.canonical_content_signature(s)
            previous = seen.get(sig)
            self.assertIsNone(
                previous,
                f"duplicate canonical content: {s['id']} and {previous}",
            )
            seen[sig] = s["id"]

    def test_canonical_content_disjoint_across_splits(self) -> None:
        """Round 10: per-task canonical signatures must not leak across splits."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "generate_d2_for_disjoint_test", GENERATOR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        signatures = {
            split: {module.canonical_content_signature(s)
                    for s in samples}
            for split, samples in (("train", self.train),
                                   ("dev", self.dev),
                                   ("test", self.test))
        }
        for task in EXPECTED_TASK_TYPES:
            task_sigs = {
                split: {module.canonical_content_signature(s)
                        for s in samples
                        if s["metadata"]["task_type"] == task}
                for split, samples in (("train", self.train),
                                       ("dev", self.dev),
                                       ("test", self.test))
            }
            for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
                self.assertEqual(
                    set(), task_sigs[a] & task_sigs[b],
                    f"{task} {a}/{b} canonical overlap",
                )

    def test_split_counts_match_manifest(self) -> None:
        for split, samples in (("train", self.train),
                               ("dev", self.dev),
                               ("test", self.test)):
            manifest = json.loads(
                (D2 / f"MANIFEST-{split}.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["count"], len(samples),
                             f"{split} manifest count != on-disk count")
            self.assertEqual(
                manifest["data_version"], "D2",
                f"{split} manifest data_version != D2")

    def test_aggregate_sha_matches_recomputed(self) -> None:
        for split, samples in (("train", self.train),
                               ("dev", self.dev),
                               ("test", self.test)):
            manifest = json.loads(
                (D2 / f"MANIFEST-{split}.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["aggregate_sha256"],
                             _aggregate_sha(samples),
                             f"{split} aggregate_sha256 stale")

    def test_per_file_sha_matches_manifest(self) -> None:
        for split, samples in (("train", self.train),
                               ("dev", self.dev),
                               ("test", self.test)):
            manifest = json.loads(
                (D2 / f"MANIFEST-{split}.json").read_text(encoding="utf-8"))
            for entry in manifest["samples"]:
                file_path = D2 / entry["path"]
                actual_sha = hashlib.sha256(
                    file_path.read_bytes()).hexdigest()
                self.assertEqual(
                    entry["sha256"], actual_sha,
                    f"{entry['path']} sha256 mismatch",
                )


@unittest.skipUnless(_dataset_present(), "D2 dataset not generated yet")
class D2IdFormatTests(unittest.TestCase):
    """Sample ids follow strict prefixes per split so consumers can
    identify partition membership without reading metadata."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.train = _load_split("train")
        cls.dev = _load_split("dev")
        cls.test = _load_split("test")

    def test_train_ids_have_train_prefix(self) -> None:
        for s in self.train:
            self.assertTrue(s["id"].startswith("d2-train-"),
                            f"{s['id']} missing d2-train- prefix")

    def test_dev_ids_have_dev_prefix(self) -> None:
        for s in self.dev:
            self.assertTrue(s["id"].startswith("d2-dev-"),
                            f"{s['id']} missing d2-dev- prefix")

    def test_test_ids_have_test_prefix(self) -> None:
        for s in self.test:
            self.assertTrue(s["id"].startswith("d2-test-"),
                            f"{s['id']} missing d2-test- prefix")


class D2TimestampContractTests(unittest.TestCase):
    """Per-sample ``created_at`` and MANIFEST ``created_at`` must follow
    the deterministic formula shared with the D1 generator::

        datetime.fromtimestamp(1785000000 + seed + index, tz=UTC)
            .isoformat().replace("+00:00", "Z")

    ``index`` is the sample's 1-based position in the ordered train /
dev / test split (1 for d2-train-0001, 1 for d2-dev-0001, etc.). The
exact string equality check guards against off-by-one drift and the
``+00:00`` vs ``Z`` formatting convention difference."""

    def setUp(self) -> None:
        # Trigger the lazy loader used by the rest of the test file so
        # we can call generate_d2_dataset private helpers.
        if not hasattr(D2TimestampContractTests, "_module"):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "_generate_d2_dataset_for_tests",
                ROOT / "scripts" / "generate_d2_dataset.py",
            )
            D2TimestampContractTests._module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(D2TimestampContractTests._module)
        self.mod = D2TimestampContractTests._module

    def test_now_ts_is_canonical_for_known_pairs(self) -> None:
        for seed, index in [(2026, 1), (2026, 7), (2027, 90), (0, 420)]:
            actual = self.mod._now_ts(seed, index)
            expected = (
                datetime.fromtimestamp(1785000000 + seed + index, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
            self.assertEqual(expected, actual,
                             f"seed={seed} index={index}: {actual!r} != {expected!r}")

    def test_now_ts_differs_between_seeds(self) -> None:
        self.assertNotEqual(self.mod._now_ts(2026, 1),
                            self.mod._now_ts(2027, 1),
                            "timestamps must depend on seed")

    def test_now_ts_differs_between_indices(self) -> None:
        self.assertNotEqual(self.mod._now_ts(2026, 1),
                            self.mod._now_ts(2026, 2),
                            "timestamps must depend on index")

    def test_now_ts_is_byte_identical_for_repeated_calls(self) -> None:
        self.assertEqual(self.mod._now_ts(2026, 1),
                         self.mod._now_ts(2026, 1),
                         "same seed+index must produce byte-identical timestamps")

    def test_on_disk_samples_use_canonical_formula(self) -> None:
        # Round 10: each split's sample id is 1-based within the split
        # (d2-train-0001 .. d2-train-0420, d2-dev-0001 .. d2-dev-0090,
        # d2-test-0001 .. d2-test-0090), and ``created_at`` uses that
        # split-local position as the index into the canonical formula.
        # This is independent of the global build-time variant identity.
        for split_name in ("train", "dev", "test"):
            split = _load_split(split_name)
            for sample in split:
                number = int(sample["id"].rsplit("-", 1)[-1])
                self.assertEqual(
                    number, sample["metadata"]["created_at_pos"],
                    f"{sample['id']} created_at_pos does not match id number",
                )
                expected = (
                    datetime.fromtimestamp(1785000000 + 2026 + number,
                                           tz=timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
                actual = sample["metadata"]["created_at"]
                self.assertEqual(
                    expected, actual,
                    f"{sample['id']} created_at={actual!r} != {expected!r}",
                )

    def test_manifest_created_at_uses_canonical_formula(self) -> None:
        for split_name in ("train", "dev", "test"):
            manifest = json.loads(
                (D2 / f"MANIFEST-{split_name}.json").read_text(encoding="utf-8")
            )
            expected = (
                datetime.fromtimestamp(1785000000 + manifest["seed"],
                                       tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
            self.assertEqual(expected, manifest["created_at"],
                             f"MANIFEST-{split_name} created_at drift")


@unittest.skipUnless(_dataset_present(), "D2 dataset not generated yet")
class D2ExpectedAnswerContractTests(unittest.TestCase):
    """Round 13: ``expected_answer`` must equal the final assistant message.

    The protocol states ``expected_answer`` is the assistant's terminal
    reply that closes the entire transcript. The generator's
    ``_insufficient_result_search`` originally emitted a *mid-conversation*
    clarification as ``expected_answer`` while the transcript then
    appended a second user turn and a closing assistant acknowledgement;
    the two diverged across 100/100 samples. This class asserts the
    invariant holds dataset-wide.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.train = _load_split("train")
        cls.dev = _load_split("dev")
        cls.test = _load_split("test")

    def _final_assistant_content(self, sample: dict) -> str | None:
        finals = [m for m in sample["messages"]
                  if m.get("role") == "assistant"
                  and not (m.get("tool_calls") or [])
                  and m.get("content") not in (None, "")]
        if not finals:
            return None
        return finals[-1].get("content")

    def test_expected_answer_equals_final_assistant_content(self) -> None:
        for split, samples in (("train", self.train),
                               ("dev", self.dev),
                               ("test", self.test)):
            for sample in samples:
                final = self._final_assistant_content(sample)
                self.assertIsNotNone(
                    final,
                    f"{sample['id']} ({split}) has no final assistant "
                    f"message but expected_answer is set",
                )
                self.assertEqual(
                    final, sample.get("expected_answer"),
                    f"{sample['id']} ({split}, "
                    f"{sample['metadata']['task_type']}) "
                    f"expected_answer does not match final assistant content",
                )

    def test_transcript_well_formedness_flags_expected_answer_mismatch(self) -> None:
        """The validator must catch a deliberate expected_answer drift."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_d2_for_expected_test", GENERATOR)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with_calls = [s for s in self.train + self.dev + self.test
                      if s.get("expected_tool_calls")]
        self.assertTrue(with_calls,
                        "expected at least one D2 sample with expected_tool_calls")
        sample = deepcopy(with_calls[0])
        original = sample["expected_answer"]
        sample["expected_answer"] = "this is not what the transcript says"
        try:
            errors = module.transcript_well_formedness_errors(sample)
            self.assertTrue(
                any("expected_answer" in e for e in errors),
                f"expected_answer drift not flagged: {errors}",
            )
        finally:
            sample["expected_answer"] = original


if __name__ == "__main__":
    unittest.main()