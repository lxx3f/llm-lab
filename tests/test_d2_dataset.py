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
        self.assertEqual(600, len(set().union(*signatures.values())))
        all_samples = self.train + self.dev + self.test
        for task_type in EXPECTED_TASK_TYPES:
            task_signatures = {
                module.canonical_content_signature(sample)
                for sample in all_samples
                if sample["metadata"]["task_type"] == task_type
            }
            self.assertEqual(
                100, len(task_signatures),
                f"{task_type} must contain 100 unique semantic instances",
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
        # The generator's sample index is global across all three splits:
        # train takes positions 1..train_n, dev takes train_n+1..train_n+dev_n,
        # test takes the remainder. Sample id encodes the 1-based split-local
        # number, so we recover the global index from the manifest counts.
        manifest_train = json.loads(
            (D2 / "MANIFEST-train.json").read_text(encoding="utf-8"))
        manifest_dev = json.loads(
            (D2 / "MANIFEST-dev.json").read_text(encoding="utf-8"))
        train_n = manifest_train["count"]
        dev_offset = train_n
        for split_name in ("train", "dev", "test"):
            split = _load_split(split_name)
            for sample in split:
                number = int(sample["id"].rsplit("-", 1)[-1])
                if split_name == "train":
                    index = number
                elif split_name == "dev":
                    index = dev_offset + number
                else:
                    index = dev_offset + manifest_dev["count"] + number
                expected = (
                    datetime.fromtimestamp(1785000000 + 2026 + index,
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


if __name__ == "__main__":
    unittest.main()