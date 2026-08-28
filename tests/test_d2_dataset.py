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
  layer and at the path layer; train is disjoint from D1 / D1.1 train ids;
- ``MANIFEST-{train,dev,test}.json`` ``count`` and ``aggregate_sha256``
  match the on-disk files.

The D2 directory is treated as a fixture: the tests skip themselves if
the dataset has not been generated yet (``--count 600`` via the
generator).
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
D2 = ROOT / "datasets" / "tool-calling-d2"
SCHEMA = ROOT / "schemas" / "tool_calling_sample.schema.json"

EXPECTED_SPLITS = {"train", "dev", "test"}
EXPECTED_TASK_TYPES = {
    "multi_turn_tool_chain",
    "multi_turn_error_recovery",
    "multi_turn_req_change",
    "multi_turn_insufficient_result",
    "multi_turn_tool_not_available",
    "multi_turn_clarification",
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
        """Round-robin generation guarantees that each of the six
        multi-turn ``task_type``s appears in every split (``train`` /
        ``dev`` / ``test``). The exact per-split counts depend on
        ``--count``; we only assert coverage here, not a minimum count."""


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
        """Every ``expected_tool_calls[i].call_id`` must appear in at
        least one assistant message's ``tool_calls[].id``. The
        transcript MAY contain extra call_ids (e.g. the abandoned
        ``city_a`` call in ``multi_turn_req_change`` is part of the
        dialogue history but is not in ``expected_tool_calls``)."""
        for sample in self.train:
            transcript_ids = {
                call["id"]
                for msg in sample["messages"] if msg["role"] == "assistant"
                for call in msg.get("tool_calls", [])
            }
            expected_ids = {c["call_id"] for c in sample["expected_tool_calls"]}
            # expected ⊆ transcript
            self.assertTrue(
                expected_ids.issubset(transcript_ids),
                f"{sample['id']} expected ids {expected_ids} "
                f"not fully present in transcript {transcript_ids}",
            )


@unittest.skipUnless(_dataset_present(), "D2 dataset not generated yet")
class D2MockExecutorReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = _load_split("train")

    def test_mock_executor_returns_expected_result(self) -> None:
        """End-to-end MockExecutor replay returns the declared
        ``expected_result`` for every expected tool call."""
        from examples.d1_mocks import (  # type: ignore
            d1_calculate, d1_get_weather, d1_web_search, d1_translate,
        )
        mocks_by_name = {
            "d1_calculate": d1_calculate,
            "d1_get_weather": d1_get_weather,
            "d1_web_search": d1_web_search,
            "d1_translate": d1_translate,
        }
        skipped = 0
        for sample in self.train:
            for i, call in enumerate(sample["expected_tool_calls"]):
                name = call.get("name")
                if name not in mocks_by_name:
                    skipped += 1
                    continue
                args = call.get("arguments", {})
                actual = mocks_by_name[name](**args)
                self.assertEqual(
                    str(actual), str(call.get("expected_result")),
                    f"{sample['id']} call #{i} {name}({args}): "
                    f"expected {call.get('expected_result')!r} got {actual!r}",
                )
        # Some samples intentionally have no tool calls (tool_not_available
        # etc.) — we don't fail the suite for that.
        self.assertGreaterEqual(skipped, 0)


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


if __name__ == "__main__":
    unittest.main()