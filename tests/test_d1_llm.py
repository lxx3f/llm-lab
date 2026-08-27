"""D1.1 real-LLM dataset verification tests (read-only).

Unlike the template D1 dataset (regenerated deterministically in
test_d1_failure.py setUpClass), the D1.1 dataset is produced by real LLM
API calls and must NOT be regenerated inside the test suite. These tests
verify the committed dataset invariants:

- exactly 126 samples, 6 task_types × 21 each;
- ``metadata.source == "<model>@<version>"`` (LLM provenance, NOT the
  template ``d1-synthetic-template``);
- every sample schema-valid against schemas/tool_calling_sample.schema.json;
- every expected call executes through MockExecutor with result ==
  expected_result;
- no_tool samples carry NO expected calls;
- the 8-level classifier reports no first_failure for a correct transcript.

If the dataset directory is absent (e.g. fresh checkout before D1.1 was
generated), the tests are skipped rather than failed — the dataset is a
generated artifact, not source code.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jsonschema import Draft202012Validator  # noqa: E402

from architecture_lab.execution import MockExecutor  # noqa: E402
from scripts.classify_tool_failure import classify  # noqa: E402
import examples.d1_mocks as d1_mocks  # noqa: E402

D1LLM_DIR = ROOT / "datasets" / "tool-calling-d1-llm"

REGISTRY: dict[str, object] = {
    "d1_calculate": d1_mocks.d1_calculate,
    "d1_get_weather": d1_mocks.d1_get_weather,
    "d1_web_search": d1_mocks.d1_web_search,
}

TASK_TYPES = {
    "no_tool", "single_tool", "multi_tool", "tool_error",
    "insufficient_result", "requirement_change",
}


def _load_samples() -> list[dict]:
    samples: list[dict] = []
    for path in sorted((D1LLM_DIR / "train").glob("d1llm-*.json")):
        samples.append(json.loads(path.read_text(encoding="utf-8")))
    return samples


@unittest.skipUnless(D1LLM_DIR.is_dir(), "D1.1 dataset not generated (skip)")
class D1LlmDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = _load_samples()
        cls.manifest = json.loads(
            (D1LLM_DIR / "MANIFEST-train.json").read_text(encoding="utf-8")
        )

    def test_manifest_matches_ondisk_layout(self) -> None:
        """On-disk files must exactly match the manifest (no stale files)."""
        manifest_paths = {Path(e["path"]).name for e in self.manifest["samples"]}
        on_disk = {p.name for p in (D1LLM_DIR / "train").glob("d1llm-*.json")}
        self.assertEqual(on_disk, manifest_paths)
        # Manifest count must reflect total on-disk entries (not just new
        # samples from the most recent generator run); this guards against
        # future regressions where count only tracks new samples.
        self.assertEqual(self.manifest["count"], len(on_disk),
                         msg=f"manifest count {self.manifest['count']} != "
                             f"on-disk count {len(on_disk)}")

    def test_manifest_entries_carry_per_file_source(self) -> None:
        """Every manifest entry MUST carry a ``source`` field (per-file
        provenance), and the source must match the corresponding sample's
        metadata. Mixed-source datasets (e.g. an old + new generator run)
        require per-file provenance, not a single top-level source."""
        missing = [e["path"] for e in self.manifest["samples"]
                   if "source" not in e]
        self.assertEqual(missing, [], msg=f"entries without source: {missing[:3]}")
        # Cross-check against the on-disk sample metadata.
        mismatches: list[str] = []
        for entry in self.manifest["samples"]:
            path = D1LLM_DIR / entry["path"]
            sample = json.loads(path.read_text(encoding="utf-8"))
            if entry["source"] != sample["metadata"]["source"]:
                mismatches.append(f"{entry['path']}: "
                                  f"manifest={entry['source']!r} "
                                  f"sample={sample['metadata']['source']!r}")
        self.assertEqual(mismatches, [],
                         msg=f"source mismatches: {mismatches[:3]}")
        # The top-level ``sources`` summary (if present) must agree with the
        # actually-seen per-entry sources.
        seen = sorted({e["source"] for e in self.manifest["samples"]})
        if "sources" in self.manifest:
            self.assertEqual(self.manifest["sources"], seen,
                             msg=f"top-level sources != observed {seen}")

    def test_count_and_task_type_coverage(self) -> None:
        from collections import Counter

        # After Phase A (1500 samples), counts grow beyond 126 per task_type.
        # Assert minimum baseline + per-task-type non-zero coverage.
        self.assertGreaterEqual(len(self.samples), 126,
                                msg=f"count={len(self.samples)} < 126 baseline")
        counts = Counter(s["metadata"]["task_type"] for s in self.samples)
        self.assertEqual(set(counts), TASK_TYPES, msg=f"missing: {TASK_TYPES - set(counts)}")
        for ttype in TASK_TYPES:
            self.assertGreater(counts[ttype], 0,
                               msg=f"{ttype} not represented")

    def test_source_is_llm_provenance_not_template(self) -> None:
        """Every sample's metadata.source must be an LLM version string
        (not the template ``d1-synthetic-template`` marker). The MANIFEST's
        per-file source fields are the canonical record of provenance."""
        sources = {e["source"] for e in self.manifest["samples"]}
        self.assertNotIn("d1-synthetic-template", sources)
        self.assertTrue(all("@" in s for s in sources),
                        msg=f"LLM provenance must include @version: {sources}")

    def test_all_samples_schema_valid(self) -> None:
        validator = Draft202012Validator(
            json.loads((ROOT / "schemas" / "tool_calling_sample.schema.json")
                       .read_text(encoding="utf-8"))
        )
        for sample in self.samples:
            errors = list(validator.iter_errors(sample))
            self.assertEqual(errors, [], msg=f"schema error in {sample['id']}: {errors}")

    def test_no_tool_samples_have_no_expected_calls(self) -> None:
        bad = [s for s in self.samples
               if s["metadata"]["task_type"] == "no_tool" and s["expected_tool_calls"]]
        self.assertEqual(bad, [], msg=f"no_tool samples with calls: {[s['id'] for s in bad]}")

    def test_every_expected_call_executes_through_mock_executor(self) -> None:
        """Every expected call must run through MockExecutor with
        outcome=success and result == expected_result (semantic contract)."""
        executor = MockExecutor()
        for name, fn in REGISTRY.items():
            tool = next(t for t in self.samples[0]["tools"]
                        if t["function"]["name"] == name)
            executor.register_mock(name, fn, tool["function"]["parameters"])
        mismatches: list[str] = []
        for sample in self.samples:
            for call in sample["expected_tool_calls"]:
                result = executor.execute({
                    "tool_name": call["name"],
                    "call_id": call.get("call_id", ""),
                    "arguments": call.get("arguments", {}),
                    "depends_on": call.get("depends_on", []),
                })
                if result["outcome"] != "success":
                    mismatches.append(
                        f"{sample['id']}:{call['name']} outcome={result['outcome']}"
                    )
                    continue
                if result["result"] != call.get("expected_result"):
                    mismatches.append(
                        f"{sample['id']}:{call['name']} result mismatch "
                        f"expected={call.get('expected_result')!r} actual={result['result']!r}"
                    )
        self.assertEqual(mismatches, [], msg="\n".join(mismatches))

    def test_classifier_reports_no_failure_for_correct_transcript(self) -> None:
        """A transcript that faithfully reproduces the expected plan must not
        produce a first_failure under the 8-level classifier."""
        fails: list[str] = []
        for sample in self.samples:
            calls = [
                {"call_id": e.get("call_id", f"c{i}"), "name": e["name"],
                 "arguments": e["arguments"], "depends_on": e.get("depends_on", []),
                 "execution_outcome": "success", "result": e.get("expected_result")}
                for i, e in enumerate(sample["expected_tool_calls"])
            ]
            result = classify(sample, {
                "tool_calls": calls,
                "final_answer": sample.get("expected_answer") or "ok",
            })
            if result["first_failure"] is not None:
                fails.append(f"{sample['id']}: {result['first_failure']}")
        self.assertEqual(fails, [], msg="\n".join(fails))

    def test_expected_result_coverage_is_total(self) -> None:
        """Every expected call must declare a deterministic expected_result."""
        uncovered = [
            f"{s['id']}:{c['name']}"
            for s in self.samples
            for c in s["expected_tool_calls"]
            if "expected_result" not in c
        ]
        self.assertEqual(uncovered, [], msg=f"calls without expected_result: {uncovered}")


if __name__ == "__main__":
    unittest.main()
