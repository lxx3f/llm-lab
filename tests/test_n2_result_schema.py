"""Schema tests for N2 benchmark and routing artifacts."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from architecture_lab.benchmarks.results import validate_n2_result
from jsonschema import Draft202012Validator


class N2ResultSchemaTests(unittest.TestCase):
    def test_benchmark_artifacts_validate(self) -> None:
        schema = json.loads(Path("schemas/n2_benchmark_result.schema.json").read_text())
        validator = Draft202012Validator(schema)
        for path in ("artifacts/n2-a-dense.json", "artifacts/n2-a-moe.json", "artifacts/n2-b-dense.json", "artifacts/n2-b-moe.json"):
            if not Path(path).is_file():
                self.skipTest(f"local N2 artifact is not present: {path}")
            result = json.loads(Path(path).read_text())
            self.assertEqual(list(validator.iter_errors(result)), [], path)
            validate_n2_result(result)

    def test_routing_artifact_validates(self) -> None:
        schema = json.loads(Path("schemas/n2_routing_stats.schema.json").read_text())
        artifact = Path("artifacts/n2-moe-routing-stats.json")
        if not artifact.is_file():
            self.skipTest(f"local N2 artifact is not present: {artifact}")
        result = json.loads(artifact.read_text())
        self.assertEqual(list(Draft202012Validator(schema).iter_errors(result)), [])
        self.assertTrue(result["collect_stats"])
        self.assertEqual(result["prefill_capacity_factor"], 1.0)
        self.assertEqual(result["decode_capacity_factor"], 2.0)


if __name__ == "__main__":
    unittest.main()
