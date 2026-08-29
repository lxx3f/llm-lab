"""Regression tests for artifact-reference extraction in
scripts/audit/run_doc_artifact_reconciliation.py.

Verifies that the regex captures:
  - concrete paths
  - paths with mid-wildcards (*)
  - paths with brace-expansions ({a,b,c})
  - paths with angle-bracket placeholders (<foo>)
  - paths with trailing slash (directory)
  - paths with trailing hyphen/underscore/... (placeholder)
And that classify_ref() assigns each kind correctly.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path
import sys

# Make the audit script importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

from run_doc_artifact_reconciliation import (
    ARTIFACT_REF_RE,
    classify_ref,
    expand_brace,
    expand_glob,
)


class ArtifactRefExtractionTests(unittest.TestCase):
    """Verify the regex captures all documented reference forms."""

    def test_concrete_path(self) -> None:
        text = "see `artifacts/foo-result.json` for details"
        refs = ARTIFACT_REF_RE.findall(text)
        self.assertEqual(refs, ["artifacts/foo-result.json"])

    def test_concrete_path_with_extension(self) -> None:
        text = "outputs to artifacts/dense-owt-formal-curve-result.json"
        refs = ARTIFACT_REF_RE.findall(text)
        self.assertEqual(refs, ["artifacts/dense-owt-formal-curve-result.json"])

    def test_path_with_mid_wildcard(self) -> None:
        text = "see `artifacts/*-result.json` for results"
        refs = ARTIFACT_REF_RE.findall(text)
        self.assertEqual(refs, ["artifacts/*-result.json"])

    def test_path_with_directory_wildcard(self) -> None:
        text = "states from `artifacts/grpo-experiment/*/state.json`"
        refs = ARTIFACT_REF_RE.findall(text)
        self.assertEqual(refs, ["artifacts/grpo-experiment/*/state.json"])

    def test_brace_expansion(self) -> None:
        text = "artifacts/{qwen2.5,huggingfacetb-smollm2}-*-eval-d2dev.json"
        refs = ARTIFACT_REF_RE.findall(text)
        self.assertEqual(
            refs,
            ["artifacts/{qwen2.5,huggingfacetb-smollm2}-*-eval-d2dev.json"],
        )

    def test_brace_expansion_with_directory_wildcard(self) -> None:
        text = "artifact path: artifacts/grpo-experiment/run-{A,B,C}-*/"
        refs = ARTIFACT_REF_RE.findall(text)
        self.assertEqual(refs, ["artifacts/grpo-experiment/run-{A,B,C}-*/"])

    def test_angle_bracket_placeholder(self) -> None:
        text = "see `artifacts/<model>-eval.json`"
        refs = ARTIFACT_REF_RE.findall(text)
        self.assertEqual(refs, ["artifacts/<model>-eval.json"])

    def test_path_with_trailing_slash(self) -> None:
        text = "see artifacts/checkpoints/"
        refs = ARTIFACT_REF_RE.findall(text)
        self.assertEqual(refs, ["artifacts/checkpoints/"])

    def test_path_with_trailing_hyphen(self) -> None:
        text = "see `artifacts/moe-owt-formal-curve-`"
        refs = ARTIFACT_REF_RE.findall(text)
        self.assertEqual(refs, ["artifacts/moe-owt-formal-curve-"])

    def test_multiple_refs_in_text(self) -> None:
        text = (
            "outputs: artifacts/a.json and artifacts/b.json; "
            "also artifacts/{x,y}-result.json and artifacts/c.png"
        )
        refs = ARTIFACT_REF_RE.findall(text)
        self.assertEqual(
            refs,
            [
                "artifacts/a.json",
                "artifacts/b.json",
                "artifacts/{x,y}-result.json",
                "artifacts/c.png",
            ],
        )

    def test_does_not_match_other_paths(self) -> None:
        text = "see scripts/foo.py and docs/experiments/README.md"
        refs = ARTIFACT_REF_RE.findall(text)
        self.assertEqual(refs, [])


class BraceExpansionTests(unittest.TestCase):

    def test_no_braces(self) -> None:
        self.assertEqual(expand_brace("artifacts/foo"), ["artifacts/foo"])

    def test_single_group(self) -> None:
        self.assertEqual(
            expand_brace("artifacts/{a,b,c}.json"),
            [
                "artifacts/a.json",
                "artifacts/b.json",
                "artifacts/c.json",
            ],
        )

    def test_nested_groups(self) -> None:
        self.assertEqual(
            sorted(expand_brace("artifacts/{a,b}-{x,y}.json")),
            sorted([
                "artifacts/a-x.json",
                "artifacts/a-y.json",
                "artifacts/b-x.json",
                "artifacts/b-y.json",
            ]),
        )

    def test_brace_with_wildcard(self) -> None:
        self.assertEqual(
            expand_brace("artifacts/{qwen2.5,huggingfacetb-smollm2}-*-eval.json"),
            [
                "artifacts/qwen2.5-*-eval.json",
                "artifacts/huggingfacetb-smollm2-*-eval.json",
            ],
        )


class ClassifyRefTests(unittest.TestCase):

    def test_placeholder_angle_brackets(self) -> None:
        self.assertEqual(classify_ref("artifacts/<model>.json"), "placeholder")

    def test_placeholder_trailing_hyphen(self) -> None:
        self.assertEqual(classify_ref("artifacts/sft-"), "placeholder")

    def test_placeholder_trailing_ellipsis(self) -> None:
        self.assertEqual(classify_ref("artifacts/foo..."), "placeholder")

    def test_directory_trailing_slash(self) -> None:
        self.assertEqual(classify_ref("artifacts/checkpoints/"), "directory")

    def test_glob_with_wildcard(self) -> None:
        self.assertEqual(classify_ref("artifacts/*-result.json"), "glob_pattern")

    def test_glob_with_brace(self) -> None:
        self.assertEqual(
            classify_ref("artifacts/{qwen2.5,huggingfacetb}-*.json"),
            "glob_pattern",
        )

    def test_exact_file_json(self) -> None:
        self.assertEqual(
            classify_ref("artifacts/dense-owt-formal-curve-result.json"),
            "exact_file",
        )

    def test_exact_file_png(self) -> None:
        self.assertEqual(classify_ref("artifacts/foo.png"), "exact_file")


class IntegrationWithRealDocsTests(unittest.TestCase):
    """Verify that ALL wildcards/braces in active docs are captured."""

    def test_p2_evaluator_brace_with_wildcard(self) -> None:
        """The auditor-cited reference must be captured by the regex."""
        text = (
            "5 模型产物位于 "
            "`artifacts/{qwen2.5,huggingfacetb-smollm2}-*-eval-d2dev.json` "
            "与 `artifacts/{qwen2.5,huggingfacetb-smollm2}-*-eval-d2dev-reward.json`"
        )
        refs = ARTIFACT_REF_RE.findall(text)
        self.assertEqual(len(refs), 2)
        self.assertTrue(all("{" in r and "}" in r and "*" in r for r in refs))

    def test_p4_grpo_wildcard_in_path(self) -> None:
        text = "all values from `artifacts/grpo-experiment/*/state.json`"
        refs = ARTIFACT_REF_RE.findall(text)
        self.assertEqual(len(refs), 1)
        self.assertIn("*", refs[0])

    def test_p4_grpo_protocol_brace_with_dir_wildcard(self) -> None:
        text = "artifact path: `artifacts/grpo-experiment/run-{A,B,C}-*/`"
        refs = ARTIFACT_REF_RE.findall(text)
        self.assertEqual(len(refs), 1)
        self.assertIn("{A,B,C}", refs[0])
        self.assertIn("*/", refs[0])


if __name__ == "__main__":
    unittest.main()