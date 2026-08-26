"""Run the repository's tiered test suites.

Examples:
    python scripts/run_tests.py fast
    python scripts/run_tests.py module training
    python scripts/run_tests.py full
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAST_MODULES = (
    ("tests", "test_batching.py"),
    ("tests", "test_experiment_metadata.py"),
    ("tests", "test_n2_benchmark.py"),
    ("tests", "test_n2_result_schema.py"),
    ("tests", "test_dense_result_schema.py"),
    ("tests", "test_plot_dense_curve.py"),
    ("tests", "test_stage0_schemas.py"),
    ("tests", "test_d0_manifest.py"),
    ("tests", "test_d1_failure.py"),
    ("tests", "test_sweep_doc_consistency.py"),
    ("tests", "test_artifact_provenance.py"),
    ("tests", "test_token_cache.py"),
    ("architecture_lab/tests", "test_dense_transformer.py"),
    ("architecture_lab/tokenization/tests", "test_bpe.py"),
)

MODULES = {
    "data": (
        ("tests", "test_batching.py"),
        ("tests", "test_token_cache.py"),
        ("tests", "test_token_cache_cli.py"),
        ("tests", "test_tokenizer_artifact_cli.py"),
    ),
    "training": (
        ("tests", "test_dense_training.py"),
        ("tests", "test_moe_training.py"),
        ("tests", "test_n2_benchmark.py"),
        ("tests", "test_n2_result_schema.py"),
        ("tests", "test_dense_result_schema.py"),
        ("tests", "test_plot_dense_curve.py"),
        ("tests", "test_experiment_metadata.py"),
        ("tests", "test_d0_manifest.py"),
        ("tests", "test_d1_failure.py"),
        ("tests", "test_sweep_doc_consistency.py"),
        ("tests", "test_artifact_provenance.py"),
    ),
    "architecture": (
        ("architecture_lab/tests", "test_dense_transformer.py"),
        ("architecture_lab/tests", "test_moe_transformer.py"),
    ),
    "tokenization": (
        ("architecture_lab/tokenization/tests", "test_bpe.py"),
        ("tests", "test_tokenizer_artifact_cli.py"),
    ),
}


def _load_test_target(directory: str, pattern: str) -> unittest.TestSuite:
    target = ROOT / directory / pattern
    if not target.is_file():
        print(f"[test] test target does not exist: {target}", file=sys.stderr)
        return unittest.TestSuite()
    relative = target.relative_to(ROOT).with_suffix("")
    module_name = ".".join(relative.parts)
    spec = importlib.util.spec_from_file_location(module_name, target)
    if spec is None or spec.loader is None:
        print(f"[test] cannot load test target: {target}", file=sys.stderr)
        return unittest.TestSuite()
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return unittest.defaultTestLoader.loadTestsFromModule(module)


def run_modules(modules: tuple[tuple[str, str], ...]) -> bool:
    suite = unittest.TestSuite()
    for directory, pattern in modules:
        loaded = _load_test_target(directory, pattern)
        if loaded.countTestCases() == 0:
            print(f"[test] no discoverable tests: {directory}/{pattern}", file=sys.stderr)
            return False
        suite.addTests(loaded)
    print(f"[test] running {len(modules)} test targets")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return result.wasSuccessful()


def run_stage0_examples() -> bool:
    command = [sys.executable, str(ROOT / "scripts/validate_stage0.py"), "--examples"]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return completed.returncode == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("fast", help="quick API/schema/model correctness checks")
    module = subparsers.add_parser("module", help="run one focused module suite")
    module.add_argument("name", choices=sorted(MODULES))
    subparsers.add_parser("full", help="run all unittest groups and schema examples")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "fast":
        modules = FAST_MODULES
    elif args.mode == "module":
        modules = MODULES[args.name]
    else:
        modules = (
            ("tests", "test_batching.py"),
            ("tests", "test_experiment_metadata.py"),
            ("tests", "test_n2_benchmark.py"),
            ("tests", "test_n2_result_schema.py"),
            ("tests", "test_dense_result_schema.py"),
            ("tests", "test_plot_dense_curve.py"),
            ("tests", "test_dense_training.py"),
            ("tests", "test_mock_executor.py"),
            ("tests", "test_moe_training.py"),
            ("tests", "test_stage0_schemas.py"),
            ("tests", "test_d0_manifest.py"),
            ("tests", "test_d1_failure.py"),
            ("tests", "test_sweep_doc_consistency.py"),
            ("tests", "test_artifact_provenance.py"),
            ("tests", "test_token_cache.py"),
            ("tests", "test_token_cache_cli.py"),
            ("tests", "test_tokenizer_artifact_cli.py"),
            ("architecture_lab/tests", "test_dense_transformer.py"),
            ("architecture_lab/tests", "test_moe_transformer.py"),
            ("architecture_lab/tokenization/tests", "test_bpe.py"),
        )

    if not run_modules(modules):
        return 1
    if args.mode == "full" and not run_stage0_examples():
        return 1
    print(f"[test] {args.mode} suite passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
