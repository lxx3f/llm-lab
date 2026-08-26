"""N2 benchmark and routing-statistics tests."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from architecture_lab.benchmarks.n2 import (
    aggregate_routing,
    build_models,
    cache_binding,
    choose_protocol_d_ff,
    load_settings,
    resolve_device,
    resolve_dtype,
    set_seed,
    train_for_benchmark,
)
from architecture_lab.benchmarks.results import validate_n2_result
from architecture_lab.tokenization import BPETokenizer
from scripts.run_n2_benchmark import benchmark


def _has_working_cuda() -> bool:
    """Return True only if a CUDA forward+backward actually executes on this machine.

    Some environments report ``torch.cuda.is_available() == True`` but the installed
    PyTorch wheel does not contain a kernel image for the GPU's compute
    capability (e.g. sm_120 RTX 5070 Ti on a PyTorch build that only ships
    up to sm_90). Running CUDA ops in such an env raises::

        RuntimeError: CUDA error: no kernel image is available for execution on the device

    Tests that drive a real CUDA benchmark should be guarded with
    ``@unittest.skipUnless(_has_working_cuda(), ...)`` so they skip cleanly
    instead of erroring out and breaking the ``full`` test suite.
    """
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.randn(2, 3, device="cuda", requires_grad=True)
        y = (x * 2).sum()
        y.backward()
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


class N2BenchmarkTests(unittest.TestCase):
    def test_routing_aggregation_reports_load_and_drop_rate(self) -> None:
        result = aggregate_routing([
            {"num_experts": 2, "capacity_factor": 1.0, "capacity": 3, "tokens": 6, "assigned_tokens_per_expert": [4, 2], "kept_tokens_per_expert": [3, 2], "dropped_tokens": 1},
            {"num_experts": 2, "capacity_factor": 1.0, "capacity": 3, "tokens": 6, "assigned_tokens_per_expert": [2, 4], "kept_tokens_per_expert": [2, 3], "dropped_tokens": 1},
        ])
        self.assertEqual(result["assigned_tokens_per_expert"], [6, 6])
        self.assertEqual(result["dropped_tokens"], 2)
        self.assertEqual(result["dropped_token_ratio"], 2 / 6)
        self.assertEqual(result["load_imbalance"], 1.0)

    def test_protocol_b_and_total_parameter_alignment(self) -> None:
        settings = {
            "model": {"vocab_size": 32, "max_seq_len": 8, "d_model": 16, "n_heads": 4, "n_layers": 1, "d_ff": 16},
            "moe": {"num_experts": 4, "capacity_factor": 1.0, "aux_loss_weight": 0.01},
        }
        protocol_a = choose_protocol_d_ff(settings, "A", 32)
        protocol_b = choose_protocol_d_ff(settings, "B", 32)
        self.assertEqual(protocol_a["moe_d_ff"], 4)
        self.assertEqual(protocol_b["dense_d_ff"], protocol_b["moe_d_ff"])

    def test_binding_validates_both_caches_and_uses_configured_budget(self) -> None:
        settings = load_settings("configs/n2_benchmark.example.yaml")
        tokenizer = BPETokenizer.load(settings["data"]["tokenizer"])
        binding = cache_binding(settings, tokenizer)
        self.assertEqual(binding["token_budget"], settings["benchmark"]["token_budget"])
        self.assertEqual(len(binding["train_cache_sha256"]), 64)
        self.assertEqual(len(binding["validation_cache_sha256"]), 64)

    def test_short_training_reads_validation_cache_and_returns_losses(self) -> None:
        settings = load_settings("configs/n2_benchmark.example.yaml")
        set_seed(settings["benchmark"]["seed"])
        device = resolve_device("cpu")
        dtype = resolve_dtype("float32")
        model, _ = build_models(settings, "DenseTransformer", 64, device, dtype, vocab_size=8192)
        metrics = train_for_benchmark(model, settings, architecture="DenseTransformer", device=device)
        self.assertGreater(metrics["optimizer_steps"], 0)
        self.assertGreater(metrics["effective_tokens"], 0)
        self.assertIsNotNone(metrics["validation_lm_loss"])


    @unittest.skipUnless(_has_working_cuda(), "no working CUDA kernel image on this host")
    def test_executed_benchmark_records_validation_and_capacity_bindings(self) -> None:
        config = Path("configs/n2_benchmark.example.yaml")
        dense = benchmark(config, "A", "DenseTransformer")
        moe = benchmark(config, "A", "MoETransformer")
        self.assertEqual(dense["binding"], moe["binding"])
        for result in (dense, moe):
            self.assertIsNotNone(result["metrics"]["validation_lm_loss"])
            self.assertEqual(result["binding"]["token_budget"], 128)
            self.assertEqual(len(result["binding"]["train_cache_sha256"]), 64)
            self.assertEqual(len(result["binding"]["validation_cache_sha256"]), 64)
            self.assertFalse(result["routing"]["collect_stats"])
        self.assertEqual(moe["routing"]["prefill_capacity_factor"], 1.0)
        self.assertEqual(moe["routing"]["decode_capacity_factor"], 2.0)

    @unittest.skipUnless(_has_working_cuda(), "no working CUDA kernel image on this host")
    def test_benchmark_result_includes_unified_metadata_block(self) -> None:
        config = Path("configs/n2_benchmark.example.yaml")
        result = benchmark(config, "A", "DenseTransformer")
        metadata = result["metadata"]
        expected_fields = {
            "git_commit", "config_sha256", "python_version", "pytorch_version",
            "cuda_version", "gpu_name", "gpu_compute_capability",
            "tokenizer_revision", "dataset_hash", "seed",
        }
        self.assertEqual(set(metadata), expected_fields)
        self.assertIsInstance(metadata["python_version"], str)
        self.assertEqual(metadata["seed"], int(metadata["seed"]))
        # git_commit may be null in environments without git available; otherwise
        # it must match the standard short or long sha pattern.
        import re
        self.assertRegex(
            metadata["git_commit"] or "0" * 40,
            r"^([0-9a-f]{40}|\d{14}-[a-z0-9]{6})$",
        )

    def test_schema_rejects_missing_capacity_metrics(self) -> None:
        with self.assertRaises(ValueError):
            validate_n2_result({"schema_version": "1.0"})


if __name__ == "__main__":
    unittest.main()
