"""N2 benchmark and routing-statistics tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from architecture_lab.benchmarks.n2 import aggregate_routing, choose_protocol_d_ff
from architecture_lab.benchmarks.results import validate_n2_result


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

    def test_schema_rejects_missing_capacity_metrics(self) -> None:
        with self.assertRaises(ValueError):
            validate_n2_result({"schema_version": "1.0"})

    def test_protocol_configuration_requires_both_widths(self) -> None:
        settings = {
            "model": {"vocab_size": 32, "max_seq_len": 8, "d_model": 16, "n_heads": 4, "n_layers": 1, "d_ff": 16},
            "moe": {"num_experts": 4, "capacity_factor": 1.0, "aux_loss_weight": 0.01},
        }
        result = choose_protocol_d_ff(settings, "A", 32)
        self.assertEqual(result["dense_d_ff"], 16)
        self.assertGreaterEqual(result["moe_d_ff"], 1)


if __name__ == "__main__":
    unittest.main()
