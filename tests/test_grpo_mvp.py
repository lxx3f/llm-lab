"""Unit tests for the P4 GRPO MVP.

These tests focus on the components that can be exercised WITHOUT
loading a real Hugging Face model:

- ``_group_relative_advantages`` (advantage standardization)
- ``_advantage_stats`` (advantage distribution summary)
- ``_hash_rollouts`` / ``_hash_rewards`` (determinism fingerprints)
- ``_assemble_step_artifact`` (artifact assembly)
- ``_save_state`` / ``_load_state`` (checkpoint I/O round-trip)
- ``_iter_prompts`` (resume-step iteration)
- ``_load_samples`` (sample loader with deterministic shuffle)
- step-artifact JSON schema validation
- resume-from previous state continues at ``global_step+1``

The full ``main`` loop is exercised in the smoke integration test
(``test_grpo_smoke_runs_on_cpu``) which uses a tiny HF model and is
gated behind the ``GRPO_SMOKE`` environment variable so it does not
run by default.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grpo_train import (  # noqa: E402
    ADVANTAGE_EPS,
    GRPO_STEP_SCHEMA_VERSION,
    _advantage_stats,
    _assemble_step_artifact,
    _group_relative_advantages,
    _hash_rewards,
    _hash_rollouts,
    _iter_prompts,
    _load_samples,
    _load_state,
    _save_state,
)


SCHEMA_PATH = ROOT / "schemas" / "grpo_step_result.schema.json"


def _validate_schema(artifact: dict) -> None:
    """Local Draft-202012 validator (avoids hard dependency on jsonschema
    package when not installed)."""
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return  # Skip if package not installed; full schema suite runs in CI
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(artifact, schema)


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------

class TestGroupRelativeAdvantages(unittest.TestCase):
    """Pure-function tests for ``_group_relative_advantages``.

    GRPO group-relative advantage = (reward - mean) / max(std, eps).
    When std is zero, the group has no learning signal — return zeros.
    """

    def test_zero_spread_returns_zeros(self):
        """All-equal rewards → no learning signal → zero advantages."""
        rewards = [{"rollout_index": i, "reward_layered": 0.5}
                   for i in range(4)]
        advantages = _group_relative_advantages(rewards)
        self.assertEqual(len(advantages), 4)
        for a in advantages:
            self.assertEqual(a, 0.0)

    def test_standardization_is_correct(self):
        """Known rewards → known advantages (mean 0, std 1)."""
        rewards = [
            {"rollout_index": 0, "reward_layered": 0.2},
            {"rollout_index": 1, "reward_layered": 0.4},
            {"rollout_index": 2, "reward_layered": 0.6},
            {"rollout_index": 3, "reward_layered": 0.8},
        ]
        advantages = _group_relative_advantages(rewards)
        self.assertEqual(len(advantages), 4)
        # mean = 0.5, std = sqrt(((0.3)^2 * 2 + (0.1)^2 * 2) / 4) = sqrt(0.05)
        import math
        expected_std = math.sqrt(0.05)
        for v, a in zip([0.2, 0.4, 0.6, 0.8], advantages):
            self.assertAlmostEqual(a, (v - 0.5) / expected_std, places=5)
        # mean of advantages ≈ 0
        self.assertAlmostEqual(sum(advantages) / 4, 0.0, places=5)

    def test_handles_empty_rewards(self):
        advantages = _group_relative_advantages([])
        self.assertEqual(advantages, [])

    def test_handles_single_reward(self):
        """K=1 group → std=0 → zero advantage (no relative signal)."""
        rewards = [{"rollout_index": 0, "reward_layered": 1.0}]
        advantages = _group_relative_advantages(rewards)
        self.assertEqual(advantages, [0.0])


class TestAdvantageStats(unittest.TestCase):
    def test_empty_input(self):
        stats = _advantage_stats([])
        self.assertEqual(stats, {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0})

    def test_known_input(self):
        stats = _advantage_stats([-1.0, 0.0, 1.0])
        self.assertAlmostEqual(stats["mean"], 0.0)
        self.assertAlmostEqual(stats["std"], 0.8164965, places=5)
        self.assertEqual(stats["min"], -1.0)
        self.assertEqual(stats["max"], 1.0)


# ---------------------------------------------------------------------------
# Determinism fingerprints
# ---------------------------------------------------------------------------

class TestHashFunctions(unittest.TestCase):
    def test_rollouts_hash_is_order_invariant(self):
        """Rollouts hash must be invariant under sort by rollout_index."""
        a = [
            {"rollout_index": 2, "generated": "hello"},
            {"rollout_index": 0, "generated": "world"},
            {"rollout_index": 1, "generated": "foo"},
        ]
        b = sorted(a, key=lambda x: x["rollout_index"])
        self.assertEqual(_hash_rollouts(a), _hash_rollouts(b))

    def test_rollouts_hash_changes_with_content(self):
        a = [{"rollout_index": 0, "generated": "hello"}]
        b = [{"rollout_index": 0, "generated": "world"}]
        self.assertNotEqual(_hash_rollouts(a), _hash_rollouts(b))

    def test_rewards_hash_uses_layered_value(self):
        a = [{"rollout_index": 0, "reward_layered": 0.5}]
        b = [{"rollout_index": 0, "reward_layered": 0.6}]
        self.assertNotEqual(_hash_rewards(a), _hash_rewards(b))


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------

class TestLoadSamples(unittest.TestCase):
    def test_load_samples_with_limit_and_seed(self, tmp_dir=None):
        """Sample loader is deterministic given the same seed and limit."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            for i in range(20):
                (td_path / f"sample-{i:03d}.json").write_text(
                    json.dumps({"id": f"sample-{i:03d}", "task_type": "x"}),
                    encoding="utf-8",
                )
            s1 = _load_samples(td_path, limit=5, seed=2026)
            s2 = _load_samples(td_path, limit=5, seed=2026)
            ids_1 = [s["id"] for s in s1]
            ids_2 = [s["id"] for s in s2]
            self.assertEqual(ids_1, ids_2)
            self.assertEqual(len(s1), 5)
            # Different seed → different shuffle order (very high probability)
            s3 = _load_samples(td_path, limit=5, seed=9999)
            ids_3 = [s["id"] for s in s3]
            self.assertNotEqual(ids_1, ids_3)


# ---------------------------------------------------------------------------
# Iteration with resume
# ---------------------------------------------------------------------------

class TestIterPrompts(unittest.TestCase):
    def test_resume_from_zero(self):
        samples = [{"id": f"s{i}"} for i in range(5)]
        pairs = _iter_prompts(samples, resume_step=0, max_steps=3, seed=0)
        self.assertEqual(len(pairs), 3)
        self.assertEqual([p[0] for p in pairs], [0, 1, 2])
        self.assertEqual([p[1]["id"] for p in pairs], ["s0", "s1", "s2"])

    def test_resume_from_middle(self):
        samples = [{"id": f"s{i}"} for i in range(5)]
        pairs = _iter_prompts(samples, resume_step=2, max_steps=5, seed=0)
        self.assertEqual(len(pairs), 3)
        self.assertEqual([p[0] for p in pairs], [2, 3, 4])

    def test_resume_past_max_steps(self):
        samples = [{"id": f"s{i}"} for i in range(5)]
        pairs = _iter_prompts(samples, resume_step=10, max_steps=5, seed=0)
        self.assertEqual(pairs, [])

    def test_negative_resume_treated_as_zero(self):
        samples = [{"id": f"s{i}"} for i in range(3)]
        pairs = _iter_prompts(samples, resume_step=-5, max_steps=2, seed=0)
        self.assertEqual([p[0] for p in pairs], [0, 1])


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

class TestCheckpointIO(unittest.TestCase):
    def test_state_round_trip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _save_state(
                td_path, global_step=4, last_step_id="0004_s1",
                rng_state=(), max_steps=8, k_rollouts=4,
                policy_model="dummy/model",
            )
            state_path = td_path / "state.json"
            self.assertTrue(state_path.exists())
            state = _load_state(state_path)
            self.assertEqual(state["global_step"], 4)
            self.assertEqual(state["last_step_id"], "0004_s1")
            self.assertEqual(state["max_steps"], 8)
            self.assertEqual(state["k_rollouts"], 4)
            self.assertEqual(state["policy_model"], "dummy/model")
            self.assertFalse(state["completed"])

    def test_state_completed_flag(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _save_state(
                td_path, global_step=7, last_step_id="0007_s8",
                rng_state=(), max_steps=8, k_rollouts=4,
                policy_model="dummy/model",
            )
            state = _load_state(td_path / "state.json")
            self.assertTrue(state["completed"])


# ---------------------------------------------------------------------------
# Artifact assembly + schema validation
# ---------------------------------------------------------------------------

class TestAssembleStepArtifact(unittest.TestCase):
    def test_minimal_artifact(self):
        sample = {"id": "s1", "metadata": {"task_type": "multi_tool_sequential"},
                  "expected_answer": "answer text"}
        rollouts = [
            {"rollout_index": 0, "generated": "hello", "extracted_calls": []},
            {"rollout_index": 1, "generated": "world", "extracted_calls": []},
        ]
        rewards = [
            {"rollout_index": 0, "reward_binary": 0.0, "reward_layered": 0.4,
             "reward_type": "argument_correct", "first_failure": "tool_name_correct"},
            {"rollout_index": 1, "reward_binary": 1.0, "reward_layered": 0.875,
             "reward_type": "execution_correct", "first_failure": None},
        ]
        advantages = [-1.0, 1.0]
        update = {
            "policy_gradient_loss": 0.5, "learning_rate": 1e-5,
            "tokens_seen": 42, "grad_norm": 0.3,
        }
        artifact = _assemble_step_artifact(
            step_id="0000_s1", global_step=0, policy_model="dummy/model",
            sample=sample, rollouts=rollouts, rewards=rewards,
            advantages=advantages, update=update, seed=2026,
        )
        self.assertEqual(artifact["schema_version"], GRPO_STEP_SCHEMA_VERSION)
        self.assertEqual(artifact["step_id"], "0000_s1")
        self.assertEqual(artifact["global_step"], 0)
        self.assertEqual(artifact["prompt_id"], "s1")
        self.assertEqual(len(artifact["rollouts"]), 2)
        self.assertEqual(len(artifact["rewards"]), 2)
        self.assertEqual(len(artifact["advantages"]), 2)
        self.assertIn("mean", artifact["advantage_stats"])
        self.assertEqual(artifact["update"]["policy_gradient_loss"], 0.5)
        self.assertIn("rollouts_text_hash", artifact["deterministic"])
        self.assertIn("rewards_text_hash", artifact["deterministic"])

    def test_artifact_validates_against_schema(self):
        sample = {"id": "s1"}
        rollouts = [{"rollout_index": 0, "generated": "ok",
                     "extracted_calls": []}]
        rewards = [{"rollout_index": 0, "reward_binary": 1.0,
                    "reward_layered": 1.0, "reward_type": "execution_correct",
                    "first_failure": None}]
        advantages = [0.0]
        update = {"policy_gradient_loss": 0.0, "learning_rate": 1e-5,
                  "tokens_seen": 0, "grad_norm": 0.0, "skipped": True,
                  "skip_reason": "all advantages zero"}
        artifact = _assemble_step_artifact(
            step_id="0000_s1", global_step=0, policy_model="dummy",
            sample=sample, rollouts=rollouts, rewards=rewards,
            advantages=advantages, update=update, seed=0,
        )
        _validate_schema(artifact)

    def test_artifact_handles_missing_extracted_calls(self):
        """Backward-compat: rollouts without extracted_calls still assemble."""
        sample = {"id": "s1"}
        rollouts = [{"rollout_index": 0, "generated": "ok"}]
        rewards = [{"rollout_index": 0, "reward_binary": 0.0,
                    "reward_layered": 0.0}]
        update = {"policy_gradient_loss": 0.0, "learning_rate": 1e-5,
                  "tokens_seen": 0, "grad_norm": 0.0}
        artifact = _assemble_step_artifact(
            step_id="0000_s1", global_step=0, policy_model="dummy",
            sample=sample, rollouts=rollouts, rewards=rewards,
            advantages=[0.0], update=update, seed=0,
        )
        self.assertEqual(len(artifact["rollouts"]), 1)


# ---------------------------------------------------------------------------
# Smoke integration (gated)
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    os.environ.get("GRPO_SMOKE") == "1",
    "Set GRPO_SMOKE=1 to run the integration smoke (requires "
    "transformers + torch + a download-capable network).",
)
class TestGrpoSmokeIntegration(unittest.TestCase):
    """End-to-end smoke that loads a tiny HF model and runs 1 step.

    Disabled by default; opt-in via ``GRPO_SMOKE=1``.
    """

    def test_smoke_runs_on_cpu(self):
        import subprocess
        cmd = [
            sys.executable, "scripts/grpo_train.py",
            "--policy-model", "sshleifer/tiny-gpt2",
            "--samples-dir", "datasets/tool-calling-d2/train",
            "--checkpoint-dir", ".tmp/grpo-smoke",
            "--max-steps", "1", "--k-rollouts", "2", "--limit", "1",
            "--max-new-tokens", "8",
            "--device", "cpu",
            "--smoke-deterministic",
        ]
        result = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(
            result.returncode, 0,
            msg=f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}",
        )
        state_path = ROOT / ".tmp" / "grpo-smoke" / "state.json"
        self.assertTrue(state_path.exists())
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["global_step"], 0)
        self.assertTrue(state["completed"])


if __name__ == "__main__":
    unittest.main()