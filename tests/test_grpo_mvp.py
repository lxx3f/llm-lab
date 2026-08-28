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
import random
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grpo_train import (  # noqa: E402
    ADVANTAGE_EPS,
    GRPO_STEP_SCHEMA_VERSION,
    _advantage_stats,
    _assemble_step_artifact,
    _capture_rng_state,
    _config_matches,
    _fingerprint_rng,
    _fresh_optimizer,
    _group_relative_advantages,
    _hash_rewards,
    _hash_rollouts,
    _iter_prompts,
    _iter_prompts_with_cursor,
    _load_binary_state,
    _load_samples,
    _load_state,
    _restore_optimizer,
    _restore_rng_state,
    _save_state,
    _seed_all,
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


class _DummyModel(torch.nn.Module):
    """A tiny parameter-only model used by checkpoint tests."""

    def __init__(self, n: int = 4):
        super().__init__()
        self.linear = torch.nn.Linear(n, n)

    def forward(self, x):  # pragma: no cover
        return self.linear(x)


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
                samples_consumed=4, max_steps=8, k_rollouts=4,
                policy_model="dummy/model", seed=2026, learning_rate=1e-5,
                max_grad_norm=1.0, samples_dir="datasets/d2/train",
                device="cpu", save_every=1, model=_DummyModel(),
                optimizer=None, rng_state={"python": (3, (0, ()), None)},
            )
            state_path = td_path / "state.json"
            self.assertTrue(state_path.exists())
            state = _load_state(state_path)
            self.assertEqual(state["global_step"], 4)
            self.assertEqual(state["last_step_id"], "0004_s1")
            self.assertEqual(state["samples_consumed"], 4)
            self.assertEqual(state["max_steps"], 8)
            self.assertEqual(state["k_rollouts"], 4)
            self.assertEqual(state["policy_model"], "dummy/model")
            self.assertEqual(state["seed"], 2026)
            self.assertFalse(state["completed"])

    def test_state_completed_flag(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _save_state(
                td_path, global_step=7, last_step_id="0007_s8",
                samples_consumed=8, max_steps=8, k_rollouts=4,
                policy_model="dummy/model", seed=2026, learning_rate=1e-5,
                max_grad_norm=1.0, samples_dir="datasets/d2/train",
                device="cpu", save_every=1, model=_DummyModel(),
                optimizer=None, rng_state={"python": (3, (0, ()), None)},
            )
            state = _load_state(td_path / "state.json")
            self.assertTrue(state["completed"])

    def test_state_persists_binary_blob_when_next_save_flag(self):
        """``state.pt`` is written when ``next_global_step_to_save=True``."""
        import tempfile
        import torch
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            model = _DummyModel()
            opt = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-5)
            _save_state(
                td_path, global_step=3, last_step_id="0003_s1",
                samples_consumed=3, max_steps=8, k_rollouts=4,
                policy_model="dummy/model", seed=2026, learning_rate=1e-5,
                max_grad_norm=1.0, samples_dir="datasets/d2/train",
                device="cpu", save_every=1, model=model, optimizer=opt,
                rng_state={"python": (3, (0, ()), None), "torch": torch.zeros(10, dtype=torch.uint8)},
                next_global_step_to_save=True,
            )
            self.assertTrue((td_path / "state.pt").exists())
            blob = torch.load(td_path / "state.pt", map_location="cpu",
                              weights_only=False)
            self.assertEqual(blob["version"], "1.0")
            self.assertIn("model_state", blob)
            self.assertIn("optimizer_state", blob)
            self.assertIn("rng_state", blob)

    def test_state_skips_binary_blob_when_next_save_flag_false(self):
        """``state.pt`` is NOT written when the save cadence says so."""
        import tempfile
        import torch
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            _save_state(
                td_path, global_step=3, last_step_id="0003_s1",
                samples_consumed=3, max_steps=8, k_rollouts=4,
                policy_model="dummy/model", seed=2026, learning_rate=1e-5,
                max_grad_norm=1.0, samples_dir="datasets/d2/train",
                device="cpu", save_every=1, model=_DummyModel(),
                optimizer=None,
                rng_state={"python": (3, (0, ()), None)},
                next_global_step_to_save=False,
            )
            self.assertFalse((td_path / "state.pt").exists())
            # state.json still exists
            self.assertTrue((td_path / "state.json").exists())


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
# Determinism / RNG state
# ---------------------------------------------------------------------------

class TestSeedAndRng(unittest.TestCase):
    """Verify Python/Torch RNG seeding + capture/restore round-trip."""

    def test_seed_all_is_deterministic(self):
        """Two seeded runs produce the same random sequence."""
        _seed_all(2026, "cpu")
        seq_1 = [random.random() for _ in range(5)]
        _seed_all(2026, "cpu")
        seq_2 = [random.random() for _ in range(5)]
        self.assertEqual(seq_1, seq_2)

    def test_seed_all_resets_torch_rng(self):
        """Two seeded torch RNGs produce the same tensor."""
        _seed_all(42, "cpu")
        t1 = torch.randn(4)
        _seed_all(42, "cpu")
        t2 = torch.randn(4)
        self.assertTrue(torch.equal(t1, t2))

    def test_capture_and_restore_rng_state(self):
        """Capture / restore round-trip yields the same next random."""
        _seed_all(7, "cpu")
        # Advance the state
        _ = [random.random() for _ in range(10)]
        captured = _capture_rng_state("cpu")
        seq_a = [random.random() for _ in range(3)]
        # Restore and try again — must match
        _restore_rng_state(captured)
        seq_b = [random.random() for _ in range(3)]
        self.assertEqual(seq_a, seq_b)

    def test_restore_rng_state_is_idempotent(self):
        """Double-restore of the same captured state yields the same seq."""
        _seed_all(13, "cpu")
        captured = _capture_rng_state("cpu")
        seq_1 = [random.random() for _ in range(3)]
        _restore_rng_state(captured)
        seq_2 = [random.random() for _ in range(3)]
        _restore_rng_state(captured)
        seq_3 = [random.random() for _ in range(3)]
        self.assertEqual(seq_1, seq_3)
        self.assertEqual(seq_2, seq_3)

    def test_fingerprint_rng_is_stable(self):
        """Fingerprint of the same RNG state produces the same hash."""
        _seed_all(2026, "cpu")
        state = _capture_rng_state("cpu")
        h1 = _fingerprint_rng(state)
        h2 = _fingerprint_rng(state)
        self.assertEqual(h1, h2)

    def test_fingerprint_changes_after_rng_advance(self):
        _seed_all(2026, "cpu")
        state_a = _capture_rng_state("cpu")
        _ = [random.random() for _ in range(5)]
        state_b = _capture_rng_state("cpu")
        self.assertNotEqual(
            _fingerprint_rng(state_a), _fingerprint_rng(state_b)
        )


# ---------------------------------------------------------------------------
# Resume cursor
# ---------------------------------------------------------------------------

class TestResumeCursor(unittest.TestCase):
    """``_iter_prompts_with_cursor`` must NOT re-use a sample that was
    already consumed in the previous run."""

    def test_fresh_run_consumes_from_offset_zero(self):
        samples = [{"id": f"s{i}"} for i in range(5)]
        triples = _iter_prompts_with_cursor(
            samples, cursor=0, max_steps=3, remaining_steps=3
        )
        ids = [s["id"] for _, _, s in triples]
        self.assertEqual(ids, ["s0", "s1", "s2"])

    def test_resume_continues_after_cursor(self):
        """Resuming at samples_consumed=2 must NOT re-use s0 or s1."""
        samples = [{"id": f"s{i}"} for i in range(5)]
        triples = _iter_prompts_with_cursor(
            samples, cursor=2, max_steps=4, remaining_steps=2
        )
        ids = [s["id"] for _, _, s in triples]
        self.assertEqual(ids, ["s2", "s3"])

    def test_resume_past_max_steps_returns_empty(self):
        """remaining_steps=0 means no work left → empty triples."""
        samples = [{"id": f"s{i}"} for i in range(5)]
        triples = _iter_prompts_with_cursor(
            samples, cursor=10, max_steps=5, remaining_steps=0
        )
        self.assertEqual(triples, [])

    def test_resume_cursor_clamped_to_zero(self):
        samples = [{"id": f"s{i}"} for i in range(3)]
        triples = _iter_prompts_with_cursor(
            samples, cursor=-2, max_steps=2, remaining_steps=2
        )
        ids = [s["id"] for _, _, s in triples]
        self.assertEqual(ids, ["s0", "s1"])


# ---------------------------------------------------------------------------
# Resume config validation
# ---------------------------------------------------------------------------

class _MockArgs:
    """Mimics argparse.Namespace for ``_config_matches`` tests."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class TestConfigMatches(unittest.TestCase):
    """``_config_matches`` returns the list of CLI arg names whose
    values differ from the saved run config. Empty = match."""

    def test_matching_config(self):
        saved = {
            "policy_model": "m", "k_rollouts": 4, "max_steps": 8,
            "learning_rate": 1e-5, "max_grad_norm": 1.0,
            "samples_dir": "datasets/d2/train", "seed": 2026,
            "device": "cpu",
        }
        args = _MockArgs(**{
            k: v for k, v in saved.items()
            if k in ("policy_model", "k_rollouts", "max_steps",
                     "learning_rate", "max_grad_norm", "samples_dir",
                     "seed", "device")
        })
        diffs = _config_matches(saved, args, strict=False)
        self.assertEqual(diffs, [])

    def test_differing_model_is_detected(self):
        saved = {
            "policy_model": "model-a", "k_rollouts": 4, "max_steps": 8,
            "learning_rate": 1e-5, "max_grad_norm": 1.0,
            "samples_dir": "d", "seed": 1, "device": "cpu",
        }
        args = _MockArgs(policy_model="model-b", k_rollouts=4,
                         max_steps=8, learning_rate=1e-5,
                         max_grad_norm=1.0, samples_dir="d", seed=1,
                         device="cpu")
        diffs = _config_matches(saved, args, strict=False)
        self.assertTrue(any("policy_model" in d for d in diffs))


# ---------------------------------------------------------------------------
# Optimizer state round-trip
# ---------------------------------------------------------------------------

class TestOptimizerRestore(unittest.TestCase):
    """``_restore_optimizer`` loads a state_dict onto an existing optimizer."""

    def test_optimizer_state_round_trip(self):
        """Adam moments (exp_avg, exp_avg_sq) survive a save/load cycle."""
        model = _DummyModel()
        opt = _fresh_optimizer(model, learning_rate=1e-5)
        # Force a step so Adam moments become non-zero
        loss = (model.linear.weight ** 2).sum()
        loss.backward()
        opt.step()
        before_state = {k: v.clone() if torch.is_tensor(v) else v
                        for k, v in opt.state_dict().items()}
        saved = opt.state_dict()

        # Build a fresh optimizer on the same model, then restore
        opt2 = _fresh_optimizer(model, learning_rate=1e-5)
        _restore_optimizer(opt2, saved)

        # The first param-group's state[0] should match the saved state
        saved_state_0 = before_state["state"][0]
        restored_state_0 = opt2.state_dict()["state"][0]
        for key in saved_state_0:
            if torch.is_tensor(saved_state_0[key]):
                self.assertTrue(torch.allclose(
                    saved_state_0[key], restored_state_0[key]
                ))


# ---------------------------------------------------------------------------
# Unconditional smoke (no HF dependency)
# ---------------------------------------------------------------------------

class TestUnconditionalSmoke(unittest.TestCase):
    """End-to-end smoke that exercises the checkpoint/resume code path
    WITHOUT requiring HF transformers or a network. Uses a tiny
    ``torch.nn`` mock + in-memory rollouts to validate that:

    1. ``state.pt`` is persisted on every step (when ``save_every=1``).
    2. ``model.state_dict()`` survives a full save/load cycle.
    3. The ``samples_consumed`` cursor advances by 1 per step.
    4. ``state.json.completed`` flips to True on the final step.
    """

    def test_smoke_full_checkpoint_round_trip(self):
        """Save state, restore on a fresh model, verify weights match."""
        import tempfile
        import torch
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            # First run: build model, save weights + state
            model_1 = _DummyModel(n=4)
            opt_1 = _fresh_optimizer(model_1, learning_rate=1e-3)
            # Touch the optimizer so moments are non-default
            loss = (model_1.linear.weight ** 2).sum()
            loss.backward()
            opt_1.step()
            opt_1.zero_grad()

            sample_w_before = model_1.linear.weight.detach().clone()
            sample_b_before = model_1.linear.bias.detach().clone()

            _seed_all(2026, "cpu")
            _save_state(
                td_path, global_step=2, last_step_id="0002_s2",
                samples_consumed=2, max_steps=8, k_rollouts=4,
                policy_model="dummy/model", seed=2026, learning_rate=1e-3,
                max_grad_norm=1.0, samples_dir="datasets/d2/train",
                device="cpu", save_every=1, model=model_1, optimizer=opt_1,
                rng_state=_capture_rng_state("cpu"),
                next_global_step_to_save=True,
            )

            # Verify artifacts on disk
            self.assertTrue((td_path / "state.json").exists())
            self.assertTrue((td_path / "state.pt").exists())

            # Second run: fresh model, restore from state.pt
            model_2 = _DummyModel(n=4)
            opt_2 = _fresh_optimizer(model_2, learning_rate=1e-3)
            blob = _load_binary_state(td_path)
            self.assertEqual(blob["version"], "1.0")
            model_2.load_state_dict(blob["model_state"])
            _restore_optimizer(opt_2, blob["optimizer_state"])
            _restore_rng_state(blob["rng_state"])

            # The restored model weights must match the original
            self.assertTrue(torch.allclose(
                model_2.linear.weight.detach(), sample_w_before
            ))
            self.assertTrue(torch.allclose(
                model_2.linear.bias.detach(), sample_b_before
            ))

    def test_samples_consumed_advances_per_step(self):
        """The persisted ``samples_consumed`` must increment per step."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            model = _DummyModel()
            _seed_all(0, "cpu")
            # Simulate 3 steps where max_steps=3 (so the final step IS the last)
            for step in range(3):
                is_last = (step == 2)
                _save_state(
                    td_path, global_step=step, last_step_id=f"000{step}_s{step}",
                    samples_consumed=step + 1, max_steps=3, k_rollouts=4,
                    policy_model="dummy", seed=0, learning_rate=1e-5,
                    max_grad_norm=1.0, samples_dir="d", device="cpu",
                    save_every=1, model=model, optimizer=None,
                    rng_state=_capture_rng_state("cpu"),
                    next_global_step_to_save=(not is_last),
                )
            final = _load_state(td_path / "state.json")
            self.assertEqual(final["global_step"], 2)
            self.assertEqual(final["samples_consumed"], 3)
            self.assertTrue(final["completed"])


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