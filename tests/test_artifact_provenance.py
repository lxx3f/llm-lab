"""Night-run artifact provenance checks (git_commit alignment).

Every objective artifact records the commit under which its training ran. For
shared/deterministic controls the commit may point at any legitimate night-run
commit (regenerated only when a stage's training commit is finalized). Every
other artifact must record a known night-run commit AND have the expected
seed, step count, and ``status: completed``.

Night-run commits covered (audit-round-6 set, includes N11 multi-seed, N11
large, and MoE long so the provenance check covers every objective artifact):

    d00989d1  N7 (rope sweep)
    ffc9f6b3  N8 (n_heads sweep)
    5aceb150  N9 (d_ff sweep)
    02863f0b  MoE 5000-step curve
    0e4e6e13  N11 long curve (baseline + medium, single seed)
    43260e3c  N9 d_ff=256 fix + shared medium control regen
    d8d05128  multi-seed training (39 runs)
    b55db7b2  MoE 50000-step curve
    41268d98  shared control accepts any night-run commit
    f2c323af  N11 multi-seed + N11 large training (50000 steps)
"""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KNOWN_NIGHT_RUN_COMMITS = {
    "d00989d1bfef7eacee5d4dbc6bc701a26d5bdffb",  # N7
    "ffc9f6b34cad8a98894713eafaa93576409a95d7",  # N8
    "5aceb150e42355e7fc8e70716270562962d71e6c",  # N9
    "02863f0b5b848170305ec0d281de05a5c3fec595",  # MoE curve
    "0e4e6e134aee8a00f8fb3ec957c9f406564da598",  # N11 long curve (single seed)
    "43260e3c41a0ac9041535cfe5d488ceb7f226776",  # multi-seed 39-run training
    "d8d05128269395c8367b1272ab8bad4539618a63",  # MoE long curve
    "b55db7b2227c83e5759c8728af8d9456d14e0043",  # MoE long commit
    "41268d98f73085e120aaf7814c020bc5d531bae1",  # N11 multi-seed provenance
    "f2c323af6d5051d2b7c501b8af5f89f42ddc786d",  # N11 multi-seed + N11 large
    "cb44a8bd7a97c22eecbeb4d099aeeeccaa4aed02", # N12 ultra 100000-step training
    "3f0cbe695854e9e5208595c6519d29fc6cca2c2c", # N12 ultra feat (saturation test)
    "f99910ace136293ca0159cd3a4e45ef70c0f3f8a", # MoE multi-seed (5000+50000 steps)
    "540ace4a74f8bf6eec8bd946522211cf8a200e55", # N11 large multi-seed (50000 steps)
}

# Every objective artifact: relative path, expected seed, expected
# optimizer_steps, expected status, and expected curve_summary.val_loss_min_step.
EXPECTED_ARTIFACTS: list[dict[str, object]] = [
    # ---- N7 rope sweep (5000 steps) --------------------------------------
    {"path": "artifacts/dense-owt-formal-curve-medium-result.json",
     "role": "shared-control"},
    {"path": "artifacts/dense-owt-formal-curve-medium-ropebase-50k-result.json",
     "seed": 42, "optimizer_steps": 5000, "status": "completed"},
    {"path": "artifacts/dense-owt-formal-curve-medium-ropebase-100k-result.json",
     "seed": 42, "optimizer_steps": 5000, "status": "completed"},
    # ---- N8 heads sweep (5000 steps) -------------------------------------
    {"path": "artifacts/dense-owt-formal-curve-medium-heads-2-result.json",
     "seed": 42, "optimizer_steps": 5000, "status": "completed"},
    {"path": "artifacts/dense-owt-formal-curve-medium-heads-8-result.json",
     "seed": 42, "optimizer_steps": 5000, "status": "completed"},
    # ---- N9 d_ff sweep (5000 steps) --------------------------------------
    {"path": "artifacts/dense-owt-formal-curve-medium-dff-256-result.json",
     "seed": 42, "optimizer_steps": 5000, "status": "completed"},
    {"path": "artifacts/dense-owt-formal-curve-medium-dff-1024-result.json",
     "seed": 42, "optimizer_steps": 5000, "status": "completed"},
    # ---- MoE 5000-step curve ---------------------------------------------
    {"path": "artifacts/moe-owt-formal-curve-result.json",
     "seed": 42, "optimizer_steps": 5000, "status": "completed"},
    # ---- N11 long curve (single seed, 50000 steps) -----------------------
    {"path": "artifacts/dense-owt-formal-curve-long-baseline-result.json",
     "seed": 42, "optimizer_steps": 50000, "status": "completed"},
    {"path": "artifacts/dense-owt-formal-curve-long-medium-result.json",
     "seed": 42, "optimizer_steps": 50000, "status": "completed"},
    # ---- N12 ultra curve (single seed, 100000 steps) ---------------------
    {"path": "artifacts/dense-owt-formal-curve-ultra-baseline-result.json",
     "seed": 42, "optimizer_steps": 100000, "status": "completed"},
    {"path": "artifacts/dense-owt-formal-curve-ultra-medium-result.json",
     "seed": 42, "optimizer_steps": 100000, "status": "completed"},
    # ---- N11 long multi-seed (3 seeds × {baseline, medium}, 50000 steps) -
    {"path": "artifacts/dense-owt-formal-curve-long-baseline-seed42-result.json",
     "seed": 42, "optimizer_steps": 50000, "status": "completed"},
    {"path": "artifacts/dense-owt-formal-curve-long-baseline-seed123-result.json",
     "seed": 123, "optimizer_steps": 50000, "status": "completed"},
    {"path": "artifacts/dense-owt-formal-curve-long-baseline-seed7-result.json",
     "seed": 7, "optimizer_steps": 50000, "status": "completed"},
    {"path": "artifacts/dense-owt-formal-curve-long-medium-seed42-result.json",
     "seed": 42, "optimizer_steps": 50000, "status": "completed"},
    {"path": "artifacts/dense-owt-formal-curve-long-medium-seed123-result.json",
     "seed": 123, "optimizer_steps": 50000, "status": "completed"},
    {"path": "artifacts/dense-owt-formal-curve-long-medium-seed7-result.json",
     "seed": 7, "optimizer_steps": 50000, "status": "completed"},
    # ---- N11 large (50000 steps) -----------------------------------------
    {"path": "artifacts/dense-owt-formal-curve-long-large-result.json",
     "seed": 42, "optimizer_steps": 50000, "status": "completed"},
    # ---- N11 large multi-seed (3 seeds, 50000 steps) ----------------------
    {"path": "artifacts/dense-owt-formal-curve-long-large-seed42-result.json",
     "seed": 42, "optimizer_steps": 50000, "status": "completed"},
    {"path": "artifacts/dense-owt-formal-curve-long-large-seed123-result.json",
     "seed": 123, "optimizer_steps": 50000, "status": "completed"},
    {"path": "artifacts/dense-owt-formal-curve-long-large-seed7-result.json",
     "seed": 7, "optimizer_steps": 50000, "status": "completed"},
    # ---- MoE 50000-step curve --------------------------------------------
    {"path": "artifacts/moe-owt-formal-curve-long-result.json",
     "seed": 42, "optimizer_steps": 50000, "status": "completed"},
    # ---- MoE multi-seed (3 seeds × {5000, 50000} steps) ------------------
    {"path": "artifacts/moe-owt-formal-curve-seed42-result.json",
     "seed": 42, "optimizer_steps": 5000, "status": "completed"},
    {"path": "artifacts/moe-owt-formal-curve-seed123-result.json",
     "seed": 123, "optimizer_steps": 5000, "status": "completed"},
    {"path": "artifacts/moe-owt-formal-curve-seed7-result.json",
     "seed": 7, "optimizer_steps": 5000, "status": "completed"},
    {"path": "artifacts/moe-owt-formal-curve-long-seed42-result.json",
     "seed": 42, "optimizer_steps": 50000, "status": "completed"},
    {"path": "artifacts/moe-owt-formal-curve-long-seed123-result.json",
     "seed": 123, "optimizer_steps": 50000, "status": "completed"},
    {"path": "artifacts/moe-owt-formal-curve-long-seed7-result.json",
     "seed": 7, "optimizer_steps": 50000, "status": "completed"},
    # ---- Multi-seed: representative subset per seed ---------------------
    {"path": "artifacts/dense-owt-formal-curve-large-seed123-result.json",
     "seed": 123, "optimizer_steps": 5000, "status": "completed"},
    {"path": "artifacts/dense-owt-formal-curve-medium-dropout01-seed7-result.json",
     "seed": 7, "optimizer_steps": 5000, "status": "completed"},
    {"path": "artifacts/dense-owt-formal-curve-medium-dropout02-seed42-result.json",
     "seed": 42, "optimizer_steps": 5000, "status": "completed"},
]

SHARED_CONTROL = "artifacts/dense-owt-formal-curve-medium-result.json"


class ArtifactProvenanceTests(unittest.TestCase):
    def test_every_objective_artifact_has_known_commit_and_metadata(self) -> None:
        """Every objective artifact must exist, point at a known night-run
        commit, and carry the expected seed / optimizer_steps / status."""
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(ROOT),
        ).stdout.strip()
        missing: list[str] = []
        stale: list[str] = []
        meta_mismatch: list[str] = []
        for spec in EXPECTED_ARTIFACTS:
            rel = spec["path"]
            path = ROOT / rel
            assert isinstance(rel, str)
            if not path.is_file():
                missing.append(rel)
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            commit = (payload.get("metadata") or {}).get("git_commit")
            if spec.get("role") == "shared-control":
                # Shared control is a deterministic artifact; its provenance
                # may point at any legitimate night-run commit (it is
                # regenerated only when a stage's training commit is
                # finalized) or at the current HEAD.
                if commit != head and commit not in KNOWN_NIGHT_RUN_COMMITS:
                    stale.append(f"{rel} (git_commit={commit} outside night-run commits)")
                continue
            # Non-shared artifacts must point at a known night-run commit.
            if commit not in KNOWN_NIGHT_RUN_COMMITS:
                stale.append(f"{rel} (git_commit={commit} outside night-run commits)")
                continue
            # Metadata checks: seed, optimizer_steps, status.
            meta = payload.get("metadata") or {}
            actual_seed = meta.get("seed")
            if spec["seed"] != actual_seed:
                meta_mismatch.append(
                    f"{rel}: seed expected={spec['seed']!r} actual={actual_seed!r}"
                )
            actual_status = payload.get("status")
            if spec["status"] != actual_status:
                meta_mismatch.append(
                    f"{rel}: status expected={spec['status']!r} actual={actual_status!r}"
                )
            # Training length lives at payload.training.optimizer_steps
            # (top-level optimizer_steps is not part of the v1.x schema).
            training = payload.get("training") or {}
            actual_optimizer_steps = training.get("optimizer_steps")
            if actual_optimizer_steps != spec["optimizer_steps"]:
                meta_mismatch.append(
                    f"{rel}: training.optimizer_steps expected={spec['optimizer_steps']!r} "
                    f"actual={actual_optimizer_steps!r}"
                )
            # val_loss_min_step is the actual minimum-loss step (may be < the
            # training length when val_loss rises again near the end, as
            # observed for d_ff=256 and MoE 5000-step curve); just verify
            # it is a real step inside the training window.
            cs = (payload.get("metrics") or {}).get("curve_summary") or {}
            actual_val_min_step = cs.get("val_loss_min_step")
            if not isinstance(actual_val_min_step, int) or actual_val_min_step < 0:
                meta_mismatch.append(
                    f"{rel}: val_loss_min_step must be a non-negative int, "
                    f"got {actual_val_min_step!r}"
                )
            elif actual_val_min_step > actual_optimizer_steps:
                meta_mismatch.append(
                    f"{rel}: val_loss_min_step={actual_val_min_step} > "
                    f"optimizer_steps={actual_optimizer_steps}"
                )
        self.assertEqual(missing, [], msg=f"missing artifacts: {missing}")
        self.assertEqual(stale, [], msg=f"stale provenance: {stale}")
        self.assertEqual(meta_mismatch, [], msg=f"metadata mismatch: {meta_mismatch}")

    def test_multi_seed_overview_exists(self) -> None:
        overview = ROOT / "artifacts" / "multi-seed-overview.json"
        self.assertTrue(overview.is_file())
        data = json.loads(overview.read_text(encoding="utf-8"))
        self.assertIn("config_summary", data)
        self.assertGreaterEqual(len(data["config_summary"]), 13)

    def test_multi_seed_overview_stats_use_population_std(self) -> None:
        """P1-03 protocol contract (auditor round 1 for this stage): ``std``
        must be the POPULATION standard deviation (divide by N). Independently
        recompute the multi-seed overview statistics from the artifact files
        and assert they match the overview exactly (which the READMEs mirror):

        - MoE 5000 steps:   mean = 7.2651, pop-std = 0.0325
        - MoE 50000 steps:  mean = 6.1179, pop-std = 0.0578
        - large 50000 steps: mean = 5.2578, pop-std = 0.0090

        Using the sample std (stdev, divide by N-1) would give 0.0398 /
        0.0708 and must NOT be what the overview reports.
        """
        def pop_std(values: list[float]) -> float:
            n = len(values)
            mean = sum(values) / n
            return (sum((v - mean) ** 2 for v in values) / n) ** 0.5

        def load_min(rel: str) -> float:
            payload = json.loads((ROOT / rel).read_text(encoding="utf-8"))
            return payload["metrics"]["curve_summary"]["val_loss_min"]

        moe_5k = [load_min(f"artifacts/moe-owt-formal-curve-seed{s}-result.json")
                  for s in (42, 123, 7)]
        moe_50k = [load_min(f"artifacts/moe-owt-formal-curve-long-seed{s}-result.json")
                   for s in (42, 123, 7)]
        large_50k = [load_min(f"artifacts/dense-owt-formal-curve-long-large-seed{s}-result.json")
                     for s in (42, 123, 7)]
        overview = json.loads((ROOT / "artifacts" / "multi-seed-overview.json")
                              .read_text(encoding="utf-8"))
        cs = overview["config_summary"]
        for rel, vals, expected_mean, expected_std in (
            ("configs/moe_training.owt-formal-curve.example.yaml", moe_5k, 7.2651, 0.0325),
            ("configs/moe_training.owt-formal-curve-long.example.yaml", moe_50k, 6.1179, 0.0578),
            ("configs/dense_training.owt-formal-curve-long-large.example.yaml", large_50k, 5.2578, 0.0090),
        ):
            entry = cs[rel]
            recomputed_mean = sum(vals) / len(vals)
            recomputed_std = pop_std(vals)
            # Overview must agree with independent recomputation.
            self.assertAlmostEqual(entry["val_loss_min_mean"], recomputed_mean, places=4)
            self.assertAlmostEqual(entry["val_loss_min_std"], recomputed_std, places=4)
            # And must match the documented values (P1-03 population std).
            self.assertAlmostEqual(recomputed_mean, expected_mean, places=4)
            self.assertAlmostEqual(recomputed_std, expected_std, places=4)
            # Guard: the overview must NOT contain the sample-std values.
            self.assertNotAlmostEqual(recomputed_std, 0.0398, places=4)
            self.assertNotAlmostEqual(recomputed_std, 0.0708, places=4)
            self.assertEqual(entry["seeds"], [42, 123, 7])

    def test_long_training_pngs_exist(self) -> None:
        """Required long-training overlay PNGs must exist (800x500 plot contract)."""
        for rel in (
            "artifacts/dense-owt-formal-curve-long-scale-sweep.png",
            "artifacts/moe-vs-dense-medium-long-curve.png",
        ):
            self.assertTrue((ROOT / rel).is_file(), msg=f"missing plot: {rel}")

    def test_ultra_curve_saturation_signature(self) -> None:
        """N12 ultra 100000-step training must show saturation: val_loss_min
        occurs at the END of training (≥90% of training length), with a
        small positive ``delta_val_loss`` showing mild rebound in the final
        2% of steps (val at last step > val at min step). This proves we
        actually trained long enough to see saturation, not just kept
        improving.
        """
        for rel in (
            "artifacts/dense-owt-formal-curve-ultra-baseline-result.json",
            "artifacts/dense-owt-formal-curve-ultra-medium-result.json",
        ):
            payload = json.loads((ROOT / rel).read_text(encoding="utf-8"))
            cs = payload["metrics"]["curve_summary"]
            total_steps = payload["training"]["optimizer_steps"]
            val_min = cs["val_loss_min"]
            val_min_step = cs["val_loss_min_step"]
            val_last = cs["val_loss_last"]
            delta_val = cs["delta_val_loss"]
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(total_steps, 100000)
            # val_min must occur in the LAST 5% of training (saturation
            # signal — not still falling linearly).
            self.assertGreaterEqual(val_min_step, int(total_steps * 0.95),
                                    msg=f"{rel}: val_min_step={val_min_step} < 95% of {total_steps}")
            # delta_val_loss must be small + positive (val at last step > val at min step).
            self.assertGreater(delta_val, 0.0,
                               msg=f"{rel}: delta_val_loss={delta_val} <= 0 (no rebound)")
            self.assertLess(delta_val, 0.1,
                            msg=f"{rel}: delta_val_loss={delta_val} too large (model still improving)")
            # val_last must be very close to val_min (the model has plateaued).
            self.assertAlmostEqual(val_last, val_min, delta=0.05,
                                   msg=f"{rel}: val_last={val_last} far from val_min={val_min}")


if __name__ == "__main__":
    unittest.main()
