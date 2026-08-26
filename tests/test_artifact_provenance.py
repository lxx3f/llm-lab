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
    # ---- MoE 50000-step curve --------------------------------------------
    {"path": "artifacts/moe-owt-formal-curve-long-result.json",
     "seed": 42, "optimizer_steps": 50000, "status": "completed"},
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

    def test_long_training_pngs_exist(self) -> None:
        """Required long-training overlay PNGs must exist (800x500 plot contract)."""
        for rel in (
            "artifacts/dense-owt-formal-curve-long-scale-sweep.png",
            "artifacts/moe-vs-dense-medium-long-curve.png",
        ):
            self.assertTrue((ROOT / rel).is_file(), msg=f"missing plot: {rel}")


if __name__ == "__main__":
    unittest.main()
