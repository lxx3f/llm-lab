"""Unified experiment metadata for every result record (N3).

Every experiment result (Dense/MoE training, N2 latency benchmark, N2 routing
stats) embeds the same `metadata` block at the top level so two runs can be
compared without re-reading the host configuration. The block covers:

- git_commit                → exact repository state used to produce the run
- config_sha256             → SHA256 of the loaded YAML/JSON config file
- python_version            → interpreter version
- pytorch_version           → torch.__version__
- cuda_version              → torch.version.cuda (or "unset" on CPU hosts)
- gpu_name                  → torch.cuda.get_device_name(0) (or "unset")
- gpu_compute_capability    → formatted as "major.minor" (or "unset")
- tokenizer_revision        → metadata.json `tokenizer.version` (or "unset")
- dataset_hash              → SHA256 of the train-cache metadata file
                              itself (i.e. the bytes of the train cache's
                              ``metadata.json``). This makes two runs that
                              bind the same cache trivially comparable. The
                              raw source-dataset hash is already recorded
                              inside the cache metadata as
                              ``source.sha256``.
- seed                      → integer seed used for the run (0 if unset)

Per the verification contract: every contract-required non-null metadata
field is a `string`. When a source is missing or unreadable the collector
returns the sentinel string `"unset"` instead of ``None``. The three
GPU/tokenizer fields explicitly allow `string|null` in the contract and
also accept `"unset"` as a synonym for null. `seed` is always `integer`.

The collector is read-only and must not initialize CUDA on CPU-only hosts.
Backed by the standard library + torch only.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any


METADATA_FIELDS: tuple[str, ...] = (
    "git_commit",
    "config_sha256",
    "python_version",
    "pytorch_version",
    "cuda_version",
    "gpu_name",
    "gpu_compute_capability",
    "tokenizer_revision",
    "dataset_hash",
    "seed",
)

# Sentinel used when a contract-required non-null string is missing.
UNSET: str = "unset"

_GIT_COMMIT_PATTERN = re.compile(r"^([0-9a-f]{40}|\d{14}-[a-z0-9]{6})$")
_GPU_CC_PATTERN = re.compile(r"^\d+\.\d+$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _safe_git_commit(cwd: str | Path | None = None) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return UNSET
    value = completed.stdout.strip()
    return value if _GIT_COMMIT_PATTERN.match(value) else UNSET


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_config_sha256(path: str | Path | None) -> str:
    if path is None:
        return UNSET
    file = Path(path)
    try:
        return _sha256_bytes(file.read_bytes())
    except (FileNotFoundError, IsADirectoryError, OSError):
        return UNSET


def _safe_dataset_hash(path: str | Path | None) -> str:
    """Return the SHA256 of the train-cache metadata file **bytes**.

    Per the verification contract, ``dataset_hash`` is the hash of the
    loaded training cache metadata file itself, not ``source.sha256``
    (which is the raw source-dataset hash recorded inside that file).
    Same content == same hash; this makes two runs trivially comparable
    when they bind the same cache metadata.
    """
    if path is None:
        return UNSET
    file = Path(path)
    try:
        return _sha256_bytes(file.read_bytes())
    except (FileNotFoundError, IsADirectoryError, OSError):
        return UNSET


def _safe_tokenizer_revision(directory: str | Path | None) -> str | None:
    """Read ``<dir>/metadata.json`` and return ``tokenizer.version``.

    Returns ``None`` when the directory, file, or field is missing. (The
    contract permits ``null`` for this optional field.)
    """
    if directory is None:
        return None
    metadata_path = Path(directory) / "metadata.json"
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, IsADirectoryError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    tokenizer_block = raw.get("tokenizer")
    if not isinstance(tokenizer_block, dict):
        return None
    version = tokenizer_block.get("version")
    return version if isinstance(version, str) and version else None


def _safe_torch_versions() -> tuple[str, str]:
    try:
        import torch
    except ImportError:
        return UNSET, UNSET
    torch_version = getattr(torch, "__version__", None)
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    return (
        torch_version if isinstance(torch_version, str) and torch_version else UNSET,
        cuda_version if isinstance(cuda_version, str) and cuda_version else UNSET,
    )


def _safe_gpu_fields() -> tuple[str | None, str | None]:
    """Return (gpu_name, gpu_compute_capability).

    Contract allows ``string|null`` for these fields. We keep returning
    ``None`` when CUDA is unavailable so the field still carries the
    contract-permitted null. Collectors that prefer the ``"unset"`` sentinel
    can map ``None`` to it explicitly.
    """
    try:
        import torch
    except ImportError:
        return None, None
    if not torch.cuda.is_available():
        return None, None
    try:
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
    except (RuntimeError, AssertionError):
        return None, None
    gpu_name = name if isinstance(name, str) and name else None
    if isinstance(capability, tuple) and len(capability) >= 2:
        cc_value = f"{int(capability[0])}.{int(capability[1])}"
        gpu_cc = cc_value if _GPU_CC_PATTERN.match(cc_value) else None
    else:
        gpu_cc = None
    return gpu_name, gpu_cc


def collect_metadata(
    *,
    config_path: str | Path | None = None,
    tokenizer_artifact_dir: str | Path | None = None,
    train_cache_dir: str | Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Collect a unified metadata block for an experiment result.

    Args:
        config_path: Filesystem path to the YAML/JSON config used by the run.
        tokenizer_artifact_dir: Filesystem path to the tokenizer artifact
            directory (the one containing ``tokenizer.json`` and
            ``metadata.json``).
        train_cache_dir: Filesystem path to the **train** token cache's
            ``metadata.json`` (or its parent directory — the function reads
            ``<train_cache_dir>/metadata.json`` when the path is a directory).
            The implementation hashes the bytes of this file to produce
            ``dataset_hash`` (the contract's "训练 cache 元数据 hash").
        seed: Integer seed used to initialize RNGs for the run. ``None`` is
            coerced to ``0`` so the contract-required ``integer`` type is
            preserved.

    Returns:
        Dict whose keys are exactly ``METADATA_FIELDS``. Every contract-
        required non-null field is a non-empty string; missing values are
        represented by the sentinel ``"unset"``. The three optional fields
        ``gpu_name`` / ``gpu_compute_capability`` / ``tokenizer_revision``
        may be ``None`` (the contract permits this) or the string
        ``"unset"`` interchangeably.
    """
    cache_path = train_cache_dir
    if cache_path is not None:
        candidate = Path(cache_path)
        # If a directory was passed, read the contained metadata.json; if a
        # direct file was passed, use it as-is. This lets callers use either
        # form of the contract parameter.
        if candidate.is_dir():
            cache_path = candidate / "metadata.json"
        elif (candidate.parent / "metadata.json").is_file() and not candidate.name.endswith(".json"):
            cache_path = candidate / "metadata.json"
    torch_version, cuda_version = _safe_torch_versions()
    gpu_name, gpu_cc = _safe_gpu_fields()
    tokenizer_revision = _safe_tokenizer_revision(tokenizer_artifact_dir)
    seed_value = int(seed) if seed is not None else 0
    if seed_value < 0:
        seed_value = 0
    return {
        "git_commit": _safe_git_commit(),
        "config_sha256": _safe_config_sha256(config_path),
        "python_version": platform.python_version(),
        "pytorch_version": torch_version,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "gpu_compute_capability": gpu_cc,
        "tokenizer_revision": tokenizer_revision,
        "dataset_hash": _safe_dataset_hash(cache_path),
        "seed": seed_value,
    }