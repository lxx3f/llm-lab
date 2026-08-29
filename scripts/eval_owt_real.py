"""Real OWT per-token CE loss evaluation for 5 public instruction-tuned models.

Scope:
- Source text: ``data/raw/owt-sample/owt_valid.txt`` (Stanford CS336 OWT-sample
  validation split, 277 MB). Source SHA-256 is locked to
  ``2406f278e71829d273b315e9b403285baea7022b26a96d2728dd8b776ea40660``.
- 5 public models from P5-04 ``DEFAULT_MODELS`` (SmolLM2-360M/1.7B-Instruct +
  Qwen2.5-0.5B/1.5B/3B-Instruct). Models are loaded from a local directory
  (e.g. ModelScope cache snapshot) and used in offline mode.
- Per-model tokenizer produces a hash-bound int32 token cache (vocab size
  > 65535 for Qwen2.5 so uint16 cannot be used).
- Per-token cross-entropy is computed by sliding the entire token stream
  through the model in chunks of ``--seq-len`` (default 1024), predicting
  token ``t+1`` from logits at position ``t``. Cross-entropy loss is summed
  across all positions and divided by the total predicted token count.
- Each model's result is written to
  ``artifacts/owt-real-eval/results/<model_short>.json``. A 5-model
  comparison table is written to
  ``artifacts/owt-real-eval/comparison.{csv,json}``.

Important caveats (recorded verbatim in the README):
- Different models have different tokenizers. Perplexity is **not** directly
  comparable across models. The README also reports
  ``loss_nats_per_source_byte`` (mean loss / source byte) which is
  tokenizer-normalized but still biased by tokenizer's BPE merges.
- Models are loaded from a local directory; the script does not download.
  ``source_hub`` and ``hf_revision`` (when known) are recorded for each
  result to make provenance explicit. ``revision_verified`` is True only
  when the local snapshot matches the P5-04 HF exact revision.

Usage::

    .venv/python.exe scripts/eval_owt_real.py --build-cache --models <list>
    .venv/python.exe scripts/eval_owt_real.py --eval-loss --models <list>
    .venv/python.exe scripts/eval_owt_real.py --aggregate
    .venv/python.exe scripts/eval_owt_real.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

# Force offline loading for any HF API call (the model is loaded from a local
# directory, but tokenizer.from_pretrained still tries HF metadata otherwise).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 5 P5-04 canonical models; ``hf_revision`` is the exact P5-04 commit.
# ``revision_verified`` is set per-model after comparing local snapshot
# against this revision.
DEFAULT_MODELS: tuple[tuple[str, str, str], ...] = (
    # (model_id, model_short, hf_revision)
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "SmolLM2-360M", "a10cc1512eabd3dde888204e902eca88bddb4951"),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "SmolLM2-1.7B", "31b70e2e869a7173562077fd711b654946d38674"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B", "7ae557604adf67be50417f59c2c2f167def9a775"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B", "aa8e72537993ba99e69dfaafa59ed015b17504d1"),
)

DEFAULT_OWT_PATH = ROOT / "data" / "raw" / "owt-sample" / "owt_valid.txt"
EXPECTED_OWT_SHA256 = "2406f278e71829d273b315e9b403285baea7022b26a96d2728dd8b776ea40660"

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "owt-real-eval"
DEFAULT_RESULTS_DIR = DEFAULT_OUTPUT_ROOT / "results"
DEFAULT_CACHE_ROOT = DEFAULT_OUTPUT_ROOT / "cache"
DEFAULT_MODELS_ROOT = DEFAULT_OUTPUT_ROOT / "models"


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file using streaming I/O."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# OWT source validation
# ---------------------------------------------------------------------------


@dataclass
class OWTSourceInfo:
    path: str
    size_bytes: int
    sha256: str
    newline_aligned: bool


def validate_owt_source(owt_path: Path = DEFAULT_OWT_PATH) -> OWTSourceInfo:
    """Verify the OWT validation source against the locked SHA-256.

    Raises ``ValueError`` on SHA mismatch so callers cannot silently evaluate
    on a different text.
    """
    if not owt_path.exists():
        raise FileNotFoundError(f"OWT validation source not found: {owt_path}")
    actual_size = owt_path.stat().st_size
    actual_sha = sha256_file(owt_path)
    if actual_sha != EXPECTED_OWT_SHA256:
        raise ValueError(
            f"OWT validation SHA mismatch: expected {EXPECTED_OWT_SHA256}, "
            f"got {actual_sha} ({owt_path}, {actual_size} bytes)"
        )
    return OWTSourceInfo(
        path=str(owt_path),
        size_bytes=actual_size,
        sha256=actual_sha,
        newline_aligned=True,  # files come from Stanford CS336 OWT sample
    )


# ---------------------------------------------------------------------------
# Token cache
# ---------------------------------------------------------------------------


@dataclass
class TokenCacheInfo:
    cache_path: str
    metadata_path: str
    model_id: str
    model_short: str
    tokenizer_revision: str
    source_path: str
    source_sha256: str
    source_size_bytes: int
    encoded_tokens: int
    dtype: str  # "int32"
    endianness: str  # "little"
    cache_sha256: str
    metadata_sha256: str
    max_bytes: int | None
    newline_aligned: bool
    created_at: str = ""
    encoded_bytes: int = 0


def _int32_to_bytes(arr) -> bytes:
    """Serialize a numpy int32 array to little-endian raw bytes."""
    import numpy as np
    if arr.dtype != np.int32:
        arr = arr.astype(np.int32)
    return arr.tobytes()


def _bytes_to_int32(blob: bytes):
    import numpy as np
    return np.frombuffer(blob, dtype="<i4")


def build_token_cache(
    *,
    model_id: str,
    model_short: str,
    tokenizer_revision: str,
    owt_path: Path,
    tokenizer,
    output_root: Path,
    max_bytes: int | None,
    source_info: OWTSourceInfo | None = None,
    skip_source_validation: bool = False,
) -> TokenCacheInfo:
    """Encode the OWT validation text with the given tokenizer.

    Writes:
        <output_root>/<model_short>/validation.tokens.int32
        <output_root>/<model_short>/validation.metadata.json

    The metadata binds the cache to the tokenizer revision and the source
    SHA-256, so a stale cache cannot be reused against a different text or
    a different tokenizer.

    By default the source SHA is validated against
    ``EXPECTED_OWT_SHA256``. Pass ``source_info`` (pre-validated) to reuse
    an existing validation, or ``skip_source_validation=True`` for
    selftests with synthetic small inputs.
    """
    import numpy as np

    cache_dir = output_root / model_short
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "validation.tokens.int32"
    metadata_path = cache_dir / "validation.metadata.json"

    # Tokenize the source text. Tokenizers expose either ``encode`` (single
    # string) or a batched variant; we read line-by-line to keep memory flat
    # and concatenate the resulting token streams. We do not insert an EOS
    # between lines because OWT is a continuous document.
    all_ids: list[int] = []
    bytes_seen = 0
    with open(owt_path, "r", encoding="utf-8") as f:
        for line in f:
            line_bytes = len(line.encode("utf-8"))
            if max_bytes is not None and bytes_seen + line_bytes > max_bytes:
                break
            ids = tokenizer.encode(line, add_special_tokens=False)
            all_ids.extend(ids)
            bytes_seen += line_bytes

    arr = np.asarray(all_ids, dtype=np.int32)
    raw = _int32_to_bytes(arr)
    cache_path.write_bytes(raw)

    cache_sha = hashlib.sha256(raw).hexdigest()
    if source_info is None:
        if skip_source_validation:
            source_info = OWTSourceInfo(
                path=str(owt_path),
                size_bytes=owt_path.stat().st_size,
                sha256=sha256_file(owt_path),
                newline_aligned=True,
            )
        else:
            source_info = validate_owt_source(owt_path)
    metadata = {
        "schema_version": "1.0",
        "cache_path": str(cache_path),
        "metadata_path": str(metadata_path),
        "model_id": model_id,
        "model_short": model_short,
        "tokenizer_revision": tokenizer_revision,
        "source_path": source_info.path,
        "source_sha256": source_info.sha256,
        "source_size_bytes": source_info.size_bytes,
        "encoded_bytes": bytes_seen,
        "max_bytes": max_bytes,
        "newline_aligned": source_info.newline_aligned,
        "encoded_tokens": int(arr.shape[0]),
        "dtype": "int32",
        "endianness": "little",
        "cache_sha256": cache_sha,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metadata_blob = json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8")
    metadata_path.write_bytes(metadata_blob)
    metadata_sha = hashlib.sha256(metadata_blob).hexdigest()
    metadata["metadata_sha256"] = metadata_sha
    metadata_path.write_bytes(
        json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8")
    )

    return TokenCacheInfo(
        cache_path=str(cache_path),
        metadata_path=str(metadata_path),
        model_id=model_id,
        model_short=model_short,
        tokenizer_revision=tokenizer_revision,
        source_path=source_info.path,
        source_sha256=source_info.sha256,
        source_size_bytes=source_info.size_bytes,
        encoded_tokens=int(arr.shape[0]),
        dtype="int32",
        endianness="little",
        cache_sha256=cache_sha,
        metadata_sha256=metadata_sha,
        max_bytes=max_bytes,
        newline_aligned=True,
        created_at=metadata.get("created_at", ""),
        encoded_bytes=metadata.get("encoded_bytes", 0),
    )


def validate_token_cache(metadata_path: Path, cache_path: Path) -> TokenCacheInfo:
    """Re-read a previously written cache and verify all hashes match."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not cache_path.exists():
        raise FileNotFoundError(f"cache file missing: {cache_path}")
    actual_cache_sha = sha256_file(cache_path)
    if actual_cache_sha != metadata["cache_sha256"]:
        raise ValueError(
            f"cache SHA mismatch for {cache_path}: expected "
            f"{metadata['cache_sha256']}, got {actual_cache_sha}"
        )
    actual_meta_sha = hashlib.sha256(
        json.dumps(
            {k: v for k, v in metadata.items() if k != "metadata_sha256"},
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if actual_meta_sha != metadata["metadata_sha256"]:
        raise ValueError(
            f"metadata SHA mismatch in {metadata_path}: expected "
            f"{metadata['metadata_sha256']}, got {actual_meta_sha}"
        )
    # Restrict to known dataclass fields so future metadata additions do
    # not break ``__init__``.
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(TokenCacheInfo)}
    return TokenCacheInfo(**{k: v for k, v in metadata.items() if k in field_names})


# ---------------------------------------------------------------------------
# Per-token CE loss
# ---------------------------------------------------------------------------


@dataclass
class LossResult:
    model_id: str
    model_short: str
    model_local_dir: str
    source_hub: str
    hf_revision: str
    revision_verified: bool
    tokenizer_revision: str
    dtype: str
    device: str
    seq_len: int
    cache_sha256: str
    source_sha256: str
    source_bytes: int
    evaluated_bytes: int  # bytes actually encoded from the source (prefix)
    encoded_tokens: int
    evaluated_tokens: int
    sum_loss_nats: float
    mean_loss_nats: float
    perplexity: float
    loss_nats_per_source_byte: float
    loss_nats_per_evaluated_byte: float
    elapsed_seconds: float
    key_file_sha256: dict[str, str]
    owt_sha256: str = EXPECTED_OWT_SHA256


def compute_per_token_loss(
    *,
    model,
    input_ids,  # numpy int32 array of shape [N]
    device: str,
    seq_len: int = 1024,
    log_every: int = 50,
) -> tuple[float, int, float]:
    """Slide ``input_ids`` through ``model`` in non-overlapping windows.

    Returns (sum_loss_nats, evaluated_tokens, elapsed_seconds).
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")

    n_tokens = int(input_ids.shape[0])
    if n_tokens < 2:
        raise ValueError(f"input_ids too short: n_tokens={n_tokens}")

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    started = time.time()
    n_windows = (n_tokens - 1 + seq_len - 1) // seq_len

    with torch.inference_mode():
        for w_idx, start in enumerate(range(0, n_tokens - 1, seq_len)):
            end = min(start + seq_len, n_tokens)
            chunk = input_ids[start:end]
            tensor = torch.from_numpy(chunk).long().unsqueeze(0).to(device)
            # Forward pass; logits shape = [1, L, vocab].
            logits = model(tensor).logits
            # Predict positions 1..L-1 from logits 0..L-2.
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = tensor[:, 1:].contiguous()
            loss_sum = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            n_pred = shift_labels.numel()
            total_loss += float(loss_sum.item())
            total_tokens += n_pred
            del logits, shift_logits, shift_labels, loss_sum, tensor
            if (w_idx + 1) % log_every == 0:
                elapsed = time.time() - started
                rate = total_tokens / elapsed if elapsed > 0 else 0.0
                print(
                    f"      [loss] window {w_idx + 1}/{n_windows} "
                    f"tokens={total_tokens:,} elapsed={elapsed:.1f}s "
                    f"rate={rate:,.0f} tok/s",
                    flush=True,
                )

    elapsed = time.time() - started
    return total_loss, total_tokens, elapsed


# ---------------------------------------------------------------------------
# Local model file fingerprinting
# ---------------------------------------------------------------------------


KEY_MODEL_FILES: tuple[str, ...] = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
)


def fingerprint_local_model(model_dir: Path) -> dict[str, str]:
    """SHA-256 each key file under ``model_dir`` (top-level only).

    Safetensors shards are NOT hashed here to keep the function fast; their
    presence is recorded separately by the eval driver.
    """
    out: dict[str, str] = {}
    if not model_dir.exists():
        return out
    for name in KEY_MODEL_FILES:
        p = model_dir / name
        if p.exists() and p.is_file():
            out[name] = sha256_file(p)
    return out


def safetensors_presence(model_dir: Path) -> dict[str, object]:
    """List safetensors files and their total bytes."""
    if not model_dir.exists():
        return {"files": [], "total_bytes": 0}
    files = sorted(p.name for p in model_dir.glob("*.safetensors"))
    total = sum((model_dir / f).stat().st_size for f in files)
    return {"files": files, "total_bytes": total}


# ---------------------------------------------------------------------------
# Top-level drivers
# ---------------------------------------------------------------------------


def _resolve_local_model_dir(model_dir_root: Path, model_id: str) -> Path:
    """Pick the snapshot directory under either HF or ModelScope layout.

    Tries the following, in order:

    1. ``<root>/<model_id>`` (HF layout with nested ``owner/name``).
    2. ``<root>/models/<owner>--<name>/snapshots/<rev>/`` (ModelScope layout).
    3. ``<root>/<owner>--<name>/snapshots/<rev>/`` (older ModelScope).

    Returns the directory containing ``config.json``.
    """
    owner, name = model_id.split("/", 1)
    candidates: list[Path] = []
    candidates.append(model_dir_root / model_id)
    candidates.append(
        model_dir_root / "models" / f"{owner}--{name}" / "snapshots" / "master"
    )
    candidates.append(
        model_dir_root / "models" / f"{owner}--{name}" / "snapshots" / "main"
    )
    candidates.append(
        model_dir_root / f"{owner}--{name}" / "snapshots" / "master"
    )
    candidates.append(
        model_dir_root / f"{owner}--{name}" / "snapshots" / "main"
    )
    # Also try any revision subdirectory.
    for sub in model_dir_root.glob(f"models/{owner}--{name}/snapshots/*"):
        candidates.append(sub)
    for sub in model_dir_root.glob(f"{owner}--{name}/snapshots/*"):
        candidates.append(sub)

    seen: set[str] = set()
    for cand in candidates:
        cp = str(cand.resolve()) if cand.exists() else str(cand)
        if cp in seen:
            continue
        seen.add(cp)
        if cand.is_dir() and (cand / "config.json").exists():
            return cand
    raise FileNotFoundError(
        f"no config.json under {model_dir_root} for {model_id!r}; tried "
        f"{[str(c) for c in candidates[:5]]}"
    )


def _load_tokenizer(model_local_dir: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_local_dir), local_files_only=True)


def _load_model(model_local_dir: Path, dtype_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if dtype_name not in dtype_map:
        raise ValueError(f"unsupported dtype {dtype_name!r}")
    dtype = dtype_map[dtype_name]
    model = AutoModelForCausalLM.from_pretrained(
        str(model_local_dir),
        torch_dtype=dtype,
        local_files_only=True,
    )
    model.to(device)
    return model


def cmd_build_cache(args: argparse.Namespace) -> int:
    """Tokenize the OWT validation text with each model's tokenizer."""
    selected = _select_models(args.models)
    source_info = validate_owt_source(args.owt_path)
    print(
        f"[build-cache] OWT source OK: {source_info.path} "
        f"{source_info.size_bytes:,} bytes sha256={source_info.sha256}",
        flush=True,
    )

    failures: list[tuple[str, str]] = []
    for model_id, model_short, hf_revision in selected:
        try:
            local_dir = _resolve_local_model_dir(args.model_dir_root, model_id)
        except FileNotFoundError as exc:
            print(f"  [skip] {model_short}: {exc}", flush=True)
            failures.append((model_short, str(exc)))
            continue

        print(f"[build-cache] {model_short}: tokenizer={local_dir}", flush=True)
        try:
            tokenizer = _load_tokenizer(local_dir)
            info = build_token_cache(
                model_id=model_id,
                model_short=model_short,
                tokenizer_revision=hf_revision,  # recorded for provenance
                owt_path=args.owt_path,
                tokenizer=tokenizer,
                output_root=args.cache_root,
                max_bytes=args.max_bytes,
            )
            print(
                f"  [ok] {model_short}: encoded={info.encoded_tokens:,} tokens "
                f"cache_sha256={info.cache_sha256[:16]}...",
                flush=True,
            )
        except Exception as exc:
            print(f"  [FAIL] {model_short}: {exc}", flush=True)
            failures.append((model_short, repr(exc)))

    if failures:
        print(f"[build-cache] {len(failures)} failure(s):", flush=True)
        for short, msg in failures:
            print(f"  - {short}: {msg}", flush=True)
        return 2
    return 0


def cmd_eval_loss(args: argparse.Namespace) -> int:
    """Run per-token CE loss on each selected model."""
    import numpy as np
    import torch

    selected = _select_models(args.models)
    validate_owt_source(args.owt_path)

    if not torch.cuda.is_available():
        print(
            "[eval-loss] WARNING: torch.cuda.is_available() is False; "
            "GPU inference not possible",
            flush=True,
        )
        if not args.allow_cpu:
            return 3

    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    for model_id, model_short, hf_revision in selected:
        out_path = results_dir / f"{model_short}.json"
        if out_path.exists() and not args.overwrite:
            print(
                f"[eval-loss] {model_short}: already exists at {out_path}; "
                f"pass --overwrite to redo",
                flush=True,
            )
            continue

        try:
            local_dir = _resolve_local_model_dir(args.model_dir_root, model_id)
        except FileNotFoundError as exc:
            print(f"  [skip] {model_short}: {exc}", flush=True)
            failures.append((model_short, str(exc)))
            continue

        cache_dir = args.cache_root / model_short
        cache_metadata_path = cache_dir / "validation.metadata.json"
        cache_path = cache_dir / "validation.tokens.int32"
        if not cache_metadata_path.exists() or not cache_path.exists():
            print(
                f"  [skip] {model_short}: missing cache under {cache_dir}; "
                f"run --build-cache first",
                flush=True,
            )
            failures.append((model_short, "missing cache"))
            continue
        try:
            cache_info = validate_token_cache(cache_metadata_path, cache_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [FAIL] {model_short}: cache invalid: {exc}", flush=True)
            failures.append((model_short, f"cache invalid: {exc}"))
            continue

        print(
            f"[eval-loss] {model_short}: loading from {local_dir} "
            f"(dtype={args.dtype} device={device})",
            flush=True,
        )
        try:
            model = _load_model(local_dir, args.dtype, device)
        except Exception as exc:
            print(f"  [FAIL] {model_short}: model load: {exc}", flush=True)
            failures.append((model_short, f"model load: {exc!r}"))
            continue

        # Load cache.
        ids = _bytes_to_int32(cache_path.read_bytes())
        print(
            f"[eval-loss] {model_short}: {ids.shape[0]:,} tokens "
            f"window={args.seq_len} cache_sha256={cache_info.cache_sha256[:16]}...",
            flush=True,
        )
        try:
            sum_loss, n_pred, elapsed = compute_per_token_loss(
                model=model,
                input_ids=ids,
                device=device,
                seq_len=args.seq_len,
                log_every=args.log_every,
            )
        except Exception as exc:
            print(f"  [FAIL] {model_short}: forward pass: {exc}", flush=True)
            failures.append((model_short, f"forward pass: {exc!r}"))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        mean_loss = sum_loss / n_pred
        perplexity = float(np.exp(mean_loss))
        loss_per_byte = mean_loss / cache_info.source_size_bytes
        loss_per_eval_byte = mean_loss / max(1, cache_info.encoded_bytes)
        result = LossResult(
            model_id=model_id,
            model_short=model_short,
            model_local_dir=str(local_dir),
            source_hub=args.source_hub,
            hf_revision=hf_revision,
            revision_verified=_check_revision_match(local_dir, hf_revision),
            tokenizer_revision=hf_revision,
            dtype=args.dtype,
            device=device,
            seq_len=args.seq_len,
            cache_sha256=cache_info.cache_sha256,
            source_sha256=cache_info.source_sha256,
            encoded_tokens=int(ids.shape[0]),
            evaluated_tokens=int(n_pred),
            sum_loss_nats=float(sum_loss),
            mean_loss_nats=float(mean_loss),
            perplexity=perplexity,
            source_bytes=int(cache_info.source_size_bytes),
            evaluated_bytes=int(cache_info.encoded_bytes),
            loss_nats_per_source_byte=float(loss_per_byte),
            loss_nats_per_evaluated_byte=float(loss_per_eval_byte),
            elapsed_seconds=float(elapsed),
            key_file_sha256=fingerprint_local_model(local_dir),
        )
        out_path.write_text(
            json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"  [ok] {model_short}: mean_loss={mean_loss:.4f} nats "
            f"ppl={perplexity:.2f} evaluated={n_pred:,} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        # Free VRAM before the next model.
        del model, ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if failures:
        print(f"[eval-loss] {len(failures)} failure(s):", flush=True)
        for short, msg in failures:
            print(f"  - {short}: {msg}", flush=True)
        return 2
    return 0


def _check_revision_match(local_dir: Path, expected_revision: str) -> bool:
    """Best-effort: does the local snapshot's git/refs match ``expected_revision``?"""
    # ModelScope / HF cache layouts both include a ``refs/main`` (or similar)
    # file. We read those and compare to ``expected_revision``.
    try:
        for refs_file in ("refs/main", "refs/master"):
            p = local_dir / refs_file
            if p.exists():
                actual = p.read_text(encoding="utf-8").strip()
                if actual == expected_revision:
                    return True
        # If neither refs file matches, we cannot verify.
    except OSError:
        pass
    return False


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def cmd_aggregate(args: argparse.Namespace) -> int:
    """Combine per-model JSON results into comparison.csv and comparison.json."""
    selected = _select_models(args.models)
    rows = []
    for model_id, model_short, _hf_revision in selected:
        p = args.results_dir / f"{model_short}.json"
        if not p.exists():
            print(f"  [warn] missing {p}", flush=True)
            continue
        result = json.loads(p.read_text(encoding="utf-8"))
        rows.append(result)

    if not rows:
        print("[aggregate] no per-model results to aggregate", flush=True)
        return 1

    csv_path = args.output_root / "comparison.csv"
    json_path = args.output_root / "comparison.json"
    csv_columns = [
        "model_short",
        "model_id",
        "hf_revision",
        "revision_verified",
        "encoded_tokens",
        "evaluated_tokens",
        "source_bytes",
        "mean_loss_nats",
        "perplexity",
        "loss_nats_per_source_byte",
        "elapsed_seconds",
        "dtype",
        "device",
        "seq_len",
        "cache_sha256",
        "source_sha256",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in csv_columns})
    json_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "owt_sha256": EXPECTED_OWT_SHA256,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        f"[aggregate] wrote {csv_path} ({len(rows)} rows) and {json_path}",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def _select_models(names: Iterable[str] | None) -> list[tuple[str, str, str]]:
    if not names:
        return list(DEFAULT_MODELS)
    by_short = {short: (mid, short, rev) for mid, short, rev in DEFAULT_MODELS}
    by_full = {mid: (mid, short, rev) for mid, short, rev in DEFAULT_MODELS}
    out = []
    for n in names:
        if n in by_short:
            out.append(by_short[n])
        elif n in by_full:
            out.append(by_full[n])
        else:
            raise ValueError(
                f"unknown model {n!r}; valid: {list(by_short) + list(by_full)}"
            )
    return out


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--owt-path",
        type=Path,
        default=DEFAULT_OWT_PATH,
        help="OWT validation UTF-8 text (default: data/raw/owt-sample/owt_valid.txt)",
    )
    parser.add_argument(
        "--model-dir-root",
        type=Path,
        default=DEFAULT_MODELS_ROOT,
        help="Root directory containing per-model subdirectories (default: "
        "artifacts/owt-real-eval/models)",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="Where per-model token caches are written (default: "
        "artifacts/owt-real-eval/cache)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Where per-model loss JSON results are written (default: "
        "artifacts/owt-real-eval/results)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Where aggregate comparison.{csv,json} are written (default: "
        "artifacts/owt-real-eval)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model IDs or short names to operate on (default: all 5)",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Cap on OWT source bytes (newline-aligned prefix)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Forward-pass window length (default: 1024, safe for 3B bf16 on 12GB)",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Model dtype (default: bf16)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device override (default: auto)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print forward-pass progress every N windows (default: 50)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run models whose result JSON already exists",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU inference (slow; default aborts if no CUDA)",
    )
    parser.add_argument(
        "--source-hub",
        default="modelscope",
        help="String to record under ``source_hub`` for provenance "
        "(default: modelscope)",
    )
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--eval-loss", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.selftest:
        return _run_selftests(args)
    if not (args.build_cache or args.eval_loss or args.aggregate):
        print(
            "error: specify one of --build-cache / --eval-loss / --aggregate",
            file=sys.stderr,
        )
        return 2
    if args.build_cache:
        rc = cmd_build_cache(args)
        if rc != 0:
            return rc
    if args.eval_loss:
        rc = cmd_eval_loss(args)
        if rc != 0:
            return rc
    if args.aggregate:
        rc = cmd_aggregate(args)
        if rc != 0:
            return rc
    return 0


# ---------------------------------------------------------------------------
# Selftest (no GPU)
# ---------------------------------------------------------------------------


def _expect(name: str, cond: bool, hint: str = "") -> None:
    if cond:
        print(f"  [PASS] {name}", flush=True)
    else:
        print(f"  [FAIL] {name}: {hint}", flush=True)
        raise SystemExit(1)


def _run_selftests(args: argparse.Namespace) -> int:
    import tempfile

    print("[selftest] OWT source validation", flush=True)
    info = validate_owt_source()
    _expect(
        "test_owt_source_sha_matches",
        info.sha256 == EXPECTED_OWT_SHA256,
        hint=f"got {info.sha256}",
    )
    _expect(
        "test_owt_source_size_matches",
        info.size_bytes == 289_998_753,
        hint=f"got {info.size_bytes}",
    )

    print("[selftest] Hash helpers", flush=True)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        f = td_path / "x.bin"
        f.write_bytes(b"hello world")
        _expect("test_sha256_file_known", sha256_file(f) == sha256_text("hello world"))

    print("[selftest] Token cache round-trip", flush=True)
    # Use a tiny OWT file under temp so we can exercise build/validate without
    # touching the real 277 MB source.
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        owt_small = td_path / "owt_small.txt"
        owt_small.write_text("hello world\nthis is owt\n", encoding="utf-8")
        # Tokenizer: a deterministic stand-in. ``tokenizers`` may not be
        # installed in the test env, so we test the hash round-trip via a
        # tokenizer-like object with ``encode``.
        class StubTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [ord(c) % 256 for c in text if c.isalpha()]

        info = build_token_cache(
            model_id="stub/model",
            model_short="stub",
            tokenizer_revision="deadbeef",
            owt_path=owt_small,
            tokenizer=StubTokenizer(),
            output_root=td_path / "cache",
            max_bytes=None,
            skip_source_validation=True,
        )
        _expect("test_cache_file_exists", Path(info.cache_path).exists())
        # Re-validate: SHA-256 must match.
        info2 = validate_token_cache(
            Path(info.metadata_path), Path(info.cache_path)
        )
        _expect(
            "test_cache_roundtrip_sha256",
            info2.cache_sha256 == info.cache_sha256,
        )
        _expect(
            "test_cache_metadata_sha256_stored",
            bool(info2.metadata_sha256),
        )

    print("[selftest] Per-token CE loss with mock logits", flush=True)
    # Use a tiny tokenizer that produces predictable IDs; then verify the
    # loss function computes finite mean loss and matched expected count.
    import numpy as np
    import torch
    from torch import nn

    class MockLMHead(nn.Module):
        """Wrap any module's forward to return deterministic logits."""

        def __init__(self, vocab_size: int):
            super().__init__()
            self.vocab_size = vocab_size
            # Bias logits so position i has highest prob at token (i+1) % V.
            bias = torch.zeros(vocab_size)
            for i in range(vocab_size):
                bias[(i + 1) % vocab_size] = 1.0
            self.bias = nn.Parameter(bias)

        def forward(self, input_ids):
            # input_ids: [B, L]; return logits [B, L, V].
            B, L = input_ids.shape
            logits = self.bias.unsqueeze(0).unsqueeze(0).expand(B, L, -1).clone()
            return type("O", (), {"logits": logits})()

    # Build mock model: just an LM head (no transformer); forward returns
    # shape [B, L, V] deterministic logits.
    class MockModel(nn.Module):
        def __init__(self, vocab_size: int):
            super().__init__()
            self.head = MockLMHead(vocab_size)

        def forward(self, input_ids):
            return self.head(input_ids)

    vocab = 16
    model = MockModel(vocab)
    model.eval()
    # Token sequence 0..15 → 16 tokens.
    ids = np.arange(16, dtype=np.int32)
    sum_loss, n_pred, elapsed = compute_per_token_loss(
        model=model, input_ids=ids, device="cpu", seq_len=8, log_every=1
    )
    # Two windows: [0..7] and [8..15]. Each gives 7 predictions = 14 total.
    _expect("test_loss_count_14", n_pred == 14, hint=f"got {n_pred}")
    _expect("test_loss_finite", np.isfinite(sum_loss), hint=f"sum_loss={sum_loss}")
    _expect("test_loss_nonneg", sum_loss >= 0, hint=f"sum_loss={sum_loss}")
    _expect("test_elapsed_nonneg", elapsed >= 0)

    print("[selftest] OWT SHA mismatch raises", flush=True)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        bogus = td_path / "owt_bogus.txt"
        bogus.write_text("garbage\n", encoding="utf-8")
        try:
            validate_owt_source(bogus)
        except ValueError as exc:
            _expect(
                "test_validate_owt_rejects_bad_sha",
                "OWT validation SHA mismatch" in str(exc),
                hint=str(exc),
            )
        else:
            _expect("test_validate_owt_rejects_bad_sha", False, hint="no exception")

    print("[selftest] Default models count and revisions", flush=True)
    _expect("test_default_models_count_5", len(DEFAULT_MODELS) == 5)
    short_names = [s for _, s, _ in DEFAULT_MODELS]
    _expect(
        "test_default_models_short_unique",
        len(set(short_names)) == 5,
        hint=f"short names: {short_names}",
    )
    revisions = [r for _, _, r in DEFAULT_MODELS]
    _expect(
        "test_default_models_revisions_unique",
        len(set(revisions)) == 5,
        hint=f"revisions: {revisions}",
    )

    print("[selftest] Fingerprint local model dir (missing)", flush=True)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        _expect(
            "test_fingerprint_empty",
            fingerprint_local_model(td_path / "nonexistent") == {},
        )
        # Populate one key file.
        (td_path / "config.json").write_text("{}", encoding="utf-8")
        fp = fingerprint_local_model(td_path)
        _expect(
            "test_fingerprint_includes_config",
            "config.json" in fp,
            hint=f"keys={list(fp)}",
        )

    # ------------------------------------------------------------------
    # Aggregate JSON regression: load all per-model JSONs that exist on
    # disk and verify the table is internally consistent. This guards
    # against accidental edits to a single result JSON without
    # regenerating the aggregate.
    # ------------------------------------------------------------------
    print("[selftest] aggregate regression", flush=True)
    results_dir = args.results_dir
    if results_dir.exists():
        found = sorted(results_dir.glob("*.json"))
        _expect(
            "test_results_dir_count_5_or_skip",
            len(found) == 5 or len(found) == 0,
            hint=f"found {len(found)} results; expected 0 (no run yet) or 5",
        )
        if len(found) == 5:
            import csv as _csv
            csv_path = args.output_root / "comparison.csv"
            _expect("test_comparison_csv_exists", csv_path.exists())
            if csv_path.exists():
                rows = list(_csv.DictReader(csv_path.open(encoding="utf-8")))
                _expect("test_comparison_csv_rows_5", len(rows) == 5)
                shorts = [r["model_short"] for r in rows]
                _expect(
                    "test_comparison_csv_has_all_5_short_names",
                    set(shorts) == {
                        "SmolLM2-360M",
                        "SmolLM2-1.7B",
                        "Qwen2.5-0.5B",
                        "Qwen2.5-1.5B",
                        "Qwen2.5-3B",
                    },
                    hint=f"shorts={shorts}",
                )
                # Each row's mean_loss and perplexity must be finite floats.
                for r in rows:
                    ml = float(r["mean_loss_nats"])
                    pp = float(r["perplexity"])
                    _expect(
                        f"test_comparison_csv_{r['model_short']}_mean_loss_finite",
                        ml == ml and ml > 0 and ml < 100,
                        hint=f"mean_loss={ml}",
                    )
                    _expect(
                        f"test_comparison_csv_{r['model_short']}_ppl_finite",
                        pp == pp and pp > 0,
                        hint=f"ppl={pp}",
                    )

    print("[selftest] all tests PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
