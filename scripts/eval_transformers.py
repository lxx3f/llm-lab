"""Evaluate an open-weights instruction-tuned LLM on D2 multi-turn held-out split.

Pipeline:
1. Load the model + tokenizer via the Hugging Face ``transformers``
   library (``AutoModelForCausalLM`` / ``AutoTokenizer``).
2. For every sample under ``--samples-dir`` (default:
   ``datasets/tool-calling-d2/dev``), render the messages history into
   the model's native chat template and greedy-generate the assistant
   continuation.
3. Reuse ``scripts.eval_sft_tool.extract_tool_calls`` to parse the
   generated text into a tool-call list, and ``classify_tool_failure``
   to walk the P1-05 8-level failure classifier over the resulting
   transcript.
4. Persist a JSON artifact with the **same schema** as
   ``scripts/eval_sft_tool.py`` (``{summary, rows}``) so
   ``scripts/reward_offline.py --transcripts`` consumes the output
   without modification.

Why a separate script:
- ``eval_sft_tool.py`` is tied to the in-repo ``DenseTransformer`` /
  ``MoETransformer`` classes whose architectures and tokenizer are
  saved alongside the checkpoint; a public instruction-tuned model has
  neither and must be loaded through its native ``transformers`` API.
- ``eval_sft_tool.py`` already implements multi-turn prompt
  serialisation (``--prompt-mode multi_turn``); the public models use
  each tokenizer's built-in ``apply_chat_template`` instead, which
  honours the model's training-time tool-call formatting. This script
  bridges the two.

Usage:
    .venv/python.exe scripts/eval_transformers.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --samples-dir datasets/tool-calling-d2/dev \\
        --output artifacts/qwen2.5-0.5b-instruct-eval-d2dev.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classify_tool_failure import classify  # noqa: E402
from scripts.eval_sft_tool import extract_tool_calls  # noqa: E402


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id or local path (e.g. Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("datasets/tool-calling-d2/dev"),
        help="Directory of D2 multi-turn sample JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the {summary, rows} JSON artifact consumed by reward_offline.py",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Greedy generation budget (default 256, matches eval_sft_tool.py)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on the number of samples evaluated (0 = all)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device override (default: auto = cuda if available else cpu)",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Model dtype (default: bf16, requires CUDA compute capability >= 8.0)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face model revision/tag/commit (default: main)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass through to AutoModel/AutoTokenizer for custom code models",
    )
    return parser


def _resolve_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def _resolve_dtype(name: str, device: str):
    resolved = torch.bfloat16 if name == "bf16" else (
        torch.float16 if name == "fp16" else torch.float32
    )
    if device == "cpu" and resolved is not torch.float32:
        print(
            "[transformers-eval] CPU does not reliably support low-precision "
            "generation; overriding dtype to torch.float32",
            flush=True,
        )
        return torch.float32
    return resolved


def _strip_terminal_assistant(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the final assistant message when it carries no tool_calls.

    D2 sample ``messages`` end with a gold ``assistant`` content message
    (``expected_answer``). The model is supposed to produce this very
    content — passing it through the chat template leaks the target
    answer into the prompt, inflating reward metrics. Mirror the
    behaviour of ``scripts.eval_sft_tool`` which does the same strip
    before serialising the prompt.
    """
    while messages and messages[-1].get("role") == "assistant":
        if messages[-1].get("tool_calls"):
            break
        messages = messages[:-1]
    return messages


def _apply_chat_template(
    tokenizer,
    sample: dict[str, Any],
) -> str:
    """Render the D2 multi-turn messages using the model's native chat template.

    Falls back to a ``str(sample['messages'])`` dump when the tokenizer
    has no chat template (older models). The fallback is deterministic but
    may not honour the model's tool-call formatting, so such models
    should be reported with appropriate caveats.

    The terminal assistant message (the gold ``expected_answer``) is
    always stripped before rendering so the model must reproduce it from
    context alone. Earlier assistant ``tool_calls`` messages and tool
    results are kept as context — they describe how the conversation
    arrived at the final answer.
    """
    messages = _strip_terminal_assistant(list(sample.get("messages", [])))
    if hasattr(tokenizer, "apply_chat_template") and getattr(
        tokenizer, "chat_template", None
    ):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=sample.get("tools"),
            )
        except Exception:
            pass
    return "\n".join(
        f"{m.get('role', 'user')}: {m.get('content','')}"
        for m in messages
        if m.get("role") in ("user", "assistant", "system")
    )


def _greedy_generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    device: str,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.inference_mode():
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
        }
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        output = model.generate(**inputs, **generation_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def _build_eval_row(
    sample: dict[str, Any],
    prompt_text: str,
    generated: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse and classify one generation, returning the artifact row and layers.

    ``extract_tool_calls`` uses ``None`` for "no JSON call found", but an
    empty list is the valid structured transcript for D2 no-tool tasks. Keep
    the normalized empty list in the persisted row as well as in the
    classifier input so reward_offline reproduces the same result.
    """
    calls = extract_tool_calls(generated)
    normalized_calls = calls if calls is not None else []
    transcript = {
        "tool_calls": normalized_calls,
        "final_answer": generated,
    }
    result = classify(sample, transcript)
    row = {
        "sample_id": sample.get("id"),
        "task_type": sample.get("metadata", {}).get("task_type"),
        "user_turn": prompt_text[:200],
        "generated": generated,
        "generated_preview": generated[:200],
        "extracted_calls": normalized_calls,
        "layers": result["layers"],
        "first_failure": result["first_failure"],
    }
    return row, result


def _update_layer_counts(
    layer_counts: dict[str, int], result: dict[str, Any]
) -> str | None:
    first = result["first_failure"]
    key = first if first is not None else "none"
    layer_counts[key] = layer_counts.get(key, 0) + 1
    return first


def main() -> int:
    args = _build_argparser().parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    cache_dir = os.environ.get("HF_HOME") or str(ROOT / "artifacts" / "huggingface")

    print(f"[transformers-eval] loading {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        cache_dir=cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        cache_dir=cache_dir,
    )
    model.to(device)
    model.eval()
    print(f"[transformers-eval] device={device} dtype={dtype}", flush=True)

    sample_paths = sorted(args.samples_dir.glob("*.json"))
    if args.limit:
        sample_paths = sample_paths[: args.limit]

    rows: list[dict[str, Any]] = []
    layer_counts: dict[str, int] = {}
    total = len(sample_paths)
    for index, path in enumerate(sample_paths, 1):
        sample = json.loads(path.read_text(encoding="utf-8"))
        prompt_text = _apply_chat_template(tokenizer, sample)
        try:
            generated = _greedy_generate(
                model, tokenizer, prompt_text,
                max_new_tokens=args.max_new_tokens, device=device,
            )
        except torch.cuda.OutOfMemoryError as exc:
            print(f"[transformers-eval] OOM on {path.name}: {exc}", flush=True)
            generated = ""
        except Exception as exc:
            print(f"[transformers-eval] generation error on {path.name}: {exc}",
                  flush=True)
            generated = ""
        row, result = _build_eval_row(sample, prompt_text, generated)
        first = _update_layer_counts(layer_counts, result)
        rows.append(row)
        calls = row["extracted_calls"]
        print(f"[transformers-eval] {index}/{total} {sample.get('id')} "
              f"first_failure={first} calls={len(calls) if calls else 'parse-fail'}",
              flush=True)

    parse_success_count = sum(
        1 for row in rows if row["layers"].get("parse_success") is True
    )
    summary = {
        "checkpoint": args.model,
        "backend": "transformers",
        "revision": args.revision,
        "device": device,
        "dtype": str(dtype),
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "samples_dir": str(args.samples_dir),
        "total": total,
        "parse_success_count": parse_success_count,
        "first_failure_distribution": layer_counts,
        "no_failure": layer_counts.get("none", 0),
        "parse_success_rate": (
            layer_counts.get("parse_success", 0) / total if total else 0.0
        ),
    }
    payload = {"summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[transformers-eval] wrote {args.output}", flush=True)
    print(f"[transformers-eval] summary: {json.dumps(summary, ensure_ascii=False)}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())