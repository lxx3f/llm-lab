"""Evaluate an SFT tool-calling model on held-out tool-calling samples.

Pipeline:
1. For each sample (default: D1 template dev split, 13 samples the SFT
   model never trained on), prompt the model with the user turn.
2. Greedy-generate the assistant continuation.
3. Try to parse a tool-call JSON array from the generated text (the SFT
   template serializes plans as ``[{"call_id", "name", "arguments"}]``).
4. Classify the transcript through the P1-05 8-level failure classifier
   and report per-layer failure distribution.

Because the model is small and imperfect, generated text is often
malformed; the evaluator deliberately exercises the classifier's
robustness (parse_success=False paths) as well as correct transcripts.

Usage:
    .venv/python.exe scripts/eval_sft_tool.py \
        --checkpoint artifacts/checkpoints/sft-tool-v1.pt \
        --samples-dir datasets/tool-calling-d1/dev \
        --output artifacts/sft-eval-result.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from architecture_lab.models.dense_transformer import DenseTransformer  # noqa: E402
from architecture_lab.tokenization import BPETokenizer  # noqa: E402
from architecture_lab.training.dense_training import (  # noqa: E402
    build_model_config,
    resolve_device,
)
from architecture_lab.training.sft_training import (  # noqa: E402
    ASSISTANT_SEP,
    USER_SEP,
    _find_user_turn,
)
from scripts.classify_tool_failure import classify  # noqa: E402

DEFAULT_TOKENIZER = ROOT / "artifacts" / "tokenizers" / "owt-bpe" / "v0.2.0" / "tokenizer.json"


def extract_tool_calls(generated: str) -> list[dict] | None:
    """Best-effort extraction of tool-call JSON object(s) from generated text.

    The SFT template serializes plans as newline-separated JSON objects
    (``{"call_id"...}``) WITHOUT an enclosing array. This parser accepts
    both the object form and a legacy array form. Returns None when no
    plausible call object is found (the classifier reports
    parse_success=False).
    """
    text = generated
    if "```" in text:
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text
    calls: list[dict] = []
    # Array form: [...].
    arr_start = text.find("[")
    if arr_start != -1:
        arr_end = text.find("]", arr_start)
        if arr_end != -1:
            try:
                parsed = json.loads(text[arr_start : arr_end + 1])
                if isinstance(parsed, list):
                    return [c for c in parsed if isinstance(c, dict)]
            except json.JSONDecodeError:
                pass
    # Object form: successive {"call_id"...} objects.
    pos = 0
    while True:
        start = text.find('{"call_id"', pos)
        if start == -1:
            start = text.find('{"', pos)
        if start == -1:
            break
        depth = 0
        in_string = False
        escape = False
        end = -1
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            break
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pos = start + 1
            continue
        if isinstance(obj, dict) and "name" in obj:
            calls.append(obj)
        pos = end + 1
    return calls if calls else None


@torch.no_grad()
def generate(
    model: DenseTransformer,
    tokenizer: BPETokenizer,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    model.eval()
    eot = tokenizer.encode("<|endoftext|>")[0]
    gen = list(prompt_ids)
    for _ in range(max_new_tokens):
        window = torch.tensor([gen[-model.config.max_seq_len :]],
                              dtype=torch.long, device=device)
        logits, _ = model(window)
        nxt = int(logits[0, -1].argmax())
        gen.append(nxt)
        if nxt == eot:
            break
    return tokenizer.decode(gen[len(prompt_ids):])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "sft-eval-result.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer)
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = build_model_config(
        {"model": {"max_seq_len": 256, "d_model": 128, "n_heads": 4,
                   "n_layers": 4, "d_ff": 512, "vocab_size": tokenizer.vocab_size}},
        tokenizer.vocab_size,
    )
    model = DenseTransformer(cfg).to(device=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sample_paths = sorted(args.samples_dir.glob("*.json"))
    if args.limit:
        sample_paths = sample_paths[: args.limit]

    rows: list[dict] = []
    layer_counts: dict[str, int] = {}
    total = len(sample_paths)
    for path in sample_paths:
        sample = json.loads(path.read_text(encoding="utf-8"))
        user_turn = _find_user_turn(sample["messages"])
        prompt_ids = tokenizer.encode(USER_SEP + user_turn + "\n" + ASSISTANT_SEP)
        generated = generate(model, tokenizer, prompt_ids,
                             max_new_tokens=args.max_new_tokens, device=device)
        calls = extract_tool_calls(generated)
        transcript = {"tool_calls": calls, "final_answer": generated}
        result = classify(sample, transcript)
        first = result["first_failure"]
        layer_counts[first if first is not None else "none"] = (
            layer_counts.get(first if first is not None else "none", 0) + 1
        )
        rows.append({
            "sample_id": sample.get("id"),
            "task_type": sample.get("metadata", {}).get("task_type"),
            "user_turn": user_turn,
            "generated": generated[:200],
            "extracted_calls": calls,
            "layers": result["layers"],
            "first_failure": first,
        })
        print(f"[eval] {sample.get('id')} first_failure={first} "
              f"calls={len(calls) if calls else 'parse-fail'}", flush=True)

    summary = {
        "checkpoint": str(args.checkpoint),
        "samples_dir": str(args.samples_dir),
        "total": total,
        "first_failure_distribution": layer_counts,
        "no_failure": layer_counts.get("none", 0),
        "parse_success_rate": round(
            sum(1 for r in rows if r["layers"].get("parse_success") is True) / total, 3
        ) if total else 0.0,
    }
    payload = {"summary": summary, "rows": rows}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"[eval] wrote {args.output}")
    print(f"[eval] summary: {json.dumps(summary, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
