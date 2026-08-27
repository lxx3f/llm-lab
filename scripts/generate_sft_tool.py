"""SFT tool-calling inference (greedy generation from a saved checkpoint).

Loads a DenseTransformer trained by scripts/train_sft.py and generates the
assistant continuation for a user turn, following the same template used
during training (### User / ### Assistant / ### Result / answer).

Usage:
    .venv/python.exe scripts/generate_sft_tool.py \
        --checkpoint artifacts/checkpoints/sft-tool-v1.pt \
        --user "请帮我查一下北京天气" \
        --max_new_tokens 128
"""

from __future__ import annotations

import argparse
import json
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
    resolve_dtype,
)
from architecture_lab.training.sft_training import USER_SEP, ASSISTANT_SEP  # noqa: E402

DEFAULT_TOKENIZER = ROOT / "artifacts" / "tokenizers" / "owt-bpe" / "v0.2.0" / "tokenizer.json"


@torch.no_grad()
def generate(
    model: DenseTransformer,
    tokenizer: BPETokenizer,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 0.0,
) -> list[int]:
    """Greedy (temperature=0) autoregressive generation."""
    model.eval()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    if input_ids.size(1) > model.config.max_seq_len:
        input_ids = input_ids[:, -model.config.max_seq_len :]
    generated = list(input_ids[0].tolist())
    for _ in range(max_new_tokens):
        window = torch.tensor([generated[-model.config.max_seq_len :]],
                              dtype=torch.long, device=device)
        logits, _ = model(window)
        next_logits = logits[0, -1, :]
        if temperature <= 0:
            next_id = int(next_logits.argmax())
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
        generated.append(next_id)
        if next_id == tokenizer.encode("<|endoftext|>")[0]:
            break
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer)
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype("float32")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model_config" in ckpt:
        model_config = build_model_config({"model": ckpt["model_config"]}, tokenizer.vocab_size)
    else:
        model_config = build_model_config(
            {"model": {"max_seq_len": 256, "d_model": 128, "n_heads": 4,
                       "n_layers": 4, "d_ff": 512, "vocab_size": tokenizer.vocab_size}},
            tokenizer.vocab_size,
        )
    model = DenseTransformer(model_config).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prefix = USER_SEP + args.user + "\n" + ASSISTANT_SEP
    prompt_ids = tokenizer.encode(prefix)
    out_ids = generate(
        model, tokenizer, prompt_ids,
        max_new_tokens=args.max_new_tokens,
        device=device, temperature=args.temperature,
    )
    text = tokenizer.decode(out_ids[len(prompt_ids):])
    print("=== prompt ===")
    print(prefix)
    print("=== generated ===")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
