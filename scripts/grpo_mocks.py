"""Mock policy + tokenizer for unconditional GRPO MVP smoke tests.

These mocks provide a tiny ``torch.nn.Module`` "policy" and a
``MockTokenizer`` that mimic the surface area required by
``scripts.grpo_train``:

- ``MockTokenizer.apply_chat_template(messages, tokenize=False,
  add_generation_prompt=True, tools=...)`` returns a deterministic
  text prompt.
- ``MockTokenizer.encode(text)`` / ``__call__(text, return_tensors="pt",
  ...)`` return ``input_ids`` and ``attention_mask`` tensors.
- ``MockTokenizer.decode(token_ids)`` returns text.
- ``MockPolicy.generate(input_ids, max_new_tokens, ...)`` returns
  ``output_ids`` with deterministic continuations.

The mocks are intentionally small (a few hundred lines) so the
unconditional smoke in ``tests/test_grpo_mvp.py`` can exercise the
full rollout → reward → advantage → policy-update → checkpoint loop
without HF transformers or network access.
"""

from __future__ import annotations

from typing import Any

import torch


class MockTokenizer:
    """Deterministic tokenizer for smoke tests.

    Maps each whitespace-separated token to a deterministic integer in
    ``[10, vocab_size)``. ``pad_token_id`` is 0; ``eos_token_id`` is 1.
    ``apply_chat_template`` returns a concatenation of
    ``f"{role}: {content}"`` lines; ``generate`` is a no-op token
    sequence that includes both a continuation and a synthetic
    tool-call hint so ``scripts.eval_sft_tool.extract_tool_calls``
    parses cleanly.
    """

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        # A simple hash so the same token always gets the same id.
        self._cache: dict[str, int] = {}
        # Mock chat_template attribute so the call-site branches
        # through apply_chat_template.
        self.chat_template = "<mock>"

    def _tok_id(self, tok: str) -> int:
        if tok in self._cache:
            return self._cache[tok]
        h = abs(hash(tok)) % (self.vocab_size - 10) + 10
        self._cache[tok] = h
        return h

    def apply_chat_template(
        self, messages: list[dict[str, Any]], *,
        tokenize: bool = False, add_generation_prompt: bool = True,
        tools: list | None = None,
    ) -> str:
        """Render messages into a deterministic text prompt."""
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            parts.append(f"{role}: {content}")
        if add_generation_prompt:
            parts.append("assistant:")
        if tools:
            for t in tools:
                parts.append(f"[tool:{t.get('name','?')}]")
        return "\n".join(parts)

    def __call__(self, text: str, *, return_tensors: str = "pt",
                 truncation: bool = True, max_length: int = 2048):
        toks = text.split()
        ids = [self._tok_id(t) for t in toks][:max_length]
        # Ensure non-empty so model.generate doesn't crash
        if not ids:
            ids = [self.eos_token_id]
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode(self, text: str) -> list[int]:
        return [self._tok_id(t) for t in text.split()]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        # Render each non-special id as a deterministic pseudo-word.
        # Including tool-call hints so extract_tool_calls picks them up.
        out_words = []
        for i, tid in enumerate(token_ids):
            if skip_special_tokens and tid in (self.pad_token_id, self.eos_token_id):
                continue
            # Cycle through a fixed dictionary to make outputs readable
            if i % 3 == 0:
                out_words.append("hello")
            elif i % 3 == 1:
                out_words.append("world")
            else:
                # Add a synthetic tool call every 4th token so the
                # extract_tool_calls regex can parse something.
                out_words.append('<tool_call>{"name":"mock_tool","arguments":{}}</tool_call>')
        return " ".join(out_words) if out_words else ""


class MockPolicy(torch.nn.Module):
    """Tiny ``torch.nn.Module`` policy that returns deterministic logits.

    Behaves like a causal-LM head: ``forward(input_ids)`` returns
    ``logits`` of shape ``(batch, seq, vocab)``. ``generate`` returns
    a continuation that is the same shape as the input plus a few
    new tokens.

    The policy parameters are exposed so the GRPO trainer can compute
    gradients on them.
    """

    def __init__(self, vocab_size: int = 256, hidden: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, hidden)
        self.head = torch.nn.Linear(hidden, vocab_size, bias=False)
        self.pad_token_id = 0
        self.eos_token_id = 1

    def forward(self, input_ids: torch.Tensor, *, labels=None,
                attention_mask=None, **_):
        hidden = self.embedding(input_ids)
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            # Cross-entropy on the unmasked positions (labels != -100).
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), ignore_index=-100,
            )
        return type("Outputs", (), {"logits": logits, "loss": loss})()

    def generate(
        self, *, input_ids: torch.Tensor, max_new_tokens: int,
        do_sample: bool = False, temperature: float = 1.0,
        pad_token_id: int | None = None,
        **_,
    ) -> torch.Tensor:
        """Return ``input_ids`` concatenated with ``max_new_tokens`` new
        ids chosen deterministically from the model's forward pass.
        """
        with torch.inference_mode():
            hidden = self.embedding(input_ids)
            logits = self.head(hidden)
            argmax = torch.argmax(logits, dim=-1)
            # Pick a deterministic continuation: cycle through argmax of
            # each position, shifted by 1.
            new_ids: list[int] = []
            for i in range(max_new_tokens):
                # Greedy pick from the last position of argmax
                pick = int(argmax[0, -1].item()) % self.vocab_size
                # Skip pad/eos so the continuation is non-empty
                if pick in (pad_token_id, self.eos_token_id):
                    pick = 2  # safe id
                new_ids.append(pick)
            new = torch.tensor([new_ids], dtype=input_ids.dtype)
        return torch.cat([input_ids, new], dim=1)

    def eval(self):  # no-op override to keep the surface uniform
        return super().eval()

    def train(self, mode: bool = True):  # no-op override
        return super().train(mode)


def make_mock_policy_and_tokenizer(
    vocab_size: int = 256,
) -> tuple[MockPolicy, MockTokenizer]:
    """Construct a (policy, tokenizer) pair for unconditional smoke."""
    tokenizer = MockTokenizer(vocab_size=vocab_size)
    model = MockPolicy(vocab_size=vocab_size)
    return model, tokenizer