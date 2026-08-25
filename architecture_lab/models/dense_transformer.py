"""Minimal decoder-only Transformer baseline for architecture experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 256
    max_seq_len: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.0
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_model // self.n_heads % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.max_seq_len < 1 or self.vocab_size < 1:
            raise ValueError("vocab_size and max_seq_len must be positive")


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, q: Tensor, k: Tensor, start_pos: int = 0) -> tuple[Tensor, Tensor]:
        seq_len = q.size(-2)
        end_pos = start_pos + seq_len
        if end_pos > self.cos.size(0):
            raise ValueError("sequence exceeds configured max_seq_len")
        cos = self.cos[start_pos:end_pos].to(device=q.device, dtype=q.dtype)[None, None]
        sin = self.sin[start_pos:end_pos].to(device=q.device, dtype=q.dtype)[None, None]
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)

    @staticmethod
    def _apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
        return x * cos.repeat_interleave(2, dim=-1) + rotated * sin.repeat_interleave(2, dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_base)
        self.dropout = config.dropout

    def forward(self, x: Tensor, start_pos: int = 0, kv_cache: Optional[dict[str, Tensor]] = None) -> Tensor:
        batch, seq_len, d_model = x.shape
        q, k, v = self.qkv(x).split(d_model, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, start_pos=start_pos)

        if kv_cache is not None:
            previous_k = kv_cache.get("k")
            previous_v = kv_cache.get("v")
            if previous_k is not None:
                if previous_k.size(-2) != start_pos:
                    raise ValueError("start_pos must equal the cached sequence length")
                k = torch.cat((previous_k, k), dim=-2)
                v = torch.cat((previous_v, v), dim=-2)
            kv_cache["k"] = k.detach()
            kv_cache["v"] = v.detach()

        past_len = k.size(-2)
        causal = torch.ones(seq_len, past_len, device=x.device, dtype=torch.bool).tril(
            diagonal=past_len - seq_len
        )
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(output)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x: Tensor, start_pos: int = 0, kv_cache: Optional[dict[str, Tensor]] = None) -> Tensor:
        x = x + self.attn(self.attn_norm(x), start_pos=start_pos, kv_cache=kv_cache)
        return x + self.ffn(self.ffn_norm(x))


class DenseTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        start_pos: int = 0,
        kv_cache: Optional[list[dict[str, Tensor]]] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if start_pos + input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("sequence exceeds configured max_seq_len")
        if kv_cache is not None and len(kv_cache) != self.config.n_layers:
            raise ValueError("kv_cache must contain one dictionary per layer")

        x = self.token_embedding(input_ids)
        for index, block in enumerate(self.blocks):
            cache = kv_cache[index] if kv_cache is not None else None
            x = block(x, start_pos=start_pos, kv_cache=cache)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            if labels.size(1) < 2:
                raise ValueError("at least two tokens are required to compute causal LM loss")
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int, temperature: float = 0.0) -> Tensor:
        self.eval()
        generated = input_ids
        cache = [dict() for _ in range(self.config.n_layers)]
        logits, _ = self(generated, kv_cache=cache)
        for _ in range(max_new_tokens):
            if generated.size(1) >= self.config.max_seq_len:
                break
            next_logits = logits[:, -1]
            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            logits, _ = self(next_token, start_pos=generated.size(1) - 1, kv_cache=cache)
        return generated


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
