"""Grouped-Query Attention (GQA) decoder-only Transformer.

Reuses `RMSNorm`, `RotaryEmbedding`, `SwiGLU` from dense_transformer to avoid
duplication. The only architectural change versus the MHA baseline is that
K/V projections use `num_kv_heads` heads instead of `n_heads`; K/V are then
broadcast (via `repeat_interleave`) to `n_heads` for SDPA.

Num_kv_heads must satisfy:
    n_heads % num_kv_heads == 0
When num_kv_heads == n_heads the model is equivalent to MHA.
When num_kv_heads == 1 the model is Multi-Query Attention (MQA).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from architecture_lab.models.dense_transformer import (
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    TransformerConfig,
    count_parameters,
)


@dataclass(frozen=True)
class GQAConfig(TransformerConfig):
    """TransformerConfig + num_kv_heads for GQA."""

    num_kv_heads: int = field(default=0)

    def __post_init__(self) -> None:
        super().__post_init__()
        # default num_kv_heads to n_heads (MHA behavior) when not set
        if self.num_kv_heads == 0:
            object.__setattr__(self, "num_kv_heads", self.n_heads)
        if self.num_kv_heads < 1:
            raise ValueError("num_kv_heads must be positive")
        if self.n_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )


class GQACausalSelfAttention(nn.Module):
    """Multi-head query + grouped KV attention.

    K/V projections produce `num_kv_heads * head_dim` channels; we
    `repeat_interleave` each KV head to cover `n_heads // num_kv_heads`
    query heads (GQA grouping).
    """

    def __init__(self, config: GQAConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_base)
        self.dropout = config.dropout
        # how many Q heads each KV head serves
        self.q_per_kv = self.n_heads // self.num_kv_heads

    def forward(
        self,
        x: Tensor,
        start_pos: int = 0,
        kv_cache: Optional[dict[str, Tensor]] = None,
    ) -> Tensor:
        batch, seq_len, d_model = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
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

        # broadcast KV heads to match Q heads (GQA grouping)
        if self.q_per_kv > 1:
            k = k.repeat_interleave(self.q_per_kv, dim=1)
            v = v.repeat_interleave(self.q_per_kv, dim=1)

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


class GQATransformerBlock(nn.Module):
    def __init__(self, config: GQAConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = GQACausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff)

    def forward(
        self,
        x: Tensor,
        start_pos: int = 0,
        kv_cache: Optional[dict[str, Tensor]] = None,
    ) -> Tensor:
        x = x + self.attn(self.attn_norm(x), start_pos=start_pos, kv_cache=kv_cache)
        return x + self.ffn(self.ffn_norm(x))


class GQATransformer(nn.Module):
    """Decoder-only Transformer with Grouped-Query Attention.

    Identical to DenseTransformer except: (1) attention uses `num_kv_heads`
    KV heads instead of `n_heads`; (2) KV cache stores only `num_kv_heads`
    heads (smaller than MHA by a factor of `q_per_kv`).
    """

    def __init__(self, config: GQAConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(GQATransformerBlock(config) for _ in range(config.n_layers))
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
    def generate(
        self, input_ids: Tensor, max_new_tokens: int, temperature: float = 0.0
    ) -> Tensor:
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
