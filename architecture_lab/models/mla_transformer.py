"""Simplified Multi-Latent Attention (MLA) decoder-only Transformer.

This is a *simplified* version of DeepSeek-V2-style MLA intended for small-scale
training comparison only (not production). The simplification:

- KV is jointly compressed to a single `latent_dim` vector per token
  (W_DKV: d_model -> latent_dim), then up-projected to per-head K and V
  (W_UK, W_UV: latent_dim -> n_heads * head_dim).
- Q is *not* compressed (W_Q: d_model -> n_heads * head_dim directly) for
  simplicity. In real MLA Q is also compressed via W_DQ/W_UQ.
- RoPE is applied to the reconstructed K (after up-projection), which
  requires re-applying RoPE to the full cached K each forward pass during
  incremental decode. For training (no incremental decode) this is moot.
- KV cache stores the compressed latent (B, L, latent_dim) instead of
  full K+V (B, L, 2 * n_heads * head_dim), reducing cache size by
  approximately 2*n_heads*head_dim / latent_dim.

Real DeepSeek-V2 MLA additionally uses decoupled RoPE (a separate RoPE-key
vector W_KR independent of the compressed latent) so that the cached latent
itself carries positional info without needing to re-materialize full K. We
omit that here for clarity; the latent_dim=64 / head_dim=32 choice keeps
the cache footprint comparable to MQA while making the architectural
trade-offs explicit in the experiment doc.
"""

from __future__ import annotations

from dataclasses import dataclass
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
class MLAConfig(TransformerConfig):
    """TransformerConfig + latent_dim for MLA."""

    latent_dim: int = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")


class MLACausalSelfAttention(nn.Module):
    """Multi-Latent Attention (simplified).

    Forward pass:
        c_kv = W_DKV(x)                # (B, L, latent_dim)
        k = W_UK(c_kv) reshape -> heads # (B, n_heads, L, head_dim)
        v = W_UV(c_kv) reshape -> heads # (B, n_heads, L, head_dim)
        q = W_Q(x) reshape -> heads     # (B, n_heads, L, head_dim)
        q, k = RoPE(q, k)
        out = SDPA(q, k, v)

    Cache stores compressed latent only (W_UK / W_UV are applied on each
    forward to reconstruct full K, V from cached latent).
    """

    def __init__(self, config: MLAConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.latent_dim = config.latent_dim
        # joint KV down-projection
        self.W_DKV = nn.Linear(config.d_model, config.latent_dim, bias=False)
        # K, V up-projections from latent
        self.W_UK = nn.Linear(config.latent_dim, config.d_model, bias=False)
        self.W_UV = nn.Linear(config.latent_dim, config.d_model, bias=False)
        # Q projection (no compression)
        self.W_Q = nn.Linear(config.d_model, config.d_model, bias=False)
        # output projection
        self.W_O = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_base)
        self.dropout = config.dropout

    def _reconstruct_kv(self, c_kv: Tensor) -> tuple[Tensor, Tensor]:
        """Up-project cached compressed latent back to per-head K, V."""
        batch, seq_len, _ = c_kv.shape
        k = self.W_UK(c_kv).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_UV(c_kv).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        return k, v

    def forward(
        self,
        x: Tensor,
        start_pos: int = 0,
        kv_cache: Optional[dict[str, Tensor]] = None,
    ) -> Tensor:
        batch, seq_len, d_model = x.shape
        # 1) compress current-step input to latent
        c_kv = self.W_DKV(x)
        # 2) cache management: append to previous latent if any
        if kv_cache is not None:
            previous_c = kv_cache.get("c_kv")
            if previous_c is not None:
                if previous_c.size(1) != start_pos:
                    raise ValueError("start_pos must equal the cached sequence length")
                c_kv = torch.cat((previous_c, c_kv), dim=1)
            kv_cache["c_kv"] = c_kv.detach()
        # 3) reconstruct K, V from full cached latent
        k, v = self._reconstruct_kv(c_kv)
        # 4) Q projection + RoPE on both Q and full K
        q = self.W_Q(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, start_pos=start_pos)
        # 5) causal SDPA
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
        return self.W_O(output)


class MLATransformerBlock(nn.Module):
    def __init__(self, config: MLAConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = MLACausalSelfAttention(config)
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


class MLATransformer(nn.Module):
    """Decoder-only Transformer with simplified Multi-Latent Attention.

    Architecture differs from DenseTransformer in the attention module only.
    Layer norm, FFN (SwiGLU), embedding tying, and loss computation are
    identical to the MHA baseline.
    """

    def __init__(self, config: MLAConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(MLATransformerBlock(config) for _ in range(config.n_layers))
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
