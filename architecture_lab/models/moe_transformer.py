"""Top-1 routed Mixture-of-Experts decoder-only Transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from architecture_lab.models.dense_transformer import (
    CausalSelfAttention,
    RMSNorm,
    TransformerConfig,
    count_parameters,
)


@dataclass(frozen=True)
class MoEConfig:
    num_experts: int = 4
    capacity_factor: float = 1.0
    aux_loss_weight: float = 0.01

    def __post_init__(self) -> None:
        if self.num_experts < 1:
            raise ValueError("num_experts must be positive")
        if self.capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive")
        if self.aux_loss_weight < 0:
            raise ValueError("aux_loss_weight must be non-negative")


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Top1MoE(nn.Module):
    """A simple capacity-limited Top-1 MoE layer.

    Each token is sent to exactly one expert selected by the router. Tokens
    beyond an expert's capacity are dropped from the expert output; the
    surrounding Transformer residual path still preserves their input.
    """

    def __init__(self, d_model: int, d_ff: int, config: MoEConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.capacity_factor = config.capacity_factor
        self.router = nn.Linear(d_model, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLUExpert(d_model, d_ff) for _ in range(config.num_experts)
        )
        self.last_stats: dict[str, object] = {}

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, dict[str, object]]:
        if x.ndim != 3:
            raise ValueError("MoE input must have shape [batch, seq_len, d_model]")

        batch, seq_len, d_model = x.shape
        tokens = batch * seq_len
        flat_x = x.reshape(tokens, d_model)
        router_logits = self.router(flat_x)
        router_probs = F.softmax(router_logits, dim=-1)
        top1_weight, top1_index = router_probs.max(dim=-1)

        capacity = max(1, math.ceil(self.capacity_factor * tokens / self.num_experts))
        flat_output = torch.zeros_like(flat_x)
        assigned_counts = torch.bincount(
            top1_index, minlength=self.num_experts
        )
        kept_counts = torch.zeros(
            self.num_experts, device=x.device, dtype=torch.long
        )

        for expert_index, expert in enumerate(self.experts):
            token_indices = torch.nonzero(
                top1_index == expert_index, as_tuple=False
            ).flatten()
            kept_indices = token_indices[:capacity]
            if kept_indices.numel() == 0:
                continue
            expert_output = expert(flat_x.index_select(0, kept_indices))
            expert_output = expert_output * top1_weight.index_select(0, kept_indices).unsqueeze(-1)
            flat_output.index_copy_(0, kept_indices, expert_output)
            kept_counts[expert_index] = kept_indices.numel()

        # Switch-style load balancing loss: E * sum(mean(router_probs) *
        # mean(one-hot top-1 assignments)). It is differentiable through the
        # router probabilities and encourages balanced expert utilization.
        mean_probs = router_probs.mean(dim=0)
        assignment_fraction = F.one_hot(
            top1_index, num_classes=self.num_experts
        ).float().mean(dim=0)
        aux_loss = self.num_experts * torch.sum(mean_probs * assignment_fraction)

        dropped_tokens = int((assigned_counts - kept_counts).sum().item())
        stats: dict[str, object] = {
            "num_experts": self.num_experts,
            "capacity": capacity,
            "tokens": tokens,
            "assigned_tokens_per_expert": assigned_counts.detach().cpu().tolist(),
            "kept_tokens_per_expert": kept_counts.detach().cpu().tolist(),
            "dropped_tokens": dropped_tokens,
            "dropped_token_rate": dropped_tokens / tokens,
            "router_prob_mean": mean_probs.detach().cpu().tolist(),
        }
        self.last_stats = stats
        return flat_output.view(batch, seq_len, d_model), aux_loss, stats


class MoETransformerBlock(nn.Module):
    def __init__(self, transformer_config: TransformerConfig, moe_config: MoEConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(transformer_config.d_model)
        self.attn = CausalSelfAttention(transformer_config)
        self.ffn_norm = RMSNorm(transformer_config.d_model)
        self.moe = Top1MoE(transformer_config.d_model, transformer_config.d_ff, moe_config)

    def forward(
        self,
        x: Tensor,
        start_pos: int = 0,
        kv_cache: Optional[dict[str, Tensor]] = None,
    ) -> tuple[Tensor, Tensor, dict[str, object]]:
        x = x + self.attn(self.attn_norm(x), start_pos=start_pos, kv_cache=kv_cache)
        moe_output, aux_loss, stats = self.moe(self.ffn_norm(x))
        return x + moe_output, aux_loss, stats


class MoETransformer(nn.Module):
    """Decoder-only Transformer with one Top-1 MoE FFN per layer."""

    def __init__(self, transformer_config: TransformerConfig, moe_config: MoEConfig) -> None:
        super().__init__()
        self.config = transformer_config
        self.moe_config = moe_config
        self.token_embedding = nn.Embedding(
            transformer_config.vocab_size, transformer_config.d_model
        )
        self.blocks = nn.ModuleList(
            MoETransformerBlock(transformer_config, moe_config)
            for _ in range(transformer_config.n_layers)
        )
        self.final_norm = RMSNorm(transformer_config.d_model)
        self.lm_head = nn.Linear(
            transformer_config.d_model, transformer_config.vocab_size, bias=False
        )
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        start_pos: int = 0,
        kv_cache: Optional[list[dict[str, Tensor]]] = None,
    ) -> tuple[Tensor, Optional[Tensor], Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if start_pos + input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("sequence exceeds configured max_seq_len")
        if kv_cache is not None and len(kv_cache) != self.config.n_layers:
            raise ValueError("kv_cache must contain one dictionary per layer")

        x = self.token_embedding(input_ids)
        aux_losses: list[Tensor] = []
        stats: list[dict[str, object]] = []
        for index, block in enumerate(self.blocks):
            cache = kv_cache[index] if kv_cache is not None else None
            x, aux_loss, layer_stats = block(
                x, start_pos=start_pos, kv_cache=cache
            )
            aux_losses.append(aux_loss)
            stats.append(layer_stats)

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

        total_aux_loss = torch.stack(aux_losses).mean()
        self.last_routing_stats = stats
        return logits, loss, total_aux_loss

    def total_loss(self, lm_loss: Tensor, aux_loss: Tensor) -> Tensor:
        return lm_loss + self.moe_config.aux_loss_weight * aux_loss


def count_moe_parameters(model: nn.Module) -> int:
    return count_parameters(model)
