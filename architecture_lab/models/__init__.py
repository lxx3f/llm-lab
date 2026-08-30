"""Architecture lab models: Dense MHA, MoE Top-1, GQA, simplified MLA."""

from architecture_lab.models.dense_transformer import (
    CausalSelfAttention,
    DenseTransformer,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    TransformerBlock,
    TransformerConfig,
    count_parameters,
)
from architecture_lab.models.gqa_transformer import (
    GQAConfig,
    GQACausalSelfAttention,
    GQATransformer,
    GQATransformerBlock,
)
from architecture_lab.models.mla_transformer import (
    MLAConfig,
    MLACausalSelfAttention,
    MLATransformer,
    MLATransformerBlock,
)
from architecture_lab.models.moe_transformer import (
    MoEConfig,
    MoETransformer,
    MoETransformerBlock,
    SwiGLUExpert,
    Top1MoE,
)

__all__ = [
    # Dense (MHA) baseline
    "DenseTransformer",
    "TransformerConfig",
    "TransformerBlock",
    "CausalSelfAttention",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "count_parameters",
    # MoE Top-1
    "MoEConfig",
    "MoETransformer",
    "MoETransformerBlock",
    "Top1MoE",
    "SwiGLUExpert",
    # GQA
    "GQAConfig",
    "GQATransformer",
    "GQATransformerBlock",
    "GQACausalSelfAttention",
    # Simplified MLA
    "MLAConfig",
    "MLATransformer",
    "MLATransformerBlock",
    "MLACausalSelfAttention",
]
