"""
Multimodal projection layers to bridge vision features → LLM embedding space.
"""
import torch
import torch.nn as nn
import math


class LinearProjection(nn.Module):
    """Simple linear projection: vision_dim → llm_dim."""

    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (B, vision_dim) pooled features.
        Returns:
            (B, 1, llm_dim) projected tokens for the LLM.
        """
        out = self.proj(vision_features)
        return out.unsqueeze(1)  # add sequence dim


class QFormerProjection(nn.Module):
    """Q-Former–style projection: learnable query tokens attend to vision features.

    A lightweight cross-attention mechanism that produces num_query_tokens
    output embeddings from spatial vision features.
    """

    def __init__(self, vision_dim: int, llm_dim: int, num_query_tokens: int = 32,
                 num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, llm_dim) * 0.02)
        self.vision_proj = nn.Linear(vision_dim, llm_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=llm_dim,
            nhead=num_heads,
            dim_feedforward=llm_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (B, N, vision_dim) spatial feature tokens
                             or (B, vision_dim) pooled features.
        Returns:
            (B, num_query_tokens, llm_dim)
        """
        if vision_features.dim() == 2:
            vision_features = vision_features.unsqueeze(1)

        B = vision_features.size(0)
        memory = self.vision_proj(vision_features)     # (B, N, llm_dim)
        queries = self.query_tokens.expand(B, -1, -1)  # (B, Q, llm_dim)
        out = self.decoder(queries, memory)             # (B, Q, llm_dim)
        return self.norm(out)
