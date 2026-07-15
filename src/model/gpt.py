"""Small decoder-only GPT model built from random initialization."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from typing import cast

from model.config import GPTConfig


class TransformerBlock(nn.Module):
    """Pre-layer-normalized self-attention plus a four-times-wide MLP."""

    def __init__(self, config: GPTConfig) -> None:
        """Build one decoder block from the shared model dimensions."""
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.embedding_dim)
        self.attention = nn.MultiheadAttention(
            config.embedding_dim, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(config.embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim),
            nn.GELU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden: Tensor, causal_mask: Tensor) -> Tensor:
        """Apply causal attention and an MLP while preserving residual paths."""
        normalized = self.norm_attention(hidden)
        attended, _ = self.attention(normalized, normalized, normalized, attn_mask=causal_mask)
        hidden = hidden + attended
        return hidden + cast(Tensor, self.mlp(self.norm_mlp(hidden)))


class GPTModel(nn.Module):
    """Small decoder-only language model initialized from scratch."""

    def __init__(self, config: GPTConfig) -> None:
        """Create embeddings, decoder blocks, and a tied language-model head."""
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        # Weight tying reduces parameters and makes input/output token
        # representations share the same learned vocabulary geometry.
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Apply the small normal initialization used by this GPT variant."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Return logits and optional next-token cross-entropy loss.

        ``input_ids`` and ``targets`` already represent shifted windows from
        the packer.  Keeping the shift in the dataset boundary avoids copying
        or shifting large batches inside every model call.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, time)")
        _, time = input_ids.shape
        if time > self.config.context_length:
            raise ValueError("input sequence exceeds model context length")
        positions = torch.arange(time, device=input_ids.device)
        hidden = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        # True entries are masked by MultiheadAttention, so the upper triangle
        # prevents each position from seeing tokens from its future.
        causal_mask = torch.triu(
            torch.ones((time, time), dtype=torch.bool, device=input_ids.device), diagonal=1
        )
        for block in self.blocks:
            hidden = block(hidden, causal_mask)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            # Flatten batch/time only for the loss API; logits retain their
            # structured shape for generation and debugging.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        """Greedily append tokens while limiting attention to model context."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        for _ in range(max_new_tokens):
            # Cropping keeps generation valid after the prompt exceeds the
            # fixed positional-embedding table.
            context = input_ids[:, -self.config.context_length :]
            logits, _ = self(context)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids
