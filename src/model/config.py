"""Validated GPT model configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    """Shape and regularization settings for a randomly initialized GPT model."""

    vocab_size: int
    context_length: int
    n_layers: int
    n_heads: int
    embedding_dim: int
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Reject invalid dimensions before PyTorch allocates model weights."""
        for name in ("vocab_size", "context_length", "n_layers", "n_heads", "embedding_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.embedding_dim % self.n_heads != 0:
            raise ValueError("embedding_dim must be divisible by n_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
