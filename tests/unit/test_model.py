import pytest
import torch

from model.config import GPTConfig
from model.gpt import GPTModel


def test_model_config_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="embedding_dim"):
        GPTConfig(vocab_size=16, context_length=8, n_layers=1, n_heads=3, embedding_dim=10)


def test_model_returns_logits_and_scalar_loss() -> None:
    model = GPTModel(
        GPTConfig(vocab_size=16, context_length=8, n_layers=1, n_heads=2, embedding_dim=8)
    )
    inputs = torch.randint(0, 16, (2, 5))
    logits, loss = model(inputs, inputs)
    assert logits.shape == (2, 5, 16)
    assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)


def test_model_rejects_context_overflow() -> None:
    model = GPTModel(
        GPTConfig(vocab_size=16, context_length=4, n_layers=1, n_heads=2, embedding_dim=8)
    )
    with pytest.raises(ValueError, match="context"):
        model(torch.ones((1, 5), dtype=torch.long))


def test_greedy_generation_extends_prompt() -> None:
    model = GPTModel(
        GPTConfig(vocab_size=16, context_length=8, n_layers=1, n_heads=2, embedding_dim=8)
    )
    generated = model.generate(torch.tensor([[1, 2]]), max_new_tokens=3)
    assert generated.shape == (1, 5)
