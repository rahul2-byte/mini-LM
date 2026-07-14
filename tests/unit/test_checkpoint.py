from pathlib import Path

import pytest
import torch

from model.config import GPTConfig
from model.gpt import GPTModel
from training.checkpoint import CheckpointManager


def test_checkpoint_restores_model_optimizer_and_step(tmp_path: Path) -> None:
    config = GPTConfig(vocab_size=8, context_length=4, n_layers=1, n_heads=2, embedding_dim=8)
    model = GPTModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    original = next(model.parameters()).detach().clone()
    manager = CheckpointManager(tmp_path)
    path = manager.save(
        model, optimizer, None, step=7, config=config, tokenizer_path="tokenizer.json"
    )
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    state = manager.load(path, model, optimizer, None, expected_config=config)
    assert state["step"] == 7
    assert torch.equal(next(model.parameters()), original)


def test_checkpoint_rejects_incompatible_config(tmp_path: Path) -> None:
    config = GPTConfig(vocab_size=8, context_length=4, n_layers=1, n_heads=2, embedding_dim=8)
    other = GPTConfig(vocab_size=9, context_length=4, n_layers=1, n_heads=2, embedding_dim=8)
    model = GPTModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = CheckpointManager(tmp_path).save(model, optimizer, None, 1, config, "tokenizer.json")
    with pytest.raises(ValueError, match="config"):
        CheckpointManager(tmp_path).load(path, model, optimizer, None, expected_config=other)
