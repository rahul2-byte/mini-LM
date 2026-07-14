from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from model.config import GPTConfig
from model.gpt import GPTModel
from training.checkpoint import CheckpointManager
from training.trainer import Trainer, TrainerConfig


def test_trainer_updates_parameters_and_advances_steps(tmp_path: Path) -> None:
    model_config = GPTConfig(
        vocab_size=16, context_length=4, n_layers=1, n_heads=2, embedding_dim=8
    )
    model = GPTModel(model_config)
    inputs = torch.randint(0, 16, (4, 4))
    loader = DataLoader(TensorDataset(inputs, inputs), batch_size=2)
    before = next(model.parameters()).detach().clone()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=2, learning_rate=1e-3, precision="fp32"),
        model_config=model_config,
        checkpoint_manager=CheckpointManager(tmp_path),
        tokenizer_path="tokenizer.json",
        device=torch.device("cpu"),
    )
    metrics = trainer.train(loader)
    assert trainer.global_step == 2
    assert torch.isfinite(torch.tensor(metrics["train_loss"]))
    assert not torch.equal(next(model.parameters()), before)
