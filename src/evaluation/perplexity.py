"""Validation loss and perplexity evaluation."""

import math

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Compute token-weighted validation loss and its exponential perplexity."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_items = 0
    for inputs, targets in data_loader:
        # Count individual target tokens, not batches, so a short final batch
        # cannot distort the mean loss.
        inputs = inputs.to(device)
        targets = targets.to(device)
        _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("model did not return a loss during evaluation")
        items = targets.numel()
        total_loss += float(loss.detach()) * items
        total_items += items
    if total_items == 0:
        raise ValueError("cannot evaluate an empty dataset")
    mean_loss = total_loss / total_items
    if was_training:
        model.train()
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss)}
