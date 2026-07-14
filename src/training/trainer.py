"""Small deterministic PyTorch training loop."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation.perplexity import evaluate
from model.config import GPTConfig
from training.checkpoint import CheckpointManager
from training.scheduler import create_warmup_decay_scheduler


@dataclass(frozen=True)
class TrainerConfig:
    """Training-loop controls kept separate from the model architecture."""

    max_steps: int = 3
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 0
    gradient_clip_norm: float = 1.0
    precision: str = "fp32"
    eval_every_steps: int = 0
    checkpoint_every_steps: int = 0

    def __post_init__(self) -> None:
        """Validate optimizer, precision, and loop values before training."""
        if self.max_steps <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("max_steps and gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("optimizer values must be positive, except weight_decay")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")


class Trainer:
    """Single-process PyTorch trainer with accumulation and resumable state."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        model_config: GPTConfig,
        checkpoint_manager: CheckpointManager,
        tokenizer_path: str | Path,
        device: torch.device,
    ) -> None:
        """Initialize model, optimizer, scheduler, and progress counters."""
        self.model = model.to(device)
        self.config = config
        self.model_config = model_config
        self.checkpoint_manager = checkpoint_manager
        self.tokenizer_path = str(tokenizer_path)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = create_warmup_decay_scheduler(
            self.optimizer, config.warmup_steps, config.max_steps
        )
        self.global_step = 0
        self.tokens_processed = 0

    def _autocast_context(self) -> Any:
        """Return a safe autocast context for the selected device/precision."""
        if self.device.type != "cuda" or self.config.precision == "fp32":
            return torch.autocast(device_type=self.device.type, enabled=False)
        dtype = torch.float16 if self.config.precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    @staticmethod
    def _next_batch(loader: DataLoader, iterator: Iterator[Any]) -> tuple[Any, Iterator[Any]]:
        """Fetch a batch and restart the iterator when an epoch is exhausted."""
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        return batch, iterator

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> dict[str, float]:
        """Train for configured optimizer steps and optionally evaluate/checkpoint.

        One optimizer step may consume multiple micro-batches.  Dividing each
        micro-batch loss before backpropagation keeps the accumulated gradient
        scale independent of the accumulation setting.
        """
        self.model.train()
        iterator = iter(train_loader)
        latest_loss = float("nan")
        while self.global_step < self.config.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            accumulation_loss = 0.0
            for _ in range(self.config.gradient_accumulation_steps):
                batch, iterator = self._next_batch(train_loader, iterator)
                # Move only the current micro-batch to the device; the dataset
                # remains disk-backed and bounded by the DataLoader.
                inputs, targets = (value.to(self.device) for value in batch)
                try:
                    with self._autocast_context():
                        _, loss = self.model(inputs, targets)
                    if loss is None:
                        raise RuntimeError("model returned no loss during training")
                    (loss / self.config.gradient_accumulation_steps).backward()
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise RuntimeError(
                            f"CUDA out of memory at step {self.global_step}; reduce batch, "
                            "sequence length, or accumulation settings"
                        ) from exc
                    raise
                accumulation_loss += float(loss.detach())
            # Clipping is applied after accumulation, which bounds the actual
            # optimizer update rather than each partial gradient.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1
            latest_loss = accumulation_loss / self.config.gradient_accumulation_steps
            self.tokens_processed += inputs.numel() * self.config.gradient_accumulation_steps
            if val_loader is not None and self.config.eval_every_steps:
                if self.global_step % self.config.eval_every_steps == 0:
                    self.last_eval = evaluate(self.model, val_loader, self.device)
            if (
                self.config.checkpoint_every_steps
                and self.global_step % self.config.checkpoint_every_steps == 0
            ):
                self.checkpoint_manager.save(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.global_step,
                    self.model_config,
                    self.tokenizer_path,
                )
        metrics = {"train_loss": latest_loss, "tokens_processed": float(self.tokens_processed)}
        if val_loader is not None:
            metrics.update(
                getattr(self, "last_eval", evaluate(self.model, val_loader, self.device))
            )
        return metrics

    def resume(self, path: str | Path) -> int:
        """Restore trainer state and return the resumed global step."""
        state = self.checkpoint_manager.load(
            path, self.model, self.optimizer, self.scheduler, expected_config=self.model_config
        )
        self.global_step = int(state["step"])
        self.tokens_processed = int(state.get("tokens_processed", 0))
        return self.global_step
