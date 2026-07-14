"""Atomic checkpoint save/load with reproducibility state."""

import os
import random
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Optimizer


class CheckpointManager:
    """Persist model state and reproducibility state with bounded retention."""

    def __init__(self, directory: str | Path, keep_last: int = 3) -> None:
        """Create the checkpoint directory and configure retention count."""
        if keep_last <= 0:
            raise ValueError("keep_last must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    @staticmethod
    def _config_dict(config: object) -> dict[str, Any]:
        """Convert a dataclass config to the serializable compatibility record."""
        if not is_dataclass(config):
            raise TypeError("checkpoint config must be a dataclass")
        return asdict(config)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        step: int,
        config: object,
        tokenizer_path: str | Path,
    ) -> Path:
        """Atomically save training state and return its final path.

        Writing to a temporary sibling and using ``os.replace`` means readers
        see either the old complete checkpoint or the new complete checkpoint,
        never a partially serialized PyTorch file.
        """
        payload: dict[str, Any] = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
            "config": self._config_dict(config),
            "tokenizer_path": str(tokenizer_path),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            # CUDA generators are separate from Python, NumPy, and CPU Torch
            # generators; all are needed for a faithful interrupted resume.
            payload["cuda_rng"] = torch.cuda.get_rng_state_all()
        destination = self.directory / f"checkpoint-{step:08d}.pt"
        with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        # Retention is applied only after the new checkpoint is durable.
        checkpoints = sorted(self.directory.glob("checkpoint-*.pt"))
        for old in checkpoints[: -self.keep_last]:
            old.unlink()
        return destination

    def load(
        self,
        path: str | Path,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        expected_config: object | None = None,
    ) -> dict[str, Any]:
        """Load and validate a checkpoint, restoring optimizer and RNG state."""
        checkpoint_path = Path(path)
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ValueError(f"could not load checkpoint: {checkpoint_path}") from exc
        if expected_config is not None and payload.get("config") != self._config_dict(
            expected_config
        ):
            raise ValueError("checkpoint config is incompatible with the requested config")
        try:
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            if scheduler is not None and payload.get("scheduler") is not None:
                scheduler.load_state_dict(payload["scheduler"])
            random.setstate(payload["python_rng"])
            np.random.set_state(payload["numpy_rng"])
            torch.set_rng_state(payload["torch_rng"])
            if torch.cuda.is_available() and "cuda_rng" in payload:
                torch.cuda.set_rng_state_all(payload["cuda_rng"])
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"checkpoint state is invalid: {checkpoint_path}") from exc
        return payload
