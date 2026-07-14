"""Learning-rate scheduling helpers."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def create_warmup_decay_scheduler(
    optimizer: Optimizer, warmup_steps: int, total_steps: int
) -> LambdaLR:
    """Create linear warmup followed by linear decay to zero."""
    if warmup_steps < 0 or total_steps <= 0:
        raise ValueError("warmup_steps must be non-negative and total_steps must be positive")

    def multiplier(step: int) -> float:
        """Return the learning-rate multiplier for one optimizer step."""
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        remaining = max(total_steps - warmup_steps, 1)
        return max(0.0, (total_steps - step) / remaining)

    return LambdaLR(optimizer, multiplier)
