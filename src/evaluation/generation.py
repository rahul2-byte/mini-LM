"""Tokenizer-backed deterministic text generation."""

from typing import cast

import torch

from model.gpt import GPTModel


def generate_text(
    model: GPTModel, tokenizer: object, prompt: str, max_new_tokens: int, seed: int
) -> str:
    """Generate a deterministic greedy sample from a tokenizer-backed model."""
    # The seed makes smoke reports comparable; generation itself is greedy, but
    # setting it also protects future sampling changes from hidden randomness.
    torch.manual_seed(seed)
    encoding = tokenizer.encode(prompt)  # type: ignore[attr-defined]
    input_ids = torch.tensor(
        [encoding.ids], dtype=torch.long, device=next(model.parameters()).device
    )
    generated = model.generate(input_ids, max_new_tokens=max_new_tokens)
    return cast(str, tokenizer.decode(generated[0].tolist()))  # type: ignore[attr-defined]
