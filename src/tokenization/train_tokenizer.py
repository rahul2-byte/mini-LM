"""Train a BPE tokenizer from local text."""

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def train_bpe(
    documents: Iterable[str],
    output_path: str | Path,
    vocab_size: int = 256,
    min_frequency: int = 1,
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """Train and save a BPE tokenizer from an iterable of clean documents.

    The iterable is intentionally accepted instead of a list so production
    callers can feed records from a streaming cleaner.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens or ["<pad>", "<unk>", "<bos>", "<eos>"],
    )
    tokenizer.train_from_iterator(documents, trainer=trainer)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(destination))
    return tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load a tokenizer artifact and fail clearly when it is missing."""
    destination = Path(path)
    if not destination.is_file():
        raise FileNotFoundError(destination)
    return Tokenizer.from_file(str(destination))
