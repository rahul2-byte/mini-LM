"""Streaming tokenizer encoding."""

from collections.abc import Iterable, Iterator
from pathlib import Path

from tokenizers import Tokenizer


def encode_documents(documents: Iterable[str], tokenizer: Tokenizer) -> Iterator[list[int]]:
    """Yield token IDs one document at a time to keep encoding streaming."""
    for document in documents:
        yield tokenizer.encode(document).ids


def write_token_ids(encoded: Iterable[list[int]], output_path: str | Path) -> int:
    """Write newline-delimited token IDs and return the total token count."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for ids in encoded:
            handle.write(" ".join(map(str, ids)) + "\n")
            count += len(ids)
    return count
