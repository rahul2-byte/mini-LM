"""Streaming text cleaning for blank-line-separated documents."""

import re
from collections.abc import Iterator
from pathlib import Path

from common.errors import EmptyDatasetError


def iter_documents(path: str | Path, min_chars: int = 20) -> Iterator[str]:
    """Yield valid documents from a blank-line-separated UTF-8 text file.

    The iterator deliberately processes one document at a time.  That keeps
    memory bounded for the full corpus and preserves a natural record boundary
    for exact deduplication and later sharding.
    """
    buffer: list[str] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                buffer.append(stripped)
            elif buffer:
                # A blank line closes the current record.  Joining lines here
                # removes source-specific wrapping without changing wording.
                document = normalize(" ".join(buffer))
                if len(document) >= min_chars:
                    yield document
                buffer = []
        if buffer:
            # Files do not always end with a blank line, so flush the final
            # buffered document explicitly at EOF.
            document = normalize(" ".join(buffer))
            if len(document) >= min_chars:
                yield document


def normalize(text: str) -> str:
    """Collapse runs of whitespace and trim document edges."""
    return re.sub(r"\s+", " ", text).strip()


def require_documents(path: str | Path, min_chars: int = 20) -> Iterator[str]:
    """Yield cleaned documents but fail explicitly when none are usable."""
    documents = iter_documents(path, min_chars)
    # Checking the first item up front turns an empty input into a useful
    # pipeline error instead of silently producing an empty downstream file.
    first = next(documents, None)
    if first is None:
        raise EmptyDatasetError(f"No valid documents found in {path}")
    yield first
    yield from documents
