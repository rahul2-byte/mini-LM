"""Streaming SHA-256 helpers."""

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory.

    Raw shards can be hundreds of megabytes, so hashing must read bounded
    chunks.  The digest is used as an identity and integrity check for both
    local files and objects uploaded to MinIO.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        # ``iter(callable, sentinel)`` keeps the read loop small while still
        # stopping cleanly at EOF and never retaining more than one chunk.
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA-256 digest of UTF-8 encoded text.

    UTF-8 is explicit here so the same text produces the same digest on every
    operating system and regardless of the process locale.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
