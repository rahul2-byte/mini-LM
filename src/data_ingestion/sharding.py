"""Bounded, record-aligned local shard construction."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO, Iterable

from data_ingestion.adapters import SourceRecord


@dataclass(frozen=True)
class LocalShard:
    """A fully written local shard waiting for object-store verification."""

    sequence: int
    path: Path
    size_bytes: int
    checksum: str
    record_count: int
    checkpoint: dict[str, int]


class ShardBuilder:
    """Build bounded shards without splitting individual source records."""

    def __init__(
        self, staging_directory: str | Path, target_size_bytes: int, maximum_size_bytes: int
    ) -> None:
        """Configure a builder with byte thresholds supplied by YAML."""
        if target_size_bytes <= 0 or maximum_size_bytes < target_size_bytes:
            raise ValueError("shard sizes must satisfy 0 < target <= maximum")
        self.staging_directory = Path(staging_directory)
        self.target_size_bytes = target_size_bytes
        self.maximum_size_bytes = maximum_size_bytes

    def build(
        self, records: Iterable[SourceRecord], start_sequence: int = 1
    ) -> Iterable[LocalShard]:
        """Stream records into `.partial` files and yield verified-ready shards.

        A shard is yielded only after it is closed and atomically renamed to
        `.ready`.  That suffix is the recovery signal used at startup: a ready
        file can be uploaded again, while a partial file must not be counted as
        complete.
        """
        self.staging_directory.mkdir(parents=True, exist_ok=True)
        sequence = start_sequence
        handle: BinaryIO | None = None
        digest = sha256()
        size = 0
        count = 0
        checkpoint: dict[str, int] = {}
        path: Path | None = None
        try:
            for record in records:
                payload = record.payload + b"\n"
                if len(payload) > self.maximum_size_bytes and count == 0:
                    raise ValueError("single record exceeds maximum shard size")
                if handle is None:
                    # Sequence numbers are stable across retries, so the same
                    # logical shard receives the same MinIO object key.
                    path = self.staging_directory / f"shard-{sequence:06d}.partial"
                    handle = path.open("wb")
                    digest = sha256()
                    size = 0
                    count = 0
                handle.write(payload)
                digest.update(payload)
                size += len(payload)
                count += 1
                checkpoint = record.checkpoint
                if size >= self.target_size_bytes:
                    # The threshold is checked after a complete record; this
                    # may make a shard larger than target but never splits data.
                    yield self._finalize(handle, path, sequence, size, digest, count, checkpoint)
                    handle = None
                    path = None
                    sequence += 1
            if handle is not None and path is not None:
                yield self._finalize(handle, path, sequence, size, digest, count, checkpoint)
        finally:
            if handle is not None:
                handle.close()

    def _finalize(
        self,
        handle: BinaryIO,
        path: Path,
        sequence: int,
        size: int,
        digest: object,
        count: int,
        checkpoint: dict[str, int],
    ) -> LocalShard:
        """Close and rename a partial file before exposing it to upload code."""
        handle.flush()
        handle.close()
        ready_path = path.with_suffix(".ready")
        # ``replace`` is atomic on the same filesystem and makes interruption
        # before this point distinguishable from a completed local shard.
        path.replace(ready_path)
        return LocalShard(sequence, ready_path, size, digest.hexdigest(), count, checkpoint)  # type: ignore[attr-defined]
