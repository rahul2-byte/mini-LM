"""Streaming fixed-length token packing and memory-mapped loading."""

import json
import os
import tempfile
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PackMetadata:
    """Sidecar metadata needed to validate and memory-map packed tokens."""

    sequence_length: int
    vocab_size: int
    num_blocks: int
    num_tokens: int
    dtype: str = "<u4"


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    """Write a metadata sidecar atomically so data and metadata stay paired."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.flush()
        # Ensure the JSON reaches the filesystem before exposing its final
        # filename; this matters if the process is interrupted immediately.
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def pack_token_ids(
    token_ids: Iterable[int],
    output_path: str | Path,
    metadata_path: str | Path,
    sequence_length: int,
    vocab_size: int,
) -> PackMetadata:
    """Pack a token iterator into fixed windows without retaining all tokens.

    Each stored row contains ``sequence_length + 1`` IDs.  The dataset later
    returns the first ``N`` as inputs and the last ``N`` as shifted targets.
    Overlapping by one token preserves the autoregressive prediction boundary.
    """
    if sequence_length <= 0 or vocab_size <= 0:
        raise ValueError("sequence_length and vocab_size must be positive")
    output = Path(output_path)
    metadata_file = Path(metadata_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    blocks = 0
    input_tokens = 0
    # A deque keeps only the small rolling window rather than the entire corpus.
    buffer: deque[int] = deque()
    with output.open("wb") as handle:
        for raw_token in token_ids:
            token = int(raw_token)
            if token < 0 or token >= vocab_size:
                raise ValueError(f"token {token} is outside vocabulary size {vocab_size}")
            buffer.append(token)
            while len(buffer) >= sequence_length + 1:
                window = np.asarray(list(buffer)[: sequence_length + 1], dtype="<u4")
                window.tofile(handle)
                blocks += 1
                input_tokens += sequence_length
                # Advance by N, leaving one token available to overlap the next
                # window and preserve continuity across packed blocks.
                for _ in range(sequence_length):
                    buffer.popleft()
    if blocks == 0:
        output.unlink(missing_ok=True)
        raise ValueError("token stream does not contain a complete training block")
    metadata = PackMetadata(sequence_length, vocab_size, blocks, input_tokens)
    _atomic_json_write(metadata_file, asdict(metadata))
    return metadata


def load_pack_metadata(metadata_path: str | Path) -> PackMetadata:
    """Load and validate the sidecar before opening a memory map."""
    path = Path(metadata_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        metadata = PackMetadata(**raw)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid pack metadata: {path}") from exc
    if metadata.sequence_length <= 0 or metadata.vocab_size <= 0 or metadata.num_blocks <= 0:
        raise ValueError(f"invalid pack metadata values: {path}")
    if metadata.num_tokens != metadata.num_blocks * metadata.sequence_length:
        raise ValueError(f"invalid token count in pack metadata: {path}")
    return metadata


class PackedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Memory-mapped packed data exposing shifted language-model examples."""

    def __init__(self, data_path: str | Path, metadata_path: str | Path) -> None:
        """Open packed uint32 rows without copying the complete dataset to RAM."""
        self.data_path = Path(data_path)
        self.metadata = load_pack_metadata(metadata_path)
        expected_items = self.metadata.num_blocks * (self.metadata.sequence_length + 1)
        if not self.data_path.is_file():
            raise FileNotFoundError(self.data_path)
        byte_size = self.data_path.stat().st_size
        expected_bytes = expected_items * np.dtype("<u4").itemsize
        if byte_size != expected_bytes:
            raise ValueError(f"packed data does not match metadata: {self.data_path}")
        self._mapped = np.memmap(
            self.data_path,
            mode="r",
            dtype="<u4",
            shape=(self.metadata.num_blocks, self.metadata.sequence_length + 1),
        )

    def __len__(self) -> int:
        """Return the number of fixed-length training examples."""
        return self.metadata.num_blocks

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return input/target tensors formed by a one-token shift."""
        row = self._mapped[index]
        values = np.asarray(row, dtype=np.int64)
        return torch.from_numpy(values[:-1]), torch.from_numpy(values[1:])
