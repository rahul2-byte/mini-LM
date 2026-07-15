"""Source adapter contracts and a resumable local line adapter for smoke runs."""

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from collections.abc import Callable
from typing import Any, Iterator, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from data_ingestion.config import DataSourceConfig


@dataclass(frozen=True)
class SourceRecord:
    """One source payload plus the checkpoint valid immediately after it."""

    payload: bytes
    checkpoint: dict[str, Any]


class DataSourceAdapter(Protocol):
    """Contract separating source access from sharding and MinIO concerns."""

    def validate_configuration(self) -> None: ...

    def stream_records(
        self, checkpoint: dict[str, Any], max_bytes: int | None = None
    ) -> Iterator[SourceRecord]: ...


class LocalTextAdapter:
    """Read UTF-8 lines and resume from the next byte offset."""

    def __init__(self, path: str | Path) -> None:
        """Store the local path; actual I/O is deferred until streaming."""
        self.path = Path(path)

    def validate_configuration(self) -> None:
        """Fail early when the smoke source file is unavailable."""
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def stream_records(
        self, checkpoint: dict[str, Any], max_bytes: int | None = None
    ) -> Iterator[SourceRecord]:
        """Yield lines from a saved byte offset without reading the whole file."""
        self.validate_configuration()
        offset = int(checkpoint.get("byte_offset", 0))
        with self.path.open("rb") as handle:
            # Byte offsets are safe here because the file is read as bytes and
            # each record is newline-delimited; no text decoder state is kept.
            handle.seek(offset)
            emitted_bytes = 0
            while line := handle.readline():
                payload = line.rstrip(b"\r\n")
                stored_bytes = len(payload) + 1
                if max_bytes is not None and emitted_bytes + stored_bytes > max_bytes:
                    return
                emitted_bytes += stored_bytes
                yield SourceRecord(payload, {"byte_offset": handle.tell()})


class HttpLineAdapter:
    """Stream newline-delimited text and resume with HTTP byte ranges."""

    def __init__(self, url: str, timeout_seconds: float = 30.0) -> None:
        """Configure a bounded-timeout HTTP line reader."""
        self.url = url
        self.timeout_seconds = timeout_seconds

    def validate_configuration(self) -> None:
        """Require a URL scheme supported by the standard HTTP client."""
        if not self.url.startswith(("http://", "https://")):
            raise ValueError(f"HTTP adapter requires an HTTP(S) URL: {self.url}")

    def stream_records(
        self, checkpoint: dict[str, Any], max_bytes: int | None = None
    ) -> Iterator[SourceRecord]:
        """Stream newline records and resume with an HTTP Range request."""
        self.validate_configuration()
        offset = int(checkpoint.get("byte_offset", 0))
        # An initial request needs no Range header; resumed requests must ask
        # the server for bytes after the last checkpointed record.
        request = Request(self.url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            response = urlopen(request, timeout=self.timeout_seconds)
        except (HTTPError, URLError) as error:
            raise ConnectionError(f"failed to open source URL: {self.url}") from error
        with response:
            if offset and response.status != 206:
                raise RuntimeError("source does not support ranged resume")
            emitted_bytes = 0
            while line := response.readline():
                offset += len(line)
                payload = line.rstrip(b"\r\n")
                stored_bytes = len(payload) + 1
                if max_bytes is not None and emitted_bytes + stored_bytes > max_bytes:
                    return
                emitted_bytes += stored_bytes
                yield SourceRecord(payload, {"byte_offset": offset})


class HuggingFaceStreamingAdapter:
    """Stream text records from a Hugging Face dataset configuration.

    Native dataset state preserves the upstream shard and record position, so
    resume does not reread the dataset from its beginning.
    """

    def __init__(
        self,
        dataset_id: str,
        *,
        dataset_config: str | None = None,
        split: str = "train",
        text_field: str = "text",
        max_records: int | None = None,
        max_bytes: int | None = None,
        resume_progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Configure a lazy dataset iterator with local quota limits."""
        self.dataset_id = dataset_id
        self.dataset_config = dataset_config
        self.split = split
        self.text_field = text_field
        self.max_records = max_records
        self.max_bytes = max_bytes
        self.resume_progress_callback = resume_progress_callback

    def validate_configuration(self) -> None:
        """Validate settings before importing the optional datasets package."""
        if not self.dataset_id or not self.split or not self.text_field:
            raise ValueError("dataset_id, split, and text_field are required")
        if self.max_records is not None and self.max_records <= 0:
            raise ValueError("max_records must be positive")
        if self.max_bytes is not None and self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")

    def stream_records(
        self, checkpoint: dict[str, Any], max_bytes: int | None = None
    ) -> Iterator[SourceRecord]:
        """Yield JSONL records and resume from Hugging Face iterator state."""
        self.validate_configuration()
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError(
                "install the ingest extra to use HuggingFaceStreamingAdapter"
            ) from error
        dataset: Any
        if self.dataset_config is None:
            dataset = load_dataset(self.dataset_id, split=self.split, streaming=True)
        else:
            dataset = load_dataset(
                self.dataset_id,
                name=self.dataset_config,
                split=self.split,
                streaming=True,
            )
        dataset_state = checkpoint.get("dataset_state")
        if isinstance(dataset_state, dict):
            dataset.load_state_dict(dataset_state)

        # Existing databases may contain the old record-index checkpoint. It
        # must replay once, but reports that scan instead of appearing frozen.
        start_index = int(checkpoint.get("record_index", 0))
        byte_limit = self.max_bytes
        if max_bytes is not None:
            byte_limit = min(byte_limit, max_bytes) if byte_limit is not None else max_bytes
        downloaded_bytes = 0
        for record_index, record in enumerate(dataset):
            if record_index < start_index:
                current = record_index + 1
                if self.resume_progress_callback and (
                    current == 1 or current == start_index or current % 1000 == 0
                ):
                    self.resume_progress_callback(current, start_index)
                continue
            emitted_index = record_index - start_index
            if self.max_records is not None and emitted_index >= self.max_records:
                return
            try:
                text = record[self.text_field]
            except (KeyError, TypeError) as error:
                raise ValueError(f"record is missing text field: {self.text_field}") from error
            if not isinstance(text, str):
                raise ValueError(f"text field is not a string: {self.text_field}")
            payload = (json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8")
            downloaded_bytes += len(payload)
            if byte_limit is not None and downloaded_bytes > byte_limit:
                return
            yield SourceRecord(payload.rstrip(b"\n"), {"dataset_state": dataset.state_dict()})


class DolmaManifestAdapter:
    """Stream Dolma JSONL.GZ shards listed by an official version manifest."""

    def __init__(self, manifest_url: str, max_bytes: int, timeout_seconds: float = 30.0) -> None:
        self.manifest_url = manifest_url
        self.max_bytes = max_bytes
        self.timeout_seconds = timeout_seconds

    def validate_configuration(self) -> None:
        if not self.manifest_url.startswith("https://") or self.max_bytes <= 0:
            raise ValueError("Dolma requires an HTTPS manifest and positive byte quota")

    def stream_records(
        self, checkpoint: dict[str, Any], max_bytes: int | None = None
    ) -> Iterator[SourceRecord]:
        """Resume at a manifest file and JSONL record without loading either fully."""
        self.validate_configuration()
        start_file = int(checkpoint.get("file_index", 0))
        start_record = int(checkpoint.get("record_index", 0))
        byte_limit = min(self.max_bytes, max_bytes) if max_bytes is not None else self.max_bytes
        downloaded_bytes = 0
        request = Request(self.manifest_url, headers={"User-Agent": "mini-llm-ingestion/0.1"})
        with urlopen(request, timeout=self.timeout_seconds) as manifest:
            for file_index, raw_url in enumerate(manifest):
                if file_index < start_file:
                    continue
                shard_url = raw_url.decode("utf-8").strip()
                if not shard_url:
                    continue
                if not shard_url.startswith("https://"):
                    raise ValueError(f"Dolma manifest contains a non-HTTPS URL: {shard_url}")
                shard_request = Request(
                    shard_url,
                    headers={"User-Agent": "mini-llm-ingestion/0.1"},
                )
                with urlopen(shard_request, timeout=self.timeout_seconds) as response:
                    with gzip.GzipFile(fileobj=response) as compressed:
                        for record_index, line in enumerate(compressed):
                            if file_index == start_file and record_index < start_record:
                                continue
                            payload = line.rstrip(b"\r\n")
                            try:
                                record = json.loads(payload)
                            except json.JSONDecodeError as error:
                                raise ValueError("Dolma shard contains invalid JSONL") from error
                            if not isinstance(record, dict) or not isinstance(
                                record.get("text"), str
                            ):
                                raise ValueError("Dolma record is missing a string text field")
                            if downloaded_bytes + len(payload) + 1 > byte_limit:
                                return
                            downloaded_bytes += len(payload) + 1
                            yield SourceRecord(
                                payload,
                                {"file_index": file_index, "record_index": record_index + 1},
                            )


def adapter_for_source(
    source: DataSourceConfig,
    resume_progress_callback: Callable[[int, int], None] | None = None,
) -> DataSourceAdapter:
    """Construct the adapter selected by declarative source configuration."""
    if source.source_type == "huggingface_dataset":
        if source.dataset_id is None:
            raise ValueError(f"missing dataset_id for {source.source_id}")
        return HuggingFaceStreamingAdapter(
            source.dataset_id,
            dataset_config=source.dataset_config,
            split=source.split,
            max_bytes=source.max_bytes,
            resume_progress_callback=resume_progress_callback,
        )
    if source.source_type == "http_lines":
        return HttpLineAdapter(source.source_url)
    if source.source_type == "filesystem":
        return LocalTextAdapter(source.source_url)
    if source.source_type == "dolma_manifest":
        return DolmaManifestAdapter(source.source_url, source.max_bytes)
    raise ValueError(f"unsupported source type: {source.source_type}")
