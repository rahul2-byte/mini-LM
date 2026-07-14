"""Source adapter contracts and a resumable local line adapter for smoke runs."""

from dataclasses import dataclass
import csv
import gzip
import io
import json
from pathlib import Path
from typing import Any, Iterator, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from data_ingestion.config import DataSourceConfig


@dataclass(frozen=True)
class SourceRecord:
    """One source payload plus the checkpoint valid immediately after it."""

    payload: bytes
    checkpoint: dict[str, int]


class DataSourceAdapter(Protocol):
    """Contract separating source access from sharding and MinIO concerns."""

    def validate_configuration(self) -> None: ...

    def discover_source_version(self) -> str | None: ...

    def stream_records(self, checkpoint: dict[str, int]) -> Iterator[SourceRecord]: ...


class LocalTextAdapter:
    """Read UTF-8 lines and resume from the next byte offset."""

    def __init__(self, path: str | Path) -> None:
        """Store the local path; actual I/O is deferred until streaming."""
        self.path = Path(path)

    def validate_configuration(self) -> None:
        """Fail early when the smoke source file is unavailable."""
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def discover_source_version(self) -> str | None:
        """Return no upstream version because a local file has no catalog ID."""
        return None

    def stream_records(self, checkpoint: dict[str, int]) -> Iterator[SourceRecord]:
        """Yield lines from a saved byte offset without reading the whole file."""
        self.validate_configuration()
        offset = int(checkpoint.get("byte_offset", 0))
        with self.path.open("rb") as handle:
            # Byte offsets are safe here because the file is read as bytes and
            # each record is newline-delimited; no text decoder state is kept.
            handle.seek(offset)
            while line := handle.readline():
                yield SourceRecord(line.rstrip(b"\r\n"), {"byte_offset": handle.tell()})


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

    def discover_source_version(self) -> str | None:
        """Return no version because generic HTTP endpoints may be mutable."""
        return None

    def stream_records(self, checkpoint: dict[str, int]) -> Iterator[SourceRecord]:
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
            while line := response.readline():
                offset += len(line)
                yield SourceRecord(line.rstrip(b"\r\n"), {"byte_offset": offset})


class HuggingFaceStreamingAdapter:
    """Stream text records from a Hugging Face dataset configuration.

    Resume is record-index based because the streaming dataset API does not
    guarantee stable byte offsets across remote shards.
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
    ) -> None:
        """Configure a lazy dataset iterator with local quota limits."""
        self.dataset_id = dataset_id
        self.dataset_config = dataset_config
        self.split = split
        self.text_field = text_field
        self.max_records = max_records
        self.max_bytes = max_bytes

    def validate_configuration(self) -> None:
        """Validate settings before importing the optional datasets package."""
        if not self.dataset_id or not self.split or not self.text_field:
            raise ValueError("dataset_id, split, and text_field are required")
        if self.max_records is not None and self.max_records <= 0:
            raise ValueError("max_records must be positive")
        if self.max_bytes is not None and self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")

    def discover_source_version(self) -> str | None:
        """Return no version because this adapter currently uses dataset tags only."""
        return None

    def stream_records(self, checkpoint: dict[str, int]) -> Iterator[SourceRecord]:
        """Yield JSONL records from a streaming dataset using record checkpoints."""
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
        # Remote streaming shards do not promise stable byte offsets, so the
        # portable resume point is the source record index.
        start_index = int(checkpoint.get("record_index", 0))
        downloaded_bytes = 0
        for record_index, record in enumerate(dataset):
            if record_index < start_index:
                continue
            if self.max_records is not None and record_index - start_index >= self.max_records:
                return
            try:
                text = record[self.text_field]
            except (KeyError, TypeError) as error:
                raise ValueError(f"record is missing text field: {self.text_field}") from error
            if not isinstance(text, str):
                raise ValueError(f"text field is not a string: {self.text_field}")
            payload = (json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8")
            downloaded_bytes += len(payload)
            if self.max_bytes is not None and downloaded_bytes > self.max_bytes:
                return
            yield SourceRecord(payload.rstrip(b"\n"), {"record_index": record_index + 1})


class ProjectGutenbergAdapter:
    """Stream English U.S.-public-domain plain-text ebooks from the PG catalog."""

    catalog_url = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"

    def __init__(self, max_bytes: int, timeout_seconds: float = 30.0) -> None:
        """Configure a catalog-driven adapter limited to public-domain English books."""
        self.max_bytes = max_bytes
        self.timeout_seconds = timeout_seconds

    def validate_configuration(self) -> None:
        """Require a positive byte quota before contacting the catalog."""
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")

    def discover_source_version(self) -> str | None:
        """Return no catalog snapshot ID because the feed is mutable."""
        return None

    def stream_records(self, checkpoint: dict[str, int]) -> Iterator[SourceRecord]:
        """Walk the catalog and yield eligible ebook payloads until the quota."""
        self.validate_configuration()
        start_index = int(checkpoint.get("catalog_index", 0))
        downloaded_bytes = 0
        with urlopen(self.catalog_url, timeout=self.timeout_seconds) as response:
            with gzip.GzipFile(fileobj=response) as compressed:
                text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
                for catalog_index, row in enumerate(csv.DictReader(text)):
                    if catalog_index < start_index:
                        continue
                    language = (row.get("Language") or row.get("Languages") or "").lower()
                    rights = (row.get("Copyright Status") or row.get("Rights") or "").lower()
                    if "en" not in language or "public domain" not in rights:
                        continue
                    ebook_id = row.get("EBook-No.") or row.get("EBook-No") or row.get("Text#")
                    if not ebook_id or not ebook_id.isdigit():
                        continue
                    text_url = f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"
                    # Gutenberg ebooks are already complete source records. A
                    # later optimization can stream this response in chunks;
                    # keeping the record intact is the current correctness rule.
                    with urlopen(text_url, timeout=self.timeout_seconds) as ebook_response:
                        payload = ebook_response.read()
                    if downloaded_bytes + len(payload) > self.max_bytes:
                        return
                    downloaded_bytes += len(payload)
                    yield SourceRecord(payload, {"catalog_index": catalog_index + 1})


def adapter_for_source(source: DataSourceConfig) -> DataSourceAdapter:
    """Construct the adapter selected by declarative source configuration."""
    if source.source_type == "huggingface_dataset":
        if source.dataset_id is None:
            raise ValueError(f"missing dataset_id for {source.source_id}")
        return HuggingFaceStreamingAdapter(
            source.dataset_id,
            dataset_config=source.dataset_config,
            split=source.split,
            max_bytes=source.max_bytes,
        )
    if source.source_type == "http_lines":
        return HttpLineAdapter(source.source_url)
    if source.source_type == "filesystem":
        return LocalTextAdapter(source.source_url)
    if source.source_type == "external_archive":
        return ProjectGutenbergAdapter(source.max_bytes)
    raise ValueError(f"unsupported source type: {source.source_type}")
