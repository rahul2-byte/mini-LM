from dataclasses import dataclass
from pathlib import Path
from threading import Event

from data_ingestion.config import DataSourceConfig
from data_ingestion.adapters import LocalTextAdapter
from data_ingestion.metadata import DuckDBMetadataStore
from data_ingestion.orchestrator import IngestionPipeline
from data_ingestion.storage import ObjectMetadata


@dataclass
class FakeObjectStore:
    root: Path

    def upload_file(self, local_path: str | Path, bucket: str, object_key: str) -> ObjectMetadata:
        destination = self.root / bucket / object_key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(local_path).read_bytes())
        return ObjectMetadata(bucket, object_key, destination.stat().st_size, "fake-etag")

    def stat(self, bucket: str, object_key: str) -> ObjectMetadata:
        destination = self.root / bucket / object_key
        return ObjectMetadata(bucket, object_key, destination.stat().st_size, "fake-etag")


def test_local_adapter_ingests_verified_shards(tmp_path: Path) -> None:
    source_path = tmp_path / "source.txt"
    source_path.write_bytes(b"one\ntwo\nthree\nfour\n")
    metadata = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    pipeline = IngestionPipeline(
        metadata=metadata,
        object_store=FakeObjectStore(tmp_path / "objects"),
        staging_directory=tmp_path / "staging",
        raw_bucket="mini-llm-raw",
        target_shard_size_bytes=8,
        maximum_shard_size_bytes=20,
    )
    source = DataSourceConfig(
        source_id="local",
        source_type="filesystem",
        source_url=str(source_path),
        license_name="test",
        license_url="https://example.test/license",
        max_bytes=100,
    )

    result = pipeline.run_source(
        source,
        LocalTextAdapter(source_path),
        ingestion_date="2026-07-14",
    )

    row = metadata.get_source("local")
    assert result == "COMPLETED"
    assert row["status"] == "COMPLETED"
    assert row["completed_shard_count"] == 2
    assert row["verified_bytes"] == len(source_path.read_bytes())


def test_local_adapter_pauses_after_verified_shard_and_resumes(tmp_path: Path) -> None:
    source_path = tmp_path / "source.txt"
    source_path.write_bytes(b"one\ntwo\nthree\nfour\n")
    metadata = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    pause_event = Event()
    pipeline = IngestionPipeline(
        metadata=metadata,
        object_store=FakeObjectStore(tmp_path / "objects"),
        staging_directory=tmp_path / "staging",
        raw_bucket="mini-llm-raw",
        target_shard_size_bytes=8,
        maximum_shard_size_bytes=20,
        pause_event=pause_event,
    )
    source = DataSourceConfig(
        source_id="local",
        source_type="filesystem",
        source_url=str(source_path),
        license_name="test",
        license_url="https://example.test/license",
        max_bytes=100,
    )

    pause_event.set()
    assert (
        pipeline.run_source(source, LocalTextAdapter(source_path), ingestion_date="2026-07-14")
        == "PAUSED"
    )
    assert metadata.get_source("local")["completed_shard_count"] == 1

    pause_event.clear()
    assert (
        pipeline.run_source(source, LocalTextAdapter(source_path), ingestion_date="2026-07-14")
        == "COMPLETED"
    )
    row = metadata.get_source("local")
    assert row["completed_shard_count"] == 2
    assert row["downloaded_bytes"] == len(source_path.read_bytes())
