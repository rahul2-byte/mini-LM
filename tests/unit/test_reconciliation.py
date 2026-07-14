from dataclasses import dataclass
from pathlib import Path

from data_ingestion.metadata import DuckDBMetadataStore
from data_ingestion.reconciliation import reconcile_startup
from data_ingestion.storage import ObjectMetadata


@dataclass
class FakeStore:
    objects: dict[tuple[str, str], int]

    def stat(self, bucket: str, object_key: str) -> ObjectMetadata:
        size = self.objects[(bucket, object_key)]
        return ObjectMetadata(bucket, object_key, size, "etag")


def test_reconcile_pauses_runs_and_quarantines_staging(tmp_path: Path) -> None:
    metadata = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    metadata.register_source("source", "Source", "test", None, 100)
    metadata.create_run("run-1", "source", "version-1")
    metadata.set_source_status("source", "RUNNING")
    metadata.set_run_status("run-1", "RUNNING")
    partial = tmp_path / "staging/source/run-1/shard-000001.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")

    reconcile_startup(metadata, FakeStore({}), tmp_path / "staging")

    assert metadata.get_source("source")["status"] == "PAUSED"
    assert not partial.exists()
    assert list((tmp_path / "staging/quarantine").rglob("*.partial"))


def test_reconcile_fails_source_when_remote_shard_is_missing(tmp_path: Path) -> None:
    metadata = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    metadata.register_source("source", "Source", "test", None, 100)
    metadata.create_run("run-1", "source", "version-1")
    metadata.record_verified_shard(
        shard_id="shard-1",
        run_id="run-1",
        source_id="source",
        version_id="version-1",
        shard_sequence=1,
        bucket="raw",
        object_key="source/shard-1.txt",
        checksum="checksum",
        stored_size_bytes=10,
    )

    reconcile_startup(metadata, FakeStore({}), tmp_path / "staging")

    assert metadata.get_source("source")["status"] == "FAILED"
