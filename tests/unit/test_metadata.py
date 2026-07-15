from pathlib import Path

import pytest

from data_ingestion.metadata import DuckDBMetadataStore


def test_metadata_store_registers_source_and_run(tmp_path: Path) -> None:
    store = DuckDBMetadataStore(tmp_path / "metadata.duckdb")

    store.register_source(
        source_id="wikipedia",
        source_name="English Wikipedia",
        source_type="huggingface_dataset",
        source_uri="https://huggingface.co/datasets/wikimedia/wikipedia",
        configured_quota_bytes=1000,
        checkpoint_type="cursor",
        license_name="CC BY-SA-3.0",
        license_url="https://example.test/license",
        source_notes="licensed source",
        dataset_id="wikimedia/wikipedia",
        dataset_config="20231101.en",
    )
    store.create_run(
        run_id="run-1",
        source_id="wikipedia",
        version_id="version-1",
    )

    source = store.get_source("wikipedia")
    run = store.get_run("run-1")
    assert source["status"] == "PENDING"
    assert source["configured_quota_bytes"] == 1000
    assert source["license_name"] == "CC BY-SA-3.0"
    assert source["license_url"] == "https://example.test/license"
    assert source["source_notes"] == "licensed source"
    assert source["dataset_id"] == "wikimedia/wikipedia"
    assert source["dataset_config"] == "20231101.en"
    assert run["status"] == "PENDING"


def test_metadata_store_rejects_invalid_state_transition(tmp_path: Path) -> None:
    store = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    store.register_source("source", "Source", "test", None, 100)
    store.set_source_status("source", "RUNNING")
    store.set_source_status("source", "COMPLETED")

    with pytest.raises(ValueError, match="invalid source state transition"):
        store.set_source_status("source", "RUNNING")


def test_metadata_store_counts_only_verified_shards(tmp_path: Path) -> None:
    store = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    store.register_source("source", "Source", "test", None, 100)
    store.create_run("run-1", "source", "version-1")

    store.record_verified_shard(
        shard_id="shard-1",
        run_id="run-1",
        source_id="source",
        version_id="version-1",
        shard_sequence=1,
        bucket="mini-llm-raw",
        object_key="source/version-1/shard-000001.txt",
        checksum="abc",
        stored_size_bytes=100,
    )

    source = store.get_source("source")
    run = store.get_run("run-1")
    assert source["verified_bytes"] == 100
    assert source["completed_shard_count"] == 1
    assert run["verified_bytes"] == 100
