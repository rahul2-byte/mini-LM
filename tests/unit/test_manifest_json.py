import json
from pathlib import Path

from data_ingestion.manifest_json import write_run_manifest
from data_ingestion.metadata import DuckDBMetadataStore


def test_run_manifest_contains_source_provenance(tmp_path: Path) -> None:
    store = DuckDBMetadataStore(tmp_path / "metadata.duckdb")
    store.register_source(
        "books",
        "books",
        "huggingface_dataset",
        "https://example.test/books",
        100,
        license_name="Public Domain",
        license_url="https://example.test/license",
        source_notes="filtered books",
        dataset_id="example/books",
        dataset_config="english",
    )
    store.create_run("run-1", "books", "version-1")

    path = write_run_manifest(store, "run-1", tmp_path / "manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["source"] == {
        "dataset_config": "english",
        "dataset_id": "example/books",
        "license_name": "Public Domain",
        "license_url": "https://example.test/license",
        "notes": "filtered books",
        "source_type": "huggingface_dataset",
        "source_uri": "https://example.test/books",
    }
