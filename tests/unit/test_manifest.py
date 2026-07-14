from pathlib import Path

from data_ingestion.manifest import ManifestStore


def test_manifest_creation_and_upsert(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("manifest me", encoding="utf-8")
    store = ManifestStore(tmp_path / "manifest.db")
    record = store.add_file(sample, "test", "CC0")
    assert store.list_files() == [record]
    assert record.size_bytes == sample.stat().st_size
