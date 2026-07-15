from pathlib import Path

import pytest

from data_ingestion.config import ConfigError, load_data_config


def test_load_data_config_resolves_paths(tmp_path: Path) -> None:
    config = tmp_path / "data.yaml"
    config.write_text(
        "project_root: .\nsample_path: sample.txt\n",
        encoding="utf-8",
    )
    loaded = load_data_config(config)
    assert loaded.sample_path == tmp_path / "sample.txt"


def test_load_data_config_rejects_invalid_threshold(tmp_path: Path) -> None:
    config = tmp_path / "data.yaml"
    config.write_text("min_document_chars: -1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_data_config(config)


def test_load_data_config_loads_source_registry(tmp_path: Path) -> None:
    config = tmp_path / "data.yaml"
    config.write_text(
        """
project_root: .
raw_budget_bytes: 100
ingestion:
  metadata_path: metadata/raw.duckdb
  staging_directory: staging
  raw_bucket: mini-llm-raw
  target_shard_size_mb: 256
  maximum_shard_size_mb: 500
sources:
  - id: wikipedia
    source_type: huggingface_dataset
    dataset_id: wikimedia/wikipedia
    source_url: https://huggingface.co/datasets/wikimedia/wikipedia
    license: CC BY-SA-3.0 / GFDL
    license_url: https://dumps.wikimedia.org/legal.html
    max_bytes: 60
  - id: books
    source_type: huggingface_dataset
    dataset_id: common-pile/project_gutenberg_filtered
    source_url: https://huggingface.co/datasets/common-pile/project_gutenberg_filtered
    license: public-domain
    license_url: https://huggingface.co/datasets/common-pile/project_gutenberg_filtered
    max_bytes: 40
""",
        encoding="utf-8",
    )

    loaded = load_data_config(config)

    assert loaded.raw_budget_bytes == 100
    assert loaded.ingestion.metadata_path == tmp_path / "metadata/raw.duckdb"
    assert loaded.ingestion.raw_bucket == "mini-llm-raw"
    assert loaded.ingestion.target_shard_size_bytes == 256 * 1024 * 1024
    assert [source.source_id for source in loaded.sources] == ["wikipedia", "books"]
    assert loaded.sources[0].max_bytes == 60
    assert loaded.sources[0].dataset_id == "wikimedia/wikipedia"


def test_load_data_config_rejects_source_budget_overflow(tmp_path: Path) -> None:
    config = tmp_path / "data.yaml"
    config.write_text(
        """
raw_budget_bytes: 10
sources:
  - id: source
    source_type: test
    source_url: https://example.test/source
    license: test
    license_url: https://example.test/license
    max_bytes: 11
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="raw budget"):
        load_data_config(config)


def test_load_data_config_rejects_duplicate_source_ids(tmp_path: Path) -> None:
    config = tmp_path / "data.yaml"
    config.write_text(
        """
sources:
  - id: duplicate
    source_type: test
    source_url: https://example.test/source
    license: test
    license_url: https://example.test/license
    max_bytes: 1
  - id: duplicate
    source_type: test
    source_url: https://example.test/source
    license: test
    license_url: https://example.test/license
    max_bytes: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unique"):
        load_data_config(config)


def test_load_data_config_rejects_invalid_shard_bounds(tmp_path: Path) -> None:
    config = tmp_path / "data.yaml"
    config.write_text(
        "ingestion:\n  target_shard_size_mb: 500\n  maximum_shard_size_mb: 200\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="shard size"):
        load_data_config(config)


def test_production_config_selects_smollm_subset() -> None:
    config = load_data_config(Path(__file__).parents[2] / "configs/data.yaml")
    smollm = next(source for source in config.sources if source.source_id == "smollm_corpus")

    assert smollm.dataset_config == "cosmopedia-v2"


def test_production_config_uses_common_pile_books() -> None:
    config = load_data_config(Path(__file__).parents[2] / "configs/data.yaml")
    books = next(source for source in config.sources if source.source_id == "public_domain_books")

    assert books.source_type == "huggingface_dataset"
    assert books.dataset_id == "common-pile/project_gutenberg_filtered"
    assert all(source.source_id != "project_gutenberg" for source in config.sources)
