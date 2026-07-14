"""Typed YAML configuration loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


@dataclass(frozen=True)
class DataSourceConfig:
    """Declarative limits and provenance for one legal upstream source."""

    source_id: str
    source_type: str
    source_url: str
    license_name: str
    license_url: str
    max_bytes: int
    dataset_id: str | None = None
    dataset_config: str | None = None
    split: str = "train"
    enabled: bool = True
    notes: str = ""


@dataclass(frozen=True)
class IngestionConfig:
    """Operational guardrails for staging, sharding, and object verification."""

    metadata_path: Path
    staging_directory: Path
    raw_bucket: str = "mini-llm-raw"
    target_shard_size_bytes: int = 256 * 1024 * 1024
    minimum_shard_size_bytes: int = 200 * 1024 * 1024
    maximum_shard_size_bytes: int = 500 * 1024 * 1024
    minimum_free_space_bytes: int = 20 * 1024 * 1024 * 1024
    maximum_staging_usage_bytes: int = 100 * 1024 * 1024 * 1024
    verify_after_upload: bool = True


@dataclass(frozen=True)
class DataConfig:
    """Resolved data settings shared by smoke runs and production ingestion."""

    project_root: Path
    manifest_path: Path
    sample_path: Path
    seed: int = 42
    min_document_chars: int = 20
    raw_budget_bytes: int = 10_000_000_000
    sources: tuple[DataSourceConfig, ...] = ()
    ingestion: IngestionConfig | None = None


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping and convert missing or malformed files to config errors."""
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration must be a YAML mapping: {path}")
    return value


def load_data_config(path: str | Path) -> DataConfig:
    """Load and validate data settings relative to the config file location.

    Relative paths are resolved from ``project_root`` rather than the current
    shell directory.  This makes VS Code launchers, cron jobs, and direct CLI
    invocations produce the same files.
    """
    config_path = Path(path).resolve()
    raw = _read_yaml(config_path)
    # Resolve the project root first because all other relative paths depend on
    # it.  ``.`` therefore means the directory containing the YAML file.
    root = Path(raw.get("project_root", "."))
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    manifest = Path(raw.get("manifest_path", "data/manifests/manifest.db"))
    sample = Path(raw.get("sample_path", "data/sample.txt"))
    if not manifest.is_absolute():
        manifest = root / manifest
    if not sample.is_absolute():
        sample = root / sample
    minimum = int(raw.get("min_document_chars", 20))
    if minimum < 0:
        raise ConfigError("min_document_chars must be non-negative")

    # Ingestion settings are optional for the tokenizer smoke path, but when
    # present they must be a mapping so invalid YAML fails early.
    raw_ingestion = raw.get("ingestion", {})
    if not isinstance(raw_ingestion, dict):
        raise ConfigError("ingestion must be a YAML mapping")
    metadata_path = Path(raw_ingestion.get("metadata_path", "data/manifests/raw_ingestion.duckdb"))
    staging_directory = Path(raw_ingestion.get("staging_directory", "data/staging"))
    if not metadata_path.is_absolute():
        metadata_path = root / metadata_path
    if not staging_directory.is_absolute():
        staging_directory = root / staging_directory
    # Keep sizes in MB in human-edited YAML and convert once at the boundary to
    # bytes, which avoids repeated unit conversions in the hot ingestion path.
    target_mb = int(raw_ingestion.get("target_shard_size_mb", 256))
    minimum_mb = int(raw_ingestion.get("minimum_shard_size_mb", 200))
    maximum_mb = int(raw_ingestion.get("maximum_shard_size_mb", 500))
    if not 0 < minimum_mb <= target_mb <= maximum_mb:
        raise ConfigError("shard size bounds must satisfy 0 < minimum <= target <= maximum")
    minimum_free_gb = int(raw_ingestion.get("minimum_free_space_gb", 20))
    maximum_staging_gb = int(raw_ingestion.get("maximum_staging_usage_gb", 100))
    if minimum_free_gb <= 0 or maximum_staging_gb <= 0:
        raise ConfigError("disk guardrails must be positive")
    ingestion = IngestionConfig(
        metadata_path=metadata_path,
        staging_directory=staging_directory,
        raw_bucket=str(raw_ingestion.get("raw_bucket", "mini-llm-raw")),
        target_shard_size_bytes=target_mb * 1024 * 1024,
        minimum_shard_size_bytes=minimum_mb * 1024 * 1024,
        maximum_shard_size_bytes=maximum_mb * 1024 * 1024,
        minimum_free_space_bytes=minimum_free_gb * 1024 * 1024 * 1024,
        maximum_staging_usage_bytes=maximum_staging_gb * 1024 * 1024 * 1024,
        verify_after_upload=bool(raw_ingestion.get("verify_after_upload", True)),
    )

    raw_budget = int(raw.get("raw_budget_bytes", 10_000_000_000))
    if raw_budget <= 0:
        raise ConfigError("raw_budget_bytes must be positive")

    raw_sources = raw.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ConfigError("sources must be a YAML list")

    sources: list[DataSourceConfig] = []
    source_ids: set[str] = set()
    for index, value in enumerate(raw_sources):
        if not isinstance(value, dict):
            raise ConfigError(f"sources[{index}] must be a YAML mapping")
        try:
            source = DataSourceConfig(
                source_id=str(value["id"]),
                source_type=str(value["source_type"]),
                source_url=str(value["source_url"]),
                license_name=str(value["license"]),
                license_url=str(value["license_url"]),
                max_bytes=int(value["max_bytes"]),
                dataset_id=(
                    str(value["dataset_id"]) if value.get("dataset_id") is not None else None
                ),
                dataset_config=(
                    str(value["dataset_config"])
                    if value.get("dataset_config") is not None
                    else None
                ),
                split=str(value.get("split", "train")),
                enabled=bool(value.get("enabled", True)),
                notes=str(value.get("notes", "")),
            )
        except KeyError as error:
            raise ConfigError(f"sources[{index}] is missing {error.args[0]}") from error
        if not source.source_id:
            raise ConfigError(f"sources[{index}].id must not be empty")
        if source.source_id in source_ids:
            raise ConfigError(f"source IDs must be unique: {source.source_id}")
        if source.max_bytes <= 0:
            raise ConfigError(f"source max_bytes must be positive: {source.source_id}")
        source_ids.add(source.source_id)
        sources.append(source)

    # Disabled sources remain visible in the config for auditability but do not
    # consume the active raw-data budget.
    configured_bytes = sum(source.max_bytes for source in sources if source.enabled)
    if configured_bytes > raw_budget:
        raise ConfigError(
            f"enabled source max_bytes exceed raw budget: {configured_bytes} > {raw_budget}"
        )
    return DataConfig(
        root,
        manifest,
        sample,
        int(raw.get("seed", 42)),
        minimum,
        raw_budget,
        tuple(sources),
        ingestion,
    )
