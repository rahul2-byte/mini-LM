"""Command-line entry points for raw data ingestion."""

import argparse
from datetime import UTC, datetime
import logging
import os
import signal
import sys
from pathlib import Path
from threading import Event

from minio import Minio

from common.logging import configure_logging
from common.progress import ProgressReporter
from data_ingestion.adapters import adapter_for_source
from data_ingestion.config import DataSourceConfig, load_data_config
from data_ingestion.ingestion import build_manifest_key
from data_ingestion.manifest_json import write_run_manifest
from data_ingestion.metadata import DuckDBMetadataStore, MetadataLockError
from data_ingestion.orchestrator import IngestionPipeline
from data_ingestion.reconciliation import reconcile_startup
from data_ingestion.storage import MinioObjectStore


logger = logging.getLogger(__name__)


def run_ingestion(config_path: str | Path, source_id: str | None = None) -> dict[str, str]:
    """Run selected sources sequentially against local MinIO.

    This is the ingestion workflow boundary. It owns infrastructure setup and
    source selection; adapters own remote reading and the pipeline owns shard
    processing and checkpoint updates.
    """
    config = load_data_config(config_path)
    if config.ingestion is None:
        raise ValueError("ingestion configuration is required")

    # Credentials remain environment-only; config files and logs never contain
    # the MinIO secret.
    username = os.environ.get("MINIO_ROOT_USER")
    password = os.environ.get("MINIO_ROOT_PASSWORD")
    if not username or not password:
        raise ValueError("MINIO_ROOT_USER and MINIO_ROOT_PASSWORD are required")
    endpoint = os.environ.get("MINIO_ENDPOINT", "127.0.0.1:9000")
    secure = os.environ.get("MINIO_SECURE", "0").lower() in {"1", "true", "yes"}
    client = Minio(endpoint, access_key=username, secret_key=password, secure=secure)

    # The CLI owns bucket bootstrap for local development. Production deploys
    # can pre-create the bucket with a least-privilege service account.
    if not client.bucket_exists(config.ingestion.raw_bucket):
        client.make_bucket(config.ingestion.raw_bucket)

    metadata = DuckDBMetadataStore(config.ingestion.metadata_path)
    object_store = MinioObjectStore(client)
    # Reconcile before scheduling new work so stale RUNNING rows cannot be
    # mistaken for healthy concurrent workers.
    reconcile_startup(metadata, object_store, config.ingestion.staging_directory)

    pause_event = Event()
    previous_handlers = {
        signal_number: signal.getsignal(signal_number)
        for signal_number in (signal.SIGINT, signal.SIGTERM, signal.SIGTSTP)
    }

    def request_pause(signum: int, _frame: object) -> None:
        """Convert terminal/process stop signals into a checkpointed pause."""
        logger.warning(
            "pause_requested",
            extra={"event": "pause_requested", "signal": signal.Signals(signum).name},
        )
        pause_event.set()

    # Ctrl-C, Ctrl-Z, and service shutdown now use the same safe pause path.
    for signal_number in previous_handlers:
        signal.signal(signal_number, request_pause)
    progress = ProgressReporter(enabled=sys.stdout.isatty(), stream=sys.stdout)
    pipeline = IngestionPipeline(
        metadata=metadata,
        object_store=object_store,
        staging_directory=config.ingestion.staging_directory,
        raw_bucket=config.ingestion.raw_bucket,
        target_shard_size_bytes=config.ingestion.target_shard_size_bytes,
        maximum_shard_size_bytes=config.ingestion.maximum_shard_size_bytes,
        minimum_free_space_bytes=config.ingestion.minimum_free_space_bytes,
        maximum_staging_usage_bytes=config.ingestion.maximum_staging_usage_bytes,
        pause_event=pause_event,
        # Keep the human progress line on stdout and JSON logs on stderr so
        # third-party warnings cannot overwrite the progress display.
        progress=progress,
    )

    configured_sources = list(config.sources)
    if source_id == "local-sample":
        configured_sources = [
            DataSourceConfig(
                source_id="local-sample",
                source_type="filesystem",
                source_url=str(config.sample_path),
                license_name="user-provided",
                license_url="",
                max_bytes=config.sample_path.stat().st_size,
            )
        ]

    # Preserve YAML order: source processing is sequential by design and the
    # order is part of the reproducible ingestion plan.
    selected = [
        source
        for source in configured_sources
        if source.enabled and (source_id is None or source.source_id == source_id)
    ]
    if source_id is not None and not selected:
        raise ValueError(f"unknown or disabled source: {source_id}")

    results: dict[str, str] = {}
    try:
        for source in selected:
            try:
                state = pipeline.run_source(
                    source,
                    adapter_for_source(source, progress.update_resume),
                    ingestion_date=datetime.now(UTC).date().isoformat(),
                )
                results[source.source_id] = str(state)
                run_id = metadata.get_source(source.source_id)["current_run_id"]
                if state == "COMPLETED" and run_id:
                    # The local manifest is written first, then copied to MinIO
                    # under the versioned key after all shard records exist.
                    manifest = write_run_manifest(
                        metadata,
                        run_id,
                        config.project_root / "artifacts/raw-manifests" / f"{run_id}.json",
                    )
                    version_id = metadata.get_run(run_id)["version_id"]
                    client.fput_object(
                        config.ingestion.raw_bucket,
                        build_manifest_key(source.source_id, version_id, run_id),
                        str(manifest),
                    )
                if state == "PAUSED":
                    break
            except Exception as error:
                results[source.source_id] = f"FAILED: {error}"
    finally:
        for signal_number, previous_handler in previous_handlers.items():
            signal.signal(signal_number, previous_handler)
        metadata.close()
    return results


def main() -> None:
    """Dispatch ingestion-only commands without importing training code."""
    parser = argparse.ArgumentParser(prog="data-ingest")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("command", choices=["ingest", "ingest-local"])
    parser.add_argument("--source-id", type=str)
    args = parser.parse_args()
    configure_logging()

    # Dataset streaming libraries are useful sources but their request-level
    # INFO logs drown out the pipeline's source/shard events. Keep warnings and
    # errors while leaving the application-owned structured logs visible.
    for logger_name in ("httpx", "httpcore", "huggingface_hub", "datasets"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # The dataset library has its own tqdm renderer. Disable it because the
    # ingestion reporter already owns the terminal progress line.
    try:
        from datasets.utils.logging import disable_progress_bar
    except ImportError:
        pass
    else:
        disable_progress_bar()
        # Hugging Face configures child loggers with their own WARNING level;
        # lower those concrete loggers after the package import as well.
        for logger_name in (
            "huggingface_hub.utils._http",
            "huggingface_hub.utils._headers",
            "datasets.utils.logging",
        ):
            logging.getLogger(logger_name).setLevel(logging.ERROR)
    source_id = "local-sample" if args.command == "ingest-local" else args.source_id
    try:
        results = run_ingestion(args.config, source_id)
        for completed_source, state in results.items():
            print(f"{completed_source}: {state}")
    except MetadataLockError as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
