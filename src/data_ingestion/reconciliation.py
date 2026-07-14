"""Startup recovery for interrupted raw-ingestion runs."""

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil

from data_ingestion.ingestion import SourceState
from data_ingestion.metadata import DuckDBMetadataStore
from data_ingestion.storage import ObjectStore


_ACTIVE_STATES = {
    SourceState.RUNNING,
    SourceState.RETRYING,
    SourceState.PAUSE_REQUESTED,
}


def reconcile_startup(
    metadata: DuckDBMetadataStore,
    object_store: ObjectStore,
    staging_directory: str | Path,
) -> None:
    """Make interrupted runs resumable without deleting local evidence.

    A process can die between any two upload/metadata operations.  Reopening
    DuckDB and checking MinIO at startup turns those ambiguous states into an
    explicit paused or failed state before new work is scheduled.
    """
    for source in metadata.list_sources():
        source_id = source["source_id"]
        run_id = source["current_run_id"]
        status = SourceState(source["status"])
        if status in _ACTIVE_STATES:
            # No process lock survives a crash, so active records are paused
            # first.  A later command can deliberately resume them.
            metadata.set_source_status(
                source_id,
                SourceState.PAUSED,
                error="startup reconciliation paused interrupted run",
            )
            if run_id:
                metadata.set_run_status(run_id, SourceState.PAUSED)
            metadata.record_event(
                event_type="startup_run_paused",
                event_level="WARNING",
                message="interrupted run made resumable",
                run_id=run_id,
                source_id=source_id,
            )
        if not run_id:
            continue
        for shard in metadata.list_run_shards(run_id):
            try:
                # DuckDB is the audit record, while MinIO is the data record;
                # verify both sides agree before trusting a completed shard.
                remote = object_store.stat(shard["minio_bucket"], shard["minio_object_key"])
                if remote.size_bytes != shard["stored_size_bytes"]:
                    raise IOError("stored shard size does not match DuckDB metadata")
            except Exception as error:
                metadata.set_source_status(source_id, SourceState.FAILED, error=str(error))
                metadata.set_run_status(run_id, SourceState.FAILED, str(error))
                metadata.record_event(
                    event_type="startup_shard_verification_failed",
                    event_level="ERROR",
                    message=str(error),
                    run_id=run_id,
                    source_id=source_id,
                    event_data=json.dumps({"object_key": shard["minio_object_key"]}),
                )

    _quarantine_incomplete_files(Path(staging_directory))


def _quarantine_incomplete_files(staging_directory: Path) -> None:
    """Move ambiguous staging files aside so recovery never destroys evidence."""
    if not staging_directory.exists():
        return
    quarantine = staging_directory / "quarantine" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for path in staging_directory.rglob("*"):
        if not path.is_file() or path.suffix not in {".partial", ".ready", ".uploading"}:
            continue
        if quarantine in path.parents:
            continue
        # Moving instead of deleting lets an operator inspect or manually
        # re-upload a file after diagnosing why the previous run stopped.
        destination = quarantine / path.relative_to(staging_directory)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), destination)
