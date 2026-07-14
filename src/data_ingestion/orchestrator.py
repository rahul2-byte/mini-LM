"""Sequential raw ingestion orchestration."""

from pathlib import Path
from threading import Event

from data_ingestion.config import DataSourceConfig
from data_ingestion.adapters import DataSourceAdapter
from data_ingestion.disk import ensure_disk_capacity
from data_ingestion.ingestion import (
    SourceState,
    build_object_key,
    create_version_id,
    decode_checkpoint,
    encode_checkpoint,
    transition_state,
)
from data_ingestion.metadata import DuckDBMetadataStore
from data_ingestion.retry import retry_call
from data_ingestion.sharding import ShardBuilder
from data_ingestion.storage import ObjectStore


class IngestionPipeline:
    """Process one configured source at a time and resume from DuckDB state.

    The pipeline owns orchestration only.  Adapters know how to read sources,
    the shard builder knows record boundaries, and the object store knows
    MinIO.  This separation keeps each piece replaceable and testable.
    """

    def __init__(
        self,
        metadata: DuckDBMetadataStore,
        object_store: ObjectStore,
        staging_directory: str | Path,
        raw_bucket: str,
        target_shard_size_bytes: int,
        maximum_shard_size_bytes: int,
        minimum_free_space_bytes: int = 0,
        maximum_staging_usage_bytes: int = 2**63 - 1,
        retry_attempts: int = 5,
        pause_event: Event | None = None,
    ) -> None:
        """Store dependencies and resource guardrails for one pipeline run."""
        self.metadata = metadata
        self.object_store = object_store
        self.staging_directory = Path(staging_directory)
        self.raw_bucket = raw_bucket
        self.target_shard_size_bytes = target_shard_size_bytes
        self.maximum_shard_size_bytes = maximum_shard_size_bytes
        self.minimum_free_space_bytes = minimum_free_space_bytes
        self.maximum_staging_usage_bytes = maximum_staging_usage_bytes
        self.retry_attempts = retry_attempts
        self.pause_event = pause_event or Event()

    def run_source(
        self,
        source: DataSourceConfig,
        adapter: DataSourceAdapter,
        *,
        dataset_name: str | None = None,
        ingestion_date: str,
    ) -> str:
        """Run or resume one source and return its terminal state.

        A shard is counted only after upload succeeds and DuckDB records the
        verified object.  That ordering is the central invariant preventing
        quota totals from claiming data that exists only in local staging.
        """
        self.metadata.register_source(
            source_id=source.source_id,
            source_name=source.source_id,
            source_type=source.source_type,
            source_uri=source.source_url,
            configured_quota_bytes=source.max_bytes,
        )
        current = SourceState(self.metadata.get_source(source.source_id)["status"])
        if self.metadata.source_quota_reached(source.source_id):
            # A previous process may have completed the quota before this
            # invocation started; do not create a duplicate run in that case.
            self.metadata.set_source_status(source.source_id, SourceState.COMPLETED)
            return SourceState.COMPLETED
        if current not in {SourceState.PENDING, SourceState.PAUSED, SourceState.FAILED}:
            raise ValueError(f"source is already active: {source.source_id} ({current})")
        if current == SourceState.FAILED:
            transition_state(current, SourceState.RETRYING)
        else:
            transition_state(current, SourceState.RUNNING)

        source_row = self.metadata.get_source(source.source_id)
        run_id = source_row["current_run_id"]
        version_id = source_row["current_version_id"]
        if not run_id or not version_id:
            run_id = uuid_hex()
            version_id = create_version_id()
            self.metadata.create_run(run_id, source.source_id, version_id)
        self.metadata.set_source_status(source.source_id, SourceState.RUNNING)
        self.metadata.set_run_status(run_id, SourceState.RUNNING)

        try:
            adapter.validate_configuration()
            checkpoint = decode_checkpoint(source_row["checkpoint_value"])
            # The adapter interprets this checkpoint according to its source
            # type: byte offset, record index, catalog cursor, or another key.
            records = adapter.stream_records(checkpoint)
            builder = ShardBuilder(
                self.staging_directory / source.source_id / run_id,
                self.target_shard_size_bytes,
                self.maximum_shard_size_bytes,
            )
            start_sequence = int(source_row["completed_shard_count"]) + 1
            for shard in builder.build(records, start_sequence=start_sequence):
                # Check capacity after the file is built but before accepting
                # another shard.  A ready file remains recoverable if this
                # guard pauses/fails the run.
                ensure_disk_capacity(
                    shard.path.parent,
                    minimum_free_space_bytes=self.minimum_free_space_bytes,
                    maximum_staging_usage_bytes=self.maximum_staging_usage_bytes,
                )
                object_key = build_object_key(
                    source_id=source.source_id,
                    dataset_name=dataset_name or source.source_id,
                    version_id=version_id,
                    ingestion_date=ingestion_date,
                    run_id=run_id,
                    shard_sequence=shard.sequence,
                    extension="txt",
                )
                stored = retry_call(
                    lambda: self.object_store.upload_file(shard.path, self.raw_bucket, object_key),
                    attempts=self.retry_attempts,
                )
                # Persist the checkpoint in the same transaction that marks the
                # object verified.  Resume therefore starts after this shard,
                # never in the middle of an object that was not committed.
                self.metadata.record_verified_shard(
                    shard_id=f"{run_id}-{shard.sequence:06d}",
                    run_id=run_id,
                    source_id=source.source_id,
                    version_id=version_id,
                    shard_sequence=shard.sequence,
                    bucket=stored.bucket,
                    object_key=stored.object_key,
                    checksum=shard.checksum,
                    stored_size_bytes=stored.size_bytes,
                    downloaded_size_bytes=shard.size_bytes,
                    etag=stored.etag,
                    checkpoint=encode_checkpoint(shard.checkpoint),
                )
                shard.path.unlink()
                if self.pause_event.is_set():
                    # Pause is checked only after the current shard is safe in
                    # MinIO and DuckDB; no partially uploaded shard is counted.
                    self.metadata.set_source_status(source.source_id, SourceState.PAUSED)
                    self.metadata.set_run_status(run_id, SourceState.PAUSED)
                    return SourceState.PAUSED
                if self.metadata.source_quota_reached(source.source_id):
                    break
        except Exception as error:
            # Preserve the exception for the caller while recording enough
            # state for a later startup reconciliation or explicit retry.
            self.metadata.set_source_status(source.source_id, SourceState.FAILED, error=str(error))
            self.metadata.set_run_status(run_id, SourceState.FAILED, str(error))
            self.metadata.record_event(
                event_type="source_failed",
                event_level="ERROR",
                message=str(error),
                run_id=run_id,
                source_id=source.source_id,
            )
            raise
        self.metadata.set_source_status(source.source_id, SourceState.COMPLETED)
        self.metadata.set_run_status(run_id, SourceState.COMPLETED)
        return SourceState.COMPLETED


def uuid_hex() -> str:
    """Return a short run identifier suitable for object keys and logs."""
    from uuid import uuid4

    return uuid4().hex[:8]
