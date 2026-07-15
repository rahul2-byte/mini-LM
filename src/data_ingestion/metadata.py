"""Transactional DuckDB metadata repository for raw ingestion."""

from datetime import UTC, datetime
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import duckdb

from data_ingestion.ingestion import SourceState, transition_state


def _now() -> str:
    """Return an explicit UTC timestamp for reproducible audit records."""
    return datetime.now(UTC).isoformat()


class MetadataLockError(RuntimeError):
    """Raised when another ingestion process owns the metadata database."""


class DuckDBMetadataStore:
    """Store source, run, shard, and audit metadata, never raw payloads.

    DuckDB is the control plane for ingestion.  MinIO remains the data plane;
    keeping those responsibilities separate lets us query progress cheaply
    without copying large text objects into the database.
    """

    def __init__(self, database_path: str | Path) -> None:
        """Open/create the metadata database and apply the initial schema."""
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.database_path.with_suffix(self.database_path.suffix + ".lock")
        self._lock_handle = self.lock_path.open("a+")
        try:
            # The OS releases this lock if the process crashes, so it cannot
            # become a stale PID file that blocks future ingestion forever.
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_handle.seek(0)
            self._lock_handle.truncate()
            self._lock_handle.write(str(os.getpid()))
            self._lock_handle.flush()
            self._connection = duckdb.connect(str(self.database_path))
            self._create_schema()
        except BlockingIOError as error:
            owner = self.lock_path.read_text(encoding="utf-8").strip() or "unknown"
            self._lock_handle.close()
            raise MetadataLockError(
                f"ingestion is already running (lock owner PID: {owner}); "
                "wait for it to finish or stop it safely"
            ) from error
        except duckdb.IOException as error:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            if "lock" in str(error).lower():
                raise MetadataLockError(
                    "the DuckDB metadata file is locked by another process; "
                    "stop the existing ingestion run safely and retry"
                ) from error
            raise
        except Exception:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            raise

    def close(self) -> None:
        """Release the DuckDB connection held by this repository instance."""
        self._connection.close()
        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        self._lock_handle.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """Commit all metadata changes together or roll them back together."""
        self._connection.execute("BEGIN TRANSACTION")
        try:
            yield
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def _create_schema(self) -> None:
        """Create idempotent tables for sources, runs, shards, and events."""
        self._connection.sql(
            """
            -- One row describes the current resumable state of each source.
            CREATE TABLE IF NOT EXISTS data_sources (
                source_id VARCHAR PRIMARY KEY,
                source_name VARCHAR NOT NULL UNIQUE,
                source_type VARCHAR NOT NULL,
                source_uri VARCHAR,
                license_name VARCHAR,
                license_url VARCHAR,
                source_notes VARCHAR,
                dataset_id VARCHAR,
                dataset_config VARCHAR,
                configured_quota_bytes UBIGINT,
                status VARCHAR NOT NULL,
                current_version_id VARCHAR,
                current_run_id VARCHAR,
                downloaded_bytes UBIGINT NOT NULL DEFAULT 0,
                verified_bytes UBIGINT NOT NULL DEFAULT 0,
                completed_shard_count UBIGINT NOT NULL DEFAULT 0,
                checkpoint_type VARCHAR,
                checkpoint_value JSON,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                last_started_at TIMESTAMP,
                last_completed_at TIMESTAMP,
                last_error VARCHAR
            );

            -- Runs preserve history when a source is retried or versioned.
            CREATE TABLE IF NOT EXISTS download_runs (
                run_id VARCHAR PRIMARY KEY,
                source_id VARCHAR NOT NULL,
                version_id VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                started_at TIMESTAMP NOT NULL,
                paused_at TIMESTAMP,
                resumed_at TIMESTAMP,
                completed_at TIMESTAMP,
                failed_at TIMESTAMP,
                downloaded_bytes UBIGINT NOT NULL DEFAULT 0,
                verified_bytes UBIGINT NOT NULL DEFAULT 0,
                shard_count UBIGINT NOT NULL DEFAULT 0,
                retry_count INTEGER NOT NULL DEFAULT 0,
                error_code VARCHAR,
                error_message VARCHAR
            );

            -- A shard is counted only after the object-store verification step.
            CREATE TABLE IF NOT EXISTS raw_shards (
                shard_id VARCHAR PRIMARY KEY,
                run_id VARCHAR NOT NULL,
                source_id VARCHAR NOT NULL,
                version_id VARCHAR NOT NULL,
                shard_sequence UBIGINT NOT NULL,
                status VARCHAR NOT NULL,
                local_path VARCHAR,
                minio_bucket VARCHAR,
                minio_object_key VARCHAR,
                source_start_offset VARCHAR,
                source_end_offset VARCHAR,
                record_count UBIGINT,
                uncompressed_size_bytes UBIGINT,
                stored_size_bytes UBIGINT,
                checksum_algorithm VARCHAR,
                checksum_value VARCHAR,
                minio_etag VARCHAR,
                download_started_at TIMESTAMP,
                download_completed_at TIMESTAMP,
                upload_started_at TIMESTAMP,
                upload_completed_at TIMESTAMP,
                verified_at TIMESTAMP,
                retry_count INTEGER NOT NULL DEFAULT 0,
                error_message VARCHAR,
                UNIQUE (run_id, shard_sequence)
            );

            -- Events provide an append-only operational trail for recovery.
            CREATE TABLE IF NOT EXISTS download_events (
                event_id VARCHAR PRIMARY KEY,
                run_id VARCHAR,
                source_id VARCHAR,
                shard_id VARCHAR,
                event_type VARCHAR NOT NULL,
                event_level VARCHAR NOT NULL,
                message VARCHAR,
                event_data JSON,
                created_at TIMESTAMP NOT NULL
            );

            ALTER TABLE data_sources ADD COLUMN IF NOT EXISTS license_name VARCHAR;
            ALTER TABLE data_sources ADD COLUMN IF NOT EXISTS license_url VARCHAR;
            ALTER TABLE data_sources ADD COLUMN IF NOT EXISTS source_notes VARCHAR;
            ALTER TABLE data_sources ADD COLUMN IF NOT EXISTS dataset_id VARCHAR;
            ALTER TABLE data_sources ADD COLUMN IF NOT EXISTS dataset_config VARCHAR;
            """
        )

    def register_source(
        self,
        source_id: str,
        source_name: str,
        source_type: str,
        source_uri: str | None,
        configured_quota_bytes: int,
        checkpoint_type: str | None = None,
        license_name: str = "",
        license_url: str = "",
        source_notes: str = "",
        dataset_id: str | None = None,
        dataset_config: str | None = None,
    ) -> None:
        """Register or refresh source configuration without resetting progress."""
        now = _now()
        self._connection.execute(
            """
            INSERT INTO data_sources (
                source_id, source_name, source_type, source_uri,
                license_name, license_url, source_notes, dataset_id, dataset_config,
                configured_quota_bytes, status, checkpoint_type, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?)
            ON CONFLICT (source_id) DO UPDATE SET
                source_name = excluded.source_name,
                source_type = excluded.source_type,
                source_uri = excluded.source_uri,
                license_name = excluded.license_name,
                license_url = excluded.license_url,
                source_notes = excluded.source_notes,
                dataset_id = excluded.dataset_id,
                dataset_config = excluded.dataset_config,
                configured_quota_bytes = excluded.configured_quota_bytes,
                checkpoint_type = excluded.checkpoint_type,
                updated_at = excluded.updated_at
            """,
            [
                source_id,
                source_name,
                source_type,
                source_uri,
                license_name,
                license_url,
                source_notes,
                dataset_id,
                dataset_config,
                configured_quota_bytes,
                checkpoint_type,
                now,
                now,
            ],
        )

    def create_run(self, run_id: str, source_id: str, version_id: str) -> None:
        """Create a run and atomically make it current for its source."""
        now = _now()
        with self._transaction():
            self._connection.execute(
                """
                INSERT INTO download_runs (run_id, source_id, version_id, status, started_at)
                VALUES (?, ?, ?, 'PENDING', ?)
                """,
                [run_id, source_id, version_id, now],
            )
            self._connection.execute(
                """
                UPDATE data_sources
                SET current_run_id = ?, current_version_id = ?, updated_at = ?
                WHERE source_id = ?
                """,
                [run_id, version_id, now, source_id],
            )

    def record_verified_shard(
        self,
        *,
        shard_id: str,
        run_id: str,
        source_id: str,
        version_id: str,
        shard_sequence: int,
        bucket: str,
        object_key: str,
        checksum: str,
        stored_size_bytes: int,
        downloaded_size_bytes: int | None = None,
        checksum_algorithm: str = "sha256",
        etag: str | None = None,
        checkpoint: str | None = None,
    ) -> None:
        """Atomically record a verified shard and advance totals/checkpoint.

        The duplicate guard makes retries idempotent: if the same logical shard
        was committed before a process interruption, replaying the operation
        does not double-count bytes or shard numbers.
        """
        now = _now()
        with self._transaction():
            existing = self._connection.execute(
                "SELECT status FROM raw_shards WHERE shard_id = ?", [shard_id]
            ).fetchone()
            if existing is not None:
                if existing[0] != "COMPLETED":
                    raise ValueError(f"shard already exists with status {existing[0]}")
                return
            downloaded_size_bytes = (
                stored_size_bytes if downloaded_size_bytes is None else downloaded_size_bytes
            )
            self._connection.execute(
                """
                INSERT INTO raw_shards (
                    shard_id, run_id, source_id, version_id, shard_sequence, status,
                    minio_bucket, minio_object_key, stored_size_bytes, checksum_algorithm,
                    checksum_value, minio_etag, upload_completed_at, verified_at
                ) VALUES (?, ?, ?, ?, ?, 'COMPLETED', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    shard_id,
                    run_id,
                    source_id,
                    version_id,
                    shard_sequence,
                    bucket,
                    object_key,
                    stored_size_bytes,
                    checksum_algorithm,
                    checksum,
                    etag,
                    now,
                    now,
                ],
            )
            self._connection.execute(
                """
                UPDATE data_sources
                SET verified_bytes = verified_bytes + ?,
                    downloaded_bytes = downloaded_bytes + ?,
                    completed_shard_count = completed_shard_count + 1,
                    checkpoint_value = COALESCE(?, checkpoint_value),
                    updated_at = ?
                WHERE source_id = ?
                """,
                [downloaded_size_bytes, stored_size_bytes, checkpoint, now, source_id],
            )
            self._connection.execute(
                """
                UPDATE download_runs
                SET verified_bytes = verified_bytes + ?,
                    downloaded_bytes = downloaded_bytes + ?,
                    shard_count = shard_count + 1
                WHERE run_id = ?
                """,
                [downloaded_size_bytes, stored_size_bytes, run_id],
            )

    def set_source_status(
        self,
        source_id: str,
        status: str,
        *,
        checkpoint: str | None = None,
        error: str | None = None,
    ) -> None:
        """Update source state while retaining its last known checkpoint."""
        status = self._validated_status("data_sources", "source_id", source_id, status)
        now = _now()
        self._connection.execute(
            """
            UPDATE data_sources
            SET status = ?, checkpoint_value = COALESCE(?, checkpoint_value),
                last_error = ?, updated_at = ?,
                last_started_at = CASE WHEN ? = 'RUNNING' THEN ? ELSE last_started_at END,
                last_completed_at = CASE WHEN ? = 'COMPLETED' THEN ? ELSE last_completed_at END
            WHERE source_id = ?
            """,
            [status, checkpoint, error, now, status, now, status, now, source_id],
        )

    def set_run_status(self, run_id: str, status: str, error_message: str | None = None) -> None:
        """Update run lifecycle timestamps without rewriting run history."""
        status = self._validated_status("download_runs", "run_id", run_id, status)
        now = _now()
        self._connection.execute(
            """
            UPDATE download_runs
            SET status = ?, error_message = ?,
                completed_at = CASE WHEN ? = 'COMPLETED' THEN ? ELSE completed_at END,
                paused_at = CASE WHEN ? = 'PAUSED' THEN ? ELSE paused_at END,
                failed_at = CASE WHEN ? = 'FAILED' THEN ? ELSE failed_at END
            WHERE run_id = ?
            """,
            [status, error_message, status, now, status, now, status, now, run_id],
        )

    def _validated_status(self, table: str, key: str, value: str, status: str) -> str:
        """Validate every persisted lifecycle transition at one boundary."""
        row = self._connection.execute(
            f"SELECT status FROM {table} WHERE {key} = ?", [value]
        ).fetchone()
        if row is None:
            raise KeyError(value)
        current = SourceState(row[0])
        target = SourceState(status)
        if current != target:
            transition_state(current, target)
        return str(target)

    def source_quota_reached(self, source_id: str) -> bool:
        """Return whether verified source bytes satisfy the configured quota."""
        row = self._connection.execute(
            """
            SELECT configured_quota_bytes IS NOT NULL
               AND verified_bytes >= configured_quota_bytes
            FROM data_sources WHERE source_id = ?
            """,
            [source_id],
        ).fetchone()
        if row is None:
            raise KeyError(source_id)
        return bool(row[0])

    def get_source(self, source_id: str) -> dict[str, Any]:
        """Return one source row as a name-keyed mapping for orchestration."""
        row = self._connection.execute(
            "SELECT * FROM data_sources WHERE source_id = ?", [source_id]
        ).fetchone()
        if row is None:
            raise KeyError(source_id)
        columns = [description[0] for description in self._connection.description]
        return dict(zip(columns, row, strict=True))

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Return one historical run row as a name-keyed mapping."""
        row = self._connection.execute(
            "SELECT * FROM download_runs WHERE run_id = ?", [run_id]
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        columns = [description[0] for description in self._connection.description]
        return dict(zip(columns, row, strict=True))

    def list_run_shards(self, run_id: str) -> list[dict[str, Any]]:
        """List completed shards in upload order for verification/manifesting."""
        result = self._connection.execute(
            """
            SELECT shard_sequence, minio_bucket, minio_object_key,
                   stored_size_bytes, checksum_algorithm, checksum_value, minio_etag
            FROM raw_shards
            WHERE run_id = ? AND status = 'COMPLETED'
            ORDER BY shard_sequence
            """,
            [run_id],
        )
        columns = [description[0] for description in result.description]
        return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]

    def list_sources(self) -> list[dict[str, Any]]:
        """Return all registered sources in deterministic order."""
        result = self._connection.execute("SELECT * FROM data_sources ORDER BY source_id")
        columns = [description[0] for description in result.description]
        return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]

    def record_event(
        self,
        *,
        event_type: str,
        event_level: str,
        message: str,
        run_id: str | None = None,
        source_id: str | None = None,
        shard_id: str | None = None,
        event_data: str | None = None,
    ) -> str:
        """Append one structured operational event and return its event ID."""
        event_id = str(uuid4())
        self._connection.execute(
            """
            INSERT INTO download_events (
                event_id, run_id, source_id, shard_id, event_type,
                event_level, message, event_data, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                event_id,
                run_id,
                source_id,
                shard_id,
                event_type,
                event_level,
                message,
                event_data,
                _now(),
            ],
        )
        return event_id
