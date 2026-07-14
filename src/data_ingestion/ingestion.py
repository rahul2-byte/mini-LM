"""Pure ingestion primitives shared by adapters and the orchestrator."""

from enum import StrEnum
import json
import re
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


class SourceState(StrEnum):
    """Persisted lifecycle states for one sequential source run."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSE_REQUESTED = "PAUSE_REQUESTED"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    CANCELLED = "CANCELLED"


_ALLOWED_TRANSITIONS: dict[SourceState, frozenset[SourceState]] = {
    # Keeping this transition graph explicit prevents a restarted process from
    # accidentally changing a completed source back to an active state.
    SourceState.PENDING: frozenset({SourceState.RUNNING, SourceState.CANCELLED}),
    SourceState.RUNNING: frozenset(
        {
            SourceState.PAUSE_REQUESTED,
            SourceState.COMPLETED,
            SourceState.FAILED,
            SourceState.RETRYING,
            SourceState.CANCELLED,
        }
    ),
    SourceState.PAUSE_REQUESTED: frozenset({SourceState.PAUSED, SourceState.FAILED}),
    SourceState.PAUSED: frozenset({SourceState.RUNNING, SourceState.CANCELLED}),
    SourceState.RETRYING: frozenset(
        {SourceState.RUNNING, SourceState.FAILED, SourceState.CANCELLED}
    ),
    SourceState.COMPLETED: frozenset(),
    SourceState.FAILED: frozenset({SourceState.RETRYING, SourceState.CANCELLED}),
    SourceState.CANCELLED: frozenset(),
}

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._=:-]+$")


def transition_state(current: SourceState, target: SourceState) -> SourceState:
    """Validate and return a source lifecycle transition."""
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"invalid source state transition: {current} -> {target}")
    return target


def _validate_component(value: str, field_name: str) -> str:
    """Validate one user/source-derived path component.

    Object keys are not filesystem paths, but rejecting slashes and traversal
    syntax still prevents ambiguous keys and makes source names safe to audit.
    """
    if not value or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"invalid {field_name}: {value!r}")
    return value


def build_object_key(
    *,
    source_id: str,
    dataset_name: str,
    version_id: str,
    ingestion_date: str,
    run_id: str,
    shard_sequence: int,
    extension: str,
) -> str:
    """Build a stable raw object key without accepting path traversal."""
    if shard_sequence < 0:
        raise ValueError("shard_sequence must be non-negative")
    extension = extension.lstrip(".")
    components = {
        "source_id": source_id,
        "dataset_name": dataset_name,
        "version_id": version_id,
        "ingestion_date": ingestion_date,
        "run_id": run_id,
        "extension": extension,
    }
    for field_name, value in components.items():
        _validate_component(value, field_name)
    return (
        f"{source_id}/{dataset_name}/version={version_id}/"
        f"ingestion_date={ingestion_date}/run_id={run_id}/"
        f"shard-{shard_sequence:06d}.{extension}"
    )


def build_manifest_key(source_id: str, version_id: str, run_id: str) -> str:
    """Return the deterministic object key for a completed run manifest."""
    for value in (source_id, version_id, run_id):
        _validate_component(value, "manifest component")
    return f"{source_id}/version={version_id}/run_id={run_id}/manifest.json"


def encode_checkpoint(checkpoint: dict[str, Any]) -> str:
    """Serialize adapter checkpoint state for DuckDB JSON columns.

    Sorted keys and compact separators make the stored representation stable,
    which helps compare checkpoints in logs and tests.
    """
    return json.dumps(checkpoint, sort_keys=True, separators=(",", ":"))


def decode_checkpoint(value: str | dict[str, Any] | None) -> dict[str, Any]:
    """Decode a stored checkpoint and require a JSON object at the boundary."""
    if value is None:
        return {}
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, dict):
        raise ValueError("checkpoint must decode to a JSON object")
    return decoded


def create_version_id() -> str:
    """Create a UTC, sortable, collision-resistant raw-data version ID."""
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f"{timestamp}-{uuid4().hex[:8]}"
