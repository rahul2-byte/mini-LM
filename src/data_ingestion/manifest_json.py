"""Immutable JSON manifest generation for completed raw runs."""

import json
from pathlib import Path
from typing import Any

from data_ingestion.metadata import DuckDBMetadataStore


def write_run_manifest(
    metadata: DuckDBMetadataStore,
    run_id: str,
    output_path: str | Path,
) -> Path:
    """Write an atomic JSON manifest for a completed or inspected run.

    The temporary sibling is replaced only after serialization succeeds, so a
    killed process cannot leave a file named ``manifest.json`` that looks
    complete but contains truncated JSON.
    """
    run = metadata.get_run(run_id)
    shards = metadata.list_run_shards(run_id)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "source_id": run["source_id"],
        "version_id": run["version_id"],
        "status": run["status"],
        "started_at": str(run["started_at"]),
        "completed_at": str(run["completed_at"]) if run["completed_at"] else None,
        "total_shards": len(shards),
        "total_downloaded_bytes": run["downloaded_bytes"],
        "total_stored_bytes": run["verified_bytes"],
        "checksum_algorithm": "SHA-256",
        "shards": [
            {
                **shard,
                "checksum": shard.pop("checksum_value"),
            }
            for shard in shards
        ],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".inprogress")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return destination
