"""Deduplication boundary."""


def deduplicate() -> None:
    """Placeholder for exact/approximate deduplication after raw ingestion.

    Deduplication is intentionally downstream of immutable raw storage so a
    cleaning policy can change without re-downloading licensed source data.
    """
    raise NotImplementedError("Deduplication is planned for Phase 2")
