"""Bounded local staging checks."""

from pathlib import Path
import shutil


class DiskSpaceError(RuntimeError):
    """Raised before ingestion would exceed local staging limits."""


def staging_usage_bytes(path: str | Path) -> int:
    """Return the total size of files currently occupying the staging tree."""
    root = Path(path)
    if not root.exists():
        return 0
    return sum(file.stat().st_size for file in root.rglob("*") if file.is_file())


def ensure_disk_capacity(
    path: str | Path,
    *,
    minimum_free_space_bytes: int,
    maximum_staging_usage_bytes: int,
) -> None:
    """Fail before a new shard starts when local disk guardrails are unsafe.

    The check is intentionally performed before each shard rather than only at
    process startup because uploads and temporary files change usage over time.
    """
    usage = staging_usage_bytes(path)
    if usage >= maximum_staging_usage_bytes:
        raise DiskSpaceError(f"staging usage limit exceeded: {usage} bytes")
    # Check the filesystem containing staging.  Creating a missing staging
    # directory is handled by the caller, so its parent is always available.
    free = shutil.disk_usage(Path(path).parent).free
    if free < minimum_free_space_bytes:
        raise DiskSpaceError(f"minimum free disk space not available: {free} bytes")
