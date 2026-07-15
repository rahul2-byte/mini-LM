"""MinIO object storage boundary with post-upload verification."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from minio import Minio


@dataclass(frozen=True)
class ObjectMetadata:
    """Metadata returned after an object-store upload or HEAD request."""

    bucket: str
    object_key: str
    size_bytes: int
    etag: str


class ObjectStore(Protocol):
    """Minimal storage contract used by ingestion and easy to fake in tests."""

    def upload_file(
        self, local_path: str | Path, bucket: str, object_key: str
    ) -> ObjectMetadata: ...

    def stat(self, bucket: str, object_key: str) -> ObjectMetadata: ...


class MinioObjectStore:
    """MinIO implementation that verifies object size after every upload."""

    def __init__(self, client: Minio) -> None:
        """Wrap an already configured client so credentials stay outside code."""
        self.client = client

    def upload_file(self, local_path: str | Path, bucket: str, object_key: str) -> ObjectMetadata:
        """Upload one local file and reject a size-mismatched remote object."""
        result = self.client.fput_object(bucket, object_key, str(local_path))
        metadata = self.client.stat_object(bucket, object_key)
        # ETags are not reliable content hashes for multipart uploads; size is
        # still a cheap mandatory sanity check before metadata is committed.
        if metadata.size != Path(local_path).stat().st_size:
            raise IOError(f"MinIO size mismatch for {bucket}/{object_key}")
        etag = result.etag or metadata.etag
        if not etag:
            raise IOError(f"MinIO returned no ETag for {bucket}/{object_key}")
        return ObjectMetadata(bucket, object_key, metadata.size, etag)

    def stat(self, bucket: str, object_key: str) -> ObjectMetadata:
        """Return remote metadata without downloading the object."""
        result: Any = self.client.stat_object(bucket, object_key)
        if not result.etag:
            raise IOError(f"MinIO returned no ETag for {bucket}/{object_key}")
        return ObjectMetadata(bucket, object_key, result.size, result.etag)
