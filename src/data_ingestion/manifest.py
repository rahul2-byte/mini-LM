"""SQLite-backed raw-file manifest."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from common.hashing import sha256_file


@dataclass(frozen=True)
class FileRecord:
    """Legacy smoke-manifest row describing one local raw file."""

    path: str
    source: str
    license: str
    sha256: str
    size_bytes: int
    status: str = "discovered"


class ManifestStore:
    """Small SQLite manifest retained for the original local smoke workflow."""

    def __init__(self, database_path: str | Path) -> None:
        """Create the parent directory and schema if needed."""
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY, source TEXT NOT NULL, license TEXT NOT NULL,
                    sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        """Open a short-lived connection so callers do not hold file locks."""
        return sqlite3.connect(self.database_path)

    def add_file(self, path: str | Path, source: str, license_name: str) -> FileRecord:
        """Hash and upsert one local file with its provenance metadata."""
        file_path = Path(path).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        record = FileRecord(
            str(file_path), source, license_name, sha256_file(file_path), file_path.stat().st_size
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO files(path, source, license, sha256, size_bytes, status) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.path,
                    record.source,
                    record.license,
                    record.sha256,
                    record.size_bytes,
                    record.status,
                ),
            )
        return record

    def list_files(self) -> list[FileRecord]:
        """Return manifest rows in stable path order."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT path, source, license, sha256, size_bytes, status FROM files ORDER BY path"
            ).fetchall()
        return [FileRecord(*row) for row in rows]
