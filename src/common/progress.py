"""Live terminal progress bars for ingestion."""

from dataclasses import dataclass
from typing import TextIO

from tqdm import tqdm


@dataclass
class ProgressReporter:
    """Display source quota and current-shard download progress."""

    enabled: bool = True
    stream: TextIO | None = None
    target_shard_size_bytes: int = 0

    def __post_init__(self) -> None:
        self._source_bar: tqdm | None = None
        self._shard_bar: tqdm | None = None
        self._resume_bar: tqdm | None = None
        self._current_shard_sequence: int | None = None
        self._last_shard_size = 0
        self._last_resume_record = 0

    def start(
        self,
        source_id: str,
        quota_bytes: int,
        initial_verified_bytes: int = 0,
        initial_shard_sequence: int = 0,
    ) -> None:
        """Open the overall quota bar from persisted verified progress."""
        if not self.enabled:
            return
        self._source_bar = tqdm(
            total=quota_bytes,
            initial=initial_verified_bytes,
            desc=f"{source_id} total (staged)",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
            file=self.stream,
            position=0,
        )
        if initial_shard_sequence:
            self._source_bar.set_postfix(shard=initial_shard_sequence)

    def start_shard(self, shard_sequence: int) -> None:
        """Open a live bar for the shard currently being built."""
        if not self.enabled:
            return
        self._close_resume_bar()
        if self._shard_bar is not None:
            self._shard_bar.close()
        self._last_shard_size = 0
        self._current_shard_sequence = shard_sequence
        self._shard_bar = tqdm(
            total=self.target_shard_size_bytes or None,
            desc=f"  shard-{shard_sequence:06d} download",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
            file=self.stream,
            position=1,
            leave=False,
        )

    def update_resume(self, current_record: int, total_records: int) -> None:
        """Show progress while converting a legacy record-index checkpoint."""
        if not self.enabled:
            return
        if self._resume_bar is None:
            self._last_resume_record = 0
            self._resume_bar = tqdm(
                total=total_records,
                desc="  legacy checkpoint scan (one time)",
                unit="record",
                unit_scale=True,
                dynamic_ncols=True,
                file=self.stream,
                position=1,
                leave=False,
            )
        assert self._resume_bar is not None
        self._resume_bar.update(max(current_record - self._last_resume_record, 0))
        self._last_resume_record = current_record
        if current_record >= total_records:
            self._close_resume_bar()

    def _close_resume_bar(self) -> None:
        """Close the temporary legacy-resume bar when normal ingestion starts."""
        if self._resume_bar is not None:
            self._resume_bar.close()
            self._resume_bar = None
            self._last_resume_record = 0

    def update_shard(self, shard_sequence: int, size_bytes: int) -> None:
        """Advance the current shard bar to the latest staged byte count."""
        if not self.enabled:
            return
        if self._shard_bar is None or self._current_shard_sequence != shard_sequence:
            self.start_shard(shard_sequence)
        delta = max(size_bytes - self._last_shard_size, 0)
        assert self._shard_bar is not None
        self._shard_bar.update(delta)
        # The source bar includes the current local shard so it moves while
        # downloading. The metadata layer still counts these bytes only after
        # MinIO upload and verification succeed.
        if self._source_bar is not None:
            self._source_bar.update(delta)
        self._last_shard_size = size_bytes

    def complete_shard(self, size_bytes: int, shard_sequence: int) -> None:
        """Close the shard bar and advance the verified source bar."""
        if not self.enabled:
            return
        self.update_shard(shard_sequence, size_bytes)
        if self._shard_bar is not None:
            self._shard_bar.close()
            self._shard_bar = None
        self._close_resume_bar()
        self._current_shard_sequence = None
        if self._source_bar is not None:
            self._source_bar.set_postfix(shard=shard_sequence)

    def finish(self, source_id: str, verified_bytes: int, shard_sequence: int) -> None:
        """Close progress bars and print the final source state."""
        if not self.enabled:
            return
        if self._shard_bar is not None:
            self._shard_bar.close()
            self._shard_bar = None
        self._current_shard_sequence = None
        if self._source_bar is not None:
            self._source_bar.n = verified_bytes
            self._source_bar.set_postfix(shard=shard_sequence, status="completed")
            self._source_bar.refresh()
            self._source_bar.close()
            self._source_bar = None

    def close(self) -> None:
        """Close and reset every bar after a failed or interrupted source."""
        if self._shard_bar is not None:
            self._shard_bar.close()
            self._shard_bar = None
        self._close_resume_bar()
        if self._source_bar is not None:
            self._source_bar.close()
            self._source_bar = None
        self._current_shard_sequence = None

    def clear(self) -> None:
        """Clear live bars before printing an error or a normal log line."""
        if self._shard_bar is not None:
            self._shard_bar.clear()
        if self._resume_bar is not None:
            self._resume_bar.clear()
        if self._source_bar is not None:
            self._source_bar.clear()
