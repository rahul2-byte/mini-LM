from datetime import UTC, datetime

import pytest

from data_ingestion.ingestion import (
    SourceState,
    build_object_key,
    decode_checkpoint,
    encode_checkpoint,
    transition_state,
)


def test_state_transition_allows_pause_request() -> None:
    assert transition_state(SourceState.RUNNING, SourceState.PAUSE_REQUESTED) == (
        SourceState.PAUSE_REQUESTED
    )


def test_state_transition_rejects_completed_resume() -> None:
    with pytest.raises(ValueError, match="invalid source state transition"):
        transition_state(SourceState.COMPLETED, SourceState.RUNNING)


def test_object_key_is_deterministic_and_versioned() -> None:
    key = build_object_key(
        source_id="fineweb_edu",
        dataset_name="fineweb-edu",
        version_id="2026-07-14T10-30-00Z-ab12cd34",
        ingestion_date="2026-07-14",
        run_id="ab12cd34",
        shard_sequence=7,
        extension="jsonl.zst",
    )
    assert key == (
        "fineweb_edu/fineweb-edu/version=2026-07-14T10-30-00Z-ab12cd34/"
        "ingestion_date=2026-07-14/run_id=ab12cd34/shard-000007.jsonl.zst"
    )


def test_checkpoint_round_trip_is_json_safe() -> None:
    checkpoint = {"cursor": "page-12", "offset": 1024, "updated_at": datetime.now(UTC).isoformat()}
    assert decode_checkpoint(encode_checkpoint(checkpoint)) == checkpoint
