from pathlib import Path

from data_ingestion.adapters import SourceRecord
from data_ingestion.sharding import ShardBuilder


def test_shard_builder_preserves_records_and_writes_ready_files(tmp_path: Path) -> None:
    records = [
        SourceRecord(b"one", {"byte_offset": 4}),
        SourceRecord(b"two", {"byte_offset": 8}),
        SourceRecord(b"three", {"byte_offset": 14}),
    ]
    shards = list(ShardBuilder(tmp_path, target_size_bytes=8, maximum_size_bytes=20).build(records))

    assert len(shards) == 2
    assert shards[0].path.name == "shard-000001.ready"
    assert shards[0].path.read_bytes() == b"one\ntwo\n"
    assert shards[0].checkpoint == {"byte_offset": 8}
    assert shards[1].path.read_bytes() == b"three\n"
