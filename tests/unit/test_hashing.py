from pathlib import Path

from common.hashing import sha256_file, sha256_text


def test_hashing_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")
    assert sha256_file(path) == sha256_text("hello")
    assert len(sha256_file(path)) == 64
