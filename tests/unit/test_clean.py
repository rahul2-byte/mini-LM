from pathlib import Path

from data_ingestion.clean import iter_documents


def test_cleaner_streams_blank_line_documents(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text(" first   document \ncontinued\n\nsecond document\n", encoding="utf-8")
    assert list(iter_documents(sample, min_chars=1)) == [
        "first document continued",
        "second document",
    ]
