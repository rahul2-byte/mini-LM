import gzip
import io
from itertools import islice
import json
from urllib.request import Request

import datasets

import data_ingestion.adapters as adapters
from data_ingestion.adapters import HuggingFaceStreamingAdapter
from data_ingestion.ingestion import decode_checkpoint, encode_checkpoint


def _streaming_dataset() -> datasets.IterableDataset:
    return datasets.Dataset.from_dict(
        {"text": ["zero", "one", "two", "three"]}
    ).to_iterable_dataset()


def test_hugging_face_adapter_resumes_from_native_dataset_state(monkeypatch) -> None:
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: _streaming_dataset())
    adapter = HuggingFaceStreamingAdapter("test/dataset")

    first_records = list(islice(adapter.stream_records({}), 2))
    checkpoint = first_records[-1].checkpoint
    resumed_record = next(adapter.stream_records(checkpoint))

    assert "dataset_state" in checkpoint
    assert decode_checkpoint(encode_checkpoint(checkpoint)) == checkpoint
    assert json.loads(resumed_record.payload) == {"text": "two"}


def test_hugging_face_adapter_reports_one_time_legacy_resume_scan(monkeypatch) -> None:
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: _streaming_dataset())
    progress: list[tuple[int, int]] = []
    adapter = HuggingFaceStreamingAdapter(
        "test/dataset",
        resume_progress_callback=lambda current, total: progress.append((current, total)),
    )

    resumed_record = next(adapter.stream_records({"record_index": 2}))

    assert json.loads(resumed_record.payload) == {"text": "two"}
    assert progress == [(1, 2), (2, 2)]


def test_dolma_manifest_adapter_streams_gzipped_jsonl(monkeypatch) -> None:
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb") as output:
        output.write(b'{"text":"first"}\n{"text":"second"}\n')

    def fake_urlopen(url, timeout):
        assert isinstance(url, Request)
        value = getattr(url, "full_url", url)
        if value == "https://example.test/manifest.txt":
            return io.BytesIO(b"https://example.test/shard.json.gz\n")
        assert value == "https://example.test/shard.json.gz"
        return io.BytesIO(compressed.getvalue())

    monkeypatch.setattr("data_ingestion.adapters.urlopen", fake_urlopen)
    adapter = adapters.DolmaManifestAdapter("https://example.test/manifest.txt", max_bytes=100)

    records = list(adapter.stream_records({}))

    assert [json.loads(record.payload) for record in records] == [
        {"text": "first"},
        {"text": "second"},
    ]
