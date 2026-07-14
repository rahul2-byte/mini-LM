from pathlib import Path

import pytest
import torch

from training.pack import PackedDataset, pack_token_ids


def test_pack_drops_incomplete_tail_and_reads_shifted_examples(tmp_path: Path) -> None:
    data_path = tmp_path / "train.bin"
    metadata_path = tmp_path / "train.json"

    metadata = pack_token_ids(
        [0, 1, 2, 3, 4], data_path, metadata_path, sequence_length=2, vocab_size=8
    )

    assert metadata.num_blocks == 2
    assert metadata.num_tokens == 4
    dataset = PackedDataset(data_path, metadata_path)
    inputs, targets = dataset[0]
    assert torch.equal(inputs, torch.tensor([0, 1]))
    assert torch.equal(targets, torch.tensor([1, 2]))


def test_pack_rejects_token_outside_vocabulary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside vocabulary"):
        pack_token_ids(
            [0, 9], tmp_path / "x.bin", tmp_path / "x.json", sequence_length=2, vocab_size=8
        )


def test_packed_dataset_rejects_corrupt_metadata(tmp_path: Path) -> None:
    data_path = tmp_path / "train.bin"
    metadata_path = tmp_path / "train.json"
    pack_token_ids([0, 1, 2], data_path, metadata_path, sequence_length=2, vocab_size=8)
    metadata_path.write_text('{"sequence_length": 3}', encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        PackedDataset(data_path, metadata_path)
