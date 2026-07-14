from pathlib import Path

from tokenization.train_tokenizer import load_tokenizer, train_bpe


def test_tokenizer_smoke_training(tmp_path: Path) -> None:
    output = tmp_path / "tokenizer.json"
    train_bpe(
        ["the quick brown fox", "the quick blue bird", "small local language model"],
        output,
        vocab_size=64,
    )
    assert output.is_file()
    restored = load_tokenizer(output)
    encoded = restored.encode("the quick fox")
    assert encoded.ids
    assert all(isinstance(token_id, int) for token_id in encoded.ids)
