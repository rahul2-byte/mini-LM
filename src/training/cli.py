"""Command-line entry points for model training workflows."""

import argparse
from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from common.logging import configure_logging
from data_ingestion.clean import require_documents
from evaluation.generation import generate_text
from model.config import GPTConfig
from model.gpt import GPTModel
from tokenization.encode import encode_documents, write_token_ids
from tokenization.train_tokenizer import train_bpe
from training.checkpoint import CheckpointManager
from training.pack import PackedDataset, pack_token_ids
from training.trainer import Trainer, TrainerConfig


def run_smoke(
    sample_path: str | Path, output_dir: str | Path, sequence_length: int = 32, max_steps: int = 3
) -> dict[str, Any]:
    """Run the local tokenizer-to-checkpoint training smoke workflow."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # The smoke fixture is deliberately tiny, so materializing its documents is
    # acceptable here; production ingestion uses the streaming iterator.
    documents = list(require_documents(sample_path, min_chars=20))
    if len(documents) < 2:
        raise ValueError("smoke training requires at least two valid documents")
    train_documents, validation_documents = documents[:-1], documents[-1:]

    tokenizer_path = output / "tokenizer.json"
    tokenizer = train_bpe(train_documents, tokenizer_path, vocab_size=256)
    train_ids = output / "train.ids"
    validation_ids = output / "validation.ids"
    write_token_ids(encode_documents(train_documents, tokenizer), train_ids)
    write_token_ids(encode_documents(validation_documents, tokenizer), validation_ids)

    def iter_ids(path: Path) -> Iterator[int]:
        """Stream newline-delimited integer IDs from a tokenization artifact."""
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                yield from (int(value) for value in line.split())

    train_pack = output / "train.bin"
    train_meta = output / "train.json"
    validation_pack = output / "validation.bin"
    validation_meta = output / "validation.json"
    pack_token_ids(
        iter_ids(train_ids), train_pack, train_meta, sequence_length, tokenizer.get_vocab_size()
    )
    pack_token_ids(
        iter_ids(validation_ids),
        validation_pack,
        validation_meta,
        sequence_length,
        tokenizer.get_vocab_size(),
    )

    # Deliberately small debugging model, not the 20M+ scaling rung.
    model_config = GPTConfig(
        vocab_size=tokenizer.get_vocab_size(),
        context_length=sequence_length,
        n_layers=2,
        n_heads=4,
        embedding_dim=64,
    )
    model = GPTModel(model_config)
    trainer_config = TrainerConfig(
        max_steps=max_steps,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        precision="fp32",
        eval_every_steps=1,
        checkpoint_every_steps=max_steps,
    )
    manager = CheckpointManager(output / "checkpoints", keep_last=3)
    trainer = Trainer(
        model, trainer_config, model_config, manager, tokenizer_path, torch.device("cpu")
    )
    train_loader = DataLoader(PackedDataset(train_pack, train_meta), batch_size=2, shuffle=False)
    validation_loader = DataLoader(
        PackedDataset(validation_pack, validation_meta), batch_size=2, shuffle=False
    )
    metrics: dict[str, Any] = trainer.train(train_loader, validation_loader)

    checkpoint_path = output / "checkpoints" / f"checkpoint-{max_steps:08d}.pt"
    # Loading into a fresh model proves the checkpoint is self-contained.
    resumed = Trainer(
        GPTModel(model_config),
        trainer_config,
        model_config,
        manager,
        tokenizer_path,
        torch.device("cpu"),
    )
    resumed.resume(checkpoint_path)
    metrics.update(
        {
            "global_step": trainer.global_step,
            "checkpoint_path": str(checkpoint_path),
            "sample": generate_text(model, tokenizer, train_documents[0][:40], 12, seed=42),
        }
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metrics


def main() -> None:
    """Run training-only commands without importing ingestion infrastructure."""
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("command", choices=["smoke-train"])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/smoke"))
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=3)
    args = parser.parse_args()
    configure_logging()
    if args.command == "smoke-train":
        print(
            json.dumps(
                run_smoke(
                    load_sample_path(args.config),
                    args.output_dir,
                    args.sequence_length,
                    args.max_steps,
                ),
                sort_keys=True,
            )
        )


def load_sample_path(config_path: str | Path) -> Path:
    """Resolve the sample fixture through the shared data configuration."""
    from data_ingestion.config import load_data_config

    return load_data_config(config_path).sample_path


if __name__ == "__main__":
    main()
