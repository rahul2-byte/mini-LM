"""Command-line entry points for the local smoke pipeline."""

import argparse
import json
import os
import signal
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any

from minio import Minio
import torch
from torch.utils.data import DataLoader

from data_ingestion.config import DataSourceConfig, load_data_config
from common.logging import configure_logging
from data_ingestion.clean import require_documents
from data_ingestion.adapters import adapter_for_source
from data_ingestion.ingestion import build_manifest_key
from data_ingestion.manifest import ManifestStore
from data_ingestion.manifest_json import write_run_manifest
from data_ingestion.metadata import DuckDBMetadataStore
from data_ingestion.orchestrator import IngestionPipeline
from training.pack import PackedDataset, pack_token_ids
from data_ingestion.reconciliation import reconcile_startup
from data_ingestion.storage import MinioObjectStore
from evaluation.generation import generate_text
from model.config import GPTConfig
from model.gpt import GPTModel
from tokenization.encode import encode_documents, write_token_ids
from tokenization.train_tokenizer import train_bpe
from training.checkpoint import CheckpointManager
from training.trainer import Trainer, TrainerConfig


def run_smoke(
    sample_path: str | Path, output_dir: str | Path, sequence_length: int = 32, max_steps: int = 3
) -> dict[str, Any]:
    """Run the complete local tokenizer-to-checkpoint smoke workflow.

    This intentionally uses the small fixture and CPU-friendly dimensions. It
    exercises the same boundaries as the larger run: clean text, tokenizer,
    token IDs, packed examples, random-init GPT, evaluation, checkpoint load,
    and generation.
    """
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
    # This is a deliberately small debugging model, not the 20M+ scaling rung.
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
    metrics = trainer.train(train_loader, validation_loader)
    checkpoint_path = output / "checkpoints" / f"checkpoint-{max_steps:08d}.pt"
    # Loading into a fresh model proves the checkpoint contains more than an
    # in-memory reference to the original trainer.
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


def run_ingestion(config_path: str | Path, source_id: str | None = None) -> dict[str, str]:
    """Run selected configured sources sequentially against local MinIO."""
    config = load_data_config(config_path)
    if config.ingestion is None:
        raise ValueError("ingestion configuration is required")
    # Credentials remain environment-only; config files and logs never contain
    # the MinIO secret.
    username = os.environ.get("MINIO_ROOT_USER")
    password = os.environ.get("MINIO_ROOT_PASSWORD")
    if not username or not password:
        raise ValueError("MINIO_ROOT_USER and MINIO_ROOT_PASSWORD are required")
    endpoint = os.environ.get("MINIO_ENDPOINT", "127.0.0.1:9000")
    secure = os.environ.get("MINIO_SECURE", "0").lower() in {"1", "true", "yes"}
    client = Minio(endpoint, access_key=username, secret_key=password, secure=secure)
    # The CLI owns bucket bootstrap for local development. Production deploys
    # can pre-create the bucket with a least-privilege service account.
    if not client.bucket_exists(config.ingestion.raw_bucket):
        client.make_bucket(config.ingestion.raw_bucket)
    metadata = DuckDBMetadataStore(config.ingestion.metadata_path)
    object_store = MinioObjectStore(client)
    # Reconcile before scheduling new work so stale RUNNING rows cannot be
    # mistaken for healthy concurrent workers.
    reconcile_startup(metadata, object_store, config.ingestion.staging_directory)
    pause_event = Event()
    previous_handler = signal.getsignal(signal.SIGINT)
    # Ctrl-C requests a safe pause; the current shard finishes and is committed
    # before the loop exits.
    signal.signal(signal.SIGINT, lambda _signum, _frame: pause_event.set())
    pipeline = IngestionPipeline(
        metadata=metadata,
        object_store=object_store,
        staging_directory=config.ingestion.staging_directory,
        raw_bucket=config.ingestion.raw_bucket,
        target_shard_size_bytes=config.ingestion.target_shard_size_bytes,
        maximum_shard_size_bytes=config.ingestion.maximum_shard_size_bytes,
        minimum_free_space_bytes=config.ingestion.minimum_free_space_bytes,
        maximum_staging_usage_bytes=config.ingestion.maximum_staging_usage_bytes,
        pause_event=pause_event,
    )
    configured_sources = list(config.sources)
    if source_id == "local-sample":
        configured_sources = [
            DataSourceConfig(
                source_id="local-sample",
                source_type="filesystem",
                source_url=str(config.sample_path),
                license_name="user-provided",
                license_url="",
                max_bytes=config.sample_path.stat().st_size,
            )
        ]
    # Preserve YAML order: source processing is sequential by design and the
    # order is part of the reproducible ingestion plan.
    selected = [
        source
        for source in configured_sources
        if source.enabled and (source_id is None or source.source_id == source_id)
    ]
    if source_id is not None and not selected:
        raise ValueError(f"unknown or disabled source: {source_id}")
    results: dict[str, str] = {}
    try:
        for source in selected:
            try:
                state = pipeline.run_source(
                    source,
                    adapter_for_source(source),
                    ingestion_date=datetime.now(UTC).date().isoformat(),
                )
                results[source.source_id] = str(state)
                run_id = metadata.get_source(source.source_id)["current_run_id"]
                if state == "COMPLETED" and run_id:
                    # The local manifest is written first, then copied to MinIO
                    # under the versioned key after all shard records exist.
                    manifest = write_run_manifest(
                        metadata,
                        run_id,
                        config.project_root / "artifacts/raw-manifests" / f"{run_id}.json",
                    )
                    version_id = metadata.get_run(run_id)["version_id"]
                    client.fput_object(
                        config.ingestion.raw_bucket,
                        build_manifest_key(source.source_id, version_id, run_id),
                        str(manifest),
                    )
                if state == "PAUSED":
                    break
            except Exception as error:
                results[source.source_id] = f"FAILED: {error}"
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        metadata.close()
    return results


def main() -> None:
    """Parse CLI arguments and dispatch one explicit pipeline command."""
    parser = argparse.ArgumentParser(prog="app")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("command", choices=["manifest", "smoke-train", "ingest", "ingest-local"])
    parser.add_argument("path", type=Path, nargs="?")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/smoke"))
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--source-id", type=str)
    args = parser.parse_args()
    configure_logging()
    config = load_data_config(args.config)
    if args.command == "manifest":
        if args.path is None:
            parser.error("manifest requires a file path")
        store = ManifestStore(config.manifest_path)
        record = store.add_file(args.path, source="local-sample", license_name="user-provided")
        print(record)
    elif args.command == "smoke-train":
        metrics = run_smoke(
            config.sample_path, args.output_dir, args.sequence_length, args.max_steps
        )
        print(json.dumps(metrics, sort_keys=True))
    elif args.command == "ingest-local":
        results = run_ingestion(args.config, source_id="local-sample")
        print(json.dumps(results, sort_keys=True))
    else:
        print(json.dumps(run_ingestion(args.config, args.source_id), sort_keys=True))
