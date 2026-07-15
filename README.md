# Local LLM

Local, reproducible decoder-only language-model pretraining platform.

The smoke workflow validates streaming cleaning, tokenizer training, token
packing, random-initialized GPT training, checkpoint resume, and evaluation.

```bash
uv sync --extra ingest --extra train
uv run pytest
```

## Package boundaries

The source tree uses independent top-level packages:

```text
src/
  data_ingestion/  # source adapters, MinIO, DuckDB, sharding, manifests
  tokenization/    # tokenizer training and encoding
  model/           # GPT model definitions
  training/        # packing, checkpoints, optimization
  evaluation/      # loss, perplexity, generation
  common/          # small shared utilities only
```

Data ingestion does not import model, training, or evaluation code.
The handoff between modules is through files and MinIO objects rather than
Python-level imports.

Module commands:

```bash
uv run train --config configs/data.yaml smoke-train
uv run data-ingest --config configs/data.yaml ingest-local
uv run data-ingest --config configs/data.yaml ingest
```
