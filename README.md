# app

Local, reproducible decoder-only language-model pretraining platform.

Phase 1 validates the pipeline on a small UTF-8 `.txt` corpus with documents
separated by blank lines. It covers configuration loading, raw-file manifesting,
streaming cleaning, tokenizer training from scratch, and token encoding.

```bash
uv sync --extra dev
uv run pytest
uv run app --config configs/data.yaml manifest data/sample.txt
```

## Package boundaries

The source tree uses independent top-level packages:

```text
src/
  data_ingestion/  # source adapters, MinIO, DuckDB, sharding, manifests
  tokenization/    # tokenizer training and encoding
  model/           # GPT model definitions
  training/        # packing, datasets, checkpoints, optimization
  evaluation/      # loss, perplexity, generation
  serving/         # local inference API boundary
  common/          # small shared utilities only
  app/           # application CLI and orchestration entry point
```

Data ingestion does not import model, training, evaluation, or serving code.
The handoff between modules is through files and MinIO objects rather than
Python-level imports.
