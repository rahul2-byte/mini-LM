"""Filesystem path helpers."""

from pathlib import Path


def ensure_project_dirs(root: str | Path) -> None:
    """Create runtime directories expected by the local pipeline.

    These directories are targets for generated data and artifacts, not Python
    package directories.  Creation is idempotent so every CLI entry point can
    safely call this during startup.
    """
    root_path = Path(root)
    for relative in (
        "data/raw",
        "data/clean",
        "data/tokenized",
        "data/packed",
        "data/manifests",
        "artifacts/tokenizers",
        "artifacts/checkpoints",
        "artifacts/eval_reports",
        "artifacts/samples",
        "logs",
    ):
        # ``parents=True`` handles a clean checkout, while ``exist_ok=True``
        # makes repeated runs safe after a previous interrupted run.
        (root_path / relative).mkdir(parents=True, exist_ok=True)
