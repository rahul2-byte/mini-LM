import json
from pathlib import Path

from app.cli import run_smoke


def test_local_training_smoke_creates_artifacts(tmp_path: Path) -> None:
    metrics = run_smoke(
        sample_path=Path("data/sample.txt"),
        output_dir=tmp_path,
        sequence_length=8,
        max_steps=2,
    )
    assert metrics["global_step"] == 2
    assert metrics["perplexity"] > 0
    assert Path(metrics["checkpoint_path"]).is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["global_step"] == 2
