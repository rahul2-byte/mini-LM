import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluation.perplexity import evaluate
from model.config import GPTConfig
from model.gpt import GPTModel


def test_evaluate_returns_finite_loss_and_perplexity() -> None:
    model = GPTModel(
        GPTConfig(vocab_size=8, context_length=4, n_layers=1, n_heads=2, embedding_dim=8)
    )
    inputs = torch.tensor([[1, 2, 3], [2, 3, 4]])
    loader = DataLoader(TensorDataset(inputs, inputs), batch_size=2)
    metrics = evaluate(model, loader, torch.device("cpu"))
    assert set(metrics) == {"loss", "perplexity"}
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_generation_decodes_prompt_and_new_tokens() -> None:
    from evaluation.generation import generate_text

    class StubTokenizer:
        def encode(self, text: str):
            return type("Encoding", (), {"ids": [1, 2]})()

        def decode(self, ids: list[int]) -> str:
            return "prompt generated"

    model = GPTModel(
        GPTConfig(vocab_size=8, context_length=4, n_layers=1, n_heads=2, embedding_dim=8)
    )
    output = generate_text(model, StubTokenizer(), "prompt", max_new_tokens=2, seed=3)
    assert output.startswith("prompt")
