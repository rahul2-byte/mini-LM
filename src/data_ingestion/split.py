"""Deterministic split boundary."""


def split_dataset() -> None:
    """Placeholder for deterministic train/validation/test splitting.

    The future implementation must split by document identity before token
    packing so near-identical records cannot leak across evaluation sets.
    """
    raise NotImplementedError("Dataset splitting is planned for Phase 2")
