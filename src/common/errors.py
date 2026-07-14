"""Domain-specific errors."""


class MiniLLMError(RuntimeError):
    """Base error for expected pipeline failures."""


class EmptyDatasetError(MiniLLMError):
    """Raised when no valid documents remain."""
