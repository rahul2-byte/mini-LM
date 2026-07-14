"""Dataset downloader boundary; legal source adapters are added in Phase 2."""


def download_sources() -> None:
    """Placeholder for the future source scheduler entry point.

    The working implementation currently lives in ``app.cli`` and
    ``IngestionPipeline``.  Keeping this module boundary records where a
    standalone downloader command can be added without putting network logic
    into the package initializer.
    """
    raise NotImplementedError("Legal dataset adapters are planned for Phase 2")
