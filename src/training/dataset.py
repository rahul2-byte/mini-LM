"""Training dataset boundary.

The current implementation is ``PackedDataset`` in ``training.pack``.  This
module remains as the natural home for future iterable datasets that read
partitioned Parquet or object-store data without materializing it in RAM.
"""
