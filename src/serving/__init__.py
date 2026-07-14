"""Local serving API package.

Serving is intentionally a separate boundary from training so an inference
process can load a selected checkpoint without importing the training loop.
"""
