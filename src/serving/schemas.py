"""Serving request and response schema boundary.

Schemas will be kept independent from the model implementation so request
validation can evolve without changing tensor or tokenizer internals.
"""
