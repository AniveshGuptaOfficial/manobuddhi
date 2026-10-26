"""Benchmark data: raw readers, gold-answer parsing, fixed splits.

Use from anywhere in the project:
    from engine.data import load_split
    items = load_split("gsm8k", "test")
"""
from .splits import BENCHMARKS, load_split, manifest

__all__ = ["BENCHMARKS", "load_split", "manifest"]
