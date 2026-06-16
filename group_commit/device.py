"""Flush-cost (device) model.

A flush of batch size K costs  F(K) = F0 + K*delta  (fixed barrier latency + per-record write).
For the real-device arm, F0 is drawn per-flush from a measured fsync-latency distribution; delta is the
per-record marginal write cost. All times in the same unit (we use milliseconds in the benches).
"""
from __future__ import annotations

import numpy as np


class FlushModel:
    def __init__(self, F0: float, delta: float = 0.0, F0_samples: np.ndarray | None = None,
                 seed: int = 0):
        """F0 = mean fixed flush cost; delta = per-record marginal. If F0_samples (a measured fsync
        latency distribution) is given, each flush draws its fixed cost from it (captures the tail)."""
        self.F0 = float(F0)
        self.delta = float(delta)
        self.samples = None if F0_samples is None else np.asarray(F0_samples, dtype=float)
        self.rng = np.random.default_rng(seed)

    def duration(self, K: int) -> float:
        """Sampled flush duration for a batch of K records."""
        base = float(self.rng.choice(self.samples)) if self.samples is not None else self.F0
        return base + K * self.delta

    def mean_F(self, K: float) -> float:
        return self.F0 + K * self.delta
