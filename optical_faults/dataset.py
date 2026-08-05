"""Builds a labeled dataset of (features, fault_type, fault_position_km) from the simulator."""
from __future__ import annotations

import numpy as np

from . import FAULT_TYPES
from .features import extract_features
from .simulate import simulate_trace


def generate_dataset(
    n: int, length_km: float = 40.0, step_km: float = 0.05, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y_type, y_position) where y_position is NaN for the 'none' class."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 10))
    y_type = np.empty(n, dtype=object)
    y_position = np.full(n, np.nan)

    for i in range(n):
        fault_type = FAULT_TYPES[rng.integers(0, len(FAULT_TYPES))]
        sample = simulate_trace(fault_type, length_km=length_km, step_km=step_km, rng=rng)
        X[i] = extract_features(sample.distance_km, sample.power_db)
        y_type[i] = sample.fault_type
        if sample.fault_position_km is not None:
            y_position[i] = sample.fault_position_km

    return X, y_type, y_position


def save_dataset(path: str, X: np.ndarray, y_type: np.ndarray, y_position: np.ndarray) -> None:
    np.savez(path, X=X, y_type=y_type.astype(str), y_position=y_position)


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return data["X"], data["y_type"], data["y_position"]
