"""Interference-fault stress test.

The classifier and localizer are trained only on single-fault traces (see
`dataset.generate_dataset`). This module asks an honest question about that choice:
what happens when a *second*, unrelated fault is already present on the link when a new
one appears? Real fiber spans accumulate minor imperfections over time; a model that only
ever sees clean-baseline-plus-one-fault traces during training may be relying on that
simplification more than it looks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error

from . import FAULT_TYPES
from .dataset import generate_dataset
from .features import FEATURE_NAMES, extract_features
from .model import train_and_evaluate
from .simulate import NOISE_STD_DB, simulate_multi_fault_trace

_INJECTABLE_FAULT_TYPES = [f for f in FAULT_TYPES if f != "none"]


@dataclass(frozen=True)
class MultiFaultResult:
    single_fault_accuracy: float
    single_fault_mae_km: float
    multi_fault_accuracy: float
    multi_fault_mae_km: float
    n_multi_fault: int


def _generate_multi_fault_eval_set(
    n: int,
    seed: int,
    min_separation_km: float,
    noise_std_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.zeros((n, len(FEATURE_NAMES)))
    y_type = np.empty(n, dtype=object)
    y_position = np.full(n, np.nan)

    for i in range(n):
        primary = _INJECTABLE_FAULT_TYPES[rng.integers(0, len(_INJECTABLE_FAULT_TYPES))]
        secondary_choices = [f for f in _INJECTABLE_FAULT_TYPES if f != primary]
        secondary = secondary_choices[rng.integers(0, len(secondary_choices))]
        sample = simulate_multi_fault_trace(
            primary,
            secondary,
            min_separation_km=min_separation_km,
            noise_std_db=noise_std_db,
            rng=rng,
        )
        X[i] = extract_features(sample.distance_km, sample.power_db)
        y_type[i] = sample.fault_type
        y_position[i] = sample.fault_position_km

    return X, y_type, y_position


def run_multi_fault_stress(
    train_n: int = 800,
    eval_n: int = 300,
    seed: int = 0,
    n_estimators: int = 80,
    min_separation_km: float = 5.0,
) -> MultiFaultResult:
    """Trains on ordinary single-fault data, then evaluates on traces with an added,
    unrelated second fault -- reporting the accuracy/localization gap honestly."""
    X, y_type, y_position = generate_dataset(train_n, seed=seed)
    trained = train_and_evaluate(X, y_type, y_position, seed=seed, n_estimators=n_estimators)

    X_multi, y_type_multi, y_position_multi = _generate_multi_fault_eval_set(
        eval_n, seed=seed + 500, min_separation_km=min_separation_km, noise_std_db=NOISE_STD_DB
    )
    pred_type = trained.classifier.predict(X_multi)
    multi_accuracy = float(accuracy_score(y_type_multi, pred_type))
    pred_position = trained.regressor.predict(X_multi)
    multi_mae = float(mean_absolute_error(y_position_multi, pred_position))

    return MultiFaultResult(
        single_fault_accuracy=trained.accuracy,
        single_fault_mae_km=trained.localization_mae_km,
        multi_fault_accuracy=multi_accuracy,
        multi_fault_mae_km=multi_mae,
        n_multi_fault=eval_n,
    )


def render_multi_fault_report(result: MultiFaultResult) -> str:
    accuracy_drop = result.single_fault_accuracy - result.multi_fault_accuracy
    mae_increase_km = result.multi_fault_mae_km - result.single_fault_mae_km
    return "\n".join(
        [
            "# Multi-Fault Interference Stress Test",
            "",
            "## Research Question",
            "",
            "A classifier trained only on single-fault traces is asked to identify and",
            "localize a fault on a trace that also has a second, unrelated fault elsewhere",
            "on the span. How much does that unmodeled interference cost, and does the",
            "model degrade gracefully or fail in a specific, informative way?",
            "",
            "## Results",
            "",
            "| eval condition | fault-type accuracy | localization MAE (km) |",
            "| --- | ---: | ---: |",
            f"| single-fault holdout | {result.single_fault_accuracy:.3f} | "
            f"{result.single_fault_mae_km:.3f} |",
            f"| + unrelated second fault (n={result.n_multi_fault}) | "
            f"{result.multi_fault_accuracy:.3f} | {result.multi_fault_mae_km:.3f} |",
            "",
            f"Accuracy drop: {accuracy_drop:.3f}. Localization MAE increase: "
            f"{mae_increase_km:.3f} km.",
            "",
            "## Interpretation",
            "",
            "The hand-engineered features (largest single jump, first/second-half slope",
            "split) implicitly assume one event per trace. The worst known failure mode is",
            "an upstream `fiber_cut`: it drives the whole downstream trace to the noise",
            "floor by construction, so any fault located past it becomes structurally",
            "unobservable -- not a model weakness, a physical one. A multi-label classifier",
            "or a per-segment sliding-window feature set are the natural next steps if",
            "multi-fault traces need to be handled well rather than just measured.",
            "",
        ]
    )
