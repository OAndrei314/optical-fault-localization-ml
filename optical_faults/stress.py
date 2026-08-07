"""Domain-shift stress tests for synthetic optical fault localization."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error

from .dataset import generate_dataset
from .model import train_and_evaluate
from .simulate import NOISE_STD_DB


@dataclass(frozen=True)
class StressScenario:
    label: str
    noise_std_db: float
    loss_scale: float


@dataclass(frozen=True)
class StressResult:
    label: str
    noise_std_db: float
    loss_scale: float
    accuracy: float
    localization_mae_km: float


DEFAULT_SCENARIOS = (
    StressScenario("source_holdout", NOISE_STD_DB, 1.0),
    StressScenario("higher_noise", NOISE_STD_DB * 2.5, 1.0),
    StressScenario("weak_faults", NOISE_STD_DB, 0.45),
    StressScenario("high_noise_weak_faults", NOISE_STD_DB * 3.0, 0.35),
)


def run_domain_shift_stress(
    train_n: int = 800,
    test_n: int = 300,
    seed: int = 0,
    n_estimators: int = 80,
    scenarios: tuple[StressScenario, ...] = DEFAULT_SCENARIOS,
) -> tuple[StressResult, ...]:
    X, y_type, y_position = generate_dataset(train_n, seed=seed)
    trained = train_and_evaluate(
        X,
        y_type,
        y_position,
        seed=seed,
        n_estimators=n_estimators,
    )

    results = []
    for idx, scenario in enumerate(scenarios):
        X_shift, y_type_shift, y_position_shift = generate_dataset(
            test_n,
            seed=seed + 100 + idx,
            noise_std_db=scenario.noise_std_db,
            loss_scale=scenario.loss_scale,
        )
        pred_type = trained.classifier.predict(X_shift)
        accuracy = float(accuracy_score(y_type_shift, pred_type))

        faulty_mask = ~np.isnan(y_position_shift)
        if faulty_mask.sum():
            pred_position = trained.regressor.predict(X_shift[faulty_mask])
            mae = float(mean_absolute_error(y_position_shift[faulty_mask], pred_position))
        else:
            mae = float("nan")

        results.append(
            StressResult(
                label=scenario.label,
                noise_std_db=scenario.noise_std_db,
                loss_scale=scenario.loss_scale,
                accuracy=accuracy,
                localization_mae_km=mae,
            )
        )
    return tuple(results)


def render_stress_report(results: tuple[StressResult, ...]) -> str:
    source = results[0]
    worst = min(results, key=lambda result: result.accuracy)
    lines = [
        "# Domain Shift Stress Test",
        "",
        "## Research Question",
        "",
        "How quickly does a classifier trained on one synthetic optical-fault distribution",
        "lose accuracy when traces become noisier or fault signatures move toward the noise",
        "floor?",
        "",
        "## Results",
        "",
        "| scenario | noise std (dB) | loss scale | accuracy | localization MAE (km) | accuracy drop |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        drop = source.accuracy - result.accuracy
        lines.append(
            f"| {result.label} | {result.noise_std_db:.3f} | {result.loss_scale:.2f} | "
            f"{result.accuracy:.3f} | {result.localization_mae_km:.3f} | {drop:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Worst scenario: `{worst.label}`. This is the next place to add synthetic",
            "data, features, or calibration experiments before claiming robustness.",
            "",
        ]
    )
    return "\n".join(lines)
