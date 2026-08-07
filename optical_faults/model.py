"""Fault-type classifier + fault-position regressor, both plain scikit-learn RandomForests."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split


@dataclass
class TrainResult:
    classifier: RandomForestClassifier
    regressor: RandomForestRegressor
    accuracy: float
    localization_mae_km: float
    report: str


def train_and_evaluate(
    X: np.ndarray,
    y_type: np.ndarray,
    y_position: np.ndarray,
    test_size: float = 0.2,
    seed: int = 0,
    n_estimators: int = 100,
) -> TrainResult:
    X_train, X_test, yt_train, yt_test, yp_train, yp_test = train_test_split(
        X, y_type, y_position, test_size=test_size, random_state=seed, stratify=y_type
    )

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(X_train, yt_train)
    yt_pred = clf.predict(X_test)
    accuracy = accuracy_score(yt_test, yt_pred)
    report = classification_report(yt_test, yt_pred, zero_division=0)

    faulty_train_mask = ~np.isnan(yp_train)
    faulty_test_mask = ~np.isnan(yp_test)

    reg = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
    reg.fit(X_train[faulty_train_mask], yp_train[faulty_train_mask])
    pos_pred = reg.predict(X_test[faulty_test_mask])
    localization_mae = (
        mean_absolute_error(yp_test[faulty_test_mask], pos_pred)
        if faulty_test_mask.sum() > 0
        else float("nan")
    )

    return TrainResult(clf, reg, float(accuracy), float(localization_mae), report)


def render_markdown_report(result: TrainResult, sample_count: int) -> str:
    return "\n".join(
        [
            "# Optical Fault Localization Report",
            "",
            "## Research Question",
            "",
            "How far can synthetic optical-link fault traces go for training and",
            "stress-testing a fault classifier/localizer before harder domain-shift tests",
            "are needed?",
            "",
            "## Money Question",
            "",
            "AI datacenter buildouts depend on high-bandwidth optical links. Faster fault",
            "localization reduces downtime, field-debug effort, and spare-module guesswork",
            "in expensive optical network deployments.",
            "",
            "## Engineering Evidence",
            "",
            f"- Synthetic samples: {sample_count}",
            f"- Fault-type accuracy: {result.accuracy:.3f}",
            f"- Localization MAE: {result.localization_mae_km:.3f} km",
            "",
            "## Classification Report",
            "",
            "```text",
            result.report.strip(),
            "```",
            "",
            "## Limitations",
            "",
            "The data is synthetic and intentionally simplified. Strong results here mean",
            "the pipeline recovered structure in the simulator; they do not prove",
            "production performance on real optical traces.",
            "",
        ]
    )
