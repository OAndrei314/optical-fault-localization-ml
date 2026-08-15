"""Evaluates `events.py`'s changepoint-scan-and-classify pipeline as an actual attempt
to *handle* multi-fault traces, not just measure how badly the old global-feature
approach degrades on them (`multi_fault.py`).

Each multi-fault trace has two ground-truth events (primary + secondary, each with a
known type and position -- see `simulate.simulate_multi_fault_trace`). The detector
proposes up to two candidate events; candidates are greedily matched to the nearest
ground-truth event within `match_tolerance_km`. This reports, honestly:
- detection recall: what fraction of the 2*n ground-truth events got a matched
  candidate at all (an upstream `fiber_cut` can make a downstream event physically
  unobservable, in which case no detector can find it -- see simulate.py).
- fault-type accuracy and localization MAE on the matched pairs only.
- false positives: detected candidates that didn't match any ground-truth event.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import FAULT_TYPES
from .events import DEFAULT_HALF_WINDOW_KM, DEFAULT_SCALES, detect_and_classify_events, train_local_event_classifier
from .simulate import NOISE_STD_DB, simulate_multi_fault_trace

_INJECTABLE_FAULT_TYPES = [f for f in FAULT_TYPES if f != "none"]


@dataclass(frozen=True)
class MultiEventResult:
    n_traces: int
    n_ground_truth_events: int
    n_matched: int
    n_false_positives: int
    detection_recall: float
    matched_type_accuracy: float
    matched_localization_mae_km: float


def _match_greedy(
    predicted: list[tuple[float, str]],
    ground_truth: list[tuple[float, str]],
    tolerance_km: float,
) -> list[tuple[tuple[float, str], tuple[float, str]]]:
    """Greedy nearest-position matching between predicted and ground-truth events,
    each used at most once. Returns matched (predicted, ground_truth) pairs."""
    pairs = []
    for p in predicted:
        for g in ground_truth:
            dist = abs(p[0] - g[0])
            if dist <= tolerance_km:
                pairs.append((dist, p, g))
    pairs.sort(key=lambda item: item[0])

    matched_pred, matched_gt, result = set(), set(), []
    for _dist, p, g in pairs:
        if p in matched_pred or g in matched_gt:
            continue
        matched_pred.add(p)
        matched_gt.add(g)
        result.append((p, g))
    return result


def run_multi_event_detection(
    train_n: int = 800,
    n_traces: int = 300,
    seed: int = 0,
    n_estimators: int = 100,
    min_separation_km: float = 5.0,
    match_tolerance_km: float = 3.0,
    half_window_km: float = DEFAULT_HALF_WINDOW_KM,
    scales: list[tuple[float, float]] = DEFAULT_SCALES,
) -> MultiEventResult:
    model = train_local_event_classifier(
        train_n=train_n, seed=seed, n_estimators=n_estimators, half_window_km=half_window_km
    )

    rng = np.random.default_rng(seed + 500)
    n_matched = 0
    n_false_positives = 0
    n_ground_truth = 0
    type_correct = 0
    position_errors: list[float] = []

    for _ in range(n_traces):
        primary = _INJECTABLE_FAULT_TYPES[rng.integers(0, len(_INJECTABLE_FAULT_TYPES))]
        secondary_choices = [f for f in _INJECTABLE_FAULT_TYPES if f != primary]
        secondary = secondary_choices[rng.integers(0, len(secondary_choices))]
        sample = simulate_multi_fault_trace(
            primary,
            secondary,
            min_separation_km=min_separation_km,
            noise_std_db=NOISE_STD_DB,
            rng=rng,
        )
        ground_truth = [
            (sample.fault_position_km, sample.fault_type),
            (sample.secondary_fault_position_km, sample.secondary_fault_type),
        ]
        n_ground_truth += len(ground_truth)

        predicted = detect_and_classify_events(
            sample.distance_km, sample.power_db, model, scales=scales, min_separation_km=min_separation_km
        )

        matches = _match_greedy(predicted, ground_truth, match_tolerance_km)
        n_matched += len(matches)
        n_false_positives += len(predicted) - len(matches)
        for (pred_pos, pred_type), (gt_pos, gt_type) in matches:
            position_errors.append(abs(pred_pos - gt_pos))
            if pred_type == gt_type:
                type_correct += 1

    return MultiEventResult(
        n_traces=n_traces,
        n_ground_truth_events=n_ground_truth,
        n_matched=n_matched,
        n_false_positives=n_false_positives,
        detection_recall=n_matched / n_ground_truth if n_ground_truth else float("nan"),
        matched_type_accuracy=type_correct / n_matched if n_matched else float("nan"),
        matched_localization_mae_km=float(np.mean(position_errors)) if position_errors else float("nan"),
    )


def render_multi_event_report(result: MultiEventResult) -> str:
    return "\n".join(
        [
            "# Multi-Event Detection",
            "",
            "## Research Question",
            "",
            "`multi_fault.py` measures how badly the single-global-feature-vector",
            "classifier degrades on a two-fault trace. This asks the follow-up: can a",
            "changepoint-scan-and-classify pipeline, designed for more than one event",
            "per trace, actually recover both faults instead of just failing",
            "informatively on them?",
            "",
            "## Results",
            "",
            f"- Traces evaluated: {result.n_traces} (each with 2 ground-truth events,",
            f"  {result.n_ground_truth_events} events total)",
            f"- Detection recall: {result.detection_recall:.3f} "
            f"({result.n_matched}/{result.n_ground_truth_events} ground-truth events matched",
            "  to a detected candidate within tolerance)",
            f"- False positives: {result.n_false_positives} detected candidates that",
            "  matched no ground-truth event",
            f"- Fault-type accuracy on matched events: {result.matched_type_accuracy:.3f}",
            f"- Localization MAE on matched events: {result.matched_localization_mae_km:.3f} km",
            "",
            "## Interpretation",
            "",
            "Recall below 1.0 is expected, not just detector error: an upstream",
            "`fiber_cut` drives the trace to the noise floor, making anything located",
            "past it physically unobservable regardless of detection method (see",
            "`simulate.simulate_multi_fault_trace`). Compare `detection_recall` and",
            "`matched_type_accuracy` against `multi_fault.py`'s single-event accuracy",
            "on the same kind of trace -- unlike that pipeline, this one is scored on",
            "recovering *both* events, not just the one it's told to look for.",
            "",
        ]
    )
