"""Joint two-event classification for close fault pairs.

`events.py`'s own Status/next-steps note left this open: "for faults closer than
`bounded_half_window_km` can fully separate, a proper joint two-event model
instead of independent per-candidate classification." Window clipping helps, but
it has a floor (`min_half_window_km`) below which it can't fully prevent one
candidate's window from picking up the other event's signature, and even before
that floor, each side's classifier is still blind to the other window entirely --
it never gets to see that the other candidate's own features look unambiguous
before deciding how much to trust its own.

This module tests the direct alternative: concatenate both sides' clipped local
features (plus the gap between them) into one feature vector, and predict both
labels from it with per-side classifier heads. If cross-window information is
useful, a joint model that can see both windows at once should do at least as
well as two classifiers that each only see their own window. If it doesn't help,
that's a real result too -- see the README for the honest comparison.

`fiber_cut` is deliberately excluded from the fault types used here (both as
"left" and "right"): an upstream `fiber_cut` makes anything downstream physically
unobservable (see `simulate.simulate_multi_fault_trace`), which is a different,
already-documented failure mode from window contamination. Mixing the two in would
conflate "the window has the wrong evidence in it" with "the window has no
evidence in it at all," muddying the comparison this module is meant to make.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import FAULT_TYPES
from .events import (
    DEFAULT_HALF_WINDOW_KM,
    DEFAULT_MIN_HALF_WINDOW_KM,
    bounded_half_window_km,
    extract_local_features,
    train_local_event_classifier,
)
from .features import FEATURE_NAMES
from .simulate import NOISE_STD_DB, simulate_multi_fault_trace

# Excludes "fiber_cut" as well as "none" -- see module docstring.
PAIR_FAULT_TYPES = [f for f in FAULT_TYPES if f not in ("none", "fiber_cut")]

JOINT_FEATURE_NAMES = (
    [f"left_{name}" for name in FEATURE_NAMES] + [f"right_{name}" for name in FEATURE_NAMES] + ["gap_km"]
)

DEFAULT_MIN_GAP_KM = 5.0
DEFAULT_MAX_GAP_KM = 8.0


def _sample_close_pair_positions(
    rng: np.random.Generator, length_km: float, min_gap_km: float, max_gap_km: float
) -> tuple[float, float]:
    """Draws a (left_km, right_km) pair with `right_km - left_km` uniform in
    `[min_gap_km, max_gap_km]`, both positions kept away from the span edges."""
    gap_km = float(rng.uniform(min_gap_km, max_gap_km))
    lo = 0.15 * length_km
    hi = 0.85 * length_km - gap_km
    if hi <= lo:
        hi = lo + 1e-6
    left_km = float(rng.uniform(lo, hi))
    return left_km, left_km + gap_km


def _joint_features(
    distance_km: np.ndarray,
    power_db: np.ndarray,
    left_km: float,
    right_km: float,
    max_half_window_km: float,
    min_half_window_km: float,
) -> np.ndarray:
    """Both sides' clipped local features, back to back, plus the gap between them.

    The gap is included explicitly (not just implied by the two positions) because
    the amount of expected cross-window contamination is a direct function of it --
    a classifier that can see the gap can in principle learn to trust each window
    less as the gap shrinks, the same way `bounded_half_window_km` already shrinks
    the window itself for that reason.
    """
    left_half_km = bounded_half_window_km(left_km, [right_km], max_half_window_km, min_half_window_km)
    right_half_km = bounded_half_window_km(right_km, [left_km], max_half_window_km, min_half_window_km)
    left_feats = extract_local_features(distance_km, power_db, left_km, left_half_km)
    right_feats = extract_local_features(distance_km, power_db, right_km, right_half_km)
    return np.concatenate([left_feats, right_feats, [right_km - left_km]])


@dataclass
class JointPairClassifier:
    left_classifier: RandomForestClassifier
    right_classifier: RandomForestClassifier
    max_half_window_km: float
    min_half_window_km: float


def _generate_close_pairs(
    train_n: int, seed: int, min_gap_km: float, max_gap_km: float, length_km: float
):
    """Yields `train_n` synthetic close-pair traces with independently, uniformly
    drawn left/right fault types and a gap uniform in `[min_gap_km, max_gap_km]`.
    Shared by `train_joint_pair_classifier` and `train_matched_single_side_classifier`
    so both see literally the same training distribution, differing only in which
    features they're allowed to use from it."""
    rng = np.random.default_rng(seed)
    for _ in range(train_n):
        left_type = PAIR_FAULT_TYPES[rng.integers(0, len(PAIR_FAULT_TYPES))]
        right_type = PAIR_FAULT_TYPES[rng.integers(0, len(PAIR_FAULT_TYPES))]
        left_km, right_km = _sample_close_pair_positions(rng, length_km, min_gap_km, max_gap_km)
        sample = simulate_multi_fault_trace(
            left_type,
            right_type,
            length_km=length_km,
            primary_position_km=left_km,
            secondary_position_km=right_km,
            noise_std_db=NOISE_STD_DB,
            rng=rng,
        )
        yield left_type, right_type, left_km, right_km, sample


def train_joint_pair_classifier(
    train_n: int = 800,
    seed: int = 0,
    n_estimators: int = 100,
    min_gap_km: float = DEFAULT_MIN_GAP_KM,
    max_gap_km: float = DEFAULT_MAX_GAP_KM,
    length_km: float = 40.0,
    max_half_window_km: float = DEFAULT_HALF_WINDOW_KM,
    min_half_window_km: float = DEFAULT_MIN_HALF_WINDOW_KM,
) -> JointPairClassifier:
    """Trains two RandomForest heads (left type, right type) on the same joint
    feature vector, using synthetic close-pair traces with gaps drawn from
    `[min_gap_km, max_gap_km]` -- the regime where `bounded_half_window_km` is
    actively shrinking windows for both events."""
    n_features = 2 * len(FEATURE_NAMES) + 1
    X = np.zeros((train_n, n_features))
    y_left = np.empty(train_n, dtype=object)
    y_right = np.empty(train_n, dtype=object)

    for i, (left_type, right_type, left_km, right_km, sample) in enumerate(
        _generate_close_pairs(train_n, seed, min_gap_km, max_gap_km, length_km)
    ):
        X[i] = _joint_features(
            sample.distance_km, sample.power_db, left_km, right_km, max_half_window_km, min_half_window_km
        )
        y_left[i] = left_type
        y_right[i] = right_type

    left_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    left_clf.fit(X, y_left)
    right_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed + 1)
    right_clf.fit(X, y_right)
    return JointPairClassifier(left_clf, right_clf, max_half_window_km, min_half_window_km)


@dataclass
class MatchedSingleSideClassifier:
    """Same role as `JointPairClassifier`, but each head only ever sees its own
    side's local features -- no access to the other window or the gap."""

    left_classifier: RandomForestClassifier
    right_classifier: RandomForestClassifier
    max_half_window_km: float
    min_half_window_km: float


def train_matched_single_side_classifier(
    train_n: int = 800,
    seed: int = 0,
    n_estimators: int = 100,
    min_gap_km: float = DEFAULT_MIN_GAP_KM,
    max_gap_km: float = DEFAULT_MAX_GAP_KM,
    length_km: float = 40.0,
    max_half_window_km: float = DEFAULT_HALF_WINDOW_KM,
    min_half_window_km: float = DEFAULT_MIN_HALF_WINDOW_KM,
) -> MatchedSingleSideClassifier:
    """An ablation: trained on the exact same close-pair distribution as
    `train_joint_pair_classifier` (same gaps, same fault-type pool), but each side's
    classifier only gets its own single-window features, never the other side's or
    the gap. This isolates *why* the joint model beats `train_local_event_classifier`
    on close pairs: is it because it can see both windows at once, or simply because
    it was trained on close-pair data instead of clean single-fault traces (which is
    what `train_local_event_classifier` uses)? If this matched-but-non-joint model
    closes most of the gap on its own, the joint feature vector isn't earning its
    keep -- distribution match would be doing the real work.
    """
    X_left = np.zeros((train_n, len(FEATURE_NAMES)))
    X_right = np.zeros((train_n, len(FEATURE_NAMES)))
    y_left = np.empty(train_n, dtype=object)
    y_right = np.empty(train_n, dtype=object)

    for i, (left_type, right_type, left_km, right_km, sample) in enumerate(
        _generate_close_pairs(train_n, seed, min_gap_km, max_gap_km, length_km)
    ):
        left_half_km = bounded_half_window_km(left_km, [right_km], max_half_window_km, min_half_window_km)
        right_half_km = bounded_half_window_km(right_km, [left_km], max_half_window_km, min_half_window_km)
        X_left[i] = extract_local_features(sample.distance_km, sample.power_db, left_km, left_half_km)
        X_right[i] = extract_local_features(sample.distance_km, sample.power_db, right_km, right_half_km)
        y_left[i] = left_type
        y_right[i] = right_type

    left_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    left_clf.fit(X_left, y_left)
    right_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed + 1)
    right_clf.fit(X_right, y_right)
    return MatchedSingleSideClassifier(left_clf, right_clf, max_half_window_km, min_half_window_km)


@dataclass(frozen=True)
class ClosePairResult:
    n_pairs: int
    min_gap_km: float
    max_gap_km: float
    independent_left_accuracy: float
    independent_right_accuracy: float
    matched_left_accuracy: float
    matched_right_accuracy: float
    joint_left_accuracy: float
    joint_right_accuracy: float

    @property
    def independent_mean_accuracy(self) -> float:
        return (self.independent_left_accuracy + self.independent_right_accuracy) / 2.0

    @property
    def matched_mean_accuracy(self) -> float:
        return (self.matched_left_accuracy + self.matched_right_accuracy) / 2.0

    @property
    def joint_mean_accuracy(self) -> float:
        return (self.joint_left_accuracy + self.joint_right_accuracy) / 2.0


def run_close_pair_comparison(
    train_n: int = 800,
    n_pairs: int = 300,
    seed: int = 0,
    n_estimators: int = 100,
    min_gap_km: float = DEFAULT_MIN_GAP_KM,
    max_gap_km: float = DEFAULT_MAX_GAP_KM,
    length_km: float = 40.0,
    max_half_window_km: float = DEFAULT_HALF_WINDOW_KM,
    min_half_window_km: float = DEFAULT_MIN_HALF_WINDOW_KM,
) -> ClosePairResult:
    """Compares independent per-candidate classification (each side sees only its
    own clipped window, as in `events.detect_and_classify_events`) against the
    joint model above, on the *same* close-pair traces and *known* fault
    positions -- isolating the classification question from detection/matching."""
    independent_model = train_local_event_classifier(
        train_n=train_n,
        seed=seed,
        n_estimators=n_estimators,
        half_window_km=max_half_window_km,
        min_half_window_km=min_half_window_km,
    )
    matched_model = train_matched_single_side_classifier(
        train_n=train_n,
        seed=seed,
        n_estimators=n_estimators,
        min_gap_km=min_gap_km,
        max_gap_km=max_gap_km,
        length_km=length_km,
        max_half_window_km=max_half_window_km,
        min_half_window_km=min_half_window_km,
    )
    joint_model = train_joint_pair_classifier(
        train_n=train_n,
        seed=seed,
        n_estimators=n_estimators,
        min_gap_km=min_gap_km,
        max_gap_km=max_gap_km,
        length_km=length_km,
        max_half_window_km=max_half_window_km,
        min_half_window_km=min_half_window_km,
    )

    rng = np.random.default_rng(seed + 1000)
    ind_left_correct = ind_right_correct = 0
    matched_left_correct = matched_right_correct = 0
    joint_left_correct = joint_right_correct = 0

    for _ in range(n_pairs):
        left_type = PAIR_FAULT_TYPES[rng.integers(0, len(PAIR_FAULT_TYPES))]
        right_type = PAIR_FAULT_TYPES[rng.integers(0, len(PAIR_FAULT_TYPES))]
        left_km, right_km = _sample_close_pair_positions(rng, length_km, min_gap_km, max_gap_km)
        sample = simulate_multi_fault_trace(
            left_type,
            right_type,
            length_km=length_km,
            primary_position_km=left_km,
            secondary_position_km=right_km,
            noise_std_db=NOISE_STD_DB,
            rng=rng,
        )

        left_half_km = bounded_half_window_km(left_km, [right_km], max_half_window_km, min_half_window_km)
        right_half_km = bounded_half_window_km(right_km, [left_km], max_half_window_km, min_half_window_km)
        left_feats = extract_local_features(sample.distance_km, sample.power_db, left_km, left_half_km)
        right_feats = extract_local_features(sample.distance_km, sample.power_db, right_km, right_half_km)
        ind_left_pred = independent_model.classifier.predict(left_feats.reshape(1, -1))[0]
        ind_right_pred = independent_model.classifier.predict(right_feats.reshape(1, -1))[0]
        matched_left_pred = matched_model.left_classifier.predict(left_feats.reshape(1, -1))[0]
        matched_right_pred = matched_model.right_classifier.predict(right_feats.reshape(1, -1))[0]

        joint_feats = _joint_features(
            sample.distance_km, sample.power_db, left_km, right_km, max_half_window_km, min_half_window_km
        )
        joint_left_pred = joint_model.left_classifier.predict(joint_feats.reshape(1, -1))[0]
        joint_right_pred = joint_model.right_classifier.predict(joint_feats.reshape(1, -1))[0]

        ind_left_correct += int(ind_left_pred == left_type)
        ind_right_correct += int(ind_right_pred == right_type)
        matched_left_correct += int(matched_left_pred == left_type)
        matched_right_correct += int(matched_right_pred == right_type)
        joint_left_correct += int(joint_left_pred == left_type)
        joint_right_correct += int(joint_right_pred == right_type)

    return ClosePairResult(
        n_pairs=n_pairs,
        min_gap_km=min_gap_km,
        max_gap_km=max_gap_km,
        independent_left_accuracy=ind_left_correct / n_pairs,
        independent_right_accuracy=ind_right_correct / n_pairs,
        matched_left_accuracy=matched_left_correct / n_pairs,
        matched_right_accuracy=matched_right_correct / n_pairs,
        joint_left_accuracy=joint_left_correct / n_pairs,
        joint_right_accuracy=joint_right_correct / n_pairs,
    )


def render_close_pair_report(result: ClosePairResult) -> str:
    total_delta = result.joint_mean_accuracy - result.independent_mean_accuracy
    distribution_delta = result.matched_mean_accuracy - result.independent_mean_accuracy
    architecture_delta = result.joint_mean_accuracy - result.matched_mean_accuracy
    verdict = (
        "The joint model measurably helped"
        if total_delta > 0.02
        else "The joint model made no meaningful difference"
        if abs(total_delta) <= 0.02
        else "The joint model measurably hurt"
    )
    if abs(distribution_delta) < 0.005 and abs(architecture_delta) < 0.005:
        attribution = (
            "Neither ablation term is large enough to draw a confident attribution at this sample size."
        )
    elif distribution_delta >= architecture_delta:
        attribution = (
            "Most of that gain is attributable to training-distribution match (matched vs. "
            "independent), not to the joint architecture itself (joint vs. matched) -- training "
            "on close-pair data the eval set actually resembles matters more here than letting "
            "each side see the other window's features."
        )
    else:
        attribution = (
            "Most of that gain is attributable to the joint architecture itself (joint vs. "
            "matched), not merely to training on close-pair data instead of clean single-fault "
            "traces -- seeing both windows' features together is carrying real weight here, not "
            "just a better-matched training distribution."
        )
    return "\n".join(
        [
            "# Close-Pair Joint Classification",
            "",
            "## Research Question",
            "",
            "For fault pairs close enough that `bounded_half_window_km` has to shrink",
            "both windows, does feeding both sides' features (plus the gap) into one",
            "joint model recover any of the accuracy independent per-candidate",
            "classification loses to window contamination -- or does each side's own",
            "local window already carry all the usable signal?",
            "",
            "## Results",
            "",
            f"- Pairs evaluated: {result.n_pairs}, gap drawn uniformly from "
            f"[{result.min_gap_km:.1f}, {result.max_gap_km:.1f}] km",
            f"- Independent (trained on clean single-fault traces, own window only): "
            f"left {result.independent_left_accuracy:.3f}, right {result.independent_right_accuracy:.3f}, "
            f"mean {result.independent_mean_accuracy:.3f}",
            f"- Matched ablation (trained on close-pair traces, own window only): "
            f"left {result.matched_left_accuracy:.3f}, right {result.matched_right_accuracy:.3f}, "
            f"mean {result.matched_mean_accuracy:.3f}",
            f"- Joint (trained on close-pair traces, both windows + gap): "
            f"left {result.joint_left_accuracy:.3f}, right {result.joint_right_accuracy:.3f}, "
            f"mean {result.joint_mean_accuracy:.3f}",
            f"- Total delta (joint - independent): {total_delta:+.3f}, of which "
            f"{distribution_delta:+.3f} is training-distribution match (matched - independent) "
            f"and {architecture_delta:+.3f} is the joint architecture on top of that "
            "(joint - matched)",
            "",
            "## Interpretation",
            "",
            f"{verdict} at this gap range. {attribution}",
            "",
            "`fiber_cut` is excluded from both fault-type pools here (see module docstring in",
            "`joint_events.py`) so this comparison isolates window-contamination effects from the",
            "separate, already-documented masking-behind-a-cut effect in `multi_event.py`.",
            "",
        ]
    )
