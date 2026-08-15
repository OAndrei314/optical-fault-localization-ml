"""Multi-event detection: find *all* fault-like changepoints on a trace instead of
assuming there's exactly one.

The global feature vector in `features.py` (biggest single jump, one slope-halves
split) is built around the single-fault assumption baked into `dataset.py`'s training
data. `multi_fault.py` measures how badly that assumption breaks when a second,
unrelated fault is present -- this module is the other half: an attempt to actually
*handle* the multi-fault case, by treating fault detection as a changepoint-scan
problem instead of a single global classification.

Approach: slide a before/after window across the trace and score every position by
how much a single-sample jump plus a two-sided slope change would explain, pick the
`top_k` local maxima of that score (with non-max suppression so two well-separated
events don't collapse into one, or one wide event doesn't produce two near-duplicate
picks), then classify each candidate independently using the *same* hand-engineered
feature extractor from `features.py`, but computed on a short local window around the
candidate rather than the full 40km span -- and a classifier trained on local windows
around known single-fault positions, so train/inference feature distributions match.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestClassifier

from . import FAULT_TYPES
from .features import FEATURE_NAMES, extract_features
from .simulate import simulate_trace

_INJECTABLE_FAULT_TYPES = [f for f in FAULT_TYPES if f != "none"]

DEFAULT_SCAN_WINDOW_KM = 3.0
DEFAULT_HALF_WINDOW_KM = 6.0
DEFAULT_MIN_SEPARATION_KM = 3.0
DEFAULT_THRESHOLD_DB = 1.0

# A single scan window trades off position precision against recall: a narrow window
# localizes discrete-jump faults (fiber_cut, connector_loss, bend_loss) tightly, but
# amp_gain_drift's slope-only signature is too weak to clear the threshold at that
# scale (0/50 detected in an offline sweep at 1.5km vs. 47/50 at 4.0km) -- and a wide
# window in turn blurs jump-type localization (fiber_cut mean error 0.78km -> 2.0km
# going from 1.5km to 4.0km windows). Scanning both scales and merging keeps the
# precision of the narrow scale for the faults it can see, while still catching the
# ones only the wide scale can.
DEFAULT_SCALES: list[tuple[float, float]] = [(1.5, 1.0), (4.0, 1.0)]


@dataclass(frozen=True)
class Candidate:
    position_km: float
    score: float


def detect_changepoints(
    distance_km: np.ndarray,
    power_db: np.ndarray,
    window_km: float = DEFAULT_SCAN_WINDOW_KM,
    min_separation_km: float = DEFAULT_MIN_SEPARATION_KM,
    top_k: int = 2,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
) -> list[Candidate]:
    """Scores every interior position by (single-sample jump) + (before/after slope
    change over `window_km` on each side, converted to an equivalent dB delta), then
    returns up to `top_k` local maxima at least `min_separation_km` apart.

    Catches both discontinuity-style faults (fiber_cut, connector_loss -- large jump
    term) and slope-change faults with no discrete jump (bend_loss, amp_gain_drift --
    large slope-diff term), because it combines both signals additively rather than
    relying on the single biggest raw jump the way the global feature vector does.
    """
    n = len(power_db)
    step_km = float(distance_km[1] - distance_km[0]) if n > 1 else 1.0
    window_samples = max(2, int(round(window_km / step_km)))

    if n < 2 * window_samples + 1:
        return []

    k = np.arange(window_samples, dtype=float)
    k_centered = k - k.mean()
    denom = float(np.sum(k_centered**2))
    # OLS slope (per km) of an evenly-spaced window, as a fixed dot-product kernel --
    # avoids an explicit per-position least-squares fit.
    slope_kernel = k_centered / (step_km * denom)

    windows = sliding_window_view(power_db, window_samples)  # windows[j] = power[j:j+window_samples]
    slopes = windows @ slope_kernel  # slope of the window starting at sample j

    num_valid = n - 2 * window_samples + 1
    if num_valid <= 0:
        return []

    slope_before = slopes[0:num_valid]  # window (i-window_samples, i)
    slope_after = slopes[window_samples : window_samples + num_valid]  # window [i, i+window_samples)
    slope_diff = np.abs(slope_after - slope_before)

    diffs = np.abs(np.diff(power_db))
    jump_at_i = diffs[window_samples - 1 : window_samples - 1 + num_valid]

    score = jump_at_i + slope_diff * window_km
    valid_idx = np.arange(window_samples, window_samples + num_valid)

    order = np.argsort(score)[::-1]
    selected: list[Candidate] = []
    for rank in order:
        s = float(score[rank])
        if s < threshold_db:
            break
        pos_km = float(distance_km[valid_idx[rank]])
        if any(abs(pos_km - c.position_km) < min_separation_km for c in selected):
            continue
        selected.append(Candidate(pos_km, s))
        if len(selected) >= top_k:
            break

    return sorted(selected, key=lambda c: c.position_km)


def detect_changepoints_multiscale(
    distance_km: np.ndarray,
    power_db: np.ndarray,
    scales: list[tuple[float, float]] = DEFAULT_SCALES,
    min_separation_km: float = DEFAULT_MIN_SEPARATION_KM,
    top_k: int = 2,
) -> list[Candidate]:
    """Runs `detect_changepoints` at each (window_km, threshold_db) in `scales`, in
    order, keeping candidates from earlier (narrower) scales preferentially and only
    adding a later-scale candidate if it isn't within `min_separation_km` of one
    already kept -- so a fault visible at the narrow scale keeps its precise position,
    while one only the wide scale can see (weak slope drift) still gets included."""
    selected: list[Candidate] = []
    for window_km, threshold_db in scales:
        for c in detect_changepoints(
            distance_km,
            power_db,
            window_km=window_km,
            min_separation_km=min_separation_km,
            top_k=top_k,
            threshold_db=threshold_db,
        ):
            if len(selected) >= top_k:
                break
            if any(abs(c.position_km - s.position_km) < min_separation_km for s in selected):
                continue
            selected.append(c)
        if len(selected) >= top_k:
            break
    return sorted(selected, key=lambda c: c.position_km)


def extract_local_features(
    distance_km: np.ndarray,
    power_db: np.ndarray,
    center_km: float,
    half_window_km: float = DEFAULT_HALF_WINDOW_KM,
) -> np.ndarray:
    """The same hand-engineered feature vector as `features.extract_features`, computed
    on a short window around `center_km` instead of the full span, so a local
    classifier sees a consistent feature distribution whether the window comes from a
    known single-fault position (training) or a detected changepoint (inference)."""
    mask = (distance_km >= center_km - half_window_km) & (distance_km <= center_km + half_window_km)
    if np.sum(mask) < 2:
        # Near a span edge with a too-small window; fall back to whatever's closest.
        idx = int(np.argmin(np.abs(distance_km - center_km)))
        lo, hi = max(0, idx - 1), min(len(distance_km), idx + 2)
        mask = np.zeros_like(distance_km, dtype=bool)
        mask[lo:hi] = True
    return extract_features(distance_km[mask], power_db[mask])


@dataclass
class LocalEventClassifier:
    classifier: RandomForestClassifier
    half_window_km: float


def train_local_event_classifier(
    train_n: int = 800,
    seed: int = 0,
    n_estimators: int = 100,
    half_window_km: float = DEFAULT_HALF_WINDOW_KM,
) -> LocalEventClassifier:
    """Trains a fault-type classifier on local windows centered on the *known* fault
    position of ordinary single-fault traces -- the train-time analogue of what
    `detect_changepoints` gives at inference time."""
    rng = np.random.default_rng(seed)
    X = np.zeros((train_n, len(FEATURE_NAMES)))
    y_type = np.empty(train_n, dtype=object)

    for i in range(train_n):
        fault_type = _INJECTABLE_FAULT_TYPES[rng.integers(0, len(_INJECTABLE_FAULT_TYPES))]
        sample = simulate_trace(fault_type, rng=rng)
        X[i] = extract_local_features(
            sample.distance_km, sample.power_db, sample.fault_position_km, half_window_km
        )
        y_type[i] = sample.fault_type

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(X, y_type)
    return LocalEventClassifier(clf, half_window_km)


def detect_and_classify_events(
    distance_km: np.ndarray,
    power_db: np.ndarray,
    model: LocalEventClassifier,
    scales: list[tuple[float, float]] = DEFAULT_SCALES,
    min_separation_km: float = DEFAULT_MIN_SEPARATION_KM,
    top_k: int = 2,
) -> list[tuple[float, str]]:
    """Runs multi-scale changepoint detection then classifies each candidate locally.
    Returns a list of (position_km, predicted_fault_type), sorted by position."""
    candidates = detect_changepoints_multiscale(distance_km, power_db, scales, min_separation_km, top_k)
    results = []
    for c in candidates:
        features = extract_local_features(distance_km, power_db, c.position_km, model.half_window_km)
        pred_type = model.classifier.predict(features.reshape(1, -1))[0]
        results.append((c.position_km, str(pred_type)))
    return results
