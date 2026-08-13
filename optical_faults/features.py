"""Hand-engineered features from a raw trace, instead of feeding raw samples to a black box.

Keeping this interpretable matters: in a real bring-up/validation setting you need to be
able to explain *why* the model flagged something, not just trust a number.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "mean_slope_db_per_km",
    "max_abs_diff",
    "max_abs_diff_position_frac",
    "residual_std_first_half",
    "residual_std_second_half",
    "slope_first_half",
    "slope_second_half",
    "slope_diff_abs",
    "frac_near_floor",
    "total_span_loss_db",
    "frac_near_floor_after_jump",
]


def _linfit_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def extract_features(distance_km: np.ndarray, power_db: np.ndarray) -> np.ndarray:
    n = len(power_db)
    diffs = np.diff(power_db)
    max_abs_diff = float(np.max(np.abs(diffs))) if n > 1 else 0.0
    jump_idx = int(np.argmax(np.abs(diffs))) if n > 1 else 0
    jump_pos_frac = jump_idx / max(n - 1, 1)

    half = n // 2
    d1, p1 = distance_km[:half], power_db[:half]
    d2, p2 = distance_km[half:], power_db[half:]

    slope1 = _linfit_slope(d1, p1)
    slope2 = _linfit_slope(d2, p2)

    def _residual_std(d, p, slope):
        if len(d) < 2:
            return 0.0
        pred = p[0] + slope * (d - d[0])
        return float(np.std(p - pred))

    resid1 = _residual_std(d1, p1, slope1)
    resid2 = _residual_std(d2, p2, slope2)

    overall_slope = _linfit_slope(distance_km, power_db)
    floor_thresh = np.min(power_db) + 2.0
    frac_near_floor = float(np.mean(power_db <= floor_thresh))
    total_span_loss = float(power_db[0] - power_db[-1])

    # Distinguishes a permanent collapse to the noise floor right after the biggest
    # jump (a fiber_cut) from a brief reflection spike followed by the trace
    # continuing at a moderately reduced level (a connector_loss's Fresnel spike,
    # see simulate.py) -- both can produce a similarly large single-step jump, but
    # only one of them stays pinned at the floor afterward.
    after_jump = power_db[jump_idx + 1 :]
    frac_near_floor_after_jump = (
        float(np.mean(after_jump <= floor_thresh)) if len(after_jump) > 0 else 0.0
    )

    return np.array(
        [
            overall_slope,
            max_abs_diff,
            jump_pos_frac,
            resid1,
            resid2,
            slope1,
            slope2,
            abs(slope1 - slope2),
            frac_near_floor,
            total_span_loss,
            frac_near_floor_after_jump,
        ],
        dtype=float,
    )
