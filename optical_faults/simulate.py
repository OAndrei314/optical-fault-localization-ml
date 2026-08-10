"""A deliberately simplified physical model of an OTDR-style power-vs-distance trace.

This is NOT a reproduction of any real instrument or proprietary model — it's a compact
stand-in that's physically *plausible* (linear dB attenuation, localized loss events,
a distance-independent noise floor) so the resulting classification/localization problem
has the right shape, without needing real fiber measurements.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALPHA_DB_PER_KM = 0.22  # typical single-mode fiber attenuation, ~C-band
LAUNCH_POWER_DB = 0.0
NOISE_STD_DB = 0.08
FLOOR_DB = -45.0  # detector noise floor


@dataclass(frozen=True)
class Sample:
    distance_km: np.ndarray
    power_db: np.ndarray
    fault_type: str
    fault_position_km: float | None  # None for "none"
    secondary_fault_type: str | None = None  # set only for simulate_multi_fault_trace


def _baseline(distance_km: np.ndarray) -> np.ndarray:
    return LAUNCH_POWER_DB - ALPHA_DB_PER_KM * distance_km


def _rng_noise(n: int, rng: np.random.Generator, noise_std_db: float = NOISE_STD_DB) -> np.ndarray:
    return rng.normal(0.0, noise_std_db, size=n)


def _inject_fault(
    power: np.ndarray,
    distance_km: np.ndarray,
    step_km: float,
    idx: int,
    fault_type: str,
    rng: np.random.Generator,
    noise_std_db: float,
    loss_scale: float,
) -> None:
    """Mutates `power` in place to add one fault's signature starting at index `idx`."""
    n = len(power)

    if fault_type == "fiber_cut":
        # Small Fresnel reflection spike right at the break, then straight to noise floor.
        spike = rng.uniform(1.5, 3.0)
        power[idx] += spike
        power[idx + 1 :] = FLOOR_DB + _rng_noise(n - idx - 1, rng, noise_std_db=noise_std_db) * 0.5

    elif fault_type == "connector_loss":
        # A discrete step loss (bad connector/splice), trace continues past it.
        step_loss_db = rng.uniform(0.8, 3.5) * loss_scale
        power[idx:] -= step_loss_db

    elif fault_type == "bend_loss":
        # Extra loss distributed over a short window (macrobend), then trace continues
        # at baseline slope, permanently shifted down by the loss accumulated in the bend.
        window_km = rng.uniform(0.5, 2.5)
        window_end_idx = min(n, idx + int(window_km / step_km))
        extra_alpha = rng.uniform(1.5, 4.0) * loss_scale  # extra dB/km inside the bend
        bend_len = window_end_idx - idx
        ramp = np.linspace(0, extra_alpha * window_km, bend_len)
        power[idx:window_end_idx] -= ramp
        total_extra = extra_alpha * window_km
        power[window_end_idx:] -= total_extra

    elif fault_type == "amp_gain_drift":
        # Amplifier under-compensating: attenuation slope past this point is steeper,
        # with no discrete step at the transition (harder to localize precisely).
        extra_alpha = rng.uniform(0.15, 0.45) * loss_scale  # additional dB/km beyond baseline
        tail = distance_km[idx:] - distance_km[idx]
        power[idx:] -= extra_alpha * tail

    else:
        raise ValueError(f"unknown fault_type {fault_type!r}")


def simulate_trace(
    fault_type: str,
    length_km: float = 40.0,
    step_km: float = 0.05,
    fault_position_km: float | None = None,
    noise_std_db: float = NOISE_STD_DB,
    loss_scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Sample:
    if rng is None:
        rng = np.random.default_rng()

    distance_km = np.arange(0.0, length_km, step_km)
    n = len(distance_km)
    power = _baseline(distance_km) + _rng_noise(n, rng, noise_std_db=noise_std_db)

    if fault_type == "none":
        power = np.maximum(power, FLOOR_DB)
        return Sample(distance_km, power, "none", None)

    if fault_position_km is None:
        fault_position_km = float(rng.uniform(0.15 * length_km, 0.85 * length_km))
    idx = int(fault_position_km / step_km)

    _inject_fault(power, distance_km, step_km, idx, fault_type, rng, noise_std_db, loss_scale)

    power = np.maximum(power, FLOOR_DB)
    return Sample(distance_km, power, fault_type, fault_position_km)


def simulate_multi_fault_trace(
    primary_fault_type: str,
    secondary_fault_type: str,
    length_km: float = 40.0,
    step_km: float = 0.05,
    primary_position_km: float | None = None,
    secondary_position_km: float | None = None,
    min_separation_km: float = 5.0,
    noise_std_db: float = NOISE_STD_DB,
    loss_scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Sample:
    """A trace with two independent, real fault events instead of one.

    `primary_fault_type`/`primary_position_km` is the fault the classifier is asked to
    identify and localize (the returned `fault_type`/`fault_position_km`); the secondary
    fault is an unrelated pre-existing imperfection elsewhere on the same span. Faults are
    injected in position order (upstream first), so a `fiber_cut` correctly masks anything
    downstream of it -- a real cut kills everything past the break, whether or not that
    downstream fault is the one the model is being asked to find.
    """
    if primary_fault_type == "none" or secondary_fault_type == "none":
        raise ValueError("multi-fault traces need two non-'none' fault types")
    if rng is None:
        rng = np.random.default_rng()

    distance_km = np.arange(0.0, length_km, step_km)
    n = len(distance_km)
    power = _baseline(distance_km) + _rng_noise(n, rng, noise_std_db=noise_std_db)

    if primary_position_km is None:
        primary_position_km = float(rng.uniform(0.15 * length_km, 0.85 * length_km))
    if secondary_position_km is None:
        secondary_position_km = None
        for _ in range(50):
            candidate = float(rng.uniform(0.05 * length_km, 0.95 * length_km))
            if abs(candidate - primary_position_km) >= min_separation_km:
                secondary_position_km = candidate
                break
        if secondary_position_km is None:
            secondary_position_km = max(0.0, primary_position_km - min_separation_km)

    events = sorted(
        [(primary_position_km, primary_fault_type), (secondary_position_km, secondary_fault_type)],
        key=lambda event: event[0],
    )
    for position_km, fault_type in events:
        idx = int(position_km / step_km)
        _inject_fault(power, distance_km, step_km, idx, fault_type, rng, noise_std_db, loss_scale)

    power = np.maximum(power, FLOOR_DB)
    return Sample(distance_km, power, primary_fault_type, primary_position_km, secondary_fault_type)
