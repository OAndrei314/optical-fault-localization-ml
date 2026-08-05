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


def _baseline(distance_km: np.ndarray) -> np.ndarray:
    return LAUNCH_POWER_DB - ALPHA_DB_PER_KM * distance_km


def _rng_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, NOISE_STD_DB, size=n)


def simulate_trace(
    fault_type: str,
    length_km: float = 40.0,
    step_km: float = 0.05,
    fault_position_km: float | None = None,
    rng: np.random.Generator | None = None,
) -> Sample:
    if rng is None:
        rng = np.random.default_rng()

    distance_km = np.arange(0.0, length_km, step_km)
    n = len(distance_km)
    power = _baseline(distance_km) + _rng_noise(n, rng)

    if fault_type == "none":
        power = np.maximum(power, FLOOR_DB)
        return Sample(distance_km, power, "none", None)

    if fault_position_km is None:
        fault_position_km = float(rng.uniform(0.15 * length_km, 0.85 * length_km))
    idx = int(fault_position_km / step_km)

    if fault_type == "fiber_cut":
        # Small Fresnel reflection spike right at the break, then straight to noise floor.
        spike = rng.uniform(1.5, 3.0)
        power[idx] += spike
        power[idx + 1 :] = FLOOR_DB + _rng_noise(n - idx - 1, rng) * 0.5

    elif fault_type == "connector_loss":
        # A discrete step loss (bad connector/splice), trace continues past it.
        step_loss_db = rng.uniform(0.8, 3.5)
        power[idx:] -= step_loss_db

    elif fault_type == "bend_loss":
        # Extra loss distributed over a short window (macrobend), then trace continues
        # at baseline slope, permanently shifted down by the loss accumulated in the bend.
        window_km = rng.uniform(0.5, 2.5)
        window_end_idx = min(n, idx + int(window_km / step_km))
        extra_alpha = rng.uniform(1.5, 4.0)  # extra dB/km inside the bend
        bend_len = window_end_idx - idx
        ramp = np.linspace(0, extra_alpha * window_km, bend_len)
        power[idx:window_end_idx] -= ramp
        total_extra = extra_alpha * window_km
        power[window_end_idx:] -= total_extra

    elif fault_type == "amp_gain_drift":
        # Amplifier under-compensating: attenuation slope past this point is steeper,
        # with no discrete step at the transition (harder to localize precisely).
        extra_alpha = rng.uniform(0.15, 0.45)  # additional dB/km beyond baseline
        tail = distance_km[idx:] - distance_km[idx]
        power[idx:] -= extra_alpha * tail

    else:
        raise ValueError(f"unknown fault_type {fault_type!r}")

    power = np.maximum(power, FLOOR_DB)
    return Sample(distance_km, power, fault_type, fault_position_km)
