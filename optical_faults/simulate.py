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

# Fresnel power reflectance at an unmated glass-air interface (SMF-28 core index at
# 1550nm, n ~= 1.4682): R = ((n-1)/(n+1))**2. This is the physical reason a poorly
# mated connector (dirty, misaligned, or a flat PC polish instead of angle-polished
# APC) reflects a visible spike back down the fiber, on top of its insertion loss,
# while a clean, index-matched or angle-polished connector reflects almost nothing.
FIBER_CORE_INDEX = 1.4682
FRESNEL_REFLECTANCE = ((FIBER_CORE_INDEX - 1) / (FIBER_CORE_INDEX + 1)) ** 2
FRESNEL_REFLECTANCE_DB = 10 * np.log10(FRESNEL_REFLECTANCE)  # ~ -14.4 dB
# Scales the full glass-air reflectance down to a plausible spike height in a logged
# OTDR trace (captured backscatter, not the raw one-way interface reflectance) --
# a calibration knob, not a claimed instrument-accurate value.
CONNECTOR_REFLECTANCE_TRACE_SCALE = 0.3

# Connector polish grades and their typical return loss (positive-dB convention:
# bigger = less reflective), per the commonly cited industry figures in Telcordia
# GR-326-CORE and the IEC 61755-3-x connector-grade classifications: PC (flat
# physical contact) ~35-45 dB, UPC (ultra physical contact) ~45-55 dB, APC (angled
# physical contact, an 8 degree angle that deflects the reflection out of the core)
# ~55-65 dB. These are illustrative typical/spec-range values from public connector
# datasheets, not measurements of any specific vendor's parts. `prevalence` is an
# assumed field-population mix (APC/UPC dominant in modern long-haul and datacenter
# links, plain PC mostly legacy) -- a stated modeling assumption, not a measured
# deployment survey.
CONNECTOR_GRADES: dict[str, tuple[float, float, float]] = {
    # grade: (mean_return_loss_db, std_db, prevalence)
    "PC": (40.0, 3.0, 0.15),
    "UPC": (50.0, 3.0, 0.35),
    "APC": (60.0, 3.0, 0.50),
}
_CONNECTOR_GRADE_NAMES = list(CONNECTOR_GRADES.keys())
_CONNECTOR_GRADE_WEIGHTS = np.array([w for _, _, w in CONNECTOR_GRADES.values()])
_CONNECTOR_GRADE_WEIGHTS = _CONNECTOR_GRADE_WEIGHTS / _CONNECTOR_GRADE_WEIGHTS.sum()

# The two physical endpoints `mating_quality` interpolates between: a fully unmated
# glass-air interface (worst case, mating_quality=1.0) has ~14.4 dB return loss
# (`FRESNEL_REFLECTANCE_DB` above, in the same positive-dB convention); a best-case
# APC connector (mating_quality=0.0) is assumed around 65 dB.
UNMATED_RETURN_LOSS_DB = abs(FRESNEL_REFLECTANCE_DB)  # ~14.4 dB
BEST_CASE_RETURN_LOSS_DB = 65.0


def sample_connector_mating_quality(rng: np.random.Generator) -> float:
    """Draws a `mating_quality` in [0, 1] from a realistic connector-grade population
    instead of a flat `Uniform(0, 1)`. Picks a polish grade (PC/UPC/APC) weighted by
    assumed field prevalence, draws that grade's return loss from a Gaussian around
    its typical spec, then maps return loss linearly onto [0, 1] between the fully
    unmated (`UNMATED_RETURN_LOSS_DB`) and best-case (`BEST_CASE_RETURN_LOSS_DB`)
    endpoints -- lower return loss (more reflective) means higher `mating_quality`.
    """
    grade = str(rng.choice(_CONNECTOR_GRADE_NAMES, p=_CONNECTOR_GRADE_WEIGHTS))
    mean_rl_db, std_rl_db, _ = CONNECTOR_GRADES[grade]
    return_loss_db = float(rng.normal(mean_rl_db, std_rl_db))
    return_loss_db = float(np.clip(return_loss_db, UNMATED_RETURN_LOSS_DB, BEST_CASE_RETURN_LOSS_DB))
    span = BEST_CASE_RETURN_LOSS_DB - UNMATED_RETURN_LOSS_DB
    mating_quality = (BEST_CASE_RETURN_LOSS_DB - return_loss_db) / span
    return float(np.clip(mating_quality, 0.0, 1.0))


@dataclass(frozen=True)
class Sample:
    distance_km: np.ndarray
    power_db: np.ndarray
    fault_type: str
    fault_position_km: float | None  # None for "none"
    secondary_fault_type: str | None = None  # set only for simulate_multi_fault_trace
    secondary_fault_position_km: float | None = None  # set only for simulate_multi_fault_trace


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
        # A discrete step loss (bad connector/splice), trace continues past it. A
        # poorly-mated connector also partially reflects light back toward the
        # source (Fresnel reflection, see FRESNEL_REFLECTANCE_DB above), producing a
        # brief spike right at the connector; a well-mated/APC connector shows
        # almost none. `mating_quality` in [0, 1] interpolates between those regimes,
        # drawn from a realistic PC/UPC/APC connector-grade population rather than a
        # flat Uniform(0, 1) -- see `sample_connector_mating_quality`.
        step_loss_db = rng.uniform(0.8, 3.5) * loss_scale
        mating_quality = sample_connector_mating_quality(rng)
        reflection_spike_db = (
            mating_quality * abs(FRESNEL_REFLECTANCE_DB) * CONNECTOR_REFLECTANCE_TRACE_SCALE
        )
        power[idx] += reflection_spike_db
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
    cut_idx = None
    for position_km, fault_type in events:
        idx = int(position_km / step_km)
        if cut_idx is not None and idx > cut_idx:
            # No light reaches past an upstream full break, so nothing downstream of
            # it -- including a reflection spike -- can show up in the trace, even
            # though the ground-truth label still records that a fault is there.
            continue
        _inject_fault(power, distance_km, step_km, idx, fault_type, rng, noise_std_db, loss_scale)
        if fault_type == "fiber_cut":
            cut_idx = idx

    power = np.maximum(power, FLOOR_DB)
    return Sample(
        distance_km,
        power,
        primary_fault_type,
        primary_position_km,
        secondary_fault_type,
        secondary_position_km,
    )
