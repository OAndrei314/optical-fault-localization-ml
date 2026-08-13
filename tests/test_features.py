import numpy as np

from optical_faults.features import FEATURE_NAMES, extract_features
from optical_faults.simulate import simulate_trace


def test_feature_vector_shape_matches_names():
    rng = np.random.default_rng(0)
    sample = simulate_trace("connector_loss", rng=rng)
    feats = extract_features(sample.distance_km, sample.power_db)
    assert feats.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(feats))


def test_fiber_cut_has_high_max_abs_diff_relative_to_none():
    rng = np.random.default_rng(0)
    none_sample = simulate_trace("none", rng=rng)
    cut_sample = simulate_trace("fiber_cut", fault_position_km=20.0, rng=rng)

    none_feats = extract_features(none_sample.distance_km, none_sample.power_db)
    cut_feats = extract_features(cut_sample.distance_km, cut_sample.power_db)

    idx = FEATURE_NAMES.index("max_abs_diff")
    assert cut_feats[idx] > none_feats[idx]


def test_fiber_cut_stays_near_floor_after_its_jump_unlike_connector_loss():
    # Both fault types can produce a similarly large single-step jump (a fiber_cut's
    # break, or a poorly-mated connector's reflection spike), but only fiber_cut
    # stays pinned at the noise floor afterward -- that's what
    # frac_near_floor_after_jump is meant to capture.
    rng = np.random.default_rng(0)
    cut_sample = simulate_trace("fiber_cut", fault_position_km=20.0, rng=rng)
    connector_sample = simulate_trace("connector_loss", fault_position_km=20.0, rng=rng)

    cut_feats = extract_features(cut_sample.distance_km, cut_sample.power_db)
    connector_feats = extract_features(connector_sample.distance_km, connector_sample.power_db)

    idx = FEATURE_NAMES.index("frac_near_floor_after_jump")
    assert cut_feats[idx] > 0.9
    assert connector_feats[idx] < cut_feats[idx]
