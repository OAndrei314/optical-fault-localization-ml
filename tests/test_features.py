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
