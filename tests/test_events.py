import numpy as np

from optical_faults.events import (
    DEFAULT_HALF_WINDOW_KM,
    DEFAULT_MIN_HALF_WINDOW_KM,
    bounded_half_window_km,
    detect_changepoints,
    detect_changepoints_multiscale,
    detect_and_classify_events,
    extract_local_features,
    train_local_event_classifier,
)
from optical_faults.features import FEATURE_NAMES
from optical_faults.simulate import simulate_multi_fault_trace, simulate_trace


def test_detect_changepoints_finds_single_fiber_cut_position():
    rng = np.random.default_rng(0)
    sample = simulate_trace("fiber_cut", fault_position_km=20.0, rng=rng)
    candidates = detect_changepoints(sample.distance_km, sample.power_db, top_k=1)
    assert len(candidates) == 1
    assert abs(candidates[0].position_km - 20.0) < 2.0


def test_detect_changepoints_finds_no_events_on_a_healthy_trace():
    rng = np.random.default_rng(1)
    sample = simulate_trace("none", rng=rng)
    candidates = detect_changepoints(sample.distance_km, sample.power_db, top_k=2)
    assert candidates == []


def test_detect_changepoints_multiscale_recovers_two_well_separated_events():
    rng = np.random.default_rng(2)
    sample = simulate_multi_fault_trace(
        "connector_loss",
        "bend_loss",
        primary_position_km=8.0,
        secondary_position_km=32.0,
        min_separation_km=1.0,
        rng=rng,
    )
    candidates = detect_changepoints_multiscale(sample.distance_km, sample.power_db, top_k=2)
    assert len(candidates) == 2
    positions = sorted(c.position_km for c in candidates)
    assert abs(positions[0] - 8.0) < 3.0
    assert abs(positions[1] - 32.0) < 3.0


def test_detect_changepoints_multiscale_catches_weak_amp_gain_drift():
    # A pure single-scale (narrow window) scan misses most amp_gain_drift traces
    # because the slope-change signature is too weak over a short window -- the
    # multi-scale merge exists specifically to still catch these. See DEFAULT_SCALES
    # in events.py for the offline sweep that motivated this.
    found = 0
    rng = np.random.default_rng(3)
    for _ in range(20):
        sample = simulate_trace("amp_gain_drift", rng=rng)
        candidates = detect_changepoints_multiscale(sample.distance_km, sample.power_db, top_k=1)
        if candidates and abs(candidates[0].position_km - sample.fault_position_km) < 3.0:
            found += 1
    assert found >= 15


def test_extract_local_features_has_expected_shape():
    rng = np.random.default_rng(4)
    sample = simulate_trace("bend_loss", fault_position_km=18.0, rng=rng)
    features = extract_local_features(sample.distance_km, sample.power_db, 18.0)
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(features))


def test_extract_local_features_handles_center_near_span_edge():
    rng = np.random.default_rng(5)
    sample = simulate_trace("connector_loss", fault_position_km=1.0, rng=rng)
    features = extract_local_features(sample.distance_km, sample.power_db, 0.2, half_window_km=6.0)
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(features))


def test_bounded_half_window_km_returns_max_width_with_no_neighbors():
    assert bounded_half_window_km(20.0, [], max_half_window_km=6.0, min_half_window_km=1.5) == 6.0


def test_bounded_half_window_km_shrinks_for_a_close_neighbor():
    # Neighbor 5km away: a full 6km half-window would reach 1km past the neighbor's
    # position, so it must shrink. Half the gap minus the default 0.5km margin is 2.0km.
    half = bounded_half_window_km(20.0, [25.0], max_half_window_km=6.0, min_half_window_km=1.5)
    assert half == 2.0


def test_bounded_half_window_km_floors_at_min_for_a_very_close_neighbor():
    half = bounded_half_window_km(20.0, [20.8], max_half_window_km=6.0, min_half_window_km=1.5)
    assert half == 1.5


def test_bounded_half_window_km_uses_the_nearest_of_several_neighbors():
    # Nearest neighbor (26.0) gives (26.0-20.0)/2 - 0.5 = 2.5km; the farther neighbor
    # (40.0) would allow a much wider 9.5km window if it were (wrongly) used instead.
    half = bounded_half_window_km(20.0, [40.0, 26.0], max_half_window_km=6.0, min_half_window_km=1.5)
    assert half == 2.5


def test_detect_and_classify_events_shrinks_windows_for_close_candidates():
    # Two faults close enough (6km apart) that the default 6km half-window would
    # overlap, but far enough apart (> 2 * min_half_window_km) that both should still
    # be detected and classified using clipped, non-overlapping windows.
    model = train_local_event_classifier(train_n=400, seed=0, n_estimators=40)
    rng = np.random.default_rng(7)
    sample = simulate_multi_fault_trace(
        "connector_loss", "bend_loss",
        primary_position_km=15.0, secondary_position_km=21.0,
        min_separation_km=1.0, rng=rng,
    )
    results = detect_and_classify_events(
        sample.distance_km, sample.power_db, model, min_separation_km=3.0, top_k=2
    )
    assert len(results) == 2
    positions = sorted(pos for pos, _ in results)
    assert abs(positions[0] - 15.0) < 3.0
    assert abs(positions[1] - 21.0) < 3.0


def test_train_local_event_classifier_stores_window_bounds_on_model():
    model = train_local_event_classifier(train_n=50, seed=0, n_estimators=10)
    assert model.half_window_km == DEFAULT_HALF_WINDOW_KM
    assert model.min_half_window_km == DEFAULT_MIN_HALF_WINDOW_KM


def test_local_event_classifier_beats_random_baseline_on_known_positions():
    # Trained and evaluated on local windows centered on the *known* fault position
    # (no detector involved) -- isolates classifier quality from detection precision.
    model = train_local_event_classifier(train_n=300, seed=0, n_estimators=30)

    rng = np.random.default_rng(99)
    fault_types = ["fiber_cut", "connector_loss", "bend_loss", "amp_gain_drift"]
    correct = 0
    total = 40
    for i in range(total):
        ft = fault_types[i % len(fault_types)]
        sample = simulate_trace(ft, rng=rng)
        features = extract_local_features(
            sample.distance_km, sample.power_db, sample.fault_position_km, model.half_window_km
        )
        pred = model.classifier.predict(features.reshape(1, -1))[0]
        if pred == ft:
            correct += 1
    accuracy = correct / total
    assert accuracy > 0.25  # random baseline over 4 classes
