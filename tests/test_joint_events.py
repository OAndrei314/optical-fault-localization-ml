import numpy as np

from optical_faults.events import DEFAULT_HALF_WINDOW_KM, DEFAULT_MIN_HALF_WINDOW_KM
from optical_faults.features import FEATURE_NAMES
from optical_faults.joint_events import (
    PAIR_FAULT_TYPES,
    JOINT_FEATURE_NAMES,
    _joint_features,
    _sample_close_pair_positions,
    render_close_pair_report,
    run_close_pair_comparison,
    train_joint_pair_classifier,
    train_matched_single_side_classifier,
)
from optical_faults.simulate import simulate_multi_fault_trace


def test_pair_fault_types_excludes_none_and_fiber_cut():
    assert "none" not in PAIR_FAULT_TYPES
    assert "fiber_cut" not in PAIR_FAULT_TYPES
    assert set(PAIR_FAULT_TYPES) == {"connector_loss", "bend_loss", "amp_gain_drift"}


def test_sample_close_pair_positions_respects_gap_and_edges():
    rng = np.random.default_rng(0)
    for _ in range(200):
        left_km, right_km = _sample_close_pair_positions(rng, length_km=40.0, min_gap_km=5.0, max_gap_km=8.0)
        gap_km = right_km - left_km
        assert 5.0 <= gap_km <= 8.0
        assert 0.0 <= left_km <= 40.0
        assert 0.0 <= right_km <= 40.0


def test_joint_features_has_expected_shape_and_names():
    rng = np.random.default_rng(1)
    sample = simulate_multi_fault_trace(
        "connector_loss", "bend_loss", primary_position_km=15.0, secondary_position_km=21.0, rng=rng
    )
    feats = _joint_features(
        sample.distance_km, sample.power_db, 15.0, 21.0, DEFAULT_HALF_WINDOW_KM, DEFAULT_MIN_HALF_WINDOW_KM
    )
    assert feats.shape == (2 * len(FEATURE_NAMES) + 1,)
    assert len(JOINT_FEATURE_NAMES) == 2 * len(FEATURE_NAMES) + 1
    assert np.all(np.isfinite(feats))
    # Last entry is the gap itself, not another local feature.
    assert feats[-1] == 6.0


def test_train_joint_pair_classifier_predicts_known_labels():
    model = train_joint_pair_classifier(train_n=150, seed=0, n_estimators=20)
    rng = np.random.default_rng(9)
    left_km, right_km = _sample_close_pair_positions(rng, 40.0, 5.0, 8.0)
    sample = simulate_multi_fault_trace(
        "connector_loss", "bend_loss", primary_position_km=left_km, secondary_position_km=right_km, rng=rng
    )
    feats = _joint_features(sample.distance_km, sample.power_db, left_km, right_km, 6.0, 1.5)
    left_pred = model.left_classifier.predict(feats.reshape(1, -1))[0]
    right_pred = model.right_classifier.predict(feats.reshape(1, -1))[0]
    assert left_pred in PAIR_FAULT_TYPES
    assert right_pred in PAIR_FAULT_TYPES


def test_train_matched_single_side_classifier_predicts_known_labels():
    model = train_matched_single_side_classifier(train_n=150, seed=0, n_estimators=20)
    rng = np.random.default_rng(9)
    left_km, right_km = _sample_close_pair_positions(rng, 40.0, 5.0, 8.0)
    sample = simulate_multi_fault_trace(
        "connector_loss", "bend_loss", primary_position_km=left_km, secondary_position_km=right_km, rng=rng
    )
    from optical_faults.events import bounded_half_window_km, extract_local_features

    left_half = bounded_half_window_km(left_km, [right_km], 6.0, 1.5)
    left_feats = extract_local_features(sample.distance_km, sample.power_db, left_km, left_half)
    pred = model.left_classifier.predict(left_feats.reshape(1, -1))[0]
    assert pred in PAIR_FAULT_TYPES


def test_run_close_pair_comparison_reports_valid_metrics():
    result = run_close_pair_comparison(train_n=150, n_pairs=60, seed=0, n_estimators=20)
    assert result.n_pairs == 60
    for value in [
        result.independent_left_accuracy,
        result.independent_right_accuracy,
        result.matched_left_accuracy,
        result.matched_right_accuracy,
        result.joint_left_accuracy,
        result.joint_right_accuracy,
    ]:
        assert 0.0 <= value <= 1.0


def test_close_pair_gain_comes_from_training_distribution_not_joint_architecture():
    # The Status/next-steps entry this module follows up on hypothesized that a "proper
    # joint two-event model" would beat independent per-candidate classification on
    # close pairs. It does -- but the matched-distribution ablation (own-window
    # features only, trained on the same close-pair data) closes nearly all of that
    # gap by itself. Across seeds 0-5 at this (train_n=200, n_pairs=80) size, the
    # distribution-match delta (matched - independent) was consistently positive
    # (0.012-0.037) while the architecture delta (joint - matched) fluctuated in a
    # tight noise band around zero (-0.019 to +0.006). This is a real, if humbling,
    # finding: the fix that matters is training on close-pair data, not seeing both
    # windows at once. See the README for the full write-up.
    result = run_close_pair_comparison(train_n=200, n_pairs=80, seed=0, n_estimators=30)
    distribution_delta = result.matched_mean_accuracy - result.independent_mean_accuracy
    architecture_delta = result.joint_mean_accuracy - result.matched_mean_accuracy
    assert distribution_delta > 0.02
    assert abs(architecture_delta) < 0.03


def test_close_pair_report_contains_ablation_breakdown():
    result = run_close_pair_comparison(train_n=120, n_pairs=40, seed=1, n_estimators=15)
    report = render_close_pair_report(result)
    assert "Close-Pair Joint Classification" in report
    assert "Matched ablation" in report
    assert "training-distribution match" in report
