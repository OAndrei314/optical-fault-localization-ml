from optical_faults.multi_event import render_multi_event_report, run_multi_event_detection


def test_multi_event_detection_reports_valid_metrics():
    # Small + fast for CI, same style as test_multi_fault.py's stress test.
    result = run_multi_event_detection(
        train_n=200, n_traces=60, seed=1, n_estimators=20, min_separation_km=5.0
    )

    assert result.n_traces == 60
    assert result.n_ground_truth_events == 120
    assert 0 <= result.n_matched <= result.n_ground_truth_events
    assert result.n_false_positives >= 0
    assert 0.0 <= result.detection_recall <= 1.0
    if result.n_matched > 0:
        assert 0.0 <= result.matched_type_accuracy <= 1.0
        assert result.matched_localization_mae_km >= 0.0


def test_matched_type_accuracy_reflects_window_contamination_fix():
    # Regression guard for the window-clipping + variable-width-training fix in
    # events.py: before it, matched_type_accuracy at this exact (seed, size) config
    # topped out at 0.818 across seeds 1/2/5/6 (0.741/0.714/0.818/0.739). After it,
    # the same seeds land at 0.83-0.92. 0.78 sits comfortably above the old ceiling
    # without being so close to the new floor that normal RandomForest variance
    # could flake it.
    result = run_multi_event_detection(
        train_n=200, n_traces=60, seed=1, n_estimators=20, min_separation_km=5.0
    )
    assert result.matched_type_accuracy > 0.78


def test_multi_event_report_contains_comparison_context():
    result = run_multi_event_detection(train_n=150, n_traces=40, seed=2, n_estimators=15)

    report = render_multi_event_report(result)

    assert "Multi-Event Detection" in report
    assert "Detection recall" in report
    assert "False positives" in report
    assert "Fault-type accuracy on matched events" in report
