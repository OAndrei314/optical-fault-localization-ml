from optical_faults.overdetection import (
    render_overdetection_report,
    run_overdetection_check,
    run_overdetection_sweep,
)


def test_run_overdetection_check_reports_a_rate_per_fault_type():
    result = run_overdetection_check(n_per_type=40, seed=0, min_separation_km=5.0)

    assert result.n_per_type == 40
    assert set(result.over_detect_rate_by_type.keys()) == {
        "none",
        "fiber_cut",
        "connector_loss",
        "bend_loss",
        "amp_gain_drift",
    }
    for rate in result.over_detect_rate_by_type.values():
        assert 0.0 <= rate <= 1.0


def test_healthy_and_fiber_cut_traces_almost_never_over_detect():
    # No structural reason for either to produce two separable local maxima: "none"
    # has no fault signature at all, and fiber_cut collapses everything downstream to
    # the noise floor, leaving nothing for a second candidate to key off.
    result = run_overdetection_check(n_per_type=150, seed=1, min_separation_km=5.0)

    assert result.over_detect_rate_by_type["none"] == 0.0
    assert result.over_detect_rate_by_type["fiber_cut"] == 0.0


def test_bend_loss_over_detects_at_the_library_default_separation():
    # Regression guard for the finding this module exists to document: bend_loss's
    # loss ramp has two real physical transitions (ramp start, ramp end), so at
    # events.py's own DEFAULT_MIN_SEPARATION_KM (3.0km) -- not the safer 5.0km that
    # multi_event.py and the CLI actually use -- a meaningful fraction of genuinely
    # single-fault traces get reported as two events by non-max suppression alone.
    result = run_overdetection_check(n_per_type=200, seed=2, min_separation_km=3.0)

    assert result.over_detect_rate_by_type["bend_loss"] > 0.15


def test_over_detection_drops_as_min_separation_grows():
    # At a wide enough separation the two scale-scan estimates of one physical event
    # collapse back into a single candidate for every fault type tested here.
    tight = run_overdetection_check(n_per_type=150, seed=3, min_separation_km=3.0)
    wide = run_overdetection_check(n_per_type=150, seed=3, min_separation_km=5.0)

    assert tight.over_detect_rate_by_type["bend_loss"] > wide.over_detect_rate_by_type["bend_loss"]
    assert wide.over_detect_rate_by_type["bend_loss"] == 0.0


def test_overdetection_sweep_runs_all_requested_separations():
    results = run_overdetection_sweep(n_per_type=30, seed=4, separations_km=(3.0, 5.0))

    assert len(results) == 2
    assert [r.min_separation_km for r in results] == [3.0, 5.0]


def test_overdetection_report_contains_all_fault_types_and_sections():
    results = run_overdetection_sweep(n_per_type=20, seed=5, separations_km=(3.0, 5.0))
    report = render_overdetection_report(results)

    assert "Single-Fault Over-Detection Check" in report
    assert "Research Question" in report
    assert "Interpretation" in report
    for fault_type in ["none", "fiber_cut", "connector_loss", "bend_loss", "amp_gain_drift"]:
        assert fault_type in report
