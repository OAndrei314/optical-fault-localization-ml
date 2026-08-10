from optical_faults.multi_fault import render_multi_fault_report, run_multi_fault_stress


def test_multi_fault_stress_reports_a_gap_or_holds_steady():
    # Small + fast on purpose so this stays cheap in CI. We can't assert the direction of
    # the effect for a *specific* tiny seed/size (that's the honest point of measuring it),
    # only that both conditions produce valid metrics in range.
    result = run_multi_fault_stress(train_n=300, eval_n=150, seed=1, n_estimators=30)

    assert 0.0 <= result.single_fault_accuracy <= 1.0
    assert 0.0 <= result.multi_fault_accuracy <= 1.0
    assert result.single_fault_mae_km >= 0.0
    assert result.multi_fault_mae_km >= 0.0
    assert result.n_multi_fault == 150


def test_multi_fault_report_contains_comparison_table():
    result = run_multi_fault_stress(train_n=200, eval_n=100, seed=2, n_estimators=20)

    report = render_multi_fault_report(result)

    assert "Multi-Fault Interference Stress Test" in report
    assert "single-fault holdout" in report
    assert "unrelated second fault" in report
    assert "Accuracy drop" in report
