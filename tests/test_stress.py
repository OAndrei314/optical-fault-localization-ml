from optical_faults.stress import render_stress_report, run_domain_shift_stress


def test_domain_shift_stress_shows_accuracy_drop():
    results = run_domain_shift_stress(train_n=300, test_n=120, seed=4, n_estimators=30)

    source = results[0]
    worst = min(results, key=lambda result: result.accuracy)

    assert source.label == "source_holdout"
    assert source.accuracy > 0.8
    assert worst.accuracy < source.accuracy


def test_stress_report_contains_worst_scenario():
    results = run_domain_shift_stress(train_n=240, test_n=100, seed=5, n_estimators=25)

    report = render_stress_report(results)

    assert "Domain Shift Stress Test" in report
    assert "accuracy drop" in report
    assert "Worst scenario" in report
