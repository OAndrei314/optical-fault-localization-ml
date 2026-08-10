from optical_faults.dataset import generate_dataset
from optical_faults.model import render_markdown_report, train_and_evaluate


def test_classifier_beats_random_baseline():
    # Small + fast on purpose so this stays cheap in CI; 5 classes -> random baseline 0.20.
    X, y_type, y_position = generate_dataset(n=400, seed=0)
    result = train_and_evaluate(X, y_type, y_position, seed=0, n_estimators=50)
    assert result.accuracy > 0.5


def test_localization_error_is_bounded_on_a_40km_span():
    X, y_type, y_position = generate_dataset(n=400, seed=1)
    result = train_and_evaluate(X, y_type, y_position, seed=1, n_estimators=50)
    # A model that's actually learning something should beat "always guess mid-span"
    # (which on a 40km span with faults in [0.15L, 0.85L] has an MAE of roughly 7-8km).
    assert result.localization_mae_km < 7.0


def test_markdown_report_contains_research_context_and_metrics():
    X, y_type, y_position = generate_dataset(n=120, seed=2)
    result = train_and_evaluate(X, y_type, y_position, seed=2, n_estimators=20)

    report = render_markdown_report(result, sample_count=len(X))

    assert "Research Question" in report
    assert "Engineering Evidence" in report
    assert "Fault-type accuracy" in report
