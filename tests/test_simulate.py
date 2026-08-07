import numpy as np

from optical_faults.simulate import FLOOR_DB, simulate_trace


def test_healthy_trace_is_roughly_monotonic_decreasing():
    rng = np.random.default_rng(0)
    sample = simulate_trace("none", rng=rng)
    # Fit a line; slope should be negative (attenuation), noise aside.
    slope = np.polyfit(sample.distance_km, sample.power_db, 1)[0]
    assert slope < 0
    assert sample.fault_position_km is None


def test_fiber_cut_drops_to_floor_after_position():
    rng = np.random.default_rng(1)
    sample = simulate_trace("fiber_cut", fault_position_km=20.0, rng=rng)
    tail = sample.power_db[sample.distance_km > 21.0]
    assert np.all(tail <= FLOOR_DB + 2.0)
    head = sample.power_db[sample.distance_km < 19.0]
    assert np.mean(head) > FLOOR_DB + 5.0


def test_connector_loss_creates_a_visible_step():
    rng = np.random.default_rng(2)
    sample = simulate_trace("connector_loss", fault_position_km=15.0, rng=rng)
    before = np.mean(sample.power_db[(sample.distance_km > 14.0) & (sample.distance_km < 14.9)])
    after = np.mean(sample.power_db[(sample.distance_km > 15.1) & (sample.distance_km < 16.0)])
    assert before - after > 0.4  # step loss should be clearly visible above noise


def test_unknown_fault_type_raises():
    try:
        simulate_trace("not_a_real_fault")
        assert False, "expected ValueError"
    except ValueError:
        pass
