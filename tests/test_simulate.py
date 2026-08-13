import numpy as np

from optical_faults.simulate import (
    FLOOR_DB,
    FRESNEL_REFLECTANCE_DB,
    simulate_multi_fault_trace,
    simulate_trace,
)


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


def test_fresnel_reflectance_is_physically_sane():
    # Glass-air Fresnel reflectance at a fiber core index (~1.47) should be a small
    # negative dB value -- a few percent of incident power reflects at an unmated
    # glass-air interface, not near-total reflection and not negligible.
    assert -20.0 < FRESNEL_REFLECTANCE_DB < -10.0


def test_connector_loss_does_not_collapse_to_floor():
    # Unlike a fiber_cut, a connector_loss trace continues past the fault at a
    # moderately reduced (not floor) level, even with a reflection spike right at
    # the fault. Check across several seeds since the reflection spike height and
    # step loss are both randomized.
    for seed in range(5):
        rng = np.random.default_rng(seed)
        sample = simulate_trace("connector_loss", fault_position_km=15.0, rng=rng)
        tail = sample.power_db[sample.distance_km > 16.0]
        assert np.mean(tail) > FLOOR_DB + 5.0


def test_connector_loss_can_show_a_reflection_spike_above_the_settled_level():
    # A poorly-mated connector reflects a visible spike right at the fault index,
    # above the level the trace settles to just past it. Noise is disabled so the
    # comparison isn't confounded by measurement noise; across enough seeds, most
    # `mating_quality` draws should produce a spike clearly above the settled level.
    spikes_seen = 0
    step_km = 0.05
    idx = int(15.0 / step_km)
    for seed in range(30):
        rng = np.random.default_rng(seed)
        sample = simulate_trace(
            "connector_loss", fault_position_km=15.0, step_km=step_km, noise_std_db=0.0, rng=rng
        )
        settled = sample.power_db[idx + 5]
        if sample.power_db[idx] > settled + 0.5:
            spikes_seen += 1
    assert spikes_seen >= 5


def test_unknown_fault_type_raises():
    try:
        simulate_trace("not_a_real_fault")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_multi_fault_reports_primary_type_and_position():
    rng = np.random.default_rng(3)
    sample = simulate_multi_fault_trace(
        "connector_loss",
        "bend_loss",
        primary_position_km=25.0,
        secondary_position_km=10.0,
        rng=rng,
    )
    assert sample.fault_type == "connector_loss"
    assert sample.fault_position_km == 25.0
    assert sample.secondary_fault_type == "bend_loss"


def test_multi_fault_none_type_rejected():
    try:
        simulate_multi_fault_trace("none", "connector_loss")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_upstream_fiber_cut_masks_downstream_secondary_fault():
    # A fiber_cut early in the span drives everything past it to the noise floor -- a real
    # cut hides any fault located further downstream, regardless of injection order.
    rng = np.random.default_rng(7)
    sample = simulate_multi_fault_trace(
        "fiber_cut",
        "connector_loss",
        primary_position_km=8.0,
        secondary_position_km=35.0,
        min_separation_km=1.0,
        rng=rng,
    )
    tail = sample.power_db[sample.distance_km > 9.0]
    assert np.all(tail <= FLOOR_DB + 2.0)
