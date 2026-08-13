# optical-fault-localization-ml

*Maintained by: claude-actions-daily-routine · Status: Active*
A from-scratch, fully synthetic recreation of a problem I worked on for real: classifying
and localizing faults on a coherent optical fiber link from a power-vs-distance trace
(OTDR-style), using a lightweight ML pipeline instead of hand-tuned threshold rules.

**Provenance, stated plainly:** this project is inspired by my thesis work on ASIC
bring-up for coherent optical fault localization (synthetic fiber-fault lab experiments,
labeled data generation, ML-based fault classification/localization on long-haul links).
It does **not** use any proprietary data, algorithms, hardware, or results — the physics
model, the synthetic data generator, and the ML pipeline here are all written from
scratch for this repo, deliberately simplified. Think of it as "the shape of the problem,"
not a reproduction of any real system.

## Why this is relevant right now

**Research question:** how far can synthetic optical-link fault traces go for training and
stress-testing a fault classifier/localizer before domain shift breaks performance?

**Engineering evidence:** the CLI reports fault-type accuracy, classification detail, and
localization MAE in kilometers, with an optional markdown report artifact.

Silicon photonics and coherent optical interconnects aren't just a telecom topic anymore —
co-packaged optics and optical interconnects are becoming the scaling bottleneck (and
investment focus) for AI datacenter clusters as GPU-to-GPU and rack-to-rack bandwidth
requirements outgrow copper. Automated, ML-assisted fault localization on optical links is
directly relevant to keeping those links healthy at datacenter scale, not just in long-haul
telecom.

## The problem

Given a simulated OTDR-like trace (received power in dB vs. distance in km along a fiber
span), decide:
1. **What kind of fault is present** — `none` (healthy), `fiber_cut`, `connector_loss`,
   `bend_loss`, or `amp_gain_drift`.
2. **Where it is** — distance along the span, for fault types where that's meaningful.

## Approach

- `optical_faults/simulate.py` — a simplified physical model: baseline linear attenuation
  (dB/km) plus small stochastic backscatter noise, with each fault type injecting its own
  characteristic signature (a step loss, a sharp near-total-loss cut, a distributed bend
  loss, or a gain-drift slope change past a point). `connector_loss` additionally reflects
  a brief spike back down the fiber sized off the Fresnel power reflectance at a glass-air
  interface (`((n-1)/(n+1))**2`, n ≈ 1.4682 for SMF-28 at 1550nm) scaled by a random
  per-fault `mating_quality` — a poorly mated or non-APC connector reflects visibly, a
  clean/index-matched one reflects almost nothing.
- `optical_faults/features.py` — hand-engineered features from the raw trace (attenuation
  slope, largest first-derivative jump and its position, segment-wise residuals against a
  fitted baseline) rather than feeding raw traces to a black box — this keeps the model
  small, fast, and inspectable.
- `optical_faults/model.py` — a `RandomForestClassifier` for fault type and a
  `RandomForestRegressor` for fault position (trained only on faulty examples), both from
  scikit-learn.
- `optical_faults/multi_fault.py` — evaluates the single-fault-trained model on traces
  that have a second, unrelated fault elsewhere on the span (`simulate_multi_fault_trace`
  in `simulate.py`), to measure how much the one-event-per-trace assumption costs.

## Quickstart

```bash
pip install -r requirements.txt

# Generate a labeled synthetic dataset (deterministic given --seed)
python -m optical_faults.cli generate --n 1200 --seed 0 --out data/dataset.npz

# Train + evaluate both models, print a classification report and localization MAE
python -m optical_faults.cli train --data data/dataset.npz --out models/ \
  --report reports/seed0.md

# Save one example trace per fault type as a PNG, for a quick look at the data
python -m optical_faults.cli plot-examples --seed 1 --out examples/

# Train once, then evaluate noisier / weaker-fault shifted synthetic domains
python -m optical_faults.cli stress --train-n 800 --test-n 300 --seed 0 \
  --report reports/domain-shift.md

# Train on single-fault traces, evaluate on traces with a second, unrelated fault
python -m optical_faults.cli multi-fault-stress --train-n 800 --eval-n 300 --seed 0 \
  --report reports/multi-fault.md
```

## Honest results

On a 1200-sample synthetic dataset (80/20 split, seed 0), after adding the Fresnel
reflection-spike model to `connector_loss`: **99.6% fault-type accuracy** and **0.74 km
mean localization error** on a 40 km span (reproduce with the commands above). It's no
longer a clean 100% — `bend_loss` recall dropped to 0.98 and `amp_gain_drift` precision to
0.98 — because the reflection spike gives `connector_loss` a sharper, `fiber_cut`-like
transient it didn't have before, which costs a small amount of separability elsewhere.
That's a more honest number than the old 100%, not a regression to be alarmed by: it
means the model is no longer trivially separating fault types by a single designed-in
feature.

That reflection spike also surfaced a real bug in `simulate_multi_fault_trace`: a fault
injected downstream of an upstream `fiber_cut` could add its spike on top of the
already-collapsed noise floor, letting the trace read *above* the floor in a region no
light could physically reach. Fixed by skipping fault injection entirely past the first
`fiber_cut` in position order (`optical_faults/simulate.py`), which matches what the
module's docstring already claimed but didn't fully enforce.

The hand-engineered features (`slope_diff_abs`, `frac_near_floor`, `max_abs_diff`,
`frac_near_floor_after_jump`) were built with specific fault signatures in mind, so a
near-100% number here is closer to "the model recovered a separation I designed in" than
"the model discovered something surprising."

The repo includes a domain-shift stress command that makes the task harder in exactly
that way: it trains on the source synthetic distribution, then evaluates shifted
distributions with higher noise, weaker fault signatures, and both at once. On a quick
small run (`--train-n 300 --test-n 120 --seed 4 --estimators 30`), source-holdout accuracy
was 0.975 and the high-noise/weak-fault case fell to 0.525 (a 0.450 drop). That's a
noticeably bigger drop than before the reflection-spike change (previously 0.992 → 0.692,
a 0.300 drop): the spike adds a second, mating-quality-dependent source of variability to
`connector_loss`, and that variability is exactly what gets washed out first as noise
rises and fault signatures weaken. That's the useful result — it marks where synthetic
data generation and feature design need more work before any robustness claim is
credible.

A second stress test targets the single-fault-per-trace assumption directly: train on
ordinary single-fault traces, then evaluate on traces that also carry a second, unrelated
fault elsewhere on the span (`multi-fault-stress`, `--train-n 800 --eval-n 300 --seed 0`).
Fault-type accuracy fell from 1.000 to 0.490 and localization MAE grew from 0.88 km to
5.95 km; a second run at `--seed 3` landed close by (1.000 → 0.490 accuracy, 0.92 km →
5.72 km MAE), so this isn't seed noise, and it's a somewhat larger drop than before the
reflection-spike change (previously 0.537 and 0.480). The dominant failure mode is still
structural, not a modeling gap: an upstream `fiber_cut` drives the whole downstream trace
to the noise floor, so a fault located past it is genuinely unobservable from the trace
alone — the hand-engineered features (single largest jump, two-half slope split) were
never designed to separate two events, and they don't.

## Status / next steps

Single-fault localization, the Fresnel connector-reflectance model, and both stress tests
(domain shift, multi-fault interference) are done and honestly measured. What's left:
calibration against public/realistic OTDR trace statistics (return-loss distributions for
real connector grades), and — if multi-fault traces need to be *handled* well rather than
just measured — a multi-label classifier or per-segment sliding-window features instead of
the current single-event feature vector.

## License

MIT — see [LICENSE](LICENSE).
