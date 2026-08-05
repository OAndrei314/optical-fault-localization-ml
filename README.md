# optical-fault-localization-ml

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

**Money question:** the AI buildout increases dependence on optical links and
transceivers. Faster optical fault localization reduces downtime, field-debug cost, and
spare-module guesswork in expensive network deployments.

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
  loss, or a gain-drift slope change past a point).
- `optical_faults/features.py` — hand-engineered features from the raw trace (attenuation
  slope, largest first-derivative jump and its position, segment-wise residuals against a
  fitted baseline) rather than feeding raw traces to a black box — this keeps the model
  small, fast, and inspectable.
- `optical_faults/model.py` — a `RandomForestClassifier` for fault type and a
  `RandomForestRegressor` for fault position (trained only on faulty examples), both from
  scikit-learn.

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
```

## Honest results

On a 1200-sample synthetic dataset (80/20 split, seed 0): **100% fault-type accuracy** and
**0.81 km mean localization error** on a 40 km span (reproduce with the commands above).

That perfect accuracy number is worth being skeptical of, not proud of — it means the
current simulator makes each fault type's signature well-separated relative to the noise
level I chose (`NOISE_STD_DB = 0.08`), not that fault classification is a solved problem.
The hand-engineered features (`slope_diff_abs`, `frac_near_floor`, `max_abs_diff`) were
built with specific fault signatures in mind, so this is closer to "the model recovered a
separation I designed in" than "the model discovered something surprising." The honest
next step, and the actual point of this repo, is to make the task harder until the model
has to work for it: raise `NOISE_STD_DB`, shrink the loss magnitudes for `connector_loss`/
`bend_loss` toward the noise floor, and add multi-fault traces — at that point I'd expect
`bend_loss` and `amp_gain_drift` to become genuinely hard to separate, since both are
"gradual extra loss" and differ only in slope shape.

## Status / next steps

Single-fault-per-trace only; multi-fault traces and a proper OTDR reflectance model
(Fresnel reflections at connectors) are the natural next extensions.

## License

MIT — see [LICENSE](LICENSE).
