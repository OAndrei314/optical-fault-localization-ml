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
  clean/index-matched one reflects almost nothing. `mating_quality` is drawn from a
  simulated population of PC/UPC/APC connector polish grades weighted by assumed field
  prevalence, with each grade's return loss centered on the typical/spec-range figures
  commonly cited in Telcordia GR-326-CORE and IEC 61755-3-x (PC ≈ 40 dB, UPC ≈ 50 dB,
  APC ≈ 60 dB return loss) — not a flat `Uniform(0, 1)` draw (`sample_connector_mating_quality`
  in `simulate.py`).
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
- `optical_faults/events.py` — an attempt to actually *handle* multi-fault traces
  instead of just measuring the damage: a changepoint scan slides a before/after
  window across the trace and scores every position by (single-sample jump) +
  (before/after slope change), picks up to `top_k` local maxima with non-max
  suppression, and classifies each one independently using the existing feature
  extractor computed on a short local window instead of the full 40km span. Runs the
  scan at two window scales (narrow for precise jump localization, wide to catch
  `amp_gain_drift`'s weak slope-only signature — a narrow window alone detects 0/50 of
  those, a wide one 47/50) and merges the results.
- `optical_faults/multi_event.py` — evaluates that pipeline on two-fault traces,
  scored against *both* ground-truth events (not just the one the old pipeline is told
  to look for), via greedy nearest-position matching.
- `optical_faults/joint_events.py` — tests whether a *joint* two-event model (both
  windows' features + the gap between them, fed to per-side classifier heads) beats
  independent per-candidate classification on fault pairs too close for
  `bounded_half_window_km` to fully separate, and includes an ablation that isolates
  *why* it wins when it does.

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

# Changepoint-scan-and-classify: detect + classify BOTH events on a two-fault trace
python -m optical_faults.cli multi-event-detect --train-n 800 --n-traces 300 --seed 0 \
  --report reports/multi-event.md

# Does the multi-event detector spuriously report a second event on single-fault or
# healthy traces? Sweeps min_separation_km to show the false-split-vs-close-fault tradeoff
python -m optical_faults.cli overdetection-check --n-per-type 300 --seed 0 \
  --report reports/overdetection.md

# For fault pairs too close to fully separate: does a joint two-event model beat
# independent per-candidate classification, and if so, why?
python -m optical_faults.cli close-pair-classify --train-n 800 --n-pairs 300 --seed 0 \
  --report reports/close-pair.md
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

### Multi-event detection: handling the multi-fault case, not just measuring it

The previous README ended with "a multi-label classifier or per-segment sliding-window
features are the natural next steps if multi-fault traces need to be *handled* well
rather than just measured." That's what `events.py` and `multi_event.py` do: a
changepoint scan proposes up to two candidate event positions per trace, and each is
classified independently on a local window, so the pipeline is no longer told in
advance that there's exactly one event to find.

On 300 two-fault evaluation traces (600 ground-truth events, `--seed 0`, otherwise
default parameters), matched against ground truth by nearest-position within 3km:
**73.7% detection recall** (442/600 ground-truth events had a matching detected
candidate), **73.5% fault-type accuracy on the matched events**, **0.691 km
localization MAE on the matched events**, and **zero false positives** (every detected
candidate matched a real event). Across seeds 0–3, recall stayed in 0.69–0.74, type
accuracy in 0.69–0.735, MAE in 0.69–0.75 km, with false positives at 0 every time —
this isn't a lucky seed.

Read those numbers against `multi-fault-stress`'s baseline carefully, because they're
not measuring quite the same thing: the old pipeline is told which fault to look for
and scores 0.490 accuracy / 5.946 km MAE finding *that one*; this pipeline isn't told
anything and has to find *both*, and it localizes the ones it does find over 8x more
precisely (0.691 km vs. 5.946 km MAE). Recall tops out well short of 1.0 for a physical
reason, not a detector weakness: an upstream `fiber_cut` makes anything downstream
genuinely unobservable, the same masking behind the old pipeline's failures.

### Closing the window-contamination gap

The matched-event type accuracy above (0.735) used to be a lot further below the
standalone local classifier's accuracy on known, uncontaminated fault positions (1.0,
verified separately in `test_local_event_classifier_beats_random_baseline_on_known_positions`).
The cause was exactly what it looked like: with a fixed 6km half-window and a 5km
`min_separation_km`, two candidates 5-8km apart routinely had overlapping feature
windows, so classifying one candidate could pick up the other fault's signature.

`events.py` now fixes this two ways: `bounded_half_window_km` clips a candidate's
window against its nearest *other* detected candidate (half the gap between them,
minus a small margin), and `train_local_event_classifier` trains on windows of
*varying* width (uniformly sampled between 1.5km and 6km) instead of only the full
6km width, so the classifier has actually seen what a clipped window looks like
instead of being evaluated on a distribution it never trained on.

Re-running `multi-event-detect` at the same config as before (`--train-n 800
--n-traces 300 --min-separation-km 5.0`, seeds 0-3) after the fix:

| seed | before | after |
|------|--------|-------|
| 0    | 0.735  | 0.835 |
| 1    | 0.728  | 0.902 |
| 2    | 0.709  | 0.861 |
| 3    | 0.689  | 0.929 |

Detection recall and localization MAE on matched events are unchanged (identical to
the decimal across all four seeds) — only the classification stage changed, so that's
the expected signature of a working fix, not a coincidence.

An ablation (isolating variable-width *training* from window *clipping* at inference,
same seeds) attributes most of the gain to training on variable widths: 0.817-0.913
with variable-width training alone but no clipping (still using the full 6km window at
inference), vs. 0.835-0.929 with both. Clipping alone is a smaller, consistent +1-2
point improvement on top — the bigger lesson was that the *classifier itself* had
never been trained on anything but pristine, uncontaminated, full-width windows, so it
generalized poorly to any narrower or noisier window shape, independent of whether
that narrower window was clipped for a good reason.

The remaining gap to 1.0 is smaller now but not zero: detection-position slop still
feeds a slightly off-center window into the classifier, and windows can still overlap
when `min_half_window_km` (1.5km) plus margin doesn't fully separate two faults closer
than ~4km apart. `bounded_half_window_km` and the variable-width training are both
exercised directly in `tests/test_events.py`, and `tests/test_multi_event.py` guards
against a regression back toward the old ~0.7-0.82 ceiling.

### Over-detection: does the detector cry wolf on traces that only have one fault?

Every result above scores the multi-event pipeline on traces that genuinely have two
faults. That leaves an obvious, more common case unmeasured: fed a trace with only
*one* real fault, or none at all, does `detect_changepoints_multiscale` spuriously
report a second event? `optical_faults/overdetection.py` checks this directly —
no classifier involved, since over-detection is purely a property of the changepoint
scan, not the classification stage on top of it.

Running the detector (no classifier) on 300 traces per fault type at each of three
`min_separation_km` settings (`--n-per-type 300 --seed 0`):

| min_separation_km | none | fiber_cut | connector_loss | bend_loss | amp_gain_drift |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3.0 | 0.000 | 0.000 | 0.310 | 0.333 | 0.000 |
| 4.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

At `events.py`'s own `DEFAULT_MIN_SEPARATION_KM` (3.0 — the value anyone gets calling
`detect_changepoints_multiscale` without overriding it), a genuinely single-fault
`bend_loss` or `connector_loss` trace gets reported as two events roughly a third of
the time. `bend_loss` has a real structural reason: its loss ramps in over
`window_km`, so a single fault has two physical transitions (ramp start, ramp end),
and if they're farther apart than `min_separation_km`, non-max suppression can't tell
they're the same event. `connector_loss` is subtler: it's a single step discontinuity,
but `detect_changepoints`'s before/after slope term is an OLS fit over a
`window_km`-wide box, so the step biases that slope estimate over a `window_km`-wide
range around it, not just exactly at it — which can produce a second local maximum a
few km away. `fiber_cut`, `amp_gain_drift`, and healthy traces stay at 0.000 at every
separation tested, because neither a total-loss collapse nor a smooth slope-only
drift gives the scan two separable local maxima to find.

This is exactly why `multi_event.py`'s own evaluation and the CLI both default to
`min_separation_km=5.0` rather than the module-level 3.0 default — that choice was
already load-bearing, just never measured or written down before now. It also puts a
number on the other side of the "closing the window-contamination gap" tradeoff:
shrinking `min_separation_km` to resolve two *real*, closely-spaced faults directly
buys more single-fault traces getting reported as two.

### Connector-grade calibration: a more realistic reflectance prior changes two numbers, not zero

The previous `mating_quality` draw was `Uniform(0, 1)` — every simulated connector was
equally likely to be pristine or catastrophically misaligned. That's not what a real
fiber plant looks like: most modern links use UPC or APC connectors specifically
*because* they're low-reflectance, with plain PC mostly confined to legacy patch
panels. `sample_connector_mating_quality` (`simulate.py`) replaces the flat draw with a
population of PC/UPC/APC grades (weights 0.15/0.35/0.50) whose return loss is drawn
from a Gaussian around the typical/spec-range figures commonly cited in Telcordia
GR-326-CORE and IEC 61755-3-x — a stated modeling assumption about field mix, not a
measured deployment survey, but grounded in real connector-grade physics rather than
an arbitrary interpolation. The mean drawn `mating_quality` is ~0.23 (measured directly:
`test_sample_connector_mating_quality_reflects_realistic_grade_mix`), down from a
`Uniform(0,1)` mean of 0.5 — most simulated connectors now reflect only a little.

Re-running every experiment above at its original seed/size after this change, most
numbers barely move — the global classifier's train/test accuracy (99.6-100% across
seeds 0-3), the domain-shift stress drop (0.967 → 0.558, a 0.408-point drop at
`--seed 4`, vs. 0.975 → 0.525 / 0.450 before), and the multi-fault-interference stress
(1.000 → 0.500 at `--seed 0`, vs. 0.490 before) all land in the same range already
documented above, well within normal seed-to-seed variance.

Two numbers moved for a real, traceable reason, though:

- **Multi-event matched-type accuracy fell**, from the post-window-fix 0.835-0.929
  range down to **0.764-0.870** across seeds 0-3 (`--train-n 800 --n-traces 300
  --min-separation-km 5.0`). Detection recall (0.697-0.720) and false positives (0 in
  every seed) were unaffected — this is purely a classification-stage effect.
- **`connector_loss` over-detection at `min_separation_km=3.0` fell**, from 0.310 to
  **0.137** (`overdetection-check --n-per-type 300 --seed 0`); `bend_loss` (0.333) and
  every other fault type were unaffected.

Both come from the same mechanism, and it's worth being honest about what it says
about the *previous* results: under the old uniform prior, a large fraction of
`connector_loss` traces had a big, easy-to-see reflection spike, which was acting as a
shortcut feature — it made those traces both *more separable* from other fault types
inside a contaminated or clipped window (inflating matched-type accuracy) and *more
likely* to look like two discrete events to the changepoint scan (inflating
over-detection). With a realistic connector population, most `connector_loss` traces
no longer have that crutch, so both numbers move in the direction you'd expect once
it's gone — worse classification accuracy under contamination, but also less spurious
over-detection. Neither the previous nor the current number was wrong, but the
previous one was measuring a synthetic population that doesn't match a real fiber
plant, and this is a concrete example of exactly the failure mode this repo's domain
shift work is about: an unrealistic training/eval distribution can flatter a metric
without anyone noticing until the distribution is corrected.

### The "proper joint two-event model": a real gain, but not for the reason expected

The previous version of this README's next-steps section proposed "a proper joint
two-event model instead of independent per-candidate classification" for faults
closer together than `bounded_half_window_km` can fully separate. `joint_events.py`
builds and tests exactly that: a classifier that sees *both* candidates' clipped
local features plus the gap between them, instead of each side being classified
blind to the other.

On 300 close-pair traces (gap uniform in [5, 8] km, `--seed 0`, `connector_loss` /
`bend_loss` / `amp_gain_drift` only — see below for why `fiber_cut` is excluded),
scored against the *known* fault positions (isolating classification from
detection): independent per-candidate classification (the existing
`train_local_event_classifier`, trained on clean single-fault traces) reaches
**0.958 mean accuracy**; the joint model reaches **0.998** — a real, seed-stable
**+0.040** improvement (seeds 0-3 ranged +0.038 to +0.057).

But *why* it wins turned out not to be what the original next-steps note assumed.
An ablation — a classifier trained on the *same* close-pair data as the joint
model, but given only its own window's features, never the other side's or the
gap — reaches **1.000** mean accuracy on the same eval set, matching or slightly
*beating* the joint model. Decomposing the total gain: **+0.042** of the +0.040
(all of it, functionally) comes from training on close-pair data instead of clean
single-fault traces; the joint architecture's own contribution on top of that is
**-0.002** — noise, not signal. This held at a second, much wider gap range too
([14, 18] km, where windows aren't clipped at all): distribution match still
bought +0.050 to +0.070, and the joint architecture's contribution stayed in a
tight band around zero (-0.019 to +0.006 across seeds 0-5 at a smaller/faster
test size).

That's a more useful finding than "the joint model works" would have been: the
actual bottleneck was never that each side's classifier *couldn't see* the other
window — it was that `train_local_event_classifier` had never been trained on
anything but clean, isolated single-fault traces, the same class of train/eval
mismatch the window-contamination fix earlier in this README already fixed once
for window *width*. The fix that matters is mixing close-pair examples into that
training distribution, not adding cross-window architecture. `fiber_cut` is
excluded from this experiment's fault-type pool specifically because an upstream
`fiber_cut` makes a downstream event *unobservable*, not *contaminated* — a
different failure mode this comparison isn't designed to measure (see
`multi_event.py`'s own results above).

## Status / next steps

Single-fault localization, the Fresnel connector-reflectance model, both stress tests
(domain shift, multi-fault interference), a first pass at actually *handling*
multi-fault traces (changepoint-scan-and-classify), the window-contamination fix that
closed most of the matched-type-accuracy gap, the single-fault over-detection check
that quantifies why `min_separation_km=5.0` was the right default, calibrating the
connector-reflectance prior against realistic PC/UPC/APC field statistics, and testing
the "proper joint two-event model" this section used to propose are done and honestly
measured — including the fact that the joint model's own architecture turned out not
to be the reason it beats independent classification (see above): training
`train_local_event_classifier` on close-pair data, not cross-window features, is what
actually closes the gap. What's left, following directly from that result: fold a
close-pair training regime into `train_local_event_classifier` itself (mixing in
examples like `train_matched_single_side_classifier`'s, rather than keeping it a
separate ablation-only code path) so `detect_and_classify_events`'s general pipeline
gets this accuracy gain in practice, not just in this isolated comparison — plus
re-measuring `multi-event-detect`'s matched-type accuracy afterward to see how much of
it that recovers. The matched-type-accuracy drop from the connector-grade calibration
change (documented above) is a candidate root cause to check first, since it's the
same class of train/eval-mismatch problem this section just resolved for window width.

## License

MIT — see [LICENSE](LICENSE).
