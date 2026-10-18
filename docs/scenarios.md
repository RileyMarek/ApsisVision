# Seeded failure experiments

`scenarios/demo.json` defines the same seven conditions used by the viewer's
presets and the command-line suite. Each run uses the actual raster → corners →
association → PnP → EKF pipeline. Ground truth is read only for scoring.

```sh
cargo build --release --locked -p apsis-cli
python scripts/scenario_suite.py
# Short CI subset:
python scripts/scenario_suite.py --seeds 42 --cases baseline delay occlusion
```

Windows can invoke the workspace's WSL executable through the same adapter as the
demo server. `output/suite/report.csv` compares runs; `report.json` includes full
conditions, state counts, rejection ticks, guard failures, and recording hashes.
Per-run `data.json` and `telemetry.csv` retain every tick. PNG/HTML export is skipped
with `--no-images`; rendering and image processing still run normally.

## What PASS means

The **contract** audit checks timestamps, finite complete estimates, quaternion
normalization, uncertainty bounds, and validity/measurement-age rules. Baseline
runs must also meet the tracking guards below. The delayed-stream contract requires
zero estimates and explicit stale rejection. Other stress cases check integrity
while recording tracking failures; their failures must not disappear from averages.
The process exits nonzero on a contract violation.

The separate **tracking guard** requires position RMS ≤ 8 cm, worst position ≤ 15 cm,
attitude RMS ≤ 8°, recovery ≤ 1 s, and GNC-valid coverage ≥ 70%. Missing estimates
or missing recovery fail that guard. These are synthetic regression thresholds,
not mission requirements. CI checks one seed for baseline, delay, and occlusion;
the full local suite checks three seeds for all seven cases.

Errors cover all available estimates, including invalid coasting estimates. Absent
estimates remain null rather than zero. The report also gives worst position error
while GNC-valid, estimate coverage, p95 error, and longest lost interval. Rejection
counts are navigation ticks carrying that reason, not independent camera events.
Host timings exclude rendering, export, and scheduling guarantees.

## Before boundary verification (`2a3a98c`)

Local release runs, seeds 42–44, 20 seconds per trial, three-second outage at 8 s:

| Scenario | Position RMS range | Tracking guards met | Observation |
| --- | --- | --- | --- |
| Baseline | 1.90–2.68 cm | 3/3 | GNC-valid recovery in 0.5 s |
| Noise: 12 ADU | 2.58–19.54 cm | 2/3 | One trial retained a poor estimate |
| Occlusion: 35% | No estimate | 0/3 | Warm-start matcher never acquired |
| False features: 16 | 2.32–34.42 cm | 2/3 | Worst GNC-valid error reached 86.6 cm |
| Focal bias: +5% | 9.69–10.48 cm | 0/3 | Calibration bias persisted through filtering |
| Delay: 300 ms | No estimate | 0/3 | Correctly rejected beyond the 25 ms age limit |
| Glare: 3× | 1.77–3.19 cm | 2/3 | One worst error exceeded 15 cm |

All 21 runs met their declared contract; only 9 met the tracking guards. This small,
fixed seed set is repeatable characterization, not a statistical reliability claim.
Changing the model, conditions, or algorithm requires rerunning the suite.

## With boundary verification and integrity hold

Same conditions and seeds, using the image-backed boundary gate and requiring
three accepted poses after integrity loss. All 21 declared contracts pass; 10 runs
meet the tracking guards. These are independent columns in the report.

| Scenario | Position RMS range, all estimates | Worst GNC-valid position error | Tracking guards met |
| --- | --- | --- | --- |
| Baseline | 1.90–2.68 cm | 6.17 cm | 3/3 |
| Noise: 12 ADU | 2.58–175.38 cm | 7.21 cm | 2/3 |
| Occlusion: 35% | No estimate | No valid estimate | 0/3 |
| False features: 16 | 2.32–5.10 cm | 10.82 cm | 2/3 |
| Focal bias: +5% | 9.62–17.15 cm | 22.27 cm | 0/3 |
| Delay: 300 ms | No estimate | No valid estimate | 0/3 |
| Glare: 3× | 1.77–2.18 cm | 7.52 cm | 3/3 |

In clutter/42, worst GNC-valid error fell from 86.6 to 10.8 cm; valid coverage fell
from 66.2% to 64.1%. Whole-recording worst error is still 16.46 cm and the tracking
guard still fails. Baseline/42 retains 2.68 cm RMS and 0.5 s outage recovery, with
valid coverage falling from 83.6% to 80.5% due to conservative integrity holds.

The noisy seed 43 drifts as far as 3.07 m while invalid/lost and never recovers.
Those predictions remain in whole-recording RMS instead of being hidden. A previous
one-tick false-valid result after isolated reacquisition is now covered by a
regression test: a timer alone cannot certify reacquisition. Calibration seed 43
still reaches 22.27 cm while valid. The guard assumes a known opaque convex body
against dark background and is not applied to arbitrary public-dataset keypoints.

**The GNC-valid flag does not guarantee an accurate estimate.** The clutter
result demonstrates that timestamp and covariance checks can accept a wrong
correspondence. Further work must strengthen correspondence ambiguity handling and
independent estimator consistency checks before expanding accuracy claims. Do not
retune these guards or read scoring truth into navigation to conceal that failure.

Real images, motion blur, exposure dynamics, domain shifts, processor faults, and
hardware deadlines remain outside this characterization. Public-dataset evaluation
and hardware measurements are separate gates.
