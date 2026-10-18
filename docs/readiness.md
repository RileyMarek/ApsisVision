# Integration gates: measured status

This is the baseline experiment at `f1cfa92`. The latest
[visual-admission and analytic-visibility follow-up](visual-admission.md) records
paired/independent trials, failed learned hypotheses and selective-INT8 limitations.

Measured 2026-10-18. **The research tools are usable; the proposed operating envelope
is not validated. A trustworthy real-image rendezvous demo is not ready.**

| Gate | Implemented and exercised | Remaining blocker |
| --- | --- | --- |
| Estimator integrity | Defined candidate envelope, 128 independent randomized pixel trials, full-state NEES, marginal uncertainty coverage, false-valid trial intervals | False-validity, acquisition and covariance consistency fail |
| Real-image acquisition | Official calibration, 500-image training subset, 160 held-out images, ORB semantic registration and seedless PnP | No held-out poses recovered; dependable correspondences and reacquisition remain unsolved |
| Rendezvous | HCW relative translation, torque-free target rotation, approach/alignment corridor, estimated FOV checks, hold/abort and virtual capture | Pixel-based approach never acquires; tumbling-target alignment control and physical docking/contact remain open |
| Embedded integration | Allocation-free core, bounded timestamp replay, fault tests, paced host timing, constrained-stack smoke test and cross-builds | Selected board, camera drivers, ARM stack/timing, power and hardware fault/radiation evidence |

## Candidate operating envelope

[Machine-readable limits](../scenarios/envelope.json) define a **screening proposal**,
not an accepted mission specification. Population: the known 3U body and asymmetric
plate, calibrated 640 × 480 pinhole camera with 1500 px focal length, fixed approximate
initial pose. Range 1.4–3 m; tumble 0–0.25 rad/s; initial roll offset ±0.15 rad.
Noise 0–5 ADU; occlusion 0–15%; 0–4 clutter features; focal bias ±0.5%; gain 0.5–1.5;
sun direction ±0.5 rad; simulated glare 0–1.5. These illumination parameters are
renderer controls, not physical irradiance or sensor saturation qualifications.

Camera cadence is 10 Hz and propagation 50 Hz. Delivery delay is 0–20 ms. A 3 s
blackout is included; **validity expires after 0.5 s**, even while prediction continues.
Screening targets: ≤15 cm position error, ≤8° attitude error, ≥70% valid coverage
per run, 100 ms vision / 20 ms filter compute deadlines on the host.

```sh
cargo build --release --locked -p apsis-cli
python scripts/envelope_campaign.py --trials 128
```

Seed 20261018, 128 independently drawn configurations, 20 s each:

- **9/128 trials** contain at least one out-of-limit GNC-valid estimate. Wilson 95%
  interval: **3.74–12.82%**. This is a probability per sampled 20 s trial, not per
  frame, per hour, or per mission. The proposed ≤5% upper screening bound fails.
- Availability fails the ≥70% per-trial requirement.
- Mean valid-trial position marginal 3σ coverage: **77.30%**, below the proposed 95%.
- Full 12-state NEES at the predeclared 6 s tick exists for only **20/128 trials**.
  Mean **88.96**, versus expected 12 and the nominal independent-trial mean interval
  **9.95–14.24**. Missing estimates and conditioning on acquisition limit the
  chi-square interpretation. Covariance is substantially overconfident.
- No measured host vision/filter compute deadline misses in this campaign. These
  measurements exclude rendering, IO and scheduling; they are not real-time guarantees.

The [committed report](evidence/envelope-2026-10-18.json) retains every sampled
configuration and trial score. Raw recordings remain under ignored `output/envelope`.
The script completes characterization even when gates fail; consumers must inspect
`envelope_validated`, which is **false**. Do not confuse successful execution with
successful validation. NEES and truth-based scores are computed after inference.

## Official image experiment

[SPEED+](https://purl.stanford.edu/wv398fc4383) and
[SHIRT](https://purl.stanford.edu/zq716br5462) were fetched from official archives
using byte ranges, ZIP size/CRC validation and image SHA-256 provenance. Data stay
under ignored `output/datasets`; no images, annotations or appearance weights are
redistributed. The downloaded archives specify **CC BY-NC-SA 4.0**; source-code
licensing does not override dataset terms. Geometry comes from the authors'
[SPNv2 repository](https://github.com/tpark94/spnv2/tree/main/core/utils/models).

The baseline registers ORB descriptors near projected Tango landmarks on **500
synthetic training images**. Inference receives pixels, calibration and that saved
map, with **no pose prior or evaluation truth**. Unique semantic matches enter
EPNP RANSAC and LM refinement; unsupported fits are rejected. Training annotations
provide supervision; held-out annotations are accessed only for scoring afterward.
Image hashes prevent training/evaluation overlap.

| Held-out subset | Images | Recovered poses |
| --- | ---: | ---: |
| SPEED+ synthetic validation | 40 | 0 |
| SPEED+ lightbox | 40 | 0 |
| SPEED+ sunlamp | 40 | 0 |
| SHIRT ROE1 lightbox, first consecutive frames | 20 | 0 |
| SHIRT ROE2 lightbox, first consecutive frames | 20 | 0 |

Accuracy is **unavailable**, not zero. Tentative descriptor matches fail geometric
consistency. A training-only positive control recovers 6/10 poses; independent known
correspondence tests recover the correct pose despite two outliers. This supports
replacing the appearance correspondence baseline rather than weakening rejection
thresholds. It does not establish that a learned detector will succeed.

SHIRT capture steps use the archive's documented 5 s interval. The 20-frame snippets
are too short and sparse to establish robust tracking. SPEED+ samples are independent;
their 0.1 s exported replay spacing is artificial and has no temporal-validation meaning.
This is a small fixed experiment, not an official competition score or majority-use-case claim.
See the [summary report](evidence/real-images-2026-10-18.json) and
[reproduction commands](benchmarking.md#cold-start-official-image-experiment).

## Rendezvous reference

```sh
cargo run --release --locked -p apsis-mission -- --outage --out output/mission-reference
cargo run --release --locked -p apsis-mission -- --outage-start 75 --out output/mission-abort
cargo run --release --locked -p apsis-mission -- --tumble --out output/mission-tumble
cargo run --release --locked -p apsis-mission -- --pixels --out output/mission-pixels
```

The plant uses local HCW translation around a circular reference orbit (mean motion
0.0011 rad/s), RK4 integration and torque-free diagonal-inertia rigid-body rotation.
HCW is a local linear approximation, not a high-fidelity orbital propagator; see
[NASA's relative-motion analysis](https://ntrs.nasa.gov/citations/20110008472).
The synthetic camera has 320 px focal length for close-range FOV, distinct from the
navigation campaign's camera. Axes and camera/LVLH transforms are explicit in code.

The guidance API emits **target-relative camera-frame acceleration**. The simulation
transforms/sign-reverses it for its chaser plant. It is not a thruster command or
an approved collision-avoidance controller. It uses estimates, validity, covariance
and estimated body-in-FOV checks. Truth only drives the plant/sensor and scoring.

Reference runs use a declared stochastic pose sensor, not image-derived measurements:

- Nominal aligned target plus blackout at 40–43 s: hold at 40.42 s, approach resumes
  at 43.5 s, virtual capture at **86.72 s**, true center range **0.36069 m**.
- Blackout at 75–78 s: latched abort at **75.42 s**, true range **0.40199 m**.
- Tumbling target: remains in alignment mode for the 180 s run; **no capture**.
- Actual pixel pipeline: **2/1800** accepted poses, **zero valid ticks**, no capture.

Virtual capture checks alignment, angular rate, lateral offset and relative speed.
The fixed camera-to-capture-plane offset is 0.30 m and target-center-to-port offset
0.05 m. The mission stops at the virtual capture envelope, before physical contact.
An axial Kelvin–Voigt contact primitive has a unit check; coupled impact, latch,
six-axis contact, attitude actuation, propulsion delays/saturation and fuel budgets
are not modeled. Thus **a complete tumbling-target docking demonstration remains open**.
The console still shows the original navigation recording; mission trajectories are
CSV/JSON artifacts, not yet a new integrated viewer mode.

## Next engineering work

1. Establish semantic landmark identity and visibility on training/validation data;
   retain these failed test subsets as a locked audit set. Use a fresh untouched
   test split for any model selection. Evaluate missing/wrong priors and deliberately
   false correspondences, not just successful initialized tracking.
2. Audit EKF Jacobians, process/measurement models, and geometry-dependent pose
   covariance. Model correspondence uncertainty and persistent calibration bias;
   covariance inflation alone cannot repair a wrong association. Repeat independent
   trials before approving a smaller operating envelope.
3. Close the **pixel-based** relative-navigation loop before extending guidance to
   actively match a rotating docking port. Then add constrained attitude control,
   verified retreat corridors and coupled contact mechanics.
4. Select a specific board/OS/camera and budgets; repeat replay, stack high-water,
   thermal/power and deadline/fault tests on it. No hardware measurements can be
   supplied without hardware access. Radiation tolerance requires device/mission
   evidence beyond Rust portability and software fault injection.
