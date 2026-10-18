# Visual admission and analytic visibility

**Research status: nominal tracking works; the broad operating envelope and real-image
perception still fail qualification.** This follows [adaptive covariance](adaptive-perception.md).

## Admission before covariance and NIS

The original solvers already enforced 3 px RMS over fitted inliers. The new
`solve_admitted` path additionally checks **unweighted mean reprojection error over
every supplied correspondence**, including RANSAC outliers, before estimating R.
Mean error above **3.0 px** rejects the frame. Finite, unique identities and at least
six observations are required. No covariance value can relax this pixel test.
Rust pixel consumers and the Python PnP benchmark use the same all-point mean limit.
The pure RANSAC fitting primitive remains available separately; a subset fit is not admission.

The shaded CubeSat demo adds independent image evidence for competing body modes:

- Enumerate the eight proper body rotations preserving its square-section 3U bus.
  Translation stays fixed; the camera pose is not inverted.
- Check image support for the modeled asymmetric bright-rim docking plate in each mode.
  A supported body outline alone cannot resolve attitude symmetry.
- Require at least 24/36 primary plate-edge samples and a sufficient primary/runner-up
  score gap. An absent plate, tied modes or a better rival causes abstention.

The gap uses a binary edge-support score with assumed 0.85/0.15 support probabilities
and a `ln(10)` margin. **It is not calibrated odds, a certified 10 dB safety margin,
or a general Tango likelihood.** Its photometric and geometric assumptions are
specific to the demo plate; shadows, cropping and glare can reject correct poses.

`reprojection_gate` and `ambiguous_geometry` latch an integrity hold just like an
unsupported boundary. Navigation remains invalid until repeated admitted poses
and the existing recovery dwell release it. Time alone cannot clear the hold;
the original rejection reason persists through subsequent missing features.
NIS remains the second admission stage, with its unchanged six-dimensional 22.46 threshold.
External pose-sensor callers remain responsible for their own measurement admission.

Initial gate-only development exposed loss of nominal recovery. Tightening candidate
association restored it without changing pixel, ambiguity, NIS, health or regression
accuracy/recovery thresholds: fresh-prior searches use 4 px, searches after a >0.5 s
measurement gap use 12 px, and initial acquisition uses 20 px. These are local
warm-start search limits, not global acquisition or a covariance-coverage guarantee.
The new visual mean and mode score are recorded in JSON/CSV; last-camera values repeat
on propagation ticks and must not be treated as independent samples.

## Frozen implementation results

| Result | Prior adaptive R, seed 20261019 | Final admission, same seed | Final admission, fresh seed 20261021 |
| --- | ---: | ---: | ---: |
| Independent trials | 128 | 128 | 128 |
| Trials with false-valid output | 9 | 1 | 1 |
| False-valid trial Wilson 95% upper bound | 12.82% | 4.29% | 4.29% |
| Trials ever GNC-valid | 15 | 4 | 4 |
| Mean valid-time fraction | 4.31% | 1.43% | 1.01% |
| Mean NEES at the fixed 6 s tick | 183.94 | 13.48 | 24.69 |
| Trials supplying that NEES estimate | 20 | 5 | 7 |
| Mean valid-trial position 3-sigma coverage | 81.85% | 97.88% | 92.91% |

**The apparent false-valid improvement comes with severe abstention.** Neither
campaign validates the envelope: availability and uncertainty gates still fail.
The false-valid statistic is per sampled 20 s trial, including trials that never
acquire. It is not a probability conditioned on valid navigation or a mission risk rate.
The paired NEES averages use different survivor sets. On the five common anchor
trials, mean NEES changes from 145.50 to 13.48, but this selected subset cannot
establish overall consistency. The fresh subset remains overconfident.

The final nominal recording achieves **1.79 cm position RMS, 0.59 deg attitude RMS,
75.1% valid coverage and 0.5 s blackout recovery**. With the earlier 4–7 s blackout,
recovery is 0.6 s and coverage 74.5%. Original accuracy, nonzero-availability and
recovery regression guards remain intact. A development prototype with broad
association had failed these cases; those results are not promoted as the final model.

Machine-readable [paired](evidence/admission-final-paired-2026-10-18.json) and
[fresh](evidence/admission-final-independent-2026-10-18.json) reports retain all configurations.

```sh
cargo build --release --locked -p apsis-cli
python scripts/envelope_campaign.py --trials 128 --seed 20261019 --out output/admission-final-paired
python scripts/envelope_campaign.py --trials 128 --seed 20261021 --out output/admission-final-independent
```

## User-authorized analytic Tango proxy

The supplied `tangoPoints.mat` coordinates support six convex planar bus faces on
points 0–7. **The bus is tapered, not a rectangular box**: the upper and lower Y
extents differ. `analytic_visibility.py` derives outward face planes from the actual
quad vertices and validates planarity, convexity and an interior centroid.

In the established active body-to-camera convention, camera origin in body coordinates
is `-R^T t`. A bus vertex is proxy-visible when any incident face faces that origin.
For points 8–10, segment clipping against the actual six bus half-spaces determines
bus occlusion. It does not substitute an oversized axis-aligned bounding box.
Training applies image-bound checks separately. Only the selected synthetic training
pool is labeled; pose annotations do not enter inference or estimator updates.

This is a **bus/point-appendage proxy**, not verified full CAD visibility. It omits
appendage volumes, appendage-to-appendage occlusion, external occluders and illumination.
Geometric line of sight also does not imply that a shadowed landmark is measurable.
The existing quaternion mapping is preserved; the annotation field name is not a
reason to conjugate it again.

A separate optional `visibility_labels.py --mesh registered-mesh.json` supports a
supplied closed triangle mesh, explicit landmark-to-vertex registration, ray tests
and binary foreground masks. It checks metre units, source/hash provenance, topology,
size bounds and registration; those checks cannot prove physical CAD fidelity.
No external Tango mesh was downloaded or fabricated.

## Masked learning and spatial moments

The 102342-parameter model adds a binary visibility head to the previous backbone.
Known hidden landmarks train visibility but receive **zero heatmap-regression gradient**;
unknown visibility labels train neither head. In-frame visible landmarks train both.
Inference uses predicted visibility above 0.5 and a global spatial-softmax expectation,
with a spread screen to reject diffuse/multimodal heatmaps. Spatial spread is not
calibrated metric pose covariance; multiple peaks can average into empty space.
The original decoder already used a local center of mass, rather than only a hard grid argmax.

The frozen experiment uses the same 400/100 train/development split, 20 CPU epochs,
gamma range 0.2–3.0, gain/noise augmentation, and development-selected epoch 18.
**It still fails**:

- Development visibility accuracy is 83.8%, but every landmark is predicted visible.
  Precision is 83.8% and recall 100%: a majority-class result, not learned self-occlusion.
- Visible-landmark error is 43.35 px RMS in the 256x192 model-input coordinates.
- No keypoints pass both predicted visibility and the spread screen on development;
  the 160-image SPEED+/SHIRT audit produces **zero pose hypotheses**.
- All learned transport packets remain `pose: null`; neither checkpoint is enabled
  for navigation or guidance. An admitted pixel fit is not enough to qualify a model.

Full foreground compositing, stars/Earth-like backgrounds, glare and synthetic
occluders are implemented for explicitly supplied complete masks. The bus proxy has
**no complete foreground mask**, so compositing is refused for it: erasing appendages
while labeling them visible would introduce inconsistent supervision. The measured
proxy run uses photometric augmentation, not unverified foreground masks.

```sh
python scripts/analytic_visibility.py
python scripts/learned_perception.py train --visibility-labels output/visibility-proxy/labels.json --model-dir output/visibility-proxy-model --epochs 20
python scripts/learned_perception.py evaluate --model-dir output/visibility-proxy-model
python scripts/visibility_perception.py
```

Dependencies/data setup are in the [benchmark guide](benchmarking.md). Output models,
labels and masks stay under ignored `output/`. Dataset terms still apply.

## Selective quantization experiment

`python scripts/export_heatmap.py --backbone-only` quantizes 17 encoder Conv nodes
of the **original checkpoint**, leaving decoder and heatmap head FP32. Calibration
still uses training-only images, with per-channel weights. It does not claim FP16,
QAT training, or an export-qualified visibility checkpoint.

On the same 16 development images, keypoint drift is **76.40 px RMS / 343.08 px worst**,
versus the earlier full-Conv INT8 result of 61.10 px RMS. Export size is 288557 bytes.
Selective precision did not fix this weak checkpoint, and deployment remains rejected.
These experiments do not establish a universal PTQ failure or prove that sparsity
caused the drift. The model emits dense logits, and it already used per-channel weights.

The [experiment summary](evidence/visibility-proxy-2026-10-18.json) records the model
hashes, training history, failed audits, development diagnostics and quantized node list.
There is no new hardware, QEMU runtime, radiation or flight-safety evidence.

## Next gate

Improve keypoint identity/localization and acquisition coverage before promoting the
new model or expanding guidance. Visibility learning needs balanced diagnostics,
better representations/training coverage and confidence calibration; its current
83.8% headline must not be used as evidence of success. Keep the examined trial seeds
as diagnostics and use another untouched set for future acceptance. Small residuals,
mode separation on one synthetic plate and selective quantization are necessary
experiments—not proof of trustworthy general spacecraft navigation.
