# Main verification and cold acquisition experiments

**Main builds and serves its demo. Real-image navigation remains unqualified.**
These are separate findings; a working viewer is not a reliable navigation system.

## Main branch verification

On 2026-10-19, `origin/main` was `a4e5f9b` (PR #4), with a successful
[CI run](https://github.com/RileyMarek/ApsisVision/actions/runs/39172146664).
An isolated checkout of that exact revision passed:

- 31 release Rust tests, Rustfmt and strict all-target Clippy.
- 18 Python tests, JavaScript syntax and viewer Prettier checks.
- A generated nominal playback: 2.03 cm position RMS, 1.36 deg attitude RMS,
  80.5% valid coverage and 0.5 s blackout recovery.
- HTTP 200 for the viewer, telemetry and camera PNG; a POST to `/api/run`
  recomputed a noise-2/three-second-outage scenario and served its returned viewer.

No build/server hotfix was needed. Local `main`, previously still at the initial
commit, was fast-forwarded to the verified remote revision. Experimental development
was not merged into main. No public deployment URL/configuration was supplied or
found; this verifies the documented local demo, not an unknown hosted site.
HTTP and JavaScript checks do not constitute a new browser rendering inspection.

`scripts/demo_smoke.py` now automates playback, API recomputation, invalid-input
rejection and foreign-origin rejection in CI. A development noise-2 run passed this
HTTP check while achieving only **2.6% valid coverage**. That is a tracking failure,
not an HTTP failure; the smoke check deliberately makes no qualification claim.

## Image-only coarse-to-fine perception

The new experiment separates global target localization from semantic keypoints:

```text
Rectified pixels -> box regressor -> padded 256x192 crop
                 -> eleven heatmaps + balanced visibility head
                 -> original-image coordinates -> unchanged strict PnP admission
```

The box model has 692244 parameters; the landmark model has 102342. The training
split remains 400 synthetic images with 100 disjoint development images. Both
models initially trained for 80 CPU epochs; development loss selected box epoch 32
and landmark epoch 58. That initial run took 916.5 s on this host and is rejected.

Training annotations construct crop/coordinate targets, with affine jitter and
photometric augmentation. At inference, `Perception.locate(image)` accepts only
pixels. There is no pose-prior input or annotation-derived crop. Earth backgrounds
and glare make a simple brightness threshold inappropriate for this dataset.

The visibility loss gives known visible and hidden examples equal class weight.
Hidden landmarks still receive no heatmap regression gradient; unknown labels
remain excluded. A local heatmap peak center avoids averaging separated peaks
into empty space. Local mass >=0.05 and visibility >0.5 are fixed experimental
screens, **not calibrated probabilities**. The existing six-correspondence,
3 px inlier RMS and 3 px all-correspondence mean gates are unchanged.

## Initial protocol: rejected

Review found a supervision bug after this first audit: ROI padding could move a
source-out-of-FOV landmark into the crop coordinate system and incorrectly label it
visible. Of the 500 training/development images, 71 have at least one projected
landmark outside the source image. These first results are retained as rejected
history, not promoted. The corrected training path preserves source-FOV visibility
before applying crop jitter; unknown labels remain unknown.

Development diagnostics distinguish annotation-derived crops from actual inference:

| Diagnostic | Result |
| --- | ---: |
| Mean projected-landmark containment in predicted crops | 87.27% |
| Mean landmarks passing experimental screens | 7.16 |
| Raw admitted PnP hypotheses | 0/100 |
| Visible-point RMS with annotation-derived crops, 256x192 crop pixels | 75.43 px |
| Visible-point RMS with image-only crops, rectified 640-wide image pixels | 123.17 px |
| Visibility balanced accuracy with annotation-derived crops | 67.66% |
| Visibility TP / TN / FP / FN | 669 / 93 / 49 / 289 |

The head no longer predicts every landmark visible. That removes one degenerate
solution, but does not establish dependable visibility or landmark identity.
The annotation-derived crop metric is an explicit development diagnostic, never
presented as inference accuracy. Its enlarged coordinate system differs from the
earlier full-frame 43.35 px metric, so those RMS values are not a direct comparison.

The frozen model produced **0/160 hypotheses** on the previously examined audit.
After freezing checkpoint hashes, filename-only selection fetched **100 additional
disjoint images**: 20 each from SPEED+ synthetic validation, lightbox and sunlamp,
and SHIRT ROE1 and ROE2 lightbox. The selection excludes every image in the earlier
training/development/audit selection. Archive size/CRC and image SHA-256 are verified.
Transfers totaled 35.4 MB; archive transfers are bounded to 128 MiB per dataset.

The fresh audit also produced **0/100 hypotheses**, without threshold/model changes
after inspection. Neither audit has an accurate pose to report. Zero emitted poses
cannot establish a low false-valid rate for a useful estimator. SHIRT samples are
correlated sequence observations, not independent reliability trials.

A separate corrected-supervision random-initialization control memorized eight deterministic **training**
images with annotation-derived crops and no augmentation: after 400 steps it reached
**0.62 px crop-input RMS**, 100% within 5 px and 100% visibility accuracy. It took
145.2 s with two CPU threads while the main training job ran concurrently; this is
not a controlled timing benchmark. It writes no checkpoint. This confirms
that this network/loss can fit that tiny set, not that it has learned general spacecraft
features. It does not prove one particular cause of the held-out failure.

All transport packets remain `pose: null`. There is no promotion to navigation,
guidance, INT8 deployment or embedded hardware. This experiment has not improved
the synthetic envelope's 4/128 acquisition result: the Rust estimator and synthetic
association pipeline were not changed or requalified in this work.

Audit exports contain an index and `temporal_replay_allowed: false`, with no
fabricated camera timestamps. Sparse randomly selected SHIRT frames are not a
uniform 5 s replay. Actual capture timestamps are required by the replay adapter.

## Corrected protocol: still rejected

The corrected run excludes **107 training and 36 development** source-out-of-FOV
landmark visibility labels. It reuses the unchanged box checkpoint with exact data
hash checks, resets the localizer seed to 1730 independently of box reuse, and trains
the localizer for another fixed 80 epochs. Development selects epoch **49**. This run
took **771.1 s**, excluding the reused box training. Neither this repair nor the changed
initialization is an isolated causal experiment on generalization.

| Corrected development diagnostic | Result |
| --- | ---: |
| Mean projected-landmark containment in predicted crops | 87.27% |
| Mean landmarks passing experimental screens | 6.31 |
| Raw admitted PnP hypotheses | 1/100 |
| Visible-point RMS with annotation-derived crops, crop pixels | 78.62 px |
| Visible-point RMS with image-only crops, rectified image pixels | 119.85 px |
| Visibility balanced accuracy with annotation-derived crops | 67.03% |
| Visibility TP / TN / FP / FN | 599 / 123 / 55 / 323 |

It still recovers **0/160 hypotheses** on the previously examined audit. Only after
freezing these corrected weights, seed 20261023 selects another **100 images**,
excluding the original selection and the first fresh audit by filename **and SHA-256**.
All 100 image hashes are unique. This second fresh audit also recovers **0/100**;
there is no threshold/model adjustment afterward. Its 35.5 MB transfer is bounded
and archive-verified as before. No floating-point or quantized model is promoted.

These failures leave real-image keypoint identity and acquisition unresolved. The
corrected eight-image control above shows tiny-set fitting works; it cannot turn
these audit failures into evidence of robust localization.

## Pixel replay and reacquisition contract

`scripts/reacquisition.py` supplies an experimental replay adapter:

- Recent tracks use pyramidal Lucas-Kanade flow with forward/backward error <=1 px.
  Every surviving correspondence set must pass the same strict PnP admission.
- Missing frames discard flow anchors. Tracks older than 0.25 s, changed calibration,
  geometry or image dimensions trigger global image-only localization.
- An outage expires diagnostic coasting at 0.5 s; feed return uses global acquisition
  rather than searching around an old pose. Duplicate/reversed/nonfinite timestamps
  discard tracks; invalid calibration/geometry abstains. A new timestamp epoch
  requires a new replay instance rather than silently accepting clock reversal.
- Output distinguishes `hypothesis` from `pose`; **`pose` is always null and
  `gnc_valid` always false**, including during diagnostic `tracking`.

Fixture tests exercise real optical flow/PnP, feed loss/return, calibration changes,
expired tracks, malformed clocks, crop-coordinate mapping and balanced loss gradients.
The fixture detector supplies known correspondences; this tests recovery plumbing,
**not learned real-image reacquisition**. Since the new detector never passes PnP
on either audit, dependable real-image reacquisition remains unproven.

## Reproduce

Use the existing benchmark setup and `scripts/requirements-learned.txt`.

```sh
# Build and verify the local demo; a native or documented WSL executable is required.
cargo run --release --locked -p apsis-cli
python scripts/demo_smoke.py

python scripts/analytic_visibility.py
python scripts/crop_perception.py train --epochs 80 --threads 4 --out output/crop-perception-fov
python scripts/crop_perception.py development --threads 4 --out output/crop-perception-fov
python scripts/crop_perception.py overfit --steps 400 --threads 2 --out output/crop-perception-fov
python scripts/crop_perception.py evaluate --threads 4 --out output/crop-perception-fov
# Freeze weights before fetching this audit. Do not train/tune against it afterward.
python scripts/fresh_audit.py --model-dir output/crop-perception-fov --seed 20261023 --exclude-selection output/datasets/fresh-selection.json --out output/datasets/fresh-selection-20261023.json
python scripts/crop_perception.py evaluate --out output/crop-perception-fov --selection output/datasets/fresh-selection-20261023.json
python -m unittest discover -s scripts -p 'test_crop_perception.py'
```

The measured corrected run used `--reuse-box output/crop-perception` to avoid
repeating unchanged box training. A full run uses the same independent localizer seed.
The exclusion file above is the first audit's recorded selection; retain it to
reproduce this second audit. Otherwise reserve another disjoint audit and record it
as a different experiment, rather than silently changing this reported selection.

Images, models, per-image audit annotations and transport files remain under ignored
`output/`, subject to the original dataset terms. The committed
[experiment evidence](evidence/crop-reacquisition-2026-10-19.json) records hashes,
training history and aggregate results. Confidence and hardware timing remain unverified.

## Next decision

This small from-scratch model is still a failed baseline despite passing the tiny-set
overfit control. Inspect semantic landmark overlays and test a stronger representation
with a larger training-only pool. Keep crop coverage,
visibility balance and localization error separate. Then freeze the revised model
and reserve another untouched audit. Do not relax PnP gates or spend effort on INT8
until floating-point localization produces useful, correct poses.
