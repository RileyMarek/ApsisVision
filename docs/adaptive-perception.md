# Adaptive covariance and learned perception: measured follow-up

**Status: research increment; integrity and perception gates still fail.**
This follows the [original readiness report](readiness.md), baseline commit `f1cfa92`.

## Geometry-dependent measurement uncertainty

The Rust vision library now forms the full local 6x6 pose covariance:

```text
H = sum(J_pixel_poseᵀ J_pixel_pose)
pixel_variance = max(1 px², residual_sum_squares / (2*inlier_count - 6))
R_pose = pixel_variance * inverse(H)
R_update = R_pose + existing measurement model floor
```

The analytic Jacobian differentiates calibrated projection with respect to camera-frame
translation in metres and a left attitude perturbation in radians. Translation/attitude
correlations remain intact. The existing floor is 5 mm position and 0.02 rad attitude
standard deviation; it is a demo model-discrepancy allowance, not calibrated evidence.
The pixel floor is also an assumption, not inferred from simulation truth.

At least six unique inlier identities are required. Information must be positive
definite with condition number <=1e8 after scaling translation by the model RMS radius.
The eigensolver has a 96-iteration limit and rejects nonconvergence. No damping is
added to hide singular geometry. Range dependence emerges from projection;
depth variance can grow roughly with the fourth power of range for a fixed baseline.
A scalar range multiplier or isotropic residual term would lose units and correlations.

`Navigator::observe_with_covariance` transports R through initialization, the existing
NIS gate (six dimensions, threshold 22.46), reacquisition weighting and the Joseph
covariance update. Initialization maps the observed position/attitude covariance and
cross terms into the 12-state covariance; velocity/rate retain their independent prior.
Invalid covariance is rejected explicitly. The legacy observation API retains its
fixed measurement-noise convenience default.

JSON and CSV expose the last NIS, measurement standard deviations, geometry condition
and scoring-only NEES. Last-measurement values repeat across propagation ticks; they
must not be treated as independent innovations. The core remains allocation-free and
`no_std`; matrix dimensions and work are bounded.

Tests compare the analytic Jacobian to central differences, check range/noise/support
dependence, reject degenerate geometry and malformed covariance, verify gain/NIS
weighting, and exercise 256 independent noisy-pixel fits with **known correct identities**.
The last test checks conditional local pose consistency, not spacecraft correspondence integrity.

## Campaign results

No accuracy, availability, NIS or validity thresholds were relaxed. Q defaults remain
unchanged. Explicit continuous linear/angular acceleration PSD configuration is now
available, but Q was not tuned against selected successful trajectories to force NEES
toward 12. That would not establish a correct motion or correspondence model.
The comparison includes both R propagation and covariance-aware initialization;
it is not an isolated R-only ablation.

| Metric | Original seeds, fixed R | Same seeds, adaptive R | Independent seeds, adaptive R |
| --- | ---: | ---: | ---: |
| Trials | 128 | 128 | 128 |
| Trials with false-valid output | 9 | 6 | 9 |
| False-valid trial Wilson 95% upper bound | 12.82% | 9.85% | 12.82% |
| Mean NEES at 6 s | 88.96 | 55.02 | 183.94 |
| Trials with an estimate at 6 s | 20 | 20 | 20 |
| Mean valid-trial position marginal 3-sigma coverage | 77.30% | 83.82% | 81.85% |
| Trials ever GNC-valid | 19 | 19 | 15 |
| Mean valid-time fraction | 5.58% | 6.17% | 4.31% |

Seeds are 20261018 and 20261019. The independent set has no fixed-R comparator;
its NEES must not be interpreted as a measured before/after regression. Only 20
trajectories supply the NEES anchor in either set; these are **estimates, not necessarily
valid outputs**, and acquisition conditioning limits the chi-square interpretation.
All three campaigns fail availability, uncertainty and false-valid screening gates.
Host compute deadlines passed; runs are host characterization, not hardware qualification.

Reports: [paired](evidence/adaptive-r-2026-10-18.json),
[independent](evidence/adaptive-r-independent-2026-10-18.json).

```sh
cargo build --release --locked -p apsis-cli
python scripts/envelope_campaign.py --trials 128 --out output/adaptive-r
python scripts/envelope_campaign.py --trials 128 --seed 20261019 --out output/adaptive-r-independent
```

**Conclusion:** adaptive R helps the paired diagnostic but does not validate the
envelope. Wrong landmark identity, calibration bias and alternative pose modes are
not represented by this local Gaussian covariance. NIS can accept a self-consistent
wrong mode. Larger R can even make the innovation gate more permissive.

## Learned landmark experiment

`learned_perception.py` adds a 101803-parameter depthwise encoder/decoder predicting
eleven 64x48 semantic heatmaps from a 256x192 grayscale image. Full-image inference
uses no truth-based crop or pose prior. Heatmaps produce landmark candidates for
the existing seedless EPNP/RANSAC/refinement benchmark gate.

Of the existing 500 synthetic training-pool images, a deterministic filename-hash
split supplies 400 training and 100 development images. Training uses projected
geometric vertices, spatial cross-entropy and brightness/gamma/noise augmentation.
Occlusion visibility is not labeled and confidence is not calibrated. Twenty fixed
CPU epochs took about 210 s; the development loss selected epoch 16. No audit image
or audit label selects weights, thresholds, augmentation or quantization calibration.

| Audit domain | Frames | Raw FP32 pose hypotheses | Position RMS among hypotheses | Attitude RMS |
| --- | ---: | ---: | ---: | ---: |
| SPEED+ synthetic validation | 40 | 1 | 39.36 m | 153.00 deg |
| SPEED+ lightbox | 40 | 2 | 74.80 m | 152.09 deg |
| SPEED+ sunlamp | 40 | 2 | 2577.48 m | 158.36 deg |
| SHIRT ROE1 lightbox | 20 | 4 | 93.31 m | 148.13 deg |
| SHIRT ROE2 lightbox | 20 | 3 | 222.79 m | 107.05 deg |

**None of the twelve hypotheses meets 15 cm / 8 deg.** These are failed raw hypotheses,
not accepted navigation estimates. All learned replay packets are quarantined with
`pose: null`; the model is not connected to navigation or guidance. A test verifies
quarantine even when a mocked raw hypothesis has zero scoring error.

The small experiment does not establish that learned methods cannot work, nor that
all classical descriptors require Lambertian surfaces. It establishes that this
particular architecture/data/training recipe is insufficient. The original ORB
audit had no silhouette gate. The existing pixel rendezvous remained in acquisition
with no approach manoeuvre, so its failures cannot be attributed to thruster-induced
perspective changes. Successful pose-sensor reference runs likewise do not prove the
guidance policy safe for flight.

## Export and quantization

FP32 ONNX export passes numerical parity on 16 development images: maximum absolute
logit difference 2.80e-5. Static QDQ INT8 calibration uses 32 **training** images only,
with graph/shape preprocessing. Export sizes: FP32 411016 bytes; INT8 172440 bytes.
INT8 development keypoint drift is **61.10 px RMS**, **273.04 px worst** at the
640x480 reporting scale. The INT8 audit produces ten raw hypotheses, none within
15 cm / 8 deg. **INT8 deployment is not approved.**

Host ORT INT8 inference p95 was about 1.75 ms for 16 development frames with four
threads; that excludes rectification, transport, decoding, PnP and scheduling.
It is not Zynq latency, a complete embedded runtime, or a deadline guarantee.
Models/checkpoints remain under ignored `output/learned`, subject to dataset terms.
The [committed experiment summary](evidence/learned-heatmaps-2026-10-18.json) records
training history, model hashes, domain failures and export evidence.

```sh
python -m pip install -r scripts/requirements-export.txt
python scripts/learned_perception.py train --epochs 20 --threads 4
python scripts/learned_perception.py evaluate
python scripts/export_heatmap.py
python scripts/learned_perception.py evaluate --runtime int8
```

Dataset prerequisites are in the [benchmark guide](benchmarking.md). Training and
ONNX tools remain desktop-only optional dependencies; CI checks learning interfaces
on synthetic fixtures without downloading or redistributing benchmark data.
Export references: [PyTorch](https://docs.pytorch.org/docs/stable/onnx_export.html),
[ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

## Next decision

Keep the two 128-trial sets as diagnostics; changing the model after inspecting them
requires a fresh independent acceptance set. Replace the inadequate learned baseline
with a stronger, provenance-checked landmark model and substantially broader training
coverage, using training/development data for all choices. Evaluate identity, visibility,
ambiguous poses, negative images, range and confidence calibration before enabling
measurements. Then jointly revisit pixel uncertainty, calibration bias and motion
model consistency. LOS attitude control, ROI tracking, physical contact and hardware
measurements remain later gates; no hardware/QEMU execution is claimed by cross-builds.
