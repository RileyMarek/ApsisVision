# Image benchmark harness

`apsis-bench` runs the same grayscale Harris detector, prior-guided association,
robust local PnP, and navigation API on rectified PNGs. Ground truth is accessed only
after each prediction is fixed. A regression changes truth labels and verifies
identical predicted poses with different scores.

**No SPEED+ or SHIRT accuracy result is claimed.** The importer and harness have
fixture tests; the official datasets have not been downloaded or evaluated here.
Harris corners do not supply semantic Tango landmark identities or global pose
acquisition. This is a measurable starting point, not a trained spacecraft detector.

## Import a small, explicit subset

Install the optional Python import dependencies:

```sh
python -m pip install -r scripts/requirements-benchmark.txt
```

Obtain dataset images/labels separately and check their terms. The
[ESA SPEED+ data page](https://kelvins.esa.int/pose-estimation-2021/data/)
describes the image domains and dataset license. Data are not redistributed in this
repository. [Stanford SLAB](https://slab.stanford.edu/projects/datasets) provides
dataset links and describes SHIRT's sequential rendezvous imagery.

Supply verified target-body landmarks as an `N x 3` JSON array in meters (`6 <= N <= 32`).
The demo's 3U geometry is not Tango geometry. Supply a separate, independently chosen
seed JSON with `position_m` and a unit `quaternion_wxyz`; never initialize from test
ground-truth labels.

```sh
python scripts/prepare_benchmark.py --dataset speedplus --images DATA/lightbox/images --labels DATA/lightbox/test.json --camera DATA/camera.json --model tango-landmarks.json --seed acquisition-prior.json --out output/benchmark --limit 50
cargo run --release --locked -p apsis-bench -- output/benchmark/manifest.json output/benchmark/report.json
```

The importer reads `cameraMatrix` and OpenCV-order `distCoeffs` (`k1,k2,p1,p2,k3`),
rectifies using bilinear Brown-Conrady resampling, converts to grayscale PNG, and
optionally resizes to 640 pixels wide with updated focal lengths/principal point.
Pixels outside the source image become black. It records hashes of calibration,
annotations, landmarks, and seed. Unsupported calibration or annotation data fail
explicitly. Importer rectification still needs independent comparison on official
calibration data; it is not a calibrated real-camera validation result.

The SPEED+ annotation mapping follows the author's
[preprocessor](https://github.com/tpark94/speedplusbaseline/blob/master/preprocess.py)
and [projection utilities](https://github.com/tpark94/speedplusbaseline/blob/master/src/utils/utils.py):

| Source | Internal meaning |
| --- | --- |
| `filename` | Image path inside the supplied image directory |
| `r_Vo2To_vbs_true` | Target origin in camera coordinates, meters, scoring only |
| `q_vbs2tango_true` | Scalar-first components; see convention below, scoring only |

The reference projector transposes its passive direction-cosine matrix. The result
equals nalgebra's active body-to-camera rotation with the same quaternion components.
The importer therefore preserves the scalar-first components; it does not blindly
conjugate based on the field name. A 90-degree rotation fixture checks that mapping.

## Independent images versus temporal tracking

SPEED+ pose samples are evaluated independently by default. The importer assigns an
index-only timestamp, resets navigation for each image, and never interprets image
order as physical motion. Pose availability/errors can be reported; the temporal
GNC-valid flag remains false without a measurement history.

For SHIRT, supply an annotation **list** with `filename` and pose fields, and an
explicit JSON mapping from filename to capture time in seconds:

```sh
python scripts/prepare_benchmark.py --dataset shirt --images DATA/images --labels annotations.json --camera DATA/camera.json --model tango-landmarks.json --seed acquisition-prior.json --timestamps capture-times.json --temporal --out output/shirt
```

Use `--position-key` and `--quaternion-key` if the exported fields differ. Quaternion
fields must follow the documented SPEED-style passive convention; convert other
conventions explicitly first. The importer does not auto-discover every SHIRT bundle
layout or invent frame timing. Sequences must be in increasing timestamp order.

The report retains every frame, including failed acquisition, lost tracking, and
missing estimates. It includes pose availability, GNC validity, scored-frame count,
RMS/worst errors, vision p95, and per-frame results. Scores over available estimates
must always be read with availability/loss counts. No official SPEED competition
score, HIL calibration threshold, or ranking is implemented.

Next: evaluate a labeled subset, inspect failed associations, establish a semantic
landmark front end/global initialization, and keep tuning/evaluation sequences separate.
