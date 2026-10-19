# Transfer learning and edge deployment

**The laptop GPU is an offline training resource. It is not a flight dependency.**
The allocation-free Rust navigation core remains separate from the Python image
experiments. Zynq UltraScale+ is an intended target, with no board measurement yet.

## Why this experiment

The [small crop model](crop-reacquisition.md) memorized eight training examples but
failed both held-out audits. The next reference uses pretrained image features,
a larger synthetic training pool, and spatial features at several resolutions.
This tests a stronger representation before trying to compress a failed detector.

## Measured result

**Synthetic acquisition improved; real-image acquisition still fails.** No model
was promoted to navigation, INT8 deployment or main. The earlier synthetic
envelope's 4/128 acquisition result was not improved or requalified by this work.

| Evaluation | Emitted hypotheses | Within 15 cm / 8 deg |
| --- | ---: | ---: |
| Synthetic development | 186/400 | 171/400 |
| Previously examined synthetic audit | 18/40 | 17/40 |
| Fresh synthetic audit, seed 20261025 | 7/20 | 6/20 |
| Previously examined real imagery | 0/120 | 0/120 |
| Fresh real imagery | 0/80 | 0/80 |

The real-image rows cover SPEED+ lightbox/sunlamp and SHIRT ROE1/ROE2 lightbox.
Their zero recovery remains a blocking failure. The synthetic model's emitted
poses reached **27.7 cm** worst-case position error on the earlier audit and
**56.1 cm** on the fresh audit despite passing pixel admission. Small reprojection
error alone does not establish a trustworthy pose. These are quarantined
hypotheses, not GNC-valid outputs. Withholding every real-image pose cannot prove
a useful estimator's false-valid rate. SHIRT samples are correlated observations.

Development crop containment is **99.64%**, visible-point RMS is **28.38 px** in
the rectified 640-wide image, and proxy visibility balanced accuracy is **88.48%**.
Per-landmark median errors span 2.07–3.38 px, but their p95 errors reach 42.81 px.
The error tail matters more than a tidy median. These development measurements
are model-selection diagnostics, not held-out reliability evidence.

Both stages trained for 60 epochs with a fixed 1,600/400 split. Development loss
selected box epoch 51 and landmark epoch 60. Training took **995.1 s**, excluding
data preparation/downloads. Peak allocated GPU memory was **623.7 MiB** for the box
stage and **712.6 MiB** for landmarks; these are training-process measurements.
Source-FOV masking excludes 484 training and 140 development visibility labels.

Frozen checkpoint SHA-256 values:

```text
box        4c8246bee3a8b3a3bb74a9fc865c10507199a048ef93b7321a702bc49ad2afe2
localizer  45c4e847d51a125e7ed83b24b844636203a22266d1fd9bb0f5fa7fc24b7b378e
```

The training-pool transfer was 367.5 MB; the fresh audit transfer was 35.2 MB.
All new training images came from the official synthetic training split. The
fresh audit was selected only after freezing weights, excluded all earlier
selections by filename and image hash, and did not change thresholds or weights.
This run froze weights only. The improved audit tool now also binds inference
source, requirements and configuration; that stronger contract is not claimed
retroactively for the reported audit.

The [committed evidence](evidence/transfer-perception-2026-10-19.json) preserves
training histories, image hashes, aggregates, timing samples and rejected results.
Images, checkpoints and per-image scoring records remain under ignored `output/`.

### Rejected image-corner refinement

A separate experiment reused the same weights and refined predictions toward
nearby image corners, with an eight-pixel displacement limit and collision
rejection. Development hypotheses fell from 186 to 177; correctly scored poses
fell from 171 to 163; worst-case position error increased from 35.5 to 42.9 cm.
It still recovered **0/120 real-image poses** on the reused audit. No further
fresh audit was spent on it. The inference path was removed; loading metadata
that enables the rejected experiment fails explicitly. Its source snapshot and
full results remain in ignored `output/refined-perception/`.

A bounded attitude/range seed-search prototype was also rejected before changing
the Rust pipeline: multiple depth fits passed its image checks. Neither prototype
justifies relaxing admission or claiming improved estimator integrity.

### Host cost

The batch-one FP32 CPU run used four threads and 20 development images after
three warmups. Tensor storage is **92.03 MiB** across both models; total checkpoint
files are 96.58 MB. Vision latency was **61.91 ms median, 65.84 ms p95, 67.61 ms
maximum**. All observed samples met the proposed 100 ms host budget. Whole-process
peak working set was **518.54 MiB**, including Python/PyTorch.

This is an uncontrolled desktop measurement, excludes decoding/rectification,
I/O and scheduling, and is not a worst-case execution bound. It provides no Zynq
latency, activation-memory, stack, power or radiation evidence. No flight memory
budget or deployment architecture has been selected. A compact failing model is
still a failing model; compression remains gated on useful perception accuracy.

## Model and data

The reference contains two separate ResNet-18 encoders: one for image-only target
boxes and one for landmark heatmaps plus visibility. It is a desktop FP32 baseline,
not the selected embedded architecture. ImageNet weights provide initialization,
not satellite-pose competence. Inference accepts pixels, calibration and known
geometry; no pose prior or evaluation annotations enter prediction.

The training pool expands from 500 to 2,000 SPEED+ synthetic training images,
split deterministically into 1,600 training and 400 development images. Filename
hashing selects new images without consulting pose or performance. Image hashes
are checked against all 360 previously selected audit images. The six-face bus
visibility proxy still excludes appendage volumes and illumination; it is not
verified CAD supervision. Off-screen landmarks cannot become visible through ROI
padding. The new reference explicitly allows partial-FOV crop extents up to twice
the image dimension; the earlier model retains its original limit.

Unweighted all-correspondence mean error and inlier RMS remain bounded at 3 px.
Local peak mass and visibility screens are unchanged. Experimental hypotheses
remain quarantined from navigation and guidance. Covariance inflation cannot
override admission, and unordered audit frames do not acquire invented timestamps.

## Reproduce

Install CPU inference and tests from `scripts/requirements-learned.txt`. For offline
training, use a separate environment with `scripts/requirements-training-gpu.txt`.
The measured training environment uses CUDA 12.8 wheels on the RTX 3070 Laptop GPU;
these are host training dependencies, not packages for a Zynq runtime.

```sh
python scripts/training_pool.py --count 2000 --seed 20261024 --exclude-selection output/datasets/fresh-selection.json --exclude-selection output/datasets/fresh-selection-20261023.json
python scripts/analytic_visibility.py --selection output/datasets/training-2000.json --out output/visibility-2000/labels.json
python scripts/transfer_perception.py --device cuda --epochs 60 --batch 16 --threads 4
python scripts/crop_perception.py development --out output/transfer-perception --labels output/visibility-2000/labels.json --threads 4
python scripts/perception_overlays.py --model-dir output/transfer-perception
python scripts/profile_perception.py --model-dir output/transfer-perception --samples 20 --threads 4
python scripts/crop_perception.py evaluate --out output/transfer-perception --threads 4
```

Training checks geometry and exact image hashes, chooses checkpoints using only
development loss, and writes portable tensor-only checkpoints. Inference performs
no weight download. Fresh audits require frozen weights and image hashes disjoint
from both training and development. Use a new output directory for a new experiment;
do not tune against a reserved audit and then call it untouched.

`perception_overlays.py` renders deterministic development samples with predictions
and a separately labeled annotation projection. Labels enter only after inference.
The artifact is a debugging aid, not an accuracy or safety claim.
Use `--domain speedplus/lightbox/test`, `speedplus/sunlamp/test`,
`shirt/roe1/lightbox` or `shirt/roe2/lightbox` to inspect the already examined audit.
Doing so makes those images development evidence for subsequent algorithm choices;
reserve another disjoint audit after freezing any revised pipeline.

## Embedded decision

`profile_perception.py` measures batch-one CPU inference after warmup, tensor storage
including buffers/dtypes, checkpoint files, and whole-process peak working set.
It records median, p95 and maximum latency against a declared host deadline.
Decode, rectification, I/O and scheduling are outside that timed region. Python/
PyTorch process memory is not standalone activation memory or an ARM stack bound.
An optional `--weight-budget-mib` compares storage with a proposed budget; it does
not establish the capacity of an unspecified flight computer.

Before choosing a deployment model, define processor allocation, camera rate,
memory, power and deadline budgets for a specific board. Then compare a smaller
model or distilled student against the same admission and accuracy requirements,
export to the selected ARM/FPGA runtime, and measure its memory and timing on that
board. INT8 must pass accuracy and integrity gates before promotion. Bare-metal
Rust portability does not establish accelerator support or radiation tolerance.

Official references: [SPEED+ baseline](https://github.com/tpark94/speedplusbaseline),
[SPNv2](https://github.com/tpark94/spnv2), and
[Torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
The authors' SPNv2 release could not be listed anonymously during this run (HTTP
401); no author-trained pose checkpoint was used. The reference initializes from
the official ImageNet checkpoint instead.

## Next implementation decision

The stronger features resolve part of the synthetic fitting failure, but not the
synthetic-to-camera appearance gap or the wrong-pose tail. The next experiment
should test stronger training-only appearance randomization or a separately
reserved, unlabeled real-image adaptation pool. Do not train on these audit labels.
Keep model changes and integrity admission independent, score availability and
incorrect emitted poses together, then reserve a new audit for the frozen pipeline.
Dependable real-image reacquisition and embedded model selection remain open.
