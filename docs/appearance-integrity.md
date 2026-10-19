# Appearance generalization and correspondence integrity

**The revised candidate still fails qualification.** Appearance randomization
recovers isolated real-image hypotheses on the previously examined audit, but no
poses on the new real-image audit. The consistency policy loses too much useful
coverage and exceeds the proposed desktop latency budget. Neither model nor
policy is promoted to navigation, INT8 deployment or main.

This follows the [transfer-learning reference](transfer-perception.md).

## What changed

Training now supports a verified warm start and deterministic sensor appearance
randomization. Geometry and the 1,600/400 synthetic training/development split stay
fixed. The policy retains 25% clean inputs and varies exposure, gamma, illumination
gradients, 3/5/7-pixel blur, read/shot-noise approximations, clipping and 5–8-bit
quantization. Partial cutouts remove affected landmarks from visible supervision;
unknown labels remain unknown. These effects are training augmentations, not a
calibrated optical model or radiation simulation.

Both stages fine-tuned for 40 epochs from the prior verified checkpoints.
Checkpoint selection uses the mean of clean and fixed-seed corrupted development
loss, without audit labels. Box epoch 40 and landmark epoch 33 were selected.
Training took 754.1 s excluding data preparation, on the laptop RTX 3070. The GPU
remains an offline training tool; inference and flight deployment do not depend on it.

The optional `roi_stability_v2` policy screens each landmark across the base crop
and two small deterministic crop perturbations. Selection uses visibility, peak
mass, source FOV and at most 3 px coordinate disagreement. It does not select by
pose residuals, covariance, or evaluation annotations. At least seven distinct
landmarks must remain, with noncoincident image points and nonplanar geometry.

Every retained correspondence then faces the existing 3 px inlier RMS and
all-correspondence mean gates. Fits from both perturbed crops and every leave-one-out
subset must agree within 5 cm translation and 2 deg attitude. There are at most
13 additional pose fits. Malformed inputs and nonfinite poses fail closed.
Rejected integrity output cannot fall back to the unfiltered pose.

These are **uncalibrated same-image repeatability limits**, not correctness
probabilities. A test explicitly demonstrates a correlated wrong pose passing
all consistency checks. The policy remains an opt-in benchmark experiment;
the Rust navigation pipeline and optical-flow replay adapter are not requalified.

## Development evidence

All rows use the same 400 synthetic development images. “Correct” means position
error <=15 cm and attitude error <=8 deg, not mission acceptance.

| Candidate | Emitted | Correct | Worst position error |
| --- | ---: | ---: | ---: |
| Original FP32 reference | 186 | 171 | 35.5 cm |
| Original weights, reject whole unstable frames | 18 | 18 | 13.8 cm |
| Original weights, per-landmark screening | 34 | 34 | 13.8 cm |
| Appearance-randomized weights, raw | 186 | 165 | 41.4 cm |
| Appearance-randomized weights, screened | 22 | 21 | 17.1 cm |

Whole-frame rejection was too restrictive and is not supported by the final
inference path. Per-landmark selection improves that control, but the joint
candidate's **5.25% correct coverage** is still unusably low. It also retains an
incorrect emitted hypothesis. Fewer outputs cannot be reported as a reliability
success while ignoring the acquisition failure.

## Paired audits

Raw and screened results come from the same frozen candidate and images. Ground
truth enters only after prediction for scoring; no truth crop or pose prior is
provided to inference. Diagnostic baseline poses are stored only in research
reports. Transport exports still contain null poses and no invented timestamps.

| Domain | Reused raw | Reused screened | Fresh raw | Fresh screened |
| --- | ---: | ---: | ---: | ---: |
| SPEED+ synthetic | 21/40 | 2/40 | 9/20 | 1/20 |
| SPEED+ lightbox | 0/40 | 0/40 | 0/20 | 0/20 |
| SPEED+ sunlamp | 1/40 | 0/40 | 0/20 | 0/20 |
| SHIRT ROE1 lightbox | 1/20 | 0/20 | 0/20 | 0/20 |
| SHIRT ROE2 lightbox | 0/20 | 0/20 | 0/20 | 0/20 |

Of the 21 raw synthetic hypotheses on the reused audit, 17 meet the accuracy
screen, matching the original model's correct count on those same 40 images.
The two real-image hypotheses also meet it: 8.05 cm and 3.36 cm position errors.
The integrity policy rejects both. This is an isolated result on already examined
images, not proof of real-image generalization.

On the untouched audit, all nine raw synthetic hypotheses meet the accuracy
screen; the screened pipeline retains one, at 1.81 cm / 0.62 deg. Both paths
recover **0/80 real-image poses**. One accepted synthetic example cannot establish
a useful false-valid rate or uncertainty calibration. SHIRT frames are correlated,
not independent mission trials. Different fresh seeds are different samples;
their counts are not a paired causal comparison with the previous experiment.

Seed 20261026 selects 100 new images only after freezing the model, policy,
inference source, requirements, camera calibration and landmark geometry. The
selection excludes the original selection, all three earlier fresh audits, and
the expanded training/development pool by filename and image hash. Archive
size/CRC and SHA-256 checks pass. Transfer totals 35.0 MB, within the 128 MiB
per-dataset limit. Nothing was tuned after selecting or inspecting this audit.

## Host cost and deployment decision

The 20-frame, four-thread CPU profile records **126.0 ms median, 138.6 ms p95 and
143.4 ms maximum** vision latency, excluding decode, rectification, I/O and
scheduling. It fails the proposed 100 ms desktop budget. Tensor storage remains
92.03 MiB across both models. Process memory includes Python/PyTorch and is not
an activation-memory or stack bound.

The candidate is not selected for embedded deployment: accuracy, coverage and the
host budget all fail. No Zynq runtime, power, stack or radiation evidence is added.
Cross-build portability remains separate from hardware performance. INT8 remains
deferred while float accuracy and useful acquisition coverage fail.

## Reproduce

Use the existing dataset setup and separate CPU inference / CUDA training
environments described in the [transfer report](transfer-perception.md#reproduce).
The original reference checkpoints and training pool must already exist.

```sh
python scripts/transfer_perception.py --device cuda --epochs 40 --threads 4 --batch 16 --appearance sensor --initialize output/transfer-perception --out output/appearance-perception
python scripts/integrity_development.py --initialize-from output/appearance-perception --model-dir output/appearance-integrity --threads 4
python scripts/crop_perception.py evaluate --out output/appearance-integrity --threads 4
python scripts/profile_perception.py --model-dir output/appearance-integrity --samples 20 --threads 4
# Freeze first. Preserve every exclusion selection and do not tune against this audit.
python scripts/fresh_audit.py --model-dir output/appearance-integrity --seed 20261026 --exclude-selection output/datasets/fresh-selection.json --exclude-selection output/datasets/fresh-selection-20261023.json --exclude-selection output/datasets/fresh-selection-20261025.json --out output/datasets/fresh-selection-20261026.json
python scripts/crop_perception.py evaluate --out output/appearance-integrity --selection output/datasets/fresh-selection-20261026.json --threads 4
```

Use fresh output directories for a new experiment. Integrity initialization
verifies parent checkpoint hashes, preserves the parent, and records its metadata
hash. Fresh audits reject changed code, configuration, calibration or weights.
The [committed evidence](evidence/appearance-integrity-2026-10-19.json) records
training history, hashes, paired aggregates, controls and host timing. Images,
checkpoints and per-image scoring records stay under ignored `output/`.

## Next experimental question

Broad appearance corruption alone did not produce dependable real-image recovery,
and post-hoc repeatability screening discards correct poses. A useful next test
is explicit multi-crop consistency training, scored jointly for correct coverage,
incorrect emissions and compute cost. Any real-image adaptation pool must remain
disjoint from these audits, with audit ground truth reserved for scoring.
The existing synthetic navigation envelope and calibrated uncertainty claims
remain unresolved; this experiment does not improve their qualification status.
