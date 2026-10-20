# Crop consistency and acquisition integrity

**Qualification still fails.** Paired-crop training improves synthetic development
coverage, but does not establish dependable real-image acquisition. The candidate
remains a research experiment; no navigation, INT8 or main promotion is approved.

This follows the [appearance experiment](appearance-integrity.md).

## Implementation

The landmark network now trains on two independently jittered and appearance-randomized
crops of each synthetic training image. Each view retains supervised heatmap and
visibility losses. An auxiliary loss, weighted 0.1, compares local peak coordinates
after mapping both crops back into their common parent frame. It excludes landmarks
that are hidden, unknown, outside either crop, or removed by the appearance cutouts.

Peak coordinates use a differentiable expectation within a detached 5×5 argmax
neighborhood, matching the inference decoder. The loss does not average distant
peaks or manufacture pseudo-labels. Two consistently wrong predictions can still
agree; agreement is not evidence of semantic correctness.

The prior appearance model initializes the run. Its box checkpoint is verified and
reused without retraining. The 1,600/400 synthetic train/development split stays
fixed. Landmark training runs for 40 epochs on the laptop RTX 3070; epoch 37 is
selected by mean clean/corrupted development loss. Training takes 709.0 seconds,
excluding preparation; peak allocated GPU memory is 998 MiB. The paired batch has
32 views from 16 source images. No GPU is required by the CPU inference path.

Paired supervision and the auxiliary loss were introduced together. This is not an
isolated causal ablation of the auxiliary loss.

## Development comparison and candidate selection

All counts below use the same 400 synthetic development images. “Correct” means
position error ≤15 cm and attitude error ≤8°, a diagnostic screen rather than a
mission requirement.

| Weights / screening | Emitted | Correct | Incorrect | Worst position error |
| --- | ---: | ---: | ---: | ---: |
| Previous appearance model / raw | 186 | 165 | 21 | 41.4 cm |
| Previous appearance model / crop stability | 22 | 21 | 1 | 17.1 cm |
| Paired-crop model / raw | 192 | 173 | 19 | 33.2 cm |
| Paired-crop model / geometry sensitivity only | 74 | 72 | 2 | 26.1 cm |
| Paired-crop model / crop stability | 31 | 31 | 0 | 13.7 cm |

The geometry-only control runs leave-one-out pose fits without two additional CNN
crop evaluations. It keeps the existing 3 px pose gates and the 5 cm / 2° pose
sensitivity limits. It provides less evidence than the crop-stability policy:
correspondence repeatability across crops is not checked. Its two inaccurate
outputs rule it out as the next audit candidate.

The stricter `roi_stability_v2` candidate is selected **before the new audit**.
Its correct development coverage rises from 5.25% to 7.75%, while observed incorrect
outputs fall from one to zero. That remains inadequate acquisition coverage;
31 accepted development images cannot establish a dependable false-valid rate.
No numerical admission threshold is relaxed or tuned using audit results.

## Audits

The previously examined 160-image audit is a regression check, not an untouched
test. The new model produces 21 raw synthetic hypotheses, of which 15 meet the
accuracy screen; its worst raw error is 53.1 cm. Crop screening retains two correct
synthetic poses. Both raw and screened paths produce **0/120 real-image poses**,
losing the previous model's two isolated real-image hypotheses. Better development
loss and synthetic coverage did not translate into real-image progress.

The untouched seed-20261027 audit freezes the checkpoints, policy, inference code,
requirements, geometry and camera calibration before selecting 100 disjoint images.
It excludes all earlier audit selections, training/development images, and the
reserved adaptation pool by filename and hash. Archive size/CRC checks pass;
35.4 MB is transferred in total. Nothing is tuned after seeing these results.

| Fresh domain | Raw hypotheses | Raw correct | Screened hypotheses | Screened correct |
| --- | ---: | ---: | ---: | ---: |
| SPEED+ synthetic | 10/20 | 8 | 1/20 | 1 |
| SPEED+ lightbox | 0/20 | 0 | 0/20 | 0 |
| SPEED+ sunlamp | 0/20 | 0 | 0/20 | 0 |
| SHIRT ROE1 lightbox | 0/20 | 0 | 0/20 | 0 |
| SHIRT ROE2 lightbox | 0/20 | 0 | 0/20 | 0 |

Worst raw synthetic position error is 24.3 cm. The single screened hypothesis has
5.60 cm position error and 1.23° attitude error. Both paths recover **0/80 real-image
poses**. There is too little accepted output to estimate dependable integrity or
uncertainty calibration. SHIRT frames are correlated, not independent mission
trials. This fresh sample differs from earlier seeds; cross-seed counts are not a
paired comparison between models.

Ground truth is used only after image-only inference for scoring. Inference has no
pose prior or truth-derived crop. The exported audit packets retain null navigation
poses, false admission, and no invented camera timestamps. The synthetic Rust
navigation path and optical-flow replay are not requalified by this experiment.

## Reserved real-image pool

`scripts/unlabeled_pool.py` reserves 200 additional real images: 50 each from SPEED+
lightbox, SPEED+ sunlamp, SHIRT ROE1 lightbox and SHIRT ROE2 lightbox. Selection uses
archive filenames and exclusion provenance; no dataset per-frame pose JSON is
opened. Each manifest row contains only a filename. Archive size, CRC and SHA-256
are checked, with a 128 MiB transfer limit per archive.

The pool is disjoint from previous audits by filename and image hash, and the new
audit explicitly excludes it. Seed 20261028 transfers 62.5 MB in total. **These
pixels have not been used for training.** This is preparation for a later adaptation
experiment, not evidence that domain adaptation works. Its future use must not
promote self-consistent wrong landmark identities into supervision.

## Desktop cost and deployment

The sequential 20-frame profiles use four CPU threads and batch one, without an
active training job. They time vision only; decoding, rectification, I/O and
scheduling are excluded.

| Policy | Median | p95 | Maximum | 100 ms budget |
| --- | ---: | ---: | ---: | --- |
| Crop stability | 175.1 ms | 198.4 ms | 206.3 ms | Fail |
| Geometry sensitivity only | 96.5 ms | 105.7 ms | 107.4 ms | Fail |

Both store 92.03 MiB of tensors. Peak host process working sets are 496 MiB and
519 MiB respectively, including Python/PyTorch; these are not standalone model
memory bounds. The cheaper control improves host cost but fails both the observed
deadline and development error screen. It is not a deployable substitute.

Neither experiment measures Zynq execution, activation-memory bounds, stack, power,
radiation tolerance, or real-time scheduling. Rust cross-builds remain portability
evidence only. Float acquisition and integrity must work before another selective
INT8 deployment attempt is justified.

## Reproduce

Use the separate CUDA-training and CPU-inference environments and existing datasets
from the [transfer report](transfer-perception.md#reproduce). Output directories must
be new; preserve all selection manifests.

```sh
python scripts/transfer_perception.py --device cuda --epochs 40 --threads 4 --batch 16 --appearance sensor --initialize output/appearance-perception --reuse-initial-box --consistency-weight 0.1 --out output/consistency-perception
python scripts/integrity_development.py --initialize-from output/consistency-perception --model-dir output/consistency-integrity --policy roi_stability_v2 --threads 4
python scripts/integrity_development.py --initialize-from output/consistency-perception --model-dir output/consistency-geometry --policy geometry_sensitivity_v1 --threads 4
python scripts/crop_perception.py evaluate --out output/consistency-integrity --threads 4
```

For a new untouched audit, freeze first and exclude every prior audit and adaptation
pool. The recorded audit uses seed 20261027 and excludes `fresh-selection.json`,
`fresh-selection-20261023.json`, `fresh-selection-20261025.json`,
`fresh-selection-20261026.json`, and `unlabeled-adaptation.json`, all under
`output/datasets/`. Original benchmark and training selections are excluded automatically.

```sh
python scripts/fresh_audit.py --model-dir output/consistency-integrity --seed 20261027 --exclude-selection output/datasets/fresh-selection.json --exclude-selection output/datasets/fresh-selection-20261023.json --exclude-selection output/datasets/fresh-selection-20261025.json --exclude-selection output/datasets/fresh-selection-20261026.json --exclude-selection output/datasets/unlabeled-adaptation.json --out output/datasets/fresh-selection-20261027.json
python scripts/crop_perception.py evaluate --out output/consistency-integrity --selection output/datasets/fresh-selection-20261027.json --threads 4
python scripts/profile_perception.py --model-dir output/consistency-integrity --samples 20 --threads 4
python scripts/profile_perception.py --model-dir output/consistency-geometry --samples 20 --threads 4
```

The [committed evidence](evidence/crop-consistency-2026-10-19.json) records training,
checkpoint and pipeline hashes, aggregate results, the unused pool and host costs.
Images, checkpoints and per-image scoring stay under ignored `output/`.

The next experiment needs real-image adaptation with explicit semantic safeguards,
plus box/landmark error diagnostics on development data. Repeating synthetic
augmentation without measuring those errors has reached diminishing returns.
An independently reserved audit must remain outside every training decision.
