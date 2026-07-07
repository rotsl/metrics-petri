# Model Card: Petri-Dish Colony Segmentation SmallUNet

## Model summary

This checkpoint is a compact U-Net for binary semantic segmentation of fungal colony
area in RGB photographs of 90 mm Petri dishes. It accepts a `256 × 256` RGB image and
returns a one-channel sigmoid probability mask. The packaged pipeline thresholds the
mask at `0.5` and resizes it to the source image with nearest-neighbour interpolation.

The network uses base width 16, four encoder/decoder stages, and a 256-channel
bottleneck. It has approximately 250,000 parameters and was trained from scratch.

## Training data

The training corpus consists of project-produced RGB photographs of filamentous fungal
colonies growing on Petri dishes. Colony boundaries were manually represented as
LabelMe polygon annotations and rasterised into binary foreground masks. The data cover
the controlled imaging setup used by the companion training project; they are not a
representative sample of all fungal species, culture media, plate formats, cameras, or
lighting conditions.

The published training data and annotations are released under the Apache License 2.0.
They are maintained in the companion training repository at
<https://github.com/rotsl/petrimodel>, which includes the training data, LabelMe JSON
annotations, trained checkpoints, sweep plots, and a PySide6 desktop tool for reviewing
manual diameter measurements against model-generated masks. That licence applies to the
distributed corpus; users remain responsible for confirming that their downstream data
collection and use comply with applicable consent, institutional, and biosafety
requirements.

## Training procedure and hyperparameters

| Property | Value |
| --- | --- |
| Architecture | SmallUNet, base channels 16 |
| Input | `256 × 256` RGB |
| Output | One-channel sigmoid mask |
| Optimizer | Adam (`β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`) |
| Learning rate at checkpoint | `1e-4` |
| Weight decay | `0` |
| Objective | Binary cross-entropy plus area-consistency loss |
| Area-consistency weight | `0.7` |
| Selected checkpoint epoch | `66` |
| Inference threshold | `0.5` |

Batch size, augmentation settings, random seed, and the maximum epoch budget are not
stored in the distributed checkpoint and are therefore not asserted here.

## Validation set

Checkpoint selection used a held-out validation partition of the same manually
annotated Petri-dish image corpus. Validation masks were not used for gradient updates.
The distributed artifact does not record the partition size, image identifiers, split
file, or whether plates from the same experimental series were grouped. Related
training and evaluation materials are documented in the companion training repository,
but this package does not publish the exact validation image IDs. These omissions limit
independent assessment of leakage and uncertainty; comparisons should reuse a
documented, plate-level split and report sample counts.

## Validation metrics

The checkpoint records the following best validation results:

| Metric | Value |
| --- | ---: |
| Intersection over Union (IoU/Jaccard) | `0.899744` |
| Dice coefficient | `0.941550` |
| Validation loss | `0.234586` |

IoU and Dice measure pixel overlap after binary segmentation. These point estimates do
not include confidence intervals, per-image dispersion, external validation, or
performance stratified by imaging condition.

## Intended use

- Segment fungal colony area in controlled, top-down RGB photographs of fully visible
  90 mm Petri dishes.
- Support research measurements such as colony area, equivalent diameter, perimeter,
  and time-series growth rates.
- Provide an initial mask for review or correction by a researcher.

Users should visually inspect masks and validate performance on their own imaging setup
before using derived measurements in an experiment.

## Out-of-scope uses

- Clinical diagnosis, treatment decisions, pathogen identification, or food-safety
  decisions.
- Species classification or inference of genotype, virulence, viability, or toxicity.
- Images outside the documented domain, including microscopy, non-circular vessels,
  partially visible dishes, severe glare, heavy occlusion, or substantially different
  media and illumination, without additional validation.
- Fully autonomous acceptance or rejection of experimental results.

## Ethical considerations and limitations

Segmentation errors can propagate into growth and morphology measurements and may bias
comparisons when image quality differs systematically between experimental groups.
Small colonies, weak contrast, glare, shadows, condensation, contamination, and unusual
morphology may increase false positives or false negatives. Researchers should preserve
source images, report the software and checkpoint version, audit representative masks,
and document exclusions or manual corrections.

The model processes laboratory images rather than personal data by design. Nevertheless,
users should remove unintended labels, names, QR-code payloads, or other identifiers
before sharing images or outputs. Work involving pathogenic organisms remains subject
to institutional biosafety procedures; this model does not replace biological risk
assessment or trained human oversight.
