# MicroStructAI evaluation protocol v1 — Session 5

Status: implemented engineering demonstration; real development-data preparation
is pending. This document freezes the initial policy, not a claim of measured
model performance. Revise the version explicitly when policy changes.

## Dataset and split policy

- Preserve Figshare DOI `10.6084/m9.figshare.30175549.v2`, original train/test
  membership, raw masks and file hashes. Consult the tracked dataset card.
- Only official training records may enter development-training/validation.
  Official test records receive `frozen_test`; never promote them to development.
- Strict grouping requires verified `source_group_id` lineage on every training
  record. The code validates presence and overlap; it cannot establish the truth
  of supplied identifiers. Do not derive independent specimen IDs from filenames.
- Sort unique development groups, shuffle with Python `random.Random(42)`, and
  select 25% of groups for validation using round (ties to even), clamped so both
  roles are nonempty. Unequal group sizes mean the image fraction can differ.
  Save every assignment; a seed alone is insufficient provenance.
- Reject duplicate sample IDs, known train/test source overlap, and exact
  cross-partition image hashes when hashes are supplied. Unknown test lineage
  remains a limitation even when training/validation grouping is verified.

Current real manifest: three official-test pairs only, zero audited official
training records, unknown lineage. The strict splitter must reject it. No real
grouped development split has been created. If lineage remains unavailable after
auditing official training data, any image-level experiment needs a separate,
explicitly limited protocol and must not claim specimen/texture independence.

## Labels and metrics

Original ten-class IDs: 0–8 are geometric pearlite lamellar orientations in nine
20-degree bins; 9 is ferrite. For a derived constituent task: original 9 becomes
0=ferrite; original 0–8 become 1=pearlite. Preserve original masks.

For each declared class compute TP, FP, FN, support, IoU, Dice, precision and
recall. Pixel accuracy is supplementary. No implicit ignore label exists.
Shapes must match, masks must contain integer IDs, and every ID must belong to
the declared schema. Both-empty class overlap is undefined: serialize null and
exclude it from macro averages. A class present only in one mask scores zero
overlap and remains included. Undefined precision or recall is also null.

The implemented evaluator operates on a single supplied mask pair. For a future
dataset benchmark, sum class TP/FP/FN across the selected partition before
computing pooled per-class overlap and its macro mean. Report per-image/group
distributions separately; do not conflate pooled metrics with means of image
scores. Archive the support counts and evaluated sample list. Report original
ten-class and derived binary constituent results separately.

Complement overlap metrics with boundary/thin-feature checks, visual failure
review, and variation across declared training seeds. Boundary metric definition,
tolerance, repeated seeds and model settings must be fixed in a later runnable
benchmark configuration before selection. Session 5 does not implement these
future model experiments or present toy scores as model accuracy.

## Fair comparison and records

Compare classical thresholds, U-Net and DeepLab on identical eligible partitions,
label mappings and metrics. Record model-specific input requirements and any
preprocessing differences explicitly. Select preprocessing and hyperparameters
on training/validation only, with a documented search budget. Freeze selected
configurations before official-test evaluation; do not iterate on test feedback.

Record dataset/version/license/hashes, group evidence and assignments, seed,
preprocessing parameters/version, architecture/hyperparameters, optimizer,
learning rate, epochs, training seeds, metric aggregation, environment/hardware,
code commit and configuration. Training and inference preprocessing must match.
Synthetic benchmark results and separately sourced real-micrograph robustness
results require separate reporting and provenance.

## Immutable protocol artifacts

`session05_protocol.py` creates JSON exclusively and refuses overwrite. Use a new
filename for each explicit revision. The report includes manifest checksum when
provided, script SHA-256, Git HEAD and dirty-tree flag, Python/NumPy/platform,
split configuration/assignments, label and metric policies, and clearly marked
toy metrics. No timestamp is injected into split logic. Exclusive creation is
an application safeguard, not a tamper-proof archive or a filesystem lock after
creation. Model/training/hardware fields remain unavailable rather than invented.
