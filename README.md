# MicroStructAI

Computational steel metallography, developed over guided learning sessions.

## Session 1: inspect a micrograph

Load a microscopy image, inspect RGB and grayscale arrays, extract a central
crop, and save a four-panel figure with a 256-bin intensity histogram.
No segmentation or model training is performed.

### Setup (Windows PowerShell, Python 3.13)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place a microscopy image at `data/raw/steel_01.png`. See `data/README.md` for the
educational sample's source and usage context.

```powershell
.\.venv\Scripts\python.exe src\numpy_basics.py
.\.venv\Scripts\python.exe src\inspect_micrograph.py
# Optional custom paths (relative paths resolve from the current directory):
.\.venv\Scripts\python.exe src\inspect_micrograph.py --image path\to\image.jpg --output outputs\custom.png
```

The default input and output paths resolve from the project root even when run
from another directory. Open `outputs/steel_01_inspection.png` after running.

### Scientific interpretation

- An array uses `[row, column]`; grayscale shape is `(height, width)`.
- The crop retains the middle portion without resizing; odd dimensions round.
- Grayscale conversion produces 8-bit values and may discard color or higher
  bit-depth information. It is for this introductory inspection task.
- Brightness and histogram peaks cannot establish ferrite or pearlite fractions.
- Pearlite is a constituent containing ferrite and cementite, not a single phase.
- Physical dimensions require verified calibration in micrometers per pixel.
- Preparation artifacts, boundaries, annotations and lighting affect intensities.

Virtual environments, raw images, generated outputs and personal session notes
are excluded from Git. Each session's learning notes live in `learning-notes/`.

## Session 2: digital images, sampling and interpolation

```powershell
.\.venv\Scripts\python.exe src\session02_image_lab.py
```

Saves `outputs/session02_image_lab.png`: original grayscale, fixed crop,
histogram, nearest-neighbor and Lanczos half-size images, and center-row profile.
Prints image mode, dtype, intensity statistics and interpolation experiment results.
Optional `--image` and `--output` arguments work as in Session 1.

This exercise accepts 8-bit `L` or `RGB` inputs at least 240 rows by 360 columns.
Higher-bit-depth inputs are rejected before conversion. The crop is
`gray[40:240, 60:360]`, with shape `(200, 300)`; it retains original sampling.
Resizing preserves the field of view but changes sampling. For the example's
exact half-size resize, micrometers per pixel would double if calibration were
verified. No physical measurements are claimed here.

The synthetic categorical-mask experiment asserts that nearest neighbor retains
IDs `{0, 2}`, while bilinear creates the invalid intermediate ID `1`. These toy
labels are not annotations of the real micrograph. Lanczos is demonstrated for
intensity images, where smoothing and possible ringing still require inspection.
Histograms discard spatial arrangement; line-profile changes do not by themselves
identify phase boundaries. Session 1 scripts and the raw image are preserved.

## Session 3: dataset provenance and validation

```powershell
# Artificial deterministic pairs: 2 train, 1 test (not steel images).
.\.venv\Scripts\python.exe src\session03_dataset_audit.py --demo
# Inspect/download Figshare v2's 115 MB test archive and audit 3 genuine pairs.
.\.venv\Scripts\python.exe src\session03_prepare_figshare.py
# Controlled success/failure cases:
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The auditor expects flat `train/images`, `train/masks`, `test/images`,
`test/masks` folders with matching PNG filenames. Empty partitions are allowed
for subset inspection; the dataset as a whole must contain at least one pair.
Use `--data-root`, `--manifest`, and `--source-dataset` for a research directory;
never use `--demo` there. The demo option is restricted to its dedicated path.

Checks include image/mask mode and dtype, dimensions, IDs 0–9, missing pairs,
unexpected files, and byte-identical images across partitions. Palette masks are
read as indices without grayscale conversion. The CSV includes source identity,
original split, dimensions, class IDs and separate image/mask SHA-256 digests.
Unknown provenance remains UNKNOWN and unverified pixel scale remains blank.
Records are sorted and CSV replacement occurs only after all checks pass.
On failure, any previous manifest remains an earlier result and is not refreshed.

See [the dataset card](data/figshare_orientation_dataset_card.md) for attribution,
license, exact schema, archive counts and limitations. The local research audit
covers three official-test pairs; it does not establish full-dataset or
source-lineage independence. No training or model evaluation occurs here.
Downloads, artificial fixtures, metadata snapshots and manifests stay out of Git.

## Session 4: preprocessing laboratory

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\session04_preprocessing.py
.\.venv\Scripts\python.exe src\session04_preprocessing.py --image data/raw/steel_01.png
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Outputs: `session04_synthetic.png`, `session04_real_comparison.png` (when an image
is supplied), and `session04_metrics.json`, all under ignored `outputs/`.
The JSON records fixed parameters, library versions, whole-image and thin-line
MAE, and the educational input's SHA-256. The replacement synthetic fixture uses
seed 42, two circles, a two-pixel line, Gaussian noise (sigma 10), 2% impulse
probability, and a separate horizontal illumination field from 0.6 to 1.2.
The handout's generator was truncated, so its exact MAE numbers are not reproduced.

Comparisons use Gaussian (5 x 5, sigma 1 pixel), median (3 x 3), CLAHE (clip limit
2, 8 x 8 tiles), heuristic multiplicative shading correction (Gaussian estimate,
sigma 55 pixels), and correction followed by CLAHE. Other methods are compared
independently; this is not a selected production pipeline. The reference level
for shading correction is the estimated field mean, with a denominator floor of 1.
This is not a calibrated blank/dark-field correction.

Only 8-bit L/RGB intensity images are accepted. Source arrays are checked for
mutation and supplied image-file hashes are checked before/after processing.
No mask inputs or processing are part of this lab; mask folders and the reserved
Figshare directory are blocked. Do not tune on official-test copies elsewhere.
Real-image comparison is qualitative, since no clean reference is available.
Lower global MAE can still hide thin-feature damage; no segmentation improvement
or physical measurement is claimed.

Algorithm references: [OpenCV smoothing](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)
and [OpenCV histogram equalization/CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html).

## Session 5: reproducible evaluation protocol

```powershell
.\.venv\Scripts\python.exe src\session05_protocol.py --output outputs/session05_protocol_demo.json
# Expected failure: current genuine manifest has official-test records only.
.\.venv\Scripts\python.exe src\session05_protocol.py --manifest outputs/session03_real_manifest.csv --output outputs/session05_real_protocol.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The artificial demonstration assigns 12 groups (60 records) to 45 development
training and 15 validation records, preserving three frozen-test placeholders.
It verifies toy binary-mask accuracy 0.8333, pearlite IoU 0.6/Dice 0.75 and macro
IoU 0.6889. These are arithmetic examples, not trained-model results.

The strict real-data path requires official training records and supplied verified
lineage. Unknown training groups, duplicates and known official train/test group
or image-hash overlap are rejected. Input manifests are unchanged. JSON output
refuses overwrite; choose a new filename for revisions. Optional `--seed` and
`--validation-fraction` control reproducible group assignment, not independence.

See [evaluation protocol v1](docs/evaluation_protocol_v1.md) for labels, absent-class
and aggregation policies, future benchmark requirements and current limitations.
No additional dependencies or real-data split are introduced in Session 5.

## Session 6: classical threshold segmentation

```powershell
.\.venv\Scripts\python.exe src\session06_threshold_lab.py
.\.venv\Scripts\python.exe src\session06_threshold_lab.py --image data/raw/steel_01.png
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Creates even-lighting and shaded synthetic six-panel comparisons, an optional
educational-image comparison, and `session06_metrics.json` in `outputs/session06/`.
Methods: fixed threshold 120, NumPy/OpenCV Otsu, adaptive mean and adaptive Gaussian
(block size 41, C=6). All masks use 0/1, with 1 meaning candidate dark foreground.
OpenCV inverted thresholding includes intensities equal to the threshold.

The reconstructed fixture uses seed 17, a circle, ellipse and rectangle in a
240 x 320 image, dark intensity 55, background 200, Gaussian noise sigma 4 and
horizontal shading factors 0.55–1.25. The referenced source package was not
attached; this reproduces the experiment design and observed failure modes,
not exact source-package arrays. The supplied shaded montage is a visual reference,
not raw numerical input or a supervised mask.

NumPy Otsu maximizes between-class variance using cumulative histogram sums and
chooses the first maximizing threshold. Constant images are explicitly rejected
because no nonempty two-class partition exists. Source arrays/files are preserved;
non-8-bit images, mask folders and reserved Figshare data are rejected.

Synthetic dark-region IoU: fixed/Otsu 1.000/1.000 in even lighting; 0.669/0.388
under shading. Adaptive mean/Gaussian score 0.631/0.446 under shading and produce
holes and speckles. These are artificial-shape scores, not steel performance.
The educational micrograph has no reference mask: only thresholds and candidate
foreground fractions are reported, with no supervised score or constituent claim.

## Session 7: mathematical morphology and connected components

```powershell
.\.venv\Scripts\python.exe src\session07_morphology_lab.py
.\.venv\Scripts\python.exe src\session07_morphology_lab.py --image data/raw/steel_01.png
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Saves synthetic and optional educational comparisons plus `session07_metrics.json`
under `outputs/session07/`. Uses a centered 3 x 3 elliptical (cross-shaped)
structuring element, one iteration, and OpenCV's default neutral morphology
border behavior. Masks must be uint8 0/1; input arrays and image files are preserved.
Kernels must be odd-sized, symmetric, binary uint8 with an active center.

Computes erosion, dilation, opening, closing, and opening followed by closing.
Closing alone is included in JSON; the six-panel figure shows the other variants
alongside the reference. Reports IoU/Dice, signed area error and thin-line
retention for synthetic masks, and foreground fraction/component counts for
educational Otsu candidates. Every variant includes component areas in pixels,
bounding boxes `(x, y, width, height)` and centroids `(x, y)` using 8-connectivity.
A diagonal toy mask separately verifies 3 components with 4-connectivity and
1 with 8-connectivity. Background label 0 is excluded from object counts.

The reconstructed synthetic fixture (seed 23) contains a rectangle, circle and
40-pixel single-column line, plus 120 added speckles and two missing interior
pixels. The referenced source package was not attached, so exact handout IoUs
are not claimed. Opening raises our IoU from 0.984174 to 0.993939 while erasing
the entire line. A cleaner mask or fewer components is not proof of better
metallurgical measurements. No physical calibration or verified particle/grain
interpretation is available; official-test data remains excluded from exploration.

## Session 8: boundaries, distance transforms and watershed

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\session08_watershed_lab.py
.\.venv\Scripts\python.exe src\session08_watershed_lab.py --image data/raw/steel_01.png
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Outputs under `outputs/session08/`: synthetic six-panel comparison, edge/boundary
comparison, optional educational comparison and metrics JSON. SciPy supplies exact
Euclidean distance transforms; scikit-image supplies peaks and marker watershed.
The three-circle fixture uses nearest eligible center to assign overlapping
pixels, with lower-ID tie breaking; seed 31 adds intensity noise. The referenced
source package was not attached, so exact handout scores are not claimed.

Uses `peak_local_max(min_distance=16, threshold_abs=7, exclude_border=False)` and
watershed on negative distance with 4-neighbor flooding and separation lines.
Binary component counts use 8-connectivity and exclude background. Distance and
peak-height parameters are in pixels. Canny uses 40/100 thresholds after Gaussian
preblur (5 x 5, sigma 1). Intensity edges are distinct from instance boundaries.

Instance IoU uses Hungarian maximum-total-IoU matching, divided by the larger of
truth/prediction counts so unmatched objects contribute zero. Both empty is null.
This introductory metric is ID-invariant; it is not AP or panoptic quality.
The synthetic fixture has 2 connected regions, 3 correct markers and 3 instances;
removing/adding a marker gives 2/4 instances. Boundary pixels left at label zero
reduce object areas. Unseeded components also remain zero and are reported
separately from watershed lines. No fallback markers are silently invented.

Educational labels are unvalidated hypotheses, not verified grains or particles.
No instance IoU is reported there or inferred from Figshare semantic class masks.
Source preservation, 16-bit rejection and reserved-test safeguards remain active.
Reference: [scikit-image marker-controlled watershed](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html).

## Session 9: quantitative region measurements

```powershell
.\.venv\Scripts\python.exe src\session09_quantitative_lab.py
.\.venv\Scripts\python.exe src\session09_quantitative_lab.py --demo-um-per-pixel 0.25
.\.venv\Scripts\python.exe src\session09_quantitative_lab.py --image data/raw/steel_01.png
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Creates `outputs/session09/session09_synthetic.png`, `session09_metrics.json`
and `session09_objects.csv`; supplying an image also creates a separate
`session09_educational_objects.csv`. No new dependencies are needed.

Reports valid-ROI area fraction, equal-area circle diameter, Crofton perimeter,
unclamped circularity, second-moment ellipse axes/aspect ratio and pixel centroids.
Objects touching the image/ROI boundary are flagged, not silently discarded.
ROI clipping preserves original IDs and reports disconnected observed fragments;
their descriptors refer to the visible union, not the unknown complete object.
Size distributions are number-weighted observations including flagged objects,
with no claim of unbiased population statistics or ASTM compliance.

The reconstructed four-object fixture uses seed 41 and a known excluded ROI.
A 5 x 5 dilation leaves four components but raises valid-field foreground fraction
from 9.5572% to 11.5545% (+1.9973 percentage points). JSON includes matched object
area/diameter errors, count and semantic IoU. The package referenced in the
handout was not attached, so exact handout fixture counts are not claimed.

`--demo-um-per-pixel` applies ONLY to synthetic descriptors: length scales by s,
area by s squared, and output fields explicitly say hypothetical. Educational
measurements always remain in pixels even when both CLI options are supplied.
The educational full-image ROI still includes annotations and yields unvalidated
candidate fractions/components. Source and reserved-test safeguards remain active.
Reference: [scikit-image regionprops](https://scikit-image.org/docs/0.26.x/api/skimage.measure.html#skimage.measure.regionprops).

## Session 10: classical CV benchmark

```powershell
.\.venv\Scripts\python.exe src\session10_classical_benchmark.py
# A repeat/revision requires a fresh output directory:
.\.venv\Scripts\python.exe src\session10_classical_benchmark.py --output-dir outputs/session10_v1
# Expected rejection: real manifest contains only official-test records.
.\.venv\Scripts\python.exe src\session10_classical_benchmark.py --readiness-manifest outputs/session03_real_manifest.csv
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Seven fixed pipelines on ten artificial scenes generate 70 CSV rows, a JSON
scorecard and a nine-panel shaded-particle comparison. Default output directory:
`outputs/session10/`; existing directories are rejected before any files change.
Three source geometries stay in development (six lighting scenes); particles and
bands stay in validation (four scenes). Lighting variants share truth and remain
together. No research/educational images are read. Readiness mode inspects CSV
metadata only, never promotes official tests, and requires supplied development lineage.

Reports foreground scene-macro IoU/Dice, signed/absolute area-fraction error in
percentage points, 8-connected region-count errors, thin-feature recall where
annotated, confusion counts and worst scenes. Pooled IoU is reported separately.
Absent thin annotations yield null, not perfect retention. Ranking uses validation
scene-macro IoU, then absolute area error, then method name. Validation selection
is not an unbiased final test estimate. Only one configuration per pipeline is
compared; no parameter search or compute-performance benchmark is claimed.

The referenced code package was not attached; our deterministic reconstruction
uses 192 x 256 scenes, seeds 100–104, intensities 55/200, noise sigma 5 and shading
0.6–1.2. Exact handout numbers are not reproduced. Fixed 120 wins this artificial
suite (mean validation IoU 0.8848), but predicts 218 components instead of eight
in shaded particles. Otsu+opening retains only 73.75% of annotated development
thin-feature pixels. These are engineering results, not ferrite–pearlite accuracy.
See [the Session 10 benchmark policy](docs/session10_artificial_benchmark.md).

## Session 11: PyTorch tensors, logits and gradients

```powershell
# Tested CPU installation (before installing the complete requirements):
.\.venv\Scripts\python.exe -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\session11_pytorch_basics.py
# Repeat with a fresh directory; existing outputs are protected:
.\.venv\Scripts\python.exe src\session11_pytorch_basics.py --output-dir outputs/session11_v1
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The artificial Dataset returns float32 `[1,64,64]` images scaled to 0–1 and long
`[64,64]` class-index targets. DataLoader uses batch size 2, no shuffle and zero
workers. TinyPixelNet is Conv2d(1,4,3,padding=1), ReLU, Conv2d(4,2,1): **50**
parameters, logits `[2,2,64,64]`, predictions `[2,64,64]`. Raw logits go directly
to CrossEntropyLoss; softmax is used only for the displayed probability map.

The CPU lab uses model seed 17 and performs exactly one SGD update at learning
rate 0.2. It checks finite gradients, weight changes and source-tensor preservation,
then measures loss on the same batch. This is neither U-Net training nor a
generalization benchmark. No file-based dataset or test imagery is accessed.

`binary_from_orientation` validates original IDs 0–9 before deriving 0=ferrite
(original 9), 1=pearlite (original 0–8), preserving the original array. The separate
toy mask means 0=bright background and 1=artificial dark object, not material labels.
Ten-class output is supported and tested without collapsing original semantics.

Outputs: `outputs/session11/session11_tensor_lab.png` and `session11_report.json`,
including shapes, types, gradients, updates, losses, versions and code provenance.
Directories are created exclusively. The source package referenced by the handout
was not attached; our reconstructed generator may produce different loss values.
References: [PyTorch Conv2d](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Conv2d.html),
[CrossEntropyLoss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html).

## Session 12: small U-Net architecture

```powershell
.\.venv\Scripts\python.exe src\session12_unet_architecture.py --output-dir outputs/session12
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Implements DoubleConv, UpBlock and SmallUNet: two pooling levels, channels 8/16/32,
learned transposed upsampling, concatenated encoder skips and a raw-logit 1 x 1
head. Two-class model: **29,330 parameters**. Ten-class output changes only the
head. Minimum spatial size is 4 x 4; input is float32 `[B,1,H,W]`.

Odd feature sizes are aligned to each skip with bilinear interpolation and
`align_corners=False`; categorical targets are never resized. JSON records every
encoder/pooling/concatenation shape, where alignment occurred, and finite gradient
norms/parameter updates. Shapes include binary `[2,2,64,64]`, ten-class
`[1,10,64,64]`, and odd `[1,2,65,67]` outputs.

The CPU smoke test reuses Session 11's artificial Dataset and performs one SGD
step (seed 12, learning rate .1). It writes `session12_report.json` and
`session12_unet_smoke.png` under the requested fresh output directory; existing
directories are refused. No new dependencies, real-image reads or model-performance
claims. This small same-padded variant is not the exact original U-Net architecture.
The referenced package was not attached, so actual loss values may differ from
the handout. No full training loop or research-data experiment is introduced yet.

## Session 13 training, validation and checkpoints

```powershell
.\.venv\Scripts\python.exe src\session13_training_lab.py --output-dir outputs/session13_v1 --epochs 5
# Optional loss comparison with a fresh output directory
.\.venv\Scripts\python.exe src\session13_training_lab.py --output-dir outputs/session13_no_dice --epochs 5 --dice-weight 0
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Trains the Session 12 SmallUNet on 16 artificial 48 x 48 images from geometry
groups 0–7 and validates on eight images from groups 8–11. Even/shaded views stay
within their source group's partition. CPU execution uses seed 13, one thread,
batch size 4, zero loader workers and Adam at learning rate 0.015.

Training combines class-weighted CE and 0.25 times mean per-image foreground soft
Dice loss. Class weights come from training labels, with foreground weight capped
at 8. Validation records unweighted pixel-mean CE, confusion counts, pooled
foreground IoU/Dice, scene-macro IoU and accuracy under eval/no_grad. Undefined
foreground overlap is serialized as null.

Selection maximizes pooled validation IoU, then minimizes validation CE. Exact
ties retain the earlier epoch. The best state is saved immediately with optimizer
state and configuration. Loading uses weights_only=True into a fresh model, and
recomputed validation metrics are verified against the selected epoch.

Each exclusive output directory contains `session13_best_state.pt`,
`session13_report.json` and `session13_training.png`. Existing directories are
protected. Optimizer state supports future continuation work. Exact RNG-state
resumption requires further implementation.

The reconstructed artificial fixture selected epoch 5 with IoU 0.8455 for the
composite loss and epoch 3 with IoU 0.7930 when Dice was disabled. These results
describe validation-selected development exercises. Audited research training
data and verified source lineage remain prerequisites for the research benchmark.
Checkpoints, outputs and personal learning notes stay excluded from Git.

## Session 14 evaluation and uncertainty

```powershell
.\.venv\Scripts\python.exe src\session14_evaluation_lab.py --checkpoint outputs/session13_final/session13_best_state.pt --output-dir outputs/session14_v1
# Separate constructed thin-feature diagnostic
.\.venv\Scripts\python.exe src\session14_evaluation_lab.py --output-dir outputs/session14_thin_demo
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Evaluates the saved Session 13 checkpoint on its four artificial validation groups
and compares Fixed 120 and all-background predictions on the same scenes. Checks
reproduction of saved foreground IoU and preservation of model/checkpoint state.
Inference uses both eval mode and inference_mode.

Confusion matrices use true rows and predicted columns. The metric functions
support validated binary and ten-class labels. Reports distinguish pooled
foreground overlap, scene-macro foreground IoU and macro-class IoU. Undefined
class overlap is null and excluded from the applicable mean. The evaluated model
uses two artificial classes.

Boundary F1 uses an inner foreground boundary and Euclidean tolerance of 1.5
pixels. Boundaries are extracted before ROI restriction, with pixels beside
excluded ROI removed from boundary scoring. A five-threshold sensitivity table
uses strict probability greater than threshold. Default 0.5 remains the reported
checkpoint setting. Group bootstrap uses 2,000 whole-group resamples, seed 14,
and percentile endpoints for scene-macro IoU.

Outputs include `session14_per_scene.csv`, `session14_report.json`,
`session14_comparison.png` and `session14_thin_diagnostic.png` under a fresh
directory. Without a checkpoint, only the constructed diagnostic is generated.
Its deleted 40-pixel line gives IoU 0.986842 and thin recall zero. The reconstructed
evaluation matches checkpoint IoU 0.845541. Fixed 120 gives perfect overlap on
these brightness-based artificial scenes.

Four generated source groups and validation-selected weights restrict the scope
of the bootstrap interval. Probability calibration requires separate evaluation
from segmentation overlap. Existing output directories are protected, and the
local checkpoints, reports and learning notes follow the repository ignore rules.

## Session 15 augmentation, normalization and robustness

```powershell
.\.venv\Scripts\python.exe src\session15_augmentation_robustness.py --output-dir outputs/session15_v1 --epochs 3
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Compares two SmallUNet runs with identical initial weights, group partitions,
loader shuffle seeds, normalization, optimizer and loss. Training groups 0–7
supply 16 images and validation groups 8–11 supply eight. Normalization statistics
come from unaugmented training images and are reused for every validation stress.

Paired transforms use quarter-turn rotations and horizontal/vertical flips.
Image-only changes use brightness/contrast factors 0.85–1.15 and Gaussian noise
sigma 0–0.03 before clipping to [0,1]. Epoch-indexed seeds reproduce each sample
transform. Original ten-class orientation semantics block geometric augmentation,
while photometric-only transforms preserve their class IDs.

Each run selects its checkpoint on nominal validation IoU and restores it before
evaluating nominal, dim (0.5 multiplier), bright (1.3 multiplier) and noisy
(sigma 0.08) conditions. Stress inputs are identical across models. The report
records both initial-state hashes and confirms checkpoint restoration.

A fresh output directory contains `session15_robustness.json`,
`session15_robustness.png`, `session15_unaugmented_best_state.pt` and
`session15_augmented_best_state.pt`. The reconstructed three-epoch exercise
selected nominal IoU 0.7568 without augmentation and 0.9411 with augmentation.
Both dim-condition scores were about 0.068. These results use four artificial
validation groups and a single seed. Real-development-data preparation and
independent acquisition-condition evaluation remain future requirements.

## Session 16 dilated convolutions and MiniDeepLabV3+

```powershell
.\.venv\Scripts\python.exe src\session16_deeplab_architecture.py --output-dir outputs/session16_v1
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Implements DepthwiseSeparableConv, MiniASPP and MiniDeepLabV3Plus with 10,642
parameters for two classes. Encoder stride-two convolutions produce eight-channel
low features and 32-channel high features. Five eight-channel ASPP branches use
a 1x1 convolution, 3x3 dilation rates 1/2/3, and global pooling. Their 40-channel
concatenation projects to 16 context channels. Four projected low-level channels
join the upsampled context before separable decoding and the class head.

Continuous features and logits use bilinear interpolation to explicit target
sizes. Outputs include [2,2,48,48], [1,2,65,67] and [1,10,48,48]. The smoke test
uses Session 13 artificial samples, seed 16, CPU and one Adam step at learning
rate .005. It checks source preservation, finite gradients and weight updates.

Fresh output directories contain `session16_report.json` with shape traces,
gradient diagnostics and code provenance, plus `session16_architecture_smoke.png`.
The written requirements were used to reconstruct the unavailable package.
This educational architecture uses a small custom encoder and simplified ASPP,
with further training and matched evaluation needed for performance comparison.
The original DeepLabV3+ combines spatial-pyramid context with decoder refinement.
[DeepLabV3+ paper](https://arxiv.org/abs/1802.02611)

## Session 17 controlled architecture comparison

```powershell
.\.venv\Scripts\python.exe src\session17_architecture_comparison.py --output-dir outputs/session17_v3 --epochs 4 --seeds 17 18
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Compares SmallUNet and MiniDeepLabV3Plus using the same Session 13 groups,
training-derived normalization and class weights. Both use four epochs, batch
size four, Adam LR .005, weighted CE plus .25 soft Dice, disabled augmentation
and one CPU thread. Each architecture receives the same seed-specific sample
order. The normalized dataset wrapper exposes its underlying source groups for
evaluation. Each selected checkpoint is loaded into a fresh architecture-specific
model and its validation metrics verified.

The report includes per-scene confusion and boundary metrics, area-fraction error,
a separate constructed thin-line probe, Fixed 120 baseline and seed summaries.
Warmed latency uses five warmups and 15 forward passes on one normalized float32
48 x 48 image. Parameter storage and training wall time are recorded separately.
Peak working memory and FLOPs require additional instrumentation.

A fresh directory contains `session17_report.json`, `session17_epochs.csv`,
`session17_comparison.png` and four best-state checkpoints. Existing directories
are protected. Completed initial results are in `outputs/session17_v2/`, and
the committed-code verification uses `outputs/session17_final/`.

Our reconstructed artificial run produced selected IoUs 0.2688/0.6540 for U-Net
and 0.7344/0.7955 for MiniDeepLab at seeds 17/18. Fixed 120 achieved 1.0. The
controlled budget exposes seed variation and thin-feature trade-offs. Further
architecture-specific tuning and independent microscopy data are needed for
broader conclusions. Generated checkpoints, reports and personal notes follow
the repository ignore rules.

## Session 18 transformer foundations

```powershell
.\.venv\Scripts\python.exe src\session18_segformer_foundations.py --output-dir outputs/session18_v2
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Integrates the supplied standalone MiniSegFormer implementation and eight tests.
The two-stage encoder uses overlapping patch embeddings, spatial-reduction
multi-head attention, residual blocks and depthwise Mix-FFN spatial mixing.
Two-scale pointwise projections fuse continuous features before logits are
resized to the input dimensions. The binary model has 19,698 parameters.

For a batch of two 48 x 48 images, stage-one tokens have shape [2,144,16],
attention [2,2,144,36], stage-two tokens [2,36,32] and logits [2,2,48,48].
Ten-class and odd-sized outputs are also checked. Seed 18 with one CPU Adam
step at LR .002 reduces same-batch CE from .833848 to .627785.

Each fresh output directory contains `session18_report.json` with shapes,
gradients, parameter changes and source hash, plus `session18_transformer_smoke.png`.
The probability panel precedes the optimizer update, while the prediction panel
follows it. Initial results are under `outputs/session18_v1/` and committed-code
verification under `outputs/session18_final/`.

The standalone artificial fixture exactly matches Session 13 group zero. This
exercise establishes architecture plumbing and gradient flow. Performance claims
require trained models, matched evaluation and audited microscopy data. The
attention heatmap visualizes numerical weights with no established physical
interpretation. Learning notes and generated artifacts follow repository ignore rules.

## Session 19 stereological sampling

```powershell
.\.venv\Scripts\python.exe src\session19_stereology_sampling_lab.py --output-dir outputs/session19_v2 --grid-step 12 --seed 19
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Integrates the supplied standalone source and eight tests. Eight artificial
128 x 128 fields support ROI-aware exact area fractions and systematic point
counts with randomized offsets. Field bootstrap uses 2,000 resamples.
Directional scans count binary interface crossings over eligible adjacent-pixel
lengths and retain foreground chords bracketed by valid background.

Seed 19 and spacing 12 produce mean exact fraction .261844 and mean point
fraction .253003. Mean absolute field error is 1.0092 percentage points.
Horizontal/vertical band scans yield 0/56 crossings and vertical complete
chords of eight pixels. The illustrative field-bootstrap interval is
[.193806,.336794]. These artificial measurements teach sampling and directional
morphology. Material inference requires representative physical sampling,
validated labels and verified calibration.

Fresh directories contain `session19_fields.csv`, `session19_report.json` and
`session19_sampling.png`. Experiments under `outputs/session19_grid6/`,
`outputs/session19_grid24/` and `outputs/session19_seed20/` vary spacing and seed.
Seed changes affect both generated fields and grid offsets. Final verification
uses `outputs/session19_final/`. Generated artifacts and learning notes follow
repository ignore rules.

## Session 20 spatial descriptors

```powershell
.\.venv\Scripts\python.exe src\session20_spatial_descriptors_lab.py --output-dir outputs/session20_v2 --seed 20 --repetitions 300 --cell-size 16
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Integrates the supplied standalone lab and eight tests. It measures ROI-aware
8-connected components, centroids, equivalent circular diameters, component
density and censoring flags. Separate random, regular and clustered fixtures
each contain 30 centers in a 15,724-pixel ROI. KD-tree nearest-neighbor summaries,
complete-cell quadrat dispersion and 300 finite-ROI Monte Carlo draws describe
each pattern. The null samples unique valid pixel centers conditional on count.

Seed 20 mean nearest distances are 11.2464, 18.0000 and 3.5004 pixels, with
cell-16 dispersion .7797, .5254 and 3.8983. The separate component mask contains
four observed regions, including two boundary-censored regions. Boundary flags
and interior-only summaries are descriptive. The nearest-neighbor edge check
uses distances to invalid pixel centers as a raster approximation.

Fresh directories contain `session20_report.json`, `session20_components.csv`,
`session20_patterns.csv` and `session20_spatial_patterns.png`. Additional runs
under `outputs/session20_cell8/`, `outputs/session20_cell32/` and
`outputs/session20_seed21/` explore cell size and fixture variability.
Committed-code verification uses `outputs/session20_final/`. Measurements use
pixel units and artificial object definitions. Physical clustering inference
requires validated instances, appropriate spatial nulls and specimen sampling.

## Session 21 specimen-level feature and property tables

```powershell
.\.venv\Scripts\python.exe src\session21_feature_property_lab.py --output-dir outputs/session21_v2 --seed 21
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Integrates the supplied source and eight tests delivered as text files. The
lab generates 54 artificial fields from 18 specimens across three batches.
Validated one-to-many joins aggregate ROI counts and component measurements
before a disjoint 12/6 specimen split. Training-only scaling precedes ridge
regression with fixed penalty 2. Augmented least squares supports rank-deficient
features even when the penalty is zero.

Seed 21 produces validation MAE 6.6817 artificial units against a 10.1592
training-mean baseline. A deliberately invalid repeated-ID memorizer achieves
zero field-row error, while six grouped holdout IDs are unseen. Simulated
responses depend on latent morphology and batch effects. Physical interpretation
requires independently measured specimen properties and audited source lineage.

Fresh output directories contain field, property, specimen and prediction CSVs,
`session21_report.json` and `session21_feature_property.png`. Initial supplied-code
results are under `outputs/session21_v1/`. Committed-code verification uses
`outputs/session21_final/`. All generated artifacts and learning notes follow
repository ignore rules.

## Session 22 measurement uncertainty and hierarchy

```powershell
.\.venv\Scripts\python.exe src\session22_uncertainty_hierarchy_lab.py --output-dir outputs/session22_v2 --seed 22 --repetitions 300 --holdout-batch B3
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Integrates the supplied retry archive, reusing Session 21's pinned seed-21
specimens and ridge implementation. Count-level binomial FP/FN perturbations
preserve ROI and specimen hierarchy. A fixed area-only ridge model isolates
fraction-error propagation. Separate specimen and batch bootstrap experiments
and leave-one-batch-out evaluation explore hierarchical dependence.

Default specimen-holdout MAEs are 10.1592, 4.5307 and 6.6817 artificial units
for train mean, area-only ridge and three-feature ridge. Holding out B3 gives
26.1190, 13.0561 and 14.3754. The two protocols use different training populations.
The plot legend follows the selected holdout batch. An assertion verifies the
exact standardized linear prediction-shift equation on all simulated draws.

Fresh directories contain perturbed field/specimen CSVs, grouped and batch
prediction CSVs, `session22_report.json` with both source hashes, and
`session22_uncertainty.png`. Initial results use `outputs/session22_v1/` and
committed-code verification uses `outputs/session22_final/`. Count perturbations
address fraction sensitivity and omit spatial error patterns. Bootstrap spans
and simulated prediction ranges require their recorded conditional interpretation.

## Session 23 model explainability

```powershell
.\.venv\Scripts\python.exe src\session23_explainability_lab.py --output-dir outputs/session23_v3 --seed 23 --repeats 200
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Reuses Session 21's generator, seed-21 specimen split and fixed ridge penalty 2.
Training coefficients, validation permutation importance, exact additive linear
contributions and a correlated-proxy experiment describe the fitted predictor.
Default permutation delta MAEs are +5.286, +.131 and -.201 artificial units for
area fraction, component density and mean NN distance. Six held-out specimens
retain ridge MAE 6.6817 against baseline 10.1592.

Each prediction is reconstructed from the training-target mean and standardized
feature contributions. These use a training-mean reference consistent with
interventional linear attribution assumptions. Correlation-aware conditional
SHAP requires a different value function. Joint proxy permutation preserves
paired feature dependence and produces larger reliance scores in this fixture.

Fresh directories contain prediction, contribution and permutation CSVs,
`session23_report.json` and `session23_explainability.png`. The completed initial
run uses `outputs/session23_v2/` and committed-code verification uses
`outputs/session23_final/`. Integration corrects Windows console encoding,
empty permutation-group validation and reconstruction-error reporting.
Generated outputs and personal notes follow repository ignore rules.

## Session 24 explanation and selection stability

```powershell
.\.venv\Scripts\python.exe src\session24_explanation_stability_lab.py --output-dir outputs/session24_v2 --seed 24 --subsamples 24 --bootstraps 200 --permutations 100
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Reconstructed the handout protocol because its source package was unavailable
locally. Reuses Session 21's specimen generator, outer split and least-squares
ridge solver, plus Session 23 grouped permutation importance. All seven nonempty
feature subsets and penalties 0/2/20 are evaluated in three inner folds using
only the 12 outer-training specimens. Scaling is refitted inside each fold.

The documented batch-stratified fold algorithm selects area fraction alone at
penalty zero, inner MAE 6.3489 and outer development MAE 4.9764. Fixed ridge
scores 6.6817 and the baseline 10.1592. Results differ from the handout example
because exact fold/subsample algorithms were unavailable. Frequencies over
24 training subsamples are 100%, 83.33% and 20.83% for fraction, density and NN.
Two hundred stratified specimen bootstrap refits hold the selected specification
fixed. Grouped density/NN permutation examines the fixed three-feature model.

Fresh outputs contain four CSVs, `session24_report.json` and
`session24_explanation_stability.png`. Final verification uses
`outputs/session24_final/`. This is inner selection with one outer development
holdout. Those specimens appeared in previous sessions, so broader independent
performance requires separate material data. Generated artifacts and personal
notes follow repository ignore rules.

## Session 25 calibration and reliability

```powershell
.\.venv\Scripts\python.exe src\session25_calibration_reliability_lab.py --output-dir outputs/session25_v2 --seed 25 --alpha 0.1 --per-batch 30
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Reconstructed the written lab because its source package was unavailable locally.
A standalone generator pools three artificial mask fields per specimen across
90 specimens and three batches. Batch-stratified partitions contain 45 fitting,
24 calibration and 21 evaluation specimens. An area-only unregularized model
learns normalization from fitting data. Split conformal uses the explicit
ceil((n+1)*(1-alpha)) order statistic and rejects unavailable finite ranks.

The default reconstruction gives radius 12.7482, rank 23 and width 25.4964
artificial units. Evaluation MAE is 5.6733 with coverage 21/21. Training-residual
calibration also covers 21/21 in this realization. B3 shift uses 30 fitting and
30 calibration specimens from B1/B2, with 30 B3 evaluation specimens. Its MAE
is 10.1899 and coverage 9/30. These generator-specific results differ from the
handout example. Marginal conformal validity requires exchangeability, while
fixed batch composition and batch shift require separate interpretation.

Fresh directories contain four CSVs, `session25_report.json` and
`session25_calibration_reliability.png`. Residuals use actual minus predicted,
with signed bias, MAE, RMSE, coverage, width and batch/prediction-bin summaries.
Final verification uses `outputs/session25_final/`. Generated artifacts and
learning notes follow repository ignore rules.

## Session 26 development provenance audit

```powershell
.\.venv\Scripts\python.exe src\session26_provenance_audit.py --output-dir outputs/session26_v2 --seed 26
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Reconstructed from the handout because the source package was unavailable
locally. Default execution generates six artificial pairs under the output
directory and audits four training/two validation entries. Every manifest row
passes partition/path/schema/calibration preflight before image content is read.
Traversal, absolute paths, protected path names, resolved escapes and reused
paths are rejected. Audit reads external inputs and writes reports under output.

External development-only mode requires both `--data-root` and `--manifest`.
The CSV columns are `pair_id`, `split`, `image_rel`, `mask_rel`, `schema`,
`texture_id`, `template_id`, `source_id`, `calibration_um_per_px`. Split accepts
`train` or `val`. Schema accepts `original10` or `derived_binary`. Empty
calibration means unavailable. Unknown lineage values remain explicitly unverified.
Use an isolated authorized development folder with stable files during audit.

The report records byte and decoded-pixel SHA-256 values for images and masks,
labels, dimensions, intensity and derived foreground-code fraction. Cross-split
image duplicates and shared declared source/texture/template IDs are blockers.
Corrupt pairs and invalid masks are blockers. Missing calibration and lineage
are warnings. An internally consistent report retains unverified source status.
Protected copies disguised under development names require provenance review.

The reconstructed fixture has train/validation means 191.0947/138.5930 intensity
and .081055/.103271 foreground-code fractions. These differ from the handout's
generator. Outputs include `session26_report.json`,
`session26_domain_diagnostics.png` and the local artificial-development fixture
with its manifest. Final verification uses `outputs/session26_final/`.
Generated fixtures, local manifests and learning notes follow output/notes ignores.
