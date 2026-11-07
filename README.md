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
