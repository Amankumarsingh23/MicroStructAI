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
