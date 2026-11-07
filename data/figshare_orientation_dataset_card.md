# Ferrite–pearlite geometric orientation dataset — version 2

## Source and attribution

Nikhil Chaurasia, Shikhar Krishn Jha and Sandeep Sangal.
*Synthetic Dataset of Ferrite-Pearlite Microstructures with specified Geometric
Orientations of lamellae in individual Pearlite Colonies.*

- Pinned DOI: https://doi.org/10.6084/m9.figshare.30175549.v2
- Pinned metadata: https://api.figshare.com/v2/articles/30175549/versions/2
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12907002/
- Figshare license checked 2026-11-07: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- No research images or masks are redistributed in this Git repository.

## Label schema

| Original ID | Meaning |
| --- | --- |
| 0 | Pearlite orientation 0–20 degrees |
| 1 | Pearlite orientation 20–40 degrees |
| 2 | Pearlite orientation 40–60 degrees |
| 3 | Pearlite orientation 60–80 degrees |
| 4 | Pearlite orientation 80–100 degrees |
| 5 | Pearlite orientation 100–120 degrees |
| 6 | Pearlite orientation 120–140 degrees |
| 7 | Pearlite orientation 140–160 degrees |
| 8 | Pearlite orientation 160–180 degrees |
| 9 | Ferrite |

ID 0 is not background. The orientation bins describe lamellar geometry in the
image frame, not a measured 3D crystallographic orientation. Original masks must
remain unchanged. A future derived binary convention can use 0=ferrite (original
9), 1=pearlite constituent (original 0–8); this session performs no relabeling.

## Archive inspection and local scope

| File | Figshare file ID | Bytes | Metadata MD5 |
| --- | --- | --- | --- |
| Training.zip | 58127689 | 2268396510 | ecd81aa1bba635b512a4be1c83004a3e |
| Testing.zip | 58127713 | 115385806 | dc00ed7011b2f097b706d8f7ef7f3f26 |

Training.zip was not downloaded or counted. Figshare describes 10,000 training
pairs and 500 testing pairs; the paper reports 499 test pairs. The downloaded
Testing.zip actually contains **499 images and 499 masks**, paired by filename.
Its byte count and MD5 match Figshare metadata. Its SHA-256 is
`35f0a0d3a64759dcad2133a701eaba2ae5fdf7a393724b17b08efb23de99ca95`.

Archive paths map without changing bytes or partition membership:

- `Testing/Synthetic Micrographs/<name>.png` → `test/images/<name>.png`
- `Testing/Ground Truth/<name>.png` → `test/masks/<name>.png`

The deterministic first three numeric filenames, `1.png`, `2.png`, `3.png`,
were copied to `data/raw/figshare_orientation/test/` and passed the validator.
Each image and mask is 512 x 512, mode L, uint8; all IDs 0–9 occur in each mask.
Normalized train folders intentionally remain empty. This is a test-only subset
for engineering validation, not a complete training dataset or benchmark audit.

## Reproduce

```powershell
.\.venv\Scripts\python.exe src\session03_prepare_figshare.py
.\.venv\Scripts\python.exe src\session03_dataset_audit.py --data-root data/raw/figshare_orientation --manifest outputs/session03_real_manifest.csv --source-dataset 10.6084/m9.figshare.30175549.v2
```

The preparer saves version-pinned metadata, downloads only Testing.zip if absent,
checks archive size/MD5, and extracts three explicit image/mask pairs. Existing
modified files are not replaced. `--pairs N` can expand the subset up to 499;
shrinking an existing subset is rejected to avoid silently mixing selections.

Local provenance: `data/metadata/figshare-30175549-v2.json` and
`data/metadata/session03_real_subset.json`, including archive members and SHA-256
digests. Manifest paths are relative to the data root supplied to the auditor.

## Scientific limits

Synthetic images are not newly acquired independent specimens. Per-file specimen
and texture-source identities are unknown; verified physical calibration is
absent. These fields remain UNKNOWN or blank. Original test membership is frozen;
there is no new random split or claim of specimen independence.

The audit checks syntax, dimensions, numerical label validity and byte-identical
image leakage among the files it receives. It does not establish semantic label
correctness, image/mask registration, visual similarity, source-texture separation,
or generalization to real micrographs. No full train/test duplicate check is
possible with this test-only subset. Numerical validity is not scientific truth.
