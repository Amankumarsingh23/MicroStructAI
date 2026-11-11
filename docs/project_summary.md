# MicroStructAI project narrative

MicroStructAI is a Python/PyTorch computational-metallography learning project.
Its 30-session implementation connects image arrays and preprocessing, classical
segmentation, educational neural architectures, quantitative measurements,
specimen-level prediction and reproducibility diagnostics.

The repository implements a small U-Net, a MiniDeepLabV3+-style model and a
two-stage SegFormer-inspired model. Architecture smoke tests verify shapes,
gradients and optimizer updates. Controlled synthetic exercises investigate
training variability, boundary accuracy, thin-feature preservation and CPU cost.

Measurement modules cover ROI-aware fractions, connected components, stereology,
spatial descriptors and grouped uncertainty. Property-model exercises demonstrate
specimen aggregation, training-only preprocessing, leakage, confounding,
explainability, feature-selection stability and conformal calibration.

The final release auditor independently recomputes the saved synthetic pipeline's
image/mask contracts, segmentation confusion counts, feature pooling, component
densities, model fit, calibration radius and property predictions. It verifies
540 input/output hashes and exports a sealed artifact manifest and claims matrix.

The seed-29 pipeline uses 36 artificial specimens with 108 fields. Its evaluated
area-only predictor has MAE 4.202506 artificial units and observed interval
coverage 9/9. Threshold-120 segmentation achieves pooled IoU 1.0 on the easy
fixture, while threshold 165 scores .933068. These results describe generated
images and simulated responses. The final independent audit validates saved
record consistency and reproducibility.

Real-material readiness remains NO-GO. Texture-source independence, independently
annotated real microscopy, verified physical calibration, specimen-matched
measured properties and independent material validation remain required.
Original Figshare IDs 0..8 encode image-plane pearlite lamellar directions and
9 ferrite. Binary conversion maps 9 to zero and 0..8 to one. These labels require
careful interpretation before downstream material measurements.

A concise interview description is “Built a tested computational-metallography
pipeline covering segmentation architectures, image measurements, specimen-level
regression and reliability diagnostics, with synthetic benchmarks, source-aware
safeguards and end-to-end traceability. Real-material claims await independent
experimental validation.”

The practical learning track is implemented. Confirmed external theory grading
remains 1080/1100 through Session 22. Later written answers require review.
