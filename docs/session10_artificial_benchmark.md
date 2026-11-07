# Session 10 artificial benchmark policy v1

Scope: fixed-configuration development/validation exercise, not the future
research-data final benchmark in evaluation_protocol_v1.md. Primary reporting
here is scene-macro foreground IoU, with pooled IoU explicitly supplementary.
This policy does not silently change the future dataset-level protocol.

Source groups: compact, elongated and crowded are development; particles and
bands are validation. Even/shaded variants stay together. Groups are verified
only by our artificial generator. All share one generation procedure and
intensity convention; this does not establish independence across real specimens,
textures, instruments or imaging protocols. No official-test pixels are read.

Each of seven pipelines receives the same eligible image, truth, ROI and binary
semantics. Configurations are fixed once: threshold 120; Otsu; adaptive mean or
Gaussian (41, C=6); Gaussian (5x5, sigma=1)+Otsu; Otsu+3x3 ellipse opening or
closing. There is one configuration per candidate and no iterative search.

Metric rows preserve TP/FP/FN/TN, IoU, Dice, precision, recall, signed and absolute
fraction error in percentage points, true/predicted 8-component counts, signed
and absolute count error and annotated thin-pixel recall. All use valid ROI.
ROI clipping may change connectivity; counts describe observed eligible binary
regions, not verified particles or unbiased material number density.

Both-empty foreground overlap and undefined precision/recall are null. Thin
recall is null when no valid thin annotation exists. Scene-macro summaries
exclude undefined values. Pooled foreground IoU uses summed TP/(TP+FP+FN).
Worst scenes and per-scene results must accompany aggregate rankings. Foreground
IoU weights overlap unions; pooled IoU weights larger foreground unions more
than a simple equal-scene average.

Illustrative selection: validation scene-macro foreground IoU descending, then
mean absolute fraction error ascending, then method name. This is a learning
exercise using validation feedback, not a held-out test result. Metrics other
than ranking IoU can reveal a winning method is unsuitable for particle counts
or fine features. Runtime, hardware throughput and repeated-training variation
are not measured by this nonlearning fixture experiment.

Output directories are exclusive; fresh directories identify explicit revisions.
JSON records fixed settings, source-group roles, versions, code hash/commit and
working-tree state. CSV rows support alternate aggregation. Generated artifacts
and learning notes remain outside Git. The real-data readiness check uses the
strict Session 5 grouped metadata rules; current test-only data must fail.

Real benchmark prerequisites remain: audit official training pairs, verify label
handling, establish a documented development split, and obtain source-lineage
evidence or explicitly narrow image-level claims if lineage is unavailable.
Reserve the official test for selected frozen pipelines; report synthetic and
real-micrograph performance separately. This session does not resolve that data dependency.
