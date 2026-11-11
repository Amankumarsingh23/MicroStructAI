# MicroStructAI research roadmap

## Establish authorized development data

Record dataset DOI/version, license, archive checksums and permitted use. Build
a development-only manifest with original mask semantics, image/mask dimensions,
source texture, generation template, transformations, specimen/section lineage,
ROI definitions and provenance evidence. Keep unresolved origins explicit.
Separate all dependent sources before model development. Preserve protected
final evaluation data under a documented access policy.

## Validate material measurements

Obtain representative independently acquired steel micrographs with expert
annotations and traceable specimen identities. Verify spatial calibration for
each coordinate system. Assess annotation disagreements, boundary validity,
small-feature identities, preparation effects and ROI exclusions. Pearlite
constituent fraction and cementite phase fraction require distinct definitions.
Connect properties only to independently measured, specimen-matched targets,
with composition, heat treatment, batch and measurement uncertainty recorded.

## Freeze a development benchmark

Specify label conversions, grouping, metrics, absent-class rules, boundary and
thin-feature annotations, aggregation and uncertainty units. Compare classical
baselines with U-Net, DeepLabV3+ and SegFormer using architecture-appropriate
tuning budgets and repeated training seeds. Perform preprocessing and selection
inside development folds. Record checkpoint selection, environment, data hashes,
source revisions, computational cost and protocol timing evidence.

## Evaluate transfer and reliability

Measure class overlap, boundary F1, feature recall, false positives, signed
fraction error and physically meaningful measurement errors. Use matched images
and verified independent source groups. Assess processing and acquisition shift,
subgroup residuals, calibration coverage/width, explanation stability and
predictor uncertainty. Use distinct fitting, calibration and evaluation cohorts
with a justified sampling design. Report uncertainty according to actual
specimen/batch dependence and available sample size.

## Require independent evidence before research claims

Remove a NO-GO criterion only when its corresponding evidence is reviewed.
A source audit can establish documented partition validity. Physical annotations
and calibration can support material measurements. Representative independent
specimens and measured targets can support generalization and property claims.
Tests and hashes provide software evidence for each stage. Their success alone
cannot certify material accuracy, provenance or causal mechanisms.

Freeze the final protocol before protected evaluation. Preserve failures,
excluded cases, uncertainty assumptions and reproducibility artifacts in the
research record. Scope conclusions to the actual sampled materials and conditions.
