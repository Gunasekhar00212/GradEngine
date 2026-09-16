# Roadmap

## PHASE 1 - STRUCTURE

- [x] Add pipeline stage folders under the backend service tree.
- [x] Add storage abstraction for local files.
- [x] Add repository layer for MongoDB-backed metadata.
- [x] Move OCR, equation OCR, diagram-label OCR, and layout helpers into stage-specific modules.

## PHASE 2 - INPUT + SPLITTING

- [x] Preserve upload flow.
- [x] Store files by evaluation/session ID.
- [x] Keep manual splitting support.
- [ ] Add question-paper and rubric upload variants if the UI needs them.
- [ ] Improve automatic split detection beyond heuristics.

## PHASE 3 - OCR + CONTENT EXTRACTION

- [x] Use Gemini Vision OCR for handwritten text regions.
- [x] Use Pix2TeX for equation-region LaTeX extraction.
- [x] Use OpenCV and Gemini OCR for diagram labels.
- [x] Persist extractor status and errors in answer quality metadata.

## PHASE 4 - ANSWER/RUBRIC JSON

- [x] Add answer JSON builder.
- [x] Add rubric JSON builder.
- [ ] Route extracted content through the new structured builders everywhere.

## PHASE 5 - GEMINI EVALUATION

- [x] Keep the evaluator interface in place.
- [ ] Replace heuristic grading with a real Gemini integration.
- [ ] Return strictly structured evaluation JSON.

## PHASE 6 - FEATURE ENGINEERING

- [ ] Define every evidence-based feature and its range.
- [ ] Build feature vectors from real pipeline outputs.
- [ ] Store feature snapshots with each evaluation.

## PHASE 7 - ML RELIABILITY MODEL

- [x] Add reliability service scaffolding.
- [ ] Collect teacher-reviewed labels.
- [ ] Train and compare baseline classifiers.
- [ ] Persist model version and predictions.

## PHASE 8 - TEACHER REVIEW

- [x] Add review repository scaffolding.
- [ ] Build teacher review UI.
- [ ] Let teachers override marks and mark evaluations as reliable or needs review.
- [ ] Save final marks as ground truth.

## PHASE 9 - FINAL EVALUATION

- [ ] Add grading performance metrics.
- [ ] Add reliability model precision/recall/F1/ROC-AUC reporting.
- [ ] Separate grading quality from classifier quality.

## PHASE 10 - PRODUCTION POLISH

- [ ] Add tests for the full upload/split/extract/grade path.
- [ ] Document MongoDB and environment setup.
- [ ] Replace placeholder UI labels with honest status indicators.
- [ ] Add migration notes for legacy scripts and helpers.
