# Pipeline Contract

## Input

Input:
Student answer sheet, rubric file, and split mode.

Output:
Evaluation/session ID and stored file references.

Technology:
FastAPI, UploadFile, pathlib, Pydantic, local storage, MongoDB repository.

Current status:
IMPLEMENTED

Dependencies:
FastAPI routes, storage layer, evaluation repository.

## PDF / Image Processing

Input:
Uploaded answer sheet PDF or page image.

Output:
Rendered page images under session folders.

Technology:
pdf2image, Pillow, placeholder page generation.

Current status:
IMPLEMENTED

Dependencies:
Input storage, page output directory.

## Question Splitting

Input:
Rendered page images and manual click annotations when available.

Output:
Question-wise crop images.

Technology:
Heuristic boundary detection, manual click replay, Pillow cropping.

Current status:
PARTIAL / IMPLEMENTED

Dependencies:
Page images, annotation file, question crop directory.

## Question Image Storage

Input:
Question crops.

Output:
Persistent per-session question image files.

Technology:
Local storage.

Current status:
IMPLEMENTED

Dependencies:
Splitting stage.

## Layout / Content Analysis

Input:
Question crop image.

Output:
Text, diagram, and equation regions.

Technology:
Heuristic layout analysis.

Current status:
PARTIAL

Dependencies:
Question image storage.

## OCR / Text Extraction

Input:
Saved full-question text crop.

Output:
Structured transcription, OCR confidence, and unreadable-word hints.

Technology:
Google Gemini Vision OCR (`google-genai`).

Current status:
IMPLEMENTED (requires `GEMINI_API_KEY`)

Dependencies:
Layout analysis, image files.

## Diagram Extraction

Input:
Diagram regions.

Output:
Diagram crops and label metadata.

Technology:
Deferred until a real diagram detector is introduced.

Current status:
DEFERRED

Dependencies:
Layout analysis.

## Equation Extraction

Input:
Equation regions.

Output:
Equation image references and LaTeX strings.

Technology:
Deferred until a real equation detector is introduced.

Current status:
DEFERRED

Dependencies:
Layout analysis.

## Answer JSON

Input:
Extracted text, diagram metadata, equation metadata, quality data.

Output:
Clean structured Answer JSON.

Technology:
Pydantic-friendly dictionary builders.

Current status:
PARTIAL

Dependencies:
OCR, diagram extraction, equation extraction.

## Rubric JSON

Input:
Teacher rubric file.

Output:
Normalized Rubric JSON.

Technology:
Rubric parser and structured builder.

Current status:
PARTIAL

Dependencies:
Input upload and rubric parser.

## LLM Evaluation

Input:
Question, Answer JSON, Rubric JSON, diagrams, equations.

Output:
Structured marks, feedback, and evaluation metrics.

Technology:
Current heuristic evaluator with mock/placeholder semantics.

Current status:
PARTIAL / MOCK

Dependencies:
Answer JSON, Rubric JSON.

## Evidence-Based Features

Input:
OCR quality, image quality, rubric coverage, semantic alignment, and related outputs.

Output:
Feature vector for reliability analysis.

Technology:
Feature builder to be implemented.

Current status:
NOT STARTED

Dependencies:
All upstream pipeline evidence.

## ML Reliability Model

Input:
Evidence-based features and historical teacher labels.

Output:
Reliable or Needs Review.

Technology:
Repository-backed model interface, future classifier.

Current status:
NOT STARTED

Dependencies:
Teacher-reviewed training data.

## Teacher Review

Input:
AI evaluation, extracted evidence, and reliability prediction.

Output:
Teacher override, approved marks, final mark.

Technology:
Review repository and future UI workflow.

Current status:
PARTIAL

Dependencies:
Evaluation results and reliability prediction.

## Final Evaluation

Input:
Teacher final marks and AI outputs.

Output:
Grading metrics and reliability model metrics.

Technology:
Reporting module.

Current status:
NOT STARTED

Dependencies:
Teacher-reviewed evaluation dataset.
