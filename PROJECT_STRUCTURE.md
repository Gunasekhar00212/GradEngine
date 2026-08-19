# GradEngine Project Structure

This document defines the target structure of GradEngine. The project is intentionally divided into clear stages so each stage can be implemented, tested, and tracked independently.

## Core pipeline

```text
INPUT
  |
  v
PDF / IMAGE INGESTION
  |
  v
QUESTION SPLITTING
  |-----------------------------|
  v                             v
AUTOMATIC                     MANUAL
  |                             |
  |-------------+---------------|
                v
QUESTION-WISE IMAGE STORAGE
                |
                v
DOCUMENT / LAYOUT ANALYSIS
                |
       +--------+---------+
       v        v         v
      TEXT    DIAGRAM   EQUATION
       |        |         |
       v        v         v
      OCR    EXTRACT    LATEX OCR
       |        |         |
       +--------+---------+
                v
         ANSWER JSON
                |
TEACHER RUBRIC -> RUBRIC JSON
                |
                v
         LLM EVALUATION
                |
       +--------+---------+
       v        v         v
     MARKS   FEEDBACK   METRICS
                |
                v
      EVIDENCE-BASED FEATURES
                |
                v
        ML RELIABILITY MODEL
           |            |
           v            v
       RELIABLE     NEEDS REVIEW
                         |
                         v
                   TEACHER REVIEW
                         |
                         v
                 FINAL TEACHER MARK
                         |
                         v
                 TRAINING DATA
```

## Target directory layout

```text
GradEngine/
├── frontend/                         # React application
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── hooks/
│   │   └── utils/
│   └── package.json
│
├── backend/
│   ├── app/
│   │   ├── api/                     # FastAPI routes/controllers
│   │   ├── core/                    # configuration and shared state
│   │   ├── schemas/                 # request/response Pydantic schemas
│   │   ├── models/                  # persistence/domain models
│   │   ├── services/
│   │   │   ├── ingestion/           # input validation and file handling
│   │   │   ├── preprocessing/       # PDF/image preparation
│   │   │   ├── splitting/           # automatic and manual splitting
│   │   │   ├── layout/              # text/diagram/equation region analysis
│   │   │   ├── extraction/           # OCR, diagram, equation extraction
│   │   │   ├── representation/       # Answer JSON and Rubric JSON
│   │   │   ├── evaluation/           # LLM evaluation
│   │   │   ├── reliability/          # ML reliability prediction
│   │   │   ├── review/               # teacher review and overrides
│   │   │   └── reporting/            # final evaluation reports
│   │   └── main.py
│   ├── tests/
│   └── requirements.txt
│
├── data/
│   ├── input/                       # original uploaded files
│   ├── pages/                       # rendered page images
│   ├── question_crops/              # question-wise images
│   ├── text/                        # OCR outputs
│   ├── diagrams/                    # cropped diagrams and metadata
│   ├── equations/                   # equation crops and LaTeX
│   ├── answer_json/                 # structured student answers
│   ├── rubric_json/                 # structured rubrics
│   ├── evaluations/                 # LLM evaluation outputs
│   ├── reliability/                 # ML predictions and feature rows
│   └── reports/                     # final reports
│
├── models/
│   ├── reliability/                 # trained ML model, scaler, metadata
│   └── checkpoints/                 # optional future CV/segmentation models
│
├── scripts/
├── notebooks/
├── docs/
├── tests/
├── .env.example
├── .gitignore
└── README.md
```

## Implementation status

Every pipeline stage should expose a clear status such as:

- `not_started`
- `scaffolded`
- `implemented`
- `tested`
- `integrated`

Do not mark a stage as completed just because a placeholder exists.

## Important rule

Do not rewrite or delete working behavior just to match this structure. Existing code should be moved or wrapped carefully. Keep compatibility where possible, then update imports and tests. Advanced modules may start as interfaces/stubs until their real implementation is added.
