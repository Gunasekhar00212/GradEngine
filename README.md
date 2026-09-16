
# GradEngine Prototype

GradEngine is a teacher-first handwritten exam evaluation prototype.

It is built to reduce grading time while keeping the teacher in control. The current codebase is a working scaffold, not a finished product. It supports PDF upload, page conversion, auto or manual question splitting, placeholder layout analysis, OCR hooks, rubric parsing, semantic grading, and a human-review flag.

## What this prototype does now

- Upload a student PDF and a rubric JSON file
- Convert PDF pages into page images
- Split pages into question crops with auto or manual mode
- Store manual click annotations as JSON
- Create a persistent full-question text crop for each split answer
- Transcribe text regions with Google Gemini Vision
- Grade saved Answer JSON against Rubric JSON with Gemini
- Format grading output into a stable JSON schema
- Mark low-confidence results for human review
- Show the flow in a simple browser UI

## What this prototype does not do yet

- It does not claim production-grade handwriting accuracy
- It does not do real diagram understanding
- It does not do real math OCR yet
- It does not use an LLM for preprocessing or splitting
- It does not include authentication, persistence, or multi-user workflows

## Project structure

```text
GradEngine/
├── backend/
│   └── app/
│       ├── api/
│       ├── core/
│       ├── models/
│       ├── services/
│       └── utils/
├── data/
│   ├── uploads/
│   ├── pages/
│   ├── crops/
│   ├── extracted/
│   └── outputs/
├── scripts/
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

## Core flow

1. Teacher uploads a student PDF and rubric JSON.
2. The PDF is converted into page images.
3. Pages are split automatically or using teacher click annotations.
4. Each question crop goes through layout analysis.
5. Text regions go through OCR.
6. Gemini returns structured OCR data, which is saved as Answer JSON.
7. The Answer JSON and Rubric JSON are sent to Gemini for structured evaluation.
8. The rubric is normalized into a grading-friendly JSON form.
9. Gemini evaluates the Answer JSON against the Rubric JSON and returns marks, feedback, and confidence.
10. A reliability layer decides whether human review is needed.

## JSON output shape

The prototype writes output in a structure like this:

```json
{
	"question_id": "Q1",
	"student_answer": {
		"text": "...",
		"diagrams": [
			{
				"image_path": "...",
				"labels": []
			}
		],
		"equations": [
			{
				"image_path": "...",
				"latex": "..."
			}
		]
	},
	"rubric": {
		"expected_concepts": [],
		"total_marks": 10
	},
	"evaluation": {
		"marks": 0,
		"feedback": "",
		"semantic_alignment": 0.0,
		"rubric_coverage": 0.0,
		"answer_completeness": 0.0,
		"evaluation_confidence": 0.0,
		"needs_human_review": false
	}
}
```

## Sample data and example output

- Example rubric: [data/rubric/expanded_rubric.json](data/rubric/expanded_rubric.json)
- Example extracted session: [data/extracted/sample_session.json](data/extracted/sample_session.json)
- Example final output: [data/reports/sample_evaluation.json](data/reports/sample_evaluation.json)

## Run locally

1. Install dependencies.

```bash
pip install -r requirements.txt
```

Set the Gemini credential before starting the API:

```bash
export GEMINI_API_KEY="your-key"
```

`GEMINI_OCR_MODEL` and `GEMINI_EVALUATION_MODEL` are optional and default to
`gemini-2.5-flash`.

2. Start the app.

```bash
python main.py
```

3. Open the UI at `http://127.0.0.1:8000`.

You can also run with Uvicorn directly:

```bash
uvicorn backend.app.main:app --reload
```

## API endpoints

- `GET /` loads the simple browser UI
- `GET /api/health` checks that the API is alive
- `POST /api/upload` uploads the student PDF and rubric JSON
- `GET /api/sessions` lists active in-memory sessions
- `POST /api/sessions/{session_id}/split` creates question crops
- `POST /api/sessions/{session_id}/extract` runs OCR and region extraction
- `POST /api/sessions/{session_id}/grade` creates the evaluation JSON
- `GET /api/sessions/{session_id}/output` returns saved output
- `GET /api/sample-output` returns a ready-made sample result

## Manual split workflow

The manual mode is intentionally simple.

1. Upload the PDF and rubric.
2. Store teacher click annotations as JSON.
3. Save clicks with `page_index`, `x`, and `y` values.
4. Reuse those clicks to create question ranges.
5. Crop each question into its own image.

This is a prototype path for teacher-guided splitting, not a polished annotation product.

## Automatic split workflow

The automatic splitter uses simple image heuristics.

- It looks for dense dark rows
- It uses whitespace gaps as rough boundaries
- It keeps the logic simple so it can be replaced later

This is a starting point, not a research-grade splitter.

## LLM usage

The LLM layer is only used for semantic grading tasks such as:

- semantic comparison
- partial marking
- feedback generation
- confidence estimation

The LLM is not used for:

- PDF splitting
- page cropping
- OCR preprocessing
- diagram detection
- equation region detection

## Future work

- Better question boundary detection
- Better manual annotation UI
- Add genuine diagram and equation detection before enabling their extractors
- Multi-page answer continuation handling
- Teacher review and override screens
- Persistent storage and user accounts
- Batch grading for many students

## Honesty note

This repository is an active prototype scaffold. It is meant to be extended step by step. It should be treated as a real starting point, not as a finished grading system.
