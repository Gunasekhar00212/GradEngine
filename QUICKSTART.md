## GradEngine Prototype — Quick Start

The FastAPI API runs at `http://127.0.0.1:8000`. The React app runs at
`http://127.0.0.1:5173` during development.

### Access Points

- **React UI**: http://127.0.0.1:5173
- **API Docs (Swagger)**: http://127.0.0.1:8000/docs
- **Health Check**: `curl http://127.0.0.1:8000/api/health`

### Sample Workflow

1. Start the API with `python main.py`.
2. In another terminal, run `cd frontend && npm install && npm run dev`.
3. Open the React UI in your browser.
4. Upload:
   - A student PDF (or use a test PDF)
   - A rubric JSON (example: `data/rubric/expanded_rubric.json`)
5. Click each question boundary on the full-page splitter.
6. Choose **Finish and grade** to split, extract text, and evaluate.
7. View the score screen.

### Example Rubric

See `data/rubric/expanded_rubric.json`:
- Question about photosynthesis
- Rubric concepts: sunlight, CO2, water, oxygen, glucose
- Each concept worth 1 mark

### Example Output

See `data/reports/sample_evaluation.json`:
- Structured JSON with student answer, rubric, and evaluation
- Includes confidence scores and human-review flag

### Debug/API Calls

To see all sample data:
```bash
curl http://127.0.0.1:8000/api/sample-output
```

To list sessions:
```bash
curl http://127.0.0.1:8000/api/sessions
```

### To Stop the Server

In the terminal where it's running, press `Ctrl+C`.

### Next Steps

- Configure `GEMINI_API_KEY` before extraction; Gemini OCR is used for handwritten text
- Add manual annotation UI for boundary drawing
- Integrate real LaTeX equation detection
- Add persistence layer (database)
- Build teacher review interface
