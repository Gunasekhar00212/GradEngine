## GradEngine Prototype — Quick Start

The FastAPI server is running at `http://127.0.0.1:8000`

### Access Points

- **UI Dashboard**: http://127.0.0.1:8000
- **API Docs (Swagger)**: http://127.0.0.1:8000/docs
- **Health Check**: `curl http://127.0.0.1:8000/api/health`

### Sample Workflow

1. Open the dashboard in your browser.
2. Upload:
   - A student PDF (or use a test PDF)
   - A rubric JSON (example: `data/rubric/expanded_rubric.json`)
3. Choose split mode: **auto** or **manual**.
4. Click **Upload files** to start the pipeline.
5. The dashboard will run through:
   - Split pages into questions
   - Extract text and regions
   - Generate evaluation JSON
6. View the results in real-time.

### Example Rubric

See `data/rubric/expanded_rubric.json`:
- Question about photosynthesis
- Rubric concepts: sunlight, CO2, water, oxygen, glucose
- Each concept worth 1 mark

### Example Output

See `data/outputs/sample_evaluation.json`:
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

- Replace placeholder OCR with real handwriting recognition
- Add manual annotation UI for boundary drawing
- Integrate real LaTeX equation detection
- Add persistence layer (database)
- Build teacher review interface
