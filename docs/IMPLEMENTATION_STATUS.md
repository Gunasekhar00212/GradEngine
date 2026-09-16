# Implementation Status

| Module | Status | Notes |
|--------|--------|-------|
| Input | IMPLEMENTED | Upload flow exists in the FastAPI routes and now stores files under session folders. |
| PDF processing | IMPLEMENTED | Real PDF-to-image conversion exists with a placeholder fallback when rendering fails. |
| Automatic splitting | PARTIAL | Heuristic dark-row splitting is implemented; no learned detector yet. |
| Manual splitting | IMPLEMENTED | Teacher clicks are stored and converted into simple page ranges. |
| Question image storage | IMPLEMENTED | Question crops are saved per session and kept separate from page images. |
| Layout analysis | PARTIAL | Current logic only approximates text, diagram, and equation regions. |
| OCR | PARTIAL | OCR wrapper exists, but it is still engine-dependent and fallback-heavy. |
| Diagram extraction | PARTIAL | Current behavior labels diagrams without full semantic understanding. |
| Equation extraction | IMPLEMENTED | Pix2TeX converts saved equation-region crops to LaTeX. |
| Answer JSON | PARTIAL | Structured builder added, but not every downstream consumer uses it yet. |
| Rubric JSON | PARTIAL | Rubric parsing exists; normalized rubric JSON builder added. |
| Gemini evaluation | PARTIAL / MOCK | The current evaluator is heuristic, not a live Gemini integration. |
| Evidence-based features | NOT_STARTED | Feature builder and calculation definitions are still missing. |
| ML reliability model | NOT_STARTED | Repository and service scaffolding exist; labelled training data is not available yet. |
| Teacher review | PARTIAL | Review persistence scaffold exists; teacher UI/override workflow is still incomplete. |
| Final grading metrics | NOT_STARTED | Need a real teacher-mark dataset before reporting grading metrics. |
| MongoDB integration | PARTIAL | Repository layer now exists with local fallback; live deployment depends on environment configuration. |
| Local file storage | IMPLEMENTED | Storage abstraction and local backend are in place. |
