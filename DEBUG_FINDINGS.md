# GradEngine Debug Findings & Phase 1 Fix

## The Root Cause ✓

**Your OCR and Gemini are working correctly.** The problem is in **question segmentation**.

### Evidence from Debug Logs

```
Q1 OCR: "UNIT-II / 3. SHTTP and SSL protocols"          ✓ CORRECT
Q2 OCR: "UNIT-III / 5. Need of Security..."             ✓ CORRECT  
Q3 OCR: "a key. / The receiver will have another key..." ❌ WRONG (starts mid-sentence)
```

### Why Q3 is Wrong

The teacher's manual click for Q3 was placed **too late on page 7**:
- Teacher clicked at: `page_index=6, y=1692`
- Actual Q3 starts before `y=1692`
- Everything before that click got assigned to Q2
- Result: Q3 crop begins mid-sentence

This is a **manual split boundary problem**, not a grading, OCR, or Gemini problem.

---

## Phase 1 Fix: Implemented ✓

### New Backend Endpoints

#### 1. Get Split Preview
**Endpoint:** `GET /api/sessions/{session_id}/split-preview`

**Returns:**
```json
{
  "session_id": "evaluation_cb31951c92",
  "validation": {
    "valid": true,
    "errors": [],
    "warnings": []
  },
  "previews": [
    {
      "question_id": "Q1",
      "crop_path": "/api/sessions/.../crop/Q1",
      "ocr_start": "UNIT-II\n3. SHTTP and SSL protocols...",
      "looks_like_continuation": false,
      "warning": null
    },
    {
      "question_id": "Q3",
      "crop_path": "/api/sessions/.../crop/Q3", 
      "ocr_start": "a key.\n* The receiver will have another...",
      "looks_like_continuation": true,
      "warning": "⚠️ Possible split error: text appears to start mid-sentence"
    }
  ]
}
```

#### 2. Adjust Boundaries & Re-crop
**Endpoint:** `POST /api/sessions/{session_id}/adjust-boundary`

**Request:**
```json
{
  "boundaries": [
    {"start_page": 1, "start_y": 1848, "end_page": 4, "end_y": 1200},
    {"start_page": 4, "start_y": 1200, "end_page": 6, "end_y": 900},
    {"start_page": 6, "start_y": 900, "end_page": 7, "end_y": null}
  ]
}
```

This re-crops all questions with the adjusted Y coordinates.

#### 3. View Crop Image
**Endpoint:** `GET /api/sessions/{session_id}/crop/{question_id}`

Return the crop image for display in the UI.

### New SplitService Methods

**`validate_boundaries(groups, num_pages)`**
- Checks for out-of-range page indices
- Detects reversed ranges (start > end)
- Returns: `{"valid": bool, "errors": [...], "warnings": [...]}`

**`detect_likely_continuation(text)`**
- Heuristic detection that OCR text starts mid-sentence
- Checks for lowercase first character
- Checks for continuation phrases: "a ", "the ", "and ", "because", "which", etc.
- Returns: `bool`

---

## Current Problems & Solutions

### Problem 1: Single Image Per Question
**Current:**
```python
source_paths={"question_image": "/path/to/merged_q3.png"}
```

**Why it's a problem:**
- Q3 spans pages 7-8 (merged into one PNG)
- If boundary is wrong, entire Q3 is wrong
- Can't fix by changing Gemini prompt

**Phase 2 Solution:**
```python
source_paths={
  "question_images": [
    "/path/to/page_7_crop.png",
    "/path/to/page_8_crop.png"
  ]
}
```

Then `LlmEvaluationService` sends both images to Gemini.

### Problem 2: No Visual Feedback Before Grading
**Current flow:**
```
Upload → Split (blindly trust teacher clicks) → Extract → Grade
```

**Phase 1 Solution:**
```
Upload → Split → PREVIEW (show crops) → Adjust if needed → Extract → Grade
```

### Problem 3: No Auto-Detection of Split Errors
**Current:** Errors only appear at grading time (too late)
**Phase 1 Solution:** Warnings appear immediately in preview

---

## How to Test Phase 1

### Test Case: Your Current PDF

1. **Upload and split** as before (pages 1-8, manual mode)
2. **Call the new preview endpoint:**
   ```bash
   curl http://localhost:8000/api/sessions/evaluation_cb31951c92/split-preview | jq
   ```

3. **Observe:**
   - Q3 shows `"looks_like_continuation": true`
   - Q3 shows `"warning": "⚠️ Possible split error..."`
   - Q3 `ocr_start` starts with "a key."

4. **View the crop image:**
   ```bash
   # For testing with curl
   curl http://localhost:8000/api/sessions/evaluation_cb31951c92/crop/Q3 > q3_current.png
   ```

5. **Call adjust-boundary** with fixed Y coordinate:
   ```bash
   # Move Q3 start earlier (before "a key.")
   curl -X POST http://localhost:8000/api/sessions/evaluation_cb31951c92/adjust-boundary \
     -H "Content-Type: application/json" \
     -d '{
       "boundaries": [
         {"start_page": 1, "start_y": 1848, "end_page": 4, "end_y": 1476},
         {"start_page": 4, "start_y": 1476, "end_page": 6, "end_y": 1692},
         {"start_page": 6, "start_y": 1500, "end_page": 7, "end_y": null}
       ]
     }' | jq
   ```

6. **View preview again:**
   ```bash
   curl http://localhost:8000/api/sessions/evaluation_cb31951c92/split-preview | jq '.previews[2]'
   ```

   Should now show:
   - `"looks_like_continuation": false`
   - `"warning": null`
   - `"ocr_start"` should start with actual Q3 content

7. **Extract and grade** - should be correct now

---

## Implementation Checklist

- [x] Added 3 new backend endpoints
- [x] Added validation and diagnostic methods in SplitService
- [x] Added logging to split/extract/grade phases
- [x] Code compiles without errors

**Next steps (for you to implement in frontend):**
- [ ] Call `/split-preview` after split completes
- [ ] Display crop previews with OCR snippets
- [ ] Show warnings for continuation-detected crops
- [ ] Provide UI to adjust boundary Y-coordinates
- [ ] Call `/adjust-boundary` when teacher adjusts
- [ ] Re-show preview until satisfied
- [ ] Only then proceed to `/extract`

---

## Why This Matters

**What's working well:**
- ✓ PDF rendering (8 pages correct)
- ✓ Question crop merging logic
- ✓ OCR extraction (accurate text)
- ✓ Gemini vision evaluation (proper grading)
- ✓ Rubric matching logic

**What needs fixing:**
- ❌ Manual boundary placement (teacher clicked wrong position)
- ❌ No visual feedback before grading (critical!)
- ❌ Single merged PNG per question (architectural limitation)

**The fix order:**
1. **Phase 1** (done): Visual preview + boundary adjustment
2. **Phase 2** (TBD): Support multi-page questions (list of images)
3. **Phase 3** (TBD): Auto-detect question numbers to propose boundaries

---

## Remember

> **Don't use the grader to detect segmentation errors. Detect segmentation errors before grading.**

The logs showed that everything downstream (OCR → Gemini → marking) is working correctly.
The problem is upstream: the teacher needs to verify the split visually **before** it goes to OCR.
