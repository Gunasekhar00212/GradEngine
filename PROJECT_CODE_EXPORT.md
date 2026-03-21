# GradEngine Code Export

This document contains the full contents of code and config files only.

## ./PROJECT_TREE.txt

```text
GradEngine/
├── config/
│   └── settings.py
├── data/
│   ├── input/
│   │   └── handwritten_test.pdf
│   ├── pages/
│   │   ├── page_0.jpg
│   │   └── page_0.png
│   ├── extracted_text/
│   │   └── output.json
│   └── rubric/
│       └── expanded_rubric.json
├── processing/
│   ├── clean_text.py
│   ├── extract_text.py
│   └── pdf_split.py
├── scoring/
│   ├── embedding.py
│   └── scoring_engine.py
├── .env
├── .gitignore
├── main.py
├── README.md
└── requirements.txt

```

## ./main.py

```python
import os
import json
import re
from dotenv import load_dotenv

from config.settings import (
    PDF_PATH,
    IMAGE_OUTPUT_DIR,
    EXTRACTED_TEXT_PATH,
    BLUEPRINT_PATH,
    VALID_SHORT_WORDS,
    DEBUG,
)

from processing.pdf_split import pdf_to_images
from processing.extract_text import extract_text
from processing.clean_text import clean_text
from scoring.embedding import get_embedding
from scoring.scoring_engine import score_answer


def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


def filter_short_tokens(text, valid_short_words):
    words = [
        word for word in text.split()
        if len(word) > 2 or word in valid_short_words
    ]
    return " ".join(words)


def load_blueprint(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["question"], data["concepts"]


def extract_pages_text(page_paths, api_key):
    full_text = ""
    all_pages_text = []

    for i, page_path in enumerate(page_paths):
        text = extract_text(page_path, api_key)

        if DEBUG:
            print(f"\nPage {i} Text:\n{text}\n")

        all_pages_text.append({
            "page": i,
            "text": text
        })

        full_text += " " + text

    return full_text.strip(), all_pages_text


def save_extracted_text(all_pages_text, output_path):
    with open(output_path, "w") as f:
        json.dump(all_pages_text, f, indent=4)


def prepare_sentence_vectors(full_text):
    sentences = re.split(r'[.\n]', full_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_vectors = [get_embedding(sentence) for sentence in sentences]
    return sentence_vectors


def build_rubric_vectors(expanded_rubric):
    rubric_vectors = {}

    for concept, concept_data in expanded_rubric.items():
        keyword_vecs = [get_embedding(k) for k in concept_data["keywords"]]
        explanation_vecs = [get_embedding(e) for e in concept_data["explanations"]]

        rubric_vectors[concept] = {
            "keywords": concept_data["keywords"],
            "keyword_vecs": keyword_vecs,
            "explanation_vecs": explanation_vecs,
            "marks": concept_data["marks"]
        }

    return rubric_vectors


def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    print("Step 1: Converting PDF to images...")
    page_paths = pdf_to_images(PDF_PATH, IMAGE_OUTPUT_DIR)

    print("Step 2: Extracting text...")
    full_text, all_pages_text = extract_pages_text(page_paths, api_key)
    save_extracted_text(all_pages_text, EXTRACTED_TEXT_PATH)

    print("\nStep 3: Generating embeddings...")
    cleaned_text = clean_text(full_text)
    normalized_text = normalize_text(cleaned_text)
    normalized_text = filter_short_tokens(normalized_text, VALID_SHORT_WORDS)

    if DEBUG:
        print("\n--- CLEANED TEXT ---\n", cleaned_text)
        print("\n--- NORMALIZED TEXT ---\n", normalized_text)

    sentence_vectors = prepare_sentence_vectors(full_text)

    question, expanded_rubric = load_blueprint(BLUEPRINT_PATH)
    question_vec = get_embedding(question)
    rubric_vectors = build_rubric_vectors(expanded_rubric)

    print("Step 4: Scoring...")
    score, detected = score_answer(
        sentence_vectors,
        rubric_vectors,
        normalized_text,
        question_vec
    )

    total_marks = sum(item["marks"] for item in expanded_rubric.values())

    print("\nDetected Concepts:", detected)
    print("Final Score:", score, "/", total_marks)


if __name__ == "__main__":
    main()

```

## ./plan_checklist.txt

```text
GradEngine Roadmap Checklist (Reality-First)

How to use:
[ ] = pending
[x] = done

PHASE 1 - Foundation Prototype (Done)
[x] OCR pipeline implemented
[x] Text cleaning implemented
[x] Scoring engine implemented
[x] Rubric-based evaluation implemented
[x] Single-answer prototype completed

PHASE 2 - Real Data Collection and Reality Validation (Current Priority)
Goal: Validate assumptions with actual exam material before new intelligence/features.

Data Collection
[ ] Collect real question papers (multi-subject if possible)
[ ] Collect real student answer booklet samples
[ ] Collect real blueprint/rubric patterns (if available)
[ ] Apply anonymization/privacy checks to all collected data
[ ] Catalog each sample with metadata (subject, grade, handwriting type, pages)

Reality Analysis
[ ] Analyze answer length patterns (short/medium/long per question type)
[ ] Analyze teacher marking patterns (strictness, partial credit behavior)
[ ] Analyze question distribution (question types and frequency)
[ ] Analyze how students continue answers across pages
[ ] Analyze real question-numbering styles in booklets
[ ] Identify OCR failure cases for real handwriting
[ ] Build failure taxonomy (e.g., merged lines, missed numbers, symbol confusion)

Phase 2 Deliverables
[ ] Produce Reality Validation Report
[ ] Produce segmentation requirements derived from real data
[ ] Produce scoring-upgrade requirements derived from real marking behavior
[ ] Approve Phase 2 gate before starting Phase 3

PHASE 3 - Question-Wise Segmentation and Mapping
Goal: Build segmentation based on validated patterns from Phase 2.

Build
[ ] Implement question boundary detection based on real numbering styles
[ ] Implement continuation detection for answers spanning multiple pages
[ ] Implement answer-to-question mapping using validated layout patterns
[ ] Handle ambiguous numbering/layout cases with fallback logic

Validation
[ ] Evaluate segmentation on held-out real booklets
[ ] Measure mapping accuracy by numbering style
[ ] Log unresolved/ambiguous mappings for rule refinement
[ ] Approve Phase 3 gate before starting Phase 4

PHASE 4 - Advanced Evaluation Intelligence
Goal: Improve scoring quality using real data and teacher marking behavior.

Scoring Enhancements
[ ] Add completeness scoring
[ ] Add depth/coverage scoring
[ ] Add expected answer-length calibration
[ ] Add irrelevance penalty
[ ] Add answer structure quality scoring

Validation
[ ] Compare model scores vs teacher marks
[ ] Measure agreement/correlation metrics
[ ] Perform per-criterion error analysis
[ ] Refine thresholds/rules based on observed gaps
[ ] Approve Phase 4 gate before starting Phase 5

PHASE 5 - Systemization / Product Layer
Goal: Turn validated pipeline into a usable system/demo.

Productization
[ ] Add batch processing workflow
[ ] Add report generation (student/question level)
[ ] Build dashboard/UI
[ ] Prepare deployment/demo packaging
[ ] Add operational monitoring/logging hooks

Release Readiness
[ ] Run end-to-end demo with multi-booklet batch
[ ] Verify outputs for technical and non-technical stakeholders
[ ] Finalize roadmap closure and next-cycle backlog

Cross-Phase Control Checklist
[ ] Do not start Phase 3 before Phase 2 validation is signed off
[ ] Do not start Phase 4 before Phase 3 validation is signed off
[ ] Do not start Phase 5 before Phase 4 validation is signed off
[ ] Keep scope reality-driven (avoid assumption-driven features)

```

## ./requirements.txt

```text
google-genai
python-dotenv
pdf2image
Pillow
sentence-transformers
scikit-learn
numpy

```

## ./config/settings.py

```python
PDF_PATH = "data/input/handwritten_test.pdf"
IMAGE_OUTPUT_DIR = "data/pages"
EXTRACTED_TEXT_PATH = "data/extracted_text/output.json"
BLUEPRINT_PATH = "data/rubric/expanded_rubric.json"

RELEVANCE_THRESHOLD = 0.5
EXPLANATION_THRESHOLD = 0.6
KEYWORD_SIM_THRESHOLD = 0.7

VALID_SHORT_WORDS = {"co2", "o2", "h2o"}
DEBUG = True

```

## ./processing/clean_text.py

```python
import re

def clean_text(text):
    text = text.lower()

    # keep letters, digits, spaces, and question marks for OCR repair
    text = re.sub(r'[^a-z0-9\s?]', ' ', text)

    # remove long numeric garbage like 0000005000
    text = re.sub(r'\d{2,}', ' ', text)

    # remove isolated single digits like 1 2 3
    text = re.sub(r'\b\d\b', ' ', text)

    # domain-aware science token fixes
    text = re.sub(r'co\s*\?', 'co2', text)
    text = re.sub(r'c0\s*2', 'co2', text)
    text = re.sub(r'c02', 'co2', text)
    text = re.sub(r'o\s*2', 'o2', text)
    text = re.sub(r'h\s*2\s*o', 'h2o', text)
    text = re.sub(r'h20', 'h2o', text)

    # normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

```

## ./processing/extract_text.py

```python
import cv2
import numpy as np
import pytesseract
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


device = "cuda" if torch.cuda.is_available() else "cpu"

_processor = None
_model = None


def get_trocr_model():
    global _processor, _model

    if _processor is None or _model is None:
        _processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        _model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        _model.to(device)
        _model.eval()

    return _processor, _model


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        10,
    )

    return img, thresh


def split_lines(image_path):
    img, thresh = preprocess_image(image_path)

    horizontal_sum = np.sum(thresh, axis=1)

    lines = []
    start = None
    min_line_height = 18

    for i, val in enumerate(horizontal_sum):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            end = i
            if end - start >= min_line_height:
                line_img = img[start:end, :]
                if line_img.shape[0] > 0 and line_img.shape[1] > 0:
                    lines.append((start, line_img))
            start = None

    if start is not None:
        end = len(horizontal_sum)
        if end - start >= min_line_height:
            line_img = img[start:end, :]
            if line_img.shape[0] > 0 and line_img.shape[1] > 0:
                lines.append((start, line_img))

    lines.sort(key=lambda x: x[0])
    return [line_img for _, line_img in lines]


def trocr_extract(image_path):
    try:
        processor, model = get_trocr_model()
        line_images = split_lines(image_path)

        if not line_images:
            return ""

        extracted_lines = []

        for line in line_images:
            pil_img = Image.fromarray(cv2.cvtColor(line, cv2.COLOR_BGR2RGB))
            pil_img.thumbnail((384, 384))

            pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

            with torch.no_grad():
                generated_ids = model.generate(pixel_values)

            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            if text:
                extracted_lines.append(text)

        return " ".join(extracted_lines)

    except Exception as e:
        print("TrOCR failed:", e)
        return ""


def tesseract_extract(image_path):
    try:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image).strip()
    except Exception as e:
        print("Tesseract failed:", e)
        return ""


def gemini_extract(image_path, api_key):
    try:
        raise Exception("Gemini disabled")
    except Exception as e:
        print("Gemini failed:", e)
        return ""


def extract_text(image_path, api_key):
    gemini_text = gemini_extract(image_path, api_key)
    if gemini_text:
        print("Using Gemini OCR")
        return gemini_text

    trocr_text = trocr_extract(image_path)
    if trocr_text:
        print("Using TrOCR OCR")
        return trocr_text

    print("Using Tesseract OCR")
    return tesseract_extract(image_path)

```

## ./processing/pdf_split.py

```python
from pdf2image import convert_from_path
import os

def pdf_to_images(pdf_path, output_folder="data/pages"):
    os.makedirs(output_folder, exist_ok=True)

    images = convert_from_path(pdf_path,dpi=300)
    paths = []

    for i, img in enumerate(images):
        path = f"{output_folder}/page_{i}.png"
        img.save(path, "PNG")
        paths.append(path)

    return paths

```

## ./scoring/embedding.py

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return model.encode(text)

```

## ./scoring/scoring_engine.py

```python
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import (
    RELEVANCE_THRESHOLD,
    EXPLANATION_THRESHOLD,
    KEYWORD_SIM_THRESHOLD,
)


def fuzzy_match(a, b):
    return SequenceMatcher(None, a, b).ratio()


def get_best_similarity(sentence_vectors, target_vectors):
    best_sim = 0

    for s_vec in sentence_vectors:
        for t_vec in target_vectors:
            sim = cosine_similarity([s_vec], [t_vec])[0][0]
            best_sim = max(best_sim, sim)

    return best_sim


def get_best_relevance(sentence_vectors, question_vec):
    best_relevance = 0

    for s_vec in sentence_vectors:
        rel_sim = cosine_similarity([s_vec], [question_vec])[0][0]
        best_relevance = max(best_relevance, rel_sim)

    return best_relevance


def has_keyword_hit(keywords, normalized_text):
    words = normalized_text.split()

    return any(
        any(fuzzy_match(keyword.lower(), word) > 0.8 for word in words)
        for keyword in keywords
    )


def score_answer(sentence_vectors, rubric_vectors, normalized_text, question_vec):
    total_score = 0
    detected = []

    for concept, data in rubric_vectors.items():
        keywords = data["keywords"]
        keyword_vecs = data["keyword_vecs"]
        explanation_vecs = data["explanation_vecs"]
        marks = data["marks"]

        keyword_hit = has_keyword_hit(keywords, normalized_text)
        best_keyword_sim = get_best_similarity(sentence_vectors, keyword_vecs)
        best_expl_sim = get_best_similarity(sentence_vectors, explanation_vecs)
        best_relevance = get_best_relevance(sentence_vectors, question_vec)

        print(
            f"{concept} → keyword_hit: {keyword_hit}, "
            f"kw_sim: {best_keyword_sim:.2f}, "
            f"expl_sim: {best_expl_sim:.2f}, "
            f"rel_sim: {best_relevance:.2f}"
        )

        concept_score = 0

        if best_relevance > RELEVANCE_THRESHOLD:
            if keyword_hit or best_keyword_sim > KEYWORD_SIM_THRESHOLD:
                concept_score += marks * 0.5

            if best_expl_sim > EXPLANATION_THRESHOLD:
                concept_score += marks * 0.5

        if concept_score > 0:
            detected.append(concept)

        total_score += concept_score

    return total_score, detected

```
