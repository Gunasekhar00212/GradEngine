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
from dotenv import load_dotenv
from google import genai
import re

from processing.pdf_split import pdf_to_images
from processing.extract_text import extract_text
from scoring.embedding import get_embedding
from scoring.scoring_engine import score_answer
from processing.clean_text import clean_text
# -----------------------
# Load API Key
# -----------------------
load_dotenv()

# -----------------------
# Hardcoded inputs
# -----------------------
PDF_PATH = "data/input/handwritten_test.pdf"

rubric = {
    "sunlight": 1,
    "CO2": 1,
    "water": 1,
    "oxygen": 1,
    "glucose": 1
}

# -----------------------
# MAIN
# -----------------------
def main():
    print("Step 1: Converting PDF to images...")
    pages = pdf_to_images(PDF_PATH)

    print("Step 2: Extracting text...")
    
    full_text = ""
    all_pages_text = []

    for i, page in enumerate(pages):
        api_key = os.getenv("GOOGLE_API_KEY")

        text = extract_text(page,api_key)

        print(f"\nPage {i} Text:\n{text}\n")

        # store page-wise
        data = {
            "page": i,
            "text": text
        }

        all_pages_text.append(data)

        full_text += " " + text

    # save after loop
    with open("data/extracted_text/output.json", "w") as f:
        json.dump(all_pages_text, f, indent=4)

    print("\nStep 3: Generating embeddings...")
    sentences = re.split(r'[.\n]', full_text)# normalize full text
    def normalize(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text
    cleaned_text = clean_text(full_text)
    normalized_text = normalize(cleaned_text)
    print("\n--- CLEANED TEXT ---\n", cleaned_text)
    print("\n--- NORMALIZED TEXT ---\n", normalized_text)
    # remove small noisy tokens
    valid_short_words = {"co2", "o2", "h2o"}

    words = [
        w for w in normalized_text.split()
        if len(w) > 2 or w in valid_short_words
    ]

    normalized_text = " ".join(words)    
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences_vec = [get_embedding(s) for s in sentences]

    with open("data/rubric/expanded_rubric.json", "r") as f:
        data = json.load(f)
        QUESTION = data["question"]
        expanded_rubric = data["concepts"]

    question_vec = get_embedding(QUESTION)

    rubric_vectors = {}

    for concept, data in expanded_rubric.items():
        keyword_vecs = [get_embedding(k) for k in data["keywords"]]
        explanation_vecs = [get_embedding(e) for e in data["explanations"]]  # Use the first explanation for now
    
        rubric_vectors[concept] = {
            "keywords": data["keywords"],
            "keyword_vecs": keyword_vecs,
            "explanation_vecs": explanation_vecs,
            "marks": data["marks"]
        }
    
    print("Step 4: Scoring...")
    score, detected = score_answer(
                                sentences_vec, 
                                rubric_vectors, 
                                normalized_text,
                                question_vec
                               )

    print("\nDetected Concepts:", detected)
    print("Final Score:", score, "/", len(rubric))


if __name__ == "__main__":
    main()

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
from sklearn.metrics.pairwise import cosine_similarity

def score_answer(sentence_vectors, rubric_vectors, normalized_text, question_vec):
    total_score = 0
    
    detected = []

    for concept, data in rubric_vectors.items():
        
        keywords = data["keywords"]
        keyword_vecs = data["keyword_vecs"]
        explanation_vecs = data["explanation_vecs"]
        marks = data["marks"]

        # ---------- KEYWORD MATCH ----------
        from difflib import SequenceMatcher

        def fuzzy_match(a, b):
            return SequenceMatcher(None, a, b).ratio()
        
        keyword_hit = any(
            any(fuzzy_match(k.lower(), word) > 0.8 for word in normalized_text.split())
            for k in keywords
        )

        # ---------- BEST SENTENCE MATCH ----------
        best_keyword_sim = 0
        best_expl_sim = 0
        best_relevance = 0
        for s_vec in sentence_vectors:
            for k_vec in keyword_vecs:
                sim = cosine_similarity([s_vec], [k_vec])[0][0]
                best_keyword_sim = max(best_keyword_sim, sim)

            for e_vec in explanation_vecs:
                expl_sim = cosine_similarity([s_vec], [e_vec])[0][0]
                best_expl_sim = max(best_expl_sim, expl_sim)

            rel_sim = cosine_similarity([s_vec], [question_vec])[0][0]
            best_relevance = max(best_relevance, rel_sim)

        print(f"{concept} → keyword_hit: {keyword_hit}, "
            f"kw_sim: {best_keyword_sim:.2f}, "
            f"expl_sim: {best_expl_sim:.2f}, "
            f"rel_sim: {best_relevance:.2f}")

        # ---------- PARTIAL SCORING ----------
        concept_score = 0

        RELEVANCE_THRESHOLD = 0.5

        if best_relevance > RELEVANCE_THRESHOLD:
        
            if keyword_hit or best_keyword_sim > 0.6:
                concept_score += 0.5

            if best_expl_sim > 0.6:
                concept_score += 0.5

        if concept_score > 0:
            detected.append(concept)

        total_score += concept_score

    return total_score, detected

```
