# GradEngine Code Export

This document contains the full contents of code and config files only.

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

# -----------------------
# Load API Key
# -----------------------
load_dotenv()

# -----------------------
# Hardcoded inputs
# -----------------------
PDF_PATH = "data/pdfs/handwritten_test.pdf"

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

    normalized_text = normalize(full_text)

    sentences = [s.strip() for s in sentences if s.strip()]
    sentences_vec = [get_embedding(s) for s in sentences]

    expanded_rubric = {
    "sunlight": {
        "keywords": ["sunlight", "light energy"],
        "explanations": [
            "sunlight provides energy for photosynthesis",
            "plants use sunlight for energy"
        ],
        "marks": 1
    },
    "CO2": {
        "keywords": ["CO2", "carbon dioxide"],
        "explanations": [
            "CO2 is used to produce glucose",
            "plants take carbon dioxide to make food"
        ],
        "marks": 1
    },
    "water": {
        "keywords": ["water", "H2O"],
        "explanations": [
            "water is used in photosynthesis",
            "plants absorb water for photosynthesis"
        ],
        "marks": 1
    },
    "oxygen": {
        "keywords": ["oxygen", "O2"],
        "explanations": [
            "oxygen is released as byproduct",
            "oxygen is produced during photosynthesis",
            "plants produce oxygen"
        ],
        "marks": 1
    },
    "glucose": {
        "keywords": ["glucose", "sugar"],
        "explanations": [
            "glucose is produced as food",
            "plants make glucose"
        ],
        "marks": 1
    }
}

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
    score, detected = score_answer(sentences_vec, rubric_vectors, normalized_text)

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

## ./processing/extract_text.py

```python
from google import genai
from google.genai import types
import pytesseract
from PIL import Image


def extract_text(image_path, api_key):
    # -------- TRY GEMINI --------
    try:
        client = genai.Client(api_key=api_key)

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                ),
                "Extract all handwritten text exactly as written."
            ]
        )

        return getattr(response, "text", "")

    # -------- FALLBACK TO TESSERACT --------
    except Exception as e:
        print("Gemini failed, using Tesseract OCR:", e)

        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)

        return text

```

## ./processing/pdf_split.py

```python
from pdf2image import convert_from_path
import os

def pdf_to_images(pdf_path, output_folder="data/pages"):
    os.makedirs(output_folder, exist_ok=True)

    images = convert_from_path(pdf_path)
    paths = []

    for i, img in enumerate(images):
        path = f"{output_folder}/page_{i}.jpg"
        img.save(path, "JPEG")
        paths.append(path)

    return paths

```

## ./rubric/rubric.json

```json
{
  "Q1": {
    "sunlight": 1,
    "CO2": 1,
    "water": 1,
    "oxygen": 1,
    "glucose": 1
  }
}

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

def score_answer(sentence_vectors, rubric_vectors, normalized_text):
    total_score = 0
    detected = []

    for concept, data in rubric_vectors.items():

        keywords = data["keywords"]
        keyword_vecs = data["keyword_vecs"]
        explanation_vecs = data["explanation_vecs"]
        marks = data["marks"]

        # ---------- KEYWORD MATCH ----------
        keyword_hit = any(k.lower() in normalized_text for k in keywords)

        # ---------- BEST SENTENCE MATCH ----------
        best_keyword_sim = 0
        best_expl_sim = 0

        for s_vec in sentence_vectors:
            for k_vec in keyword_vecs:
                sim = cosine_similarity([s_vec], [k_vec])[0][0]
                best_keyword_sim = max(best_keyword_sim, sim)

            for e_vec in explanation_vecs:
                expl_sim = cosine_similarity([s_vec], [e_vec])[0][0]
                best_expl_sim = max(best_expl_sim, expl_sim)

        print(f"{concept} → keyword_hit: {keyword_hit}, "
              f"kw_sim: {best_keyword_sim:.2f}, expl_sim: {best_expl_sim:.2f}")

        # ---------- PARTIAL SCORING ----------
        concept_score = 0

        # mention
        if keyword_hit or best_keyword_sim > 0.6:
            concept_score += 0.5

        # explanation
        if best_expl_sim > 0.6:
            concept_score += 0.5

        if concept_score > 0:
            detected.append(concept)

        total_score += concept_score

    return total_score, detected

```
