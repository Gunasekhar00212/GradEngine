# GradEngine Code Export

This document contains the full contents of code and config files only.

## ./main.py

```python
import os
import json
from dotenv import load_dotenv
from google import genai

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
    answer_vec = get_embedding(full_text)

    rubric_vectors = {}
    for concept in rubric:
        rubric_vectors[concept] = get_embedding(concept)

    print("Step 4: Scoring...")
    score, detected = score_answer(answer_vec, rubric_vectors)

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

def score_answer(answer_vec, rubric_vectors):
    score = 0
    detected = []

    for concept, vec in rubric_vectors.items():
        sim = cosine_similarity([answer_vec], [vec])[0][0]

        if sim > 0.8:
            score += 1
            detected.append(concept)

    return score, detected

```
