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

    with open("rubric/expanded_rubric.json", "r") as f:
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