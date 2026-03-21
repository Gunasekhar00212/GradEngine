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