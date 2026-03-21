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