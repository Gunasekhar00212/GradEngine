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