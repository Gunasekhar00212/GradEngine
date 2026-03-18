from sklearn.metrics.pairwise import cosine_similarity

def score_answer(sentence_vectors, rubric_vectors, normalized_text):
    score = 0
    detected = []

    for concept, vec_list in rubric_vectors.items():

        # ---------- KEYWORD MATCH ----------
        keyword_hit = concept.lower() in normalized_text

        # ---------- EMBEDDING MATCH ----------
        best_sim = 0
        for s_vec in sentence_vectors:
            for r_vec in vec_list:
                sim = cosine_similarity([s_vec], [r_vec])[0][0]
                best_sim = max(best_sim, sim)

        print(f"{concept} → sim: {best_sim:.2f}, keyword: {keyword_hit}")

        # ---------- FINAL DECISION ----------
        if keyword_hit or best_sim > 0.6:
            score += 1
            detected.append(concept)

    return score, detected