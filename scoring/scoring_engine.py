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
        keyword_hit = any(k.lower() in normalized_text for k in keywords)

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