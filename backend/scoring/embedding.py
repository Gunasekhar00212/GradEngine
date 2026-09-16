"""Backend embedding helpers for scoring."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return model.encode(text)
