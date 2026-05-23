"""
core/embedder.py — Mbështjellës i sentence-transformers për gjenerimin e embedding-eve.

Modeli shumëgjuhësh mbështet shqipen pa fine-tuning të veçantë.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL


class EmbeddingManager:
    """Prodhon vektorë embedding për tekste të dhëna."""

    def __init__(self, model_name: str = EMBED_MODEL):
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        """Embedding për një tekst të vetëm."""
        return self._model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        ).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embedding për shumë tekste në një kalon (më efikas)."""
        if not texts:
            return []
        return self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            batch_size=32, show_progress_bar=False
        ).tolist()