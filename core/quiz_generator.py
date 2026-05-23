from __future__ import annotations
from datetime import datetime
from config import QUIZ_QUESTIONS, COL_PERSONAL, TOP_K


class QuizGenerator:
    def __init__(self, embedder, vector_store, llm_client):
        self._emb = embedder
        self._vs  = vector_store
        self._llm = llm_client

    def generate(self, source: str, username: str,
                 n: int = QUIZ_QUESTIONS) -> tuple[bytes, str]:
        emb  = self._emb.embed("konceptet dhe teorite kryesore te lendes")
        hits = self._vs.query(COL_PERSONAL, emb, min(8, TOP_K), username)
        hits = [h for h in hits if h["metadata"].get("source") == source]
        if not hits:
            raise ValueError(
                f"Nuk u gjet asnje chunk per '{source}'. "
                "Sigurohuni qe dokumenti eshte ngarkuar dhe indeksuar."
            )
        quiz_text = self._llm.generate_quiz([h["text"] for h in hits], n=n)
        txt_bytes = quiz_text.encode("utf-8")
        return txt_bytes, quiz_text