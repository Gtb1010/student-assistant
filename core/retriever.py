"""
core/retriever.py — Retriever i personalizuar: embed pyetje → kërko ChromaDB → kthen chunk.

Ndryshimet ndaj versionit origjinal:
  - Semantic threshold ulet pak (0.35) për të kapur materiale shqip
  - TOP_K rritet dinamikisht për pyetje "shpjego/supozimet/listo" që
    mund të kenë 4-6 pika të shpërndarë në chunks të ndryshëm
  - De-duplikim i përmirësuar: kontrollon si tekst i njëjtë ashtu edhe
    nënvarg i gjatë (≥80% i chunk-ut) — parandalon chunk copëza
  - Mbaj BM25 boost por ulet në 0.03 (semantika dominon)

RERANKING:
  - Pas retrieval-it fillestar, Reranker-i ri-rendit chunks duke përdorur
    një skor të kombinuar: relevancë semantike + BM25 + pozicion relativ.
  - Opsionalisht, nëse sentence-transformers cross-encoder është i disponueshëm,
    përdoret ai (cross_encoder mode). Nëse jo, kthehet te score-i hibrid i zgjeruar.
  - Reranker-i është i pavarur (klasë e veçantë) dhe thirret nga CustomRetriever
    pas de-duplikimit, para kthimit të rezultateve finale.
"""
from __future__ import annotations
import concurrent.futures
from dataclasses import dataclass, field
from rank_bm25 import BM25Okapi
from config import COL_FINANCE, COL_INFORMATIKE, COL_PERSONAL, TOP_K

# Trigger-at konceptualë: pyetje "çfarë është / shpjego / listo"
CONCEPTUAL_TRIGGERS = [
    "cfare eshte", "çfare është", "what is", "define", "perkufizone",
    "përkufizo", "explain", "shpjego", "what are", "cilat jane",
    "cilat janë", "describe", "pershkruaj", "përshkruaj",
    "supozimet", "hapat", "parimet", "llojet", "karakteristikat",
    "si funksionon", "si llogaritet", "si nxirret",
]

# Pyetje "listo gjithçka" → duam më shumë chunks
EXHAUSTIVE_TRIGGERS = [
    "të gjitha", "tgjitha", "listo", "numëro", "enumero",
    "supozimet", "kushtet", "hapat", "parimet", "kriteret",
    "all", "list all", "enumerate",
]

# Threshold semantik: chunk-et nën këtë vlerë hidhen
SEMANTIC_THRESHOLD = 0.35


@dataclass
class Chunk:
    text:         str
    source:       str
    collection:   str
    distance:     float
    metadata:     dict  = field(default_factory=dict)
    rerank_score: float = field(default=0.0)   # score nga reranker-i


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _is_conceptual(query: str) -> bool:
    q = query.lower().strip()
    return any(q.startswith(t) or t in q for t in CONCEPTUAL_TRIGGERS)


def _is_exhaustive(query: str) -> bool:
    """Pyetje që kërkon listim të plotë → duam TOP_K * 2 chunks."""
    q = query.lower().strip()
    return any(t in q for t in EXHAUSTIVE_TRIGGERS)


def _overlap_ratio(a: str, b: str) -> float:
    """Sa % e tekstit të shkurtër gjendet brenda atij të gjatë."""
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < 40:
        return 0.0
    return 1.0 if short in long else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# RERANKER
# ══════════════════════════════════════════════════════════════════════════════

class Reranker:
    """
    Ri-rendit chunks pas retrieval-it fillestar.

    Dy modalitete:
      1. cross_encoder=True  → provon të ngarkojë 'cross-encoder/ms-marco-MiniLM-L-6-v2'
                               nga sentence-transformers. Nëse nuk ka paketën/modelin,
                               bie automatikisht te modaliteti 2.
      2. cross_encoder=False → Skor hibrid i zgjeruar:
                               • Semantic score (1 - distance)
                               • BM25 score i normalizuar
                               • Pozicion bonus (chunks herët në listë marrin bonus të vogël)
                               • Gjatësi bonus (chunks mesatarë preferohen ndaj shumë të shkurtrave)
                               Secili faktor ka peshë të konfiguruar.

    Përdorim:
        reranker = Reranker(use_cross_encoder=True)   # provon cross-encoder
        reranked = reranker.rerank(query, chunks, top_n=6)
    """

    # Peshat e skorit hibrid (duhet të mblidhen 1.0)
    W_SEMANTIC  = 0.60
    W_BM25      = 0.25
    W_POSITION  = 0.08
    W_LENGTH    = 0.07

    # Gjatësia ideale e chunk-ut (karaktere) — chunks afër saj marrin bonus maksimal
    IDEAL_LENGTH = 500

    def __init__(self, use_cross_encoder: bool = True):
        self._cross_encoder = None
        if use_cross_encoder:
            self._cross_encoder = self._load_cross_encoder()

    @staticmethod
    def _load_cross_encoder():
        """
        Ngarkon cross-encoder nga sentence-transformers.
        Nëse paketa ose modeli mungon, kthen None (bie te moda hibride).
        """
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
            )
            print("[Reranker] Cross-encoder u ngarkua me sukses.")
            return model
        except Exception as e:
            print(f"[Reranker] Cross-encoder nuk u ngarkua ({e}). Po përdoret moda hibride.")
            return None

    def rerank(self, query: str, chunks: list[Chunk], top_n: int | None = None) -> list[Chunk]:
        """
        Ri-rendit listën e chunk-eve dhe kthen top_n të mirët.
        Nëse top_n=None, kthen të gjitha (të rirenditura).
        """
        if not chunks:
            return []
        top_n = top_n or len(chunks)

        if self._cross_encoder is not None:
            return self._rerank_cross_encoder(query, chunks, top_n)
        return self._rerank_hybrid(query, chunks, top_n)

    def _rerank_cross_encoder(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        """
        Përdor cross-encoder për të skoruar çdo (query, chunk) çift.
        Cross-encoder-i është shumë më i saktë por më i ngadaltë se bi-encoder-i.
        """
        import math

        pairs  = [(query, c.text) for c in chunks]
        scores = self._cross_encoder.predict(pairs, show_progress_bar=False)

        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        for chunk, raw_score in zip(chunks, scores):
            chunk.rerank_score = _sigmoid(float(raw_score))

        ranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
        return ranked[:top_n]

    def _rerank_hybrid(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        """
        Skor hibrid pa cross-encoder:
          final = W_SEMANTIC * sem + W_BM25 * bm25_norm + W_POSITION * pos + W_LENGTH * len_bonus
        """
        if not chunks:
            return []

        corpus    = [_tokenize(c.text) for c in chunks]
        bm25      = BM25Okapi(corpus)
        bm25_raw  = bm25.get_scores(_tokenize(query))
        max_bm25  = max(bm25_raw) if max(bm25_raw) > 0 else 1.0
        bm25_norm = [s / max_bm25 for s in bm25_raw]

        n = len(chunks)
        for i, chunk in enumerate(chunks):
            sem_score = max(0.0, 1.0 - chunk.distance)
            pos_score = 1.0 - (i / n)

            length    = len(chunk.text)
            len_ratio = length / self.IDEAL_LENGTH
            len_bonus = 1.0 - abs(1.0 - min(len_ratio, 2.0)) * 0.5
            len_bonus = max(0.0, min(1.0, len_bonus))

            chunk.rerank_score = (
                self.W_SEMANTIC * sem_score    +
                self.W_BM25     * bm25_norm[i] +
                self.W_POSITION * pos_score    +
                self.W_LENGTH   * len_bonus
            )

        ranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
        return ranked[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
# CustomRetriever (me integrim Reranker)
# ══════════════════════════════════════════════════════════════════════════════

class CustomRetriever:
    """
    Retriever hibrid: Dense (ChromaDB cosine) + BM25 sparse + Reranking.

    Parametra:
      use_reranker      : bool  — aktivizon/çaktivizon reranker-in (default: True)
      use_cross_encoder : bool  — nëse True, provon cross-encoder (default: True)
    """

    def __init__(self, embedder, vector_store, top_k: int = TOP_K,
                 use_reranker: bool = True, use_cross_encoder: bool = True):
        self._emb      = embedder
        self._vs       = vector_store
        self._top_k    = top_k
        self._reranker = Reranker(use_cross_encoder=use_cross_encoder) if use_reranker else None

    def retrieve(self, query: str,
                 use_finance: bool = False,
                 use_informatike: bool = False,
                 use_personal: bool = False,
                 username: str | None = None) -> list[Chunk]:

        tasks = []
        if use_finance:               tasks.append((COL_FINANCE,     None))
        if use_informatike:           tasks.append((COL_INFORMATIKE, None))
        if use_personal and username: tasks.append((COL_PERSONAL,    username))
        if not tasks:
            return []

        q_emb      = self._emb.embed(query)
        conceptual = _is_conceptual(query)
        exhaustive = _is_exhaustive(query)

        effective_k      = self._top_k * 2 if exhaustive else self._top_k
        fetch_multiplier = 2 if self._reranker is not None else 1
        candidate_k      = effective_k * fetch_multiplier

        results: list[Chunk] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {
                pool.submit(
                    self._hybrid_one, col, query, q_emb, user, conceptual, candidate_k
                ): col
                for col, user in tasks
            }
            for fut in concurrent.futures.as_completed(futures):
                try:
                    results.extend(fut.result())
                except Exception as e:
                    print(f"[Retriever] Error '{futures[fut]}': {e}")

        # Rendit sipas distancës (para reranking-ut)
        results.sort(key=lambda c: c.distance)

        # De-duplikim
        seen_texts: list[str] = []
        unique: list[Chunk]   = []
        for c in results:
            is_dup = any(
                c.text == s or _overlap_ratio(c.text, s) > 0.8
                for s in seen_texts
            )
            if not is_dup:
                seen_texts.append(c.text)
                unique.append(c)

        # Reranking — ri-rendit kandidatët e de-duplikuar
        if self._reranker is not None and unique:
            unique = self._reranker.rerank(query, unique, top_n=effective_k)
        else:
            unique = unique[:effective_k]

        return unique

    def _hybrid_one(self, col: str, query: str,
                    embedding: list[float],
                    username: str | None,
                    conceptual: bool,
                    top_k: int) -> list[Chunk]:

        fetch_k    = top_k * 4
        dense_hits = self._vs.query(col, embedding, fetch_k, username)
        if not dense_hits:
            return []

        # Filtro chunks me kualitet të ulët semantik
        dense_hits = [
            h for h in dense_hits
            if (1.0 - h.get("distance", 1.0)) >= SEMANTIC_THRESHOLD
        ]
        if not dense_hits:
            return []

        all_texts   = [h["text"] for h in dense_hits]
        text_to_hit = {h["text"]: h for h in dense_hits}

        # Pyetje konceptuale: vetëm semantikë (BM25 shpesh dëmton)
        if conceptual:
            return [
                Chunk(
                    text=h["text"],
                    source=h["metadata"].get("source", "?"),
                    collection=col,
                    distance=h.get("distance", 1.0),
                    metadata=h["metadata"],
                )
                for h in dense_hits[:top_k]
            ]

        # Pyetje faktike: semantic 0.97 + BM25 boost 0.03
        tokenized_corpus = [_tokenize(t) for t in all_texts]
        bm25             = BM25Okapi(tokenized_corpus)
        bm25_scores      = bm25.get_scores(_tokenize(query))
        max_bm25  = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        norm_bm25 = [s / max_bm25 for s in bm25_scores]

        scored = []
        for i, h in enumerate(dense_hits):
            semantic_score = 1.0 - h.get("distance", 1.0)
            final_score    = 0.97 * semantic_score + 0.03 * norm_bm25[i]
            scored.append((h["text"], final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        score_map = dict(scored)
        top_texts = [t for t, _ in scored[:top_k]]

        return [
            Chunk(
                text=t,
                source=text_to_hit[t]["metadata"].get("source", "?"),
                collection=col,
                distance=1.0 - score_map[t],
                metadata=text_to_hit[t]["metadata"],
            )
            for t in top_texts
        ]