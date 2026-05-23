"""
core/vector_store.py — ChromaDB wrapper + pipeline indeksimi (parse→chunk→embed→store).

Indexer-i i vjetër u bashkua këtu sepse të dyja klasat punonin me të njëjtat
objekte dhe nuk kishte arsye logjike t'i ndanim.

Tre koleksione:
  • finance            – leksione bazë Finance (admin-i e shkruan)
  • informatike_biznesi – leksione bazë Inf. Biz. (admin-i e shkruan)
  • personal_docs      – dokumentet e studentit (filtrohen me username)
"""



from __future__ import annotations

import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings

from config import (
    CHROMA_DIR, COL_FINANCE, COL_INFORMATIKE, COL_PERSONAL,
    TOP_K, MAX_PERSONAL_DOCS, FINANCE_DIR, INFORMATIKE_DIR,
)


class VectorStore:
    # ── Inicializimi ──────────────────────────────────────────────────────────

    def __init__(self, persist_dir: str | Path = CHROMA_DIR):
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        meta = {"hnsw:space": "cosine"}
        self._cols = {
            name: self._client.get_or_create_collection(name, metadata=meta)
            for name in [COL_FINANCE, COL_INFORMATIKE, COL_PERSONAL]
        }

    def _col(self, name: str):
        if name not in self._cols:
            raise ValueError(f"Koleksioni '{name}' nuk ekziston.")
        return self._cols[name]

    # ── Shtimi ────────────────────────────────────────────────────────────────

    def add_chunks(self, col: str, chunks: list[str],
                   embeddings: list[list[float]], metadatas: list[dict]) -> None:
        if not chunks:
            return
        self._col(col).add(
            ids=[str(uuid.uuid4()) for _ in chunks],
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # ── Kontrolle ─────────────────────────────────────────────────────────────

    def already_indexed(self, col: str, source: str, username: str | None = None) -> bool:
        if username:
            where = {"$and": [{"source": {"$eq": source}}, {"username": {"$eq": username}}]}
        else:
            where = {"source": {"$eq": source}}
        return len(self._col(col).get(where=where, limit=1)["ids"]) > 0
    


    def delete_source(self, col: str, source: str, username: str | None = None) -> None:
        if username:
            where = {"$and": [{"source": {"$eq": source}}, {"username": {"$eq": username}}]}
        else:
            where = {"source": {"$eq": source}}
        self._col(col).delete(where=where)

    # ── Kërkimi ───────────────────────────────────────────────────────────────

    def query(self, col: str, embedding: list[float],
              top_k: int = TOP_K, username: str | None = None) -> list[dict]:
        """Kthe top_k chunk-et më të ngjashëm si listë dict-esh {text, metadata, distance}."""
        kwargs: dict = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if col == COL_PERSONAL and username:
            kwargs["where"] = {"username": username}
        try:
            r = self._col(col).query(**kwargs)
        except Exception:
            return []
        return [
            {"text": d, "metadata": m or {}, "distance": dist}
            for d, m, dist in zip(
                r["documents"][0], r["metadatas"][0], r["distances"][0]
            )
        ]

    # ── Listimi ───────────────────────────────────────────────────────────────

    def list_sources(self, col: str, username: str | None = None) -> list[str]:
        kwargs: dict = {"include": ["metadatas"]}
        if username:
            kwargs["where"] = {"username":{"$eq": username}}
        metas = self._col(col).get(**kwargs).get("metadatas") or []
        return sorted({m.get("source", "") for m in metas if m} - {""})

    def count(self, col: str) -> int:
        return self._col(col).count()
    
    def get_all_chunks(self, col: str, username: str | None = None) -> list[dict]:
        """Fetch all chunks from a collection (used for BM25 index building)."""
        kwargs: dict = {"include": ["documents", "metadatas"]}
        if username:
            kwargs["where"] = {"username": {"$eq": username}}
        try:
            r = self._col(col).get(**kwargs)
            return [
                {"text": d, "metadata": m or {}}
                for d, m in zip(r["documents"], r["metadatas"])
            ]
        except Exception:
            return []

    # ══ Pipeline indeksimi (ishte Indexer) ════════════════════════════════════

    def index_file(self, path: str | Path, col: str,
                   parser, chunker, embedder,
                   username: str | None = None,
                   force: bool = False) -> tuple[bool, str]:
        """
        Rrugëtimi i plotë: parse → chunk → embed → store.

        Kthen (sukses: bool, mesazh: str).
        """
        path = Path(path)
        if not path.exists():
            return False, f"Skedari nuk u gjet: {path}"

        fname = path.name

        if not force and self.already_indexed(col, fname, username=username):
            return True, f"'{fname}' është tashmë i indeksuar — u kalua."
        if force:
            self.delete_source(col, fname, username=username)

        try:
            documents = parser.parse(path)
        except Exception as e:
            return False, f"Gabim parsimi për '{fname}': {e}"

        if not documents:
            return False, f"Nuk u nxor tekst nga '{fname}'."

        all_chunks, all_meta = [], []
        for doc in documents:
            meta = {**doc.metadata}
            if username:
                meta["username"] = username
            for cd in chunker.chunk_with_metadata(doc.text, source=fname, extra_meta=meta):
                all_chunks.append(cd["text"])
                all_meta.append({k: v for k, v in cd.items() if k != "text"})

        if not all_chunks:
            return False, f"Chunking nuk prodhoi asnjë chunk për '{fname}'."

        try:
            embeddings = embedder.embed_batch(all_chunks)
        except Exception as e:
            return False, f"Gabim embedding për '{fname}': {e}"

        try:
            self.add_chunks(col, all_chunks, embeddings, all_meta)
        except Exception as e:
            return False, f"Gabim ruajtjeje për '{fname}': {e}"

        return True, f"'{fname}' u indeksua me sukses ({len(all_chunks)} chunk)."

    def index_personal(self, path: str | Path, username: str,
                       parser, chunker, embedder) -> tuple[bool, str]:
        """Indekson dokument personal, duke zbatuar kufirin MAX_PERSONAL_DOCS."""
        existing = self.list_sources(COL_PERSONAL, username=username)
        p = Path(path)
        if p.name not in existing and len(existing) >= MAX_PERSONAL_DOCS:
            return False, f"Keni arritur kufirin prej {MAX_PERSONAL_DOCS} dokumenteve."
        return self.index_file(p, COL_PERSONAL, parser, chunker, embedder, username=username)

    def index_directory(self, directory: Path, col: str,
                        parser, chunker, embedder) -> list[tuple]:
        """Indekson të gjithë PDF-të e një direktorie. Kthen listë (fname, ok, msg)."""
        results = []
        for f in sorted(directory.iterdir()):
            if f.suffix.lower() == ".pdf":
                ok, msg = self.index_file(f, col, parser, chunker, embedder)
                results.append((f.name, ok, msg))
        return results

    def startup_index(self, parser, chunker, embedder) -> None:
        """Indekson automatikisht të gjitha PDF-të bazë gjatë nisjes."""
        for col, directory in [(COL_FINANCE, FINANCE_DIR), (COL_INFORMATIKE, INFORMATIKE_DIR)]:
            self.index_directory(directory, col, parser, chunker, embedder)
