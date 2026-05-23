"""
config.py — Konfigurimet qendrore të sistemit RAG.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

# ── Direktoritë ───────────────────────────────────────────────────────────────
DATA_DIR          = BASE_DIR / "data"
FINANCE_DIR       = DATA_DIR / "Finance"
INFORMATIKE_DIR   = DATA_DIR / "Informatike_Biznesi"
PERSONAL_DOCS_DIR = BASE_DIR / "personal_docs"
CHAT_HISTORY_DIR  = BASE_DIR / "chat_histories"
USER_DATA_DIR     = BASE_DIR / "user_data"
CHROMA_DIR        = Path("/Users/test/chroma_store")

for _d in [FINANCE_DIR, INFORMATIKE_DIR, PERSONAL_DOCS_DIR,
           CHAT_HISTORY_DIR, USER_DATA_DIR, CHROMA_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── ChromaDB ──────────────────────────────────────────────────────────────────
COL_FINANCE    = "finance"
COL_INFORMATIKE = "informatike_biznesi"
COL_PERSONAL   = "personal_docs"

# ── Modelet ───────────────────────────────────────────────────────────────────
EMBED_MODEL      = "paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL       = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS  = 3000
GROQ_TEMPERATURE = 0.3

# ── Parametrat ────────────────────────────────────────────────────────────────
CHUNK_SIZE        = 800
CHUNK_OVERLAP     = 100
TOP_K             = 6
MAX_PERSONAL_DOCS = 15
QUIZ_QUESTIONS    = 5

# ── Rolet ─────────────────────────────────────────────────────────────────────
ROLE_STUDENT  = "student"
ROLE_FIN      = "finance_admin"
ROLE_INFO     = "informatike_admin"

UPLOAD_PERMS = {COL_FINANCE: [ROLE_FIN], COL_INFORMATIKE: [ROLE_INFO]}

# ── Llogaritë demo (username → {password, role}) ──────────────────────────────
DEFAULT_USERS = {
    "student1":  {"password": "student123",   "role": ROLE_STUDENT},
    "student2":  {"password": "student456",   "role": ROLE_STUDENT},
    "fin_admin": {"password": "finAdmin2024",  "role": ROLE_FIN},
    "info_admin":{"password": "infoAdmin2024", "role": ROLE_INFO},
}