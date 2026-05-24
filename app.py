"""
app.py — Streamlit app i plotë: të gjitha faqet UI janë këtu.
Struktura e skedarit:
  1. CSS inline
  2. Cache + inicializimi i objekteve
  3. _page_login()          ← përfshin lidhje me _page_register()
  4. _page_register()       ← E RE: faqe regjistrimi me SQLite
  5. _page_chat()
  6. _page_upload()
  7. _page_quiz()
  8. _page_admin()
  9. main() — routing + sidebar

NDRYSHIMET (v3):
  - auth.py migrojë nga JSON → SQLite
  - register() pranon username + email + password (unike)
  - _page_login() tregon buton "Krijo llogari" → kalon te _page_register()
  - _page_register() validimet UI + thirrje register()
  - login() kthen edhe email-in → ruhet në session_state
"""
from __future__ import annotations
import os
import json
from datetime import datetime
from pathlib import Path
import streamlit as st
st.set_page_config(
    page_title="Student Assistant",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)
from config import (
    COL_FINANCE, COL_INFORMATIKE, COL_PERSONAL,
    ROLE_STUDENT, ROLE_FIN, ROLE_INFO,
    MAX_PERSONAL_DOCS, QUIZ_QUESTIONS, PERSONAL_DOCS_DIR,
    FINANCE_DIR, INFORMATIKE_DIR,
    CHAT_HISTORY_DIR,
)
from core import (
    login, register, can_upload, is_admin,
    TextChunker, DocumentParser, EmbeddingManager,
    VectorStore, CustomRetriever, Chunk,
    LLMClient, FileChatMessageHistory, QuizGenerator,
)

# ═══════════════════════════════════════════════════════════
# 1. CSS
# ═══════════════════════════════════════════════════════════
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;600&display=swap');
:root {
  --navy: #1a3a5c;
  --navy2: #0f2740;
  --gold: #c9973e;
  --cream: #f5f0e8;
}
.stApp {
  background: linear-gradient(135deg, #f0eee8, #e8e4dc);
  font-family: 'Source Sans 3', sans-serif;
}
.stApp * { color: #000000; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--navy), var(--navy2));
  border-right: 3px solid var(--gold);
}
[data-testid="stSidebar"] * { color: var(--cream) !important; }
[data-testid="stSidebar"] hr { border-color: rgba(201,151,62,0.4) !important; }
h1, h2, h3 { font-family: 'Playfair Display', serif; color: var(--navy); }
.avatar {
  width: 44px; height: 44px; border-radius: 50%;
  background: var(--gold); color: var(--navy) !important;
  font-family: 'Playfair Display', serif;
  font-size: 1.3rem; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  border: 2px solid rgba(255,255,255,0.3);
}
.role-badge { font-size: 0.72rem; padding: 2px 8px; border-radius: 20px; display: inline-block; margin-top: 3px; }
.rb-student            { background: rgba(45,122,69,0.7);   color: #b6f0c8 !important; }
.rb-finance_admin      { background: rgba(201,151,62,0.7);  color: #fff3cd !important; }
.rb-informatike_admin { background: rgba(79,119,190,0.7);  color: #cfe2ff !important; }
.chat-u {
  background: var(--navy); color: var(--cream) !important;
  padding: 12px 16px; border-radius: 10px 10px 4px 10px;
  margin: 6px 0 6px 18%; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  font-size: 0.95rem; line-height: 1.55;
}
.chat-u * { color: var(--cream) !important; }
.chat-a {
  background: #fff; color: #222 !important;
  padding: 12px 16px; border-radius: 10px 10px 10px 4px;
  margin: 6px 18% 6px 0; border-left: 4px solid var(--gold);
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  font-size: 0.95rem; line-height: 1.6;
}
.chat-a * { color: #222 !important; }
.src-tag {
  display: inline-block; background: var(--cream);
  border: 1px solid var(--gold); color: var(--navy) !important;
  font-size: 0.7rem; padding: 1px 7px; border-radius: 20px; margin: 1px 2px;
}
.card {
  background: #fff; border-radius: 10px; padding: 16px 20px;
  border: 1px solid rgba(26,58,92,0.12);
  box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin-bottom: 12px;
}
.auth-card {
  background: #fff;
  border-radius: 16px;
  padding: 36px 40px;
  border: 1px solid rgba(26,58,92,0.10);
  box-shadow: 0 4px 24px rgba(0,0,0,0.08);
  border-top: 4px solid var(--gold);
}
.stat-card {
  background: #fff;
  border-radius: 12px;
  padding: 20px 24px;
  border: 1px solid rgba(26,58,92,0.10);
  box-shadow: 0 2px 12px rgba(0,0,0,0.07);
  margin-bottom: 16px;
  border-top: 4px solid var(--gold);
}
.stat-card h4 {
  font-family: 'Playfair Display', serif;
  color: var(--navy);
  margin: 0 0 4px 0;
  font-size: 1rem;
}
.stat-number {
  font-family: 'Playfair Display', serif;
  font-size: 2.4rem;
  color: var(--gold) !important;
  line-height: 1;
  font-weight: 700;
}
.stat-label {
  font-size: 0.78rem;
  color: #777 !important;
  margin-top: 3px;
}
.q-row {
  background: #f9f7f3;
  border-left: 3px solid var(--gold);
  border-radius: 0 8px 8px 0;
  padding: 8px 14px;
  margin-bottom: 7px;
  font-size: 0.88rem;
}
.q-row .q-meta { font-size: 0.72rem; color: #999 !important; margin-top: 2px; }
/* ── Buttons ── */
.stButton > button {
  background: var(--navy) !important; color: #ffffff !important;
  border: none !important; border-radius: 8px !important;
  font-weight: 600 !important; transition: 0.2s !important;
}
.stButton > button * { color: #ffffff !important; }
.stButton > button:hover { background: var(--gold) !important; color: var(--navy) !important; }
.stButton > button:hover,
.stButton > button:hover * { color: var(--navy) !important; }
[data-testid="stMetricValue"] {
  color: var(--gold) !important;
  font-family: 'Playfair Display', serif !important;
}
input[type="text"], input[type="password"], textarea,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea { color: #ffffff !important; caret-color: #ffffff !important; }
[data-testid="stExpander"] * { color: #000000 !important; }
.stMarkdown p, .stMarkdown div { color: #000000 !important; }
[data-baseweb="select"] *,
[data-testid="stSelectbox"] * { color: #ffffff !important; }
/* ── File uploader: label dark, internal button white ── */
[data-testid="stFileUploader"] * { color: #000000 !important; }
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button * { color: #ffffff !important; }
/* ── Sidebar SVGs (collapse arrow, X close) always white ── */
[data-testid="stSidebar"] svg { fill: var(--cream) !important; }
[data-testid="collapsedControl"] svg,
[data-testid="baseButton-headerNoPadding"] svg,
button[kind="header"] svg,
[data-testid="stSidebarCollapseButton"] svg {
  fill: #ffffff !important;
  color: #ffffff !important;
}
/* ── Chat input send arrow ── */
[data-testid="stChatInput"] button svg,
[data-testid="stChatInputSubmitButton"] {
  fill: #ffffff !important;
  color: #ffffff !important;
}
/* ── "New Chat" button column ── */
[data-testid="column"] .stButton > button,
[data-testid="column"] .stButton > button span,
[data-testid="column"] .stButton > button p { 
  color: #ffffff !important; 
}

button[kind="primary"], button[kind="secondary"],
button[kind="primary"] *, button[kind="secondary"] *,
.stButton button, .stButton button span, .stButton button p,
div[data-testid] .stButton > button,
div[data-testid] .stButton > button * {
  color: #ffffff !important;
}

/* ── Link-style secondary button ── */
.link-btn {
  background: none !important;
  border: none !important;
  color: var(--gold) !important;
  font-size: 0.88rem;
  cursor: pointer;
  text-decoration: underline;
  padding: 0 !important;
}
</style>
"""

# ═══════════════════════════════════════════════════════════
# 2. Cache / singletons
# ═══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Duke ngarkuar modelin e embedding...")
def _embedder()     -> EmbeddingManager: return EmbeddingManager()

@st.cache_resource(show_spinner="Duke u lidhur me ChromaDB...")
def _vector_store() -> VectorStore:      return VectorStore()

@st.cache_resource
def _parser()       -> DocumentParser:   return DocumentParser()

@st.cache_resource
def _chunker()      -> TextChunker:      return TextChunker()

def _llm() -> LLMClient:
    if "llm" not in st.session_state:
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            st.error("GROQ_API_KEY nuk është vendosur.\n```\nexport GROQ_API_KEY=...\n```")
            st.stop()
        st.session_state.llm = LLMClient(api_key=key)
    return st.session_state.llm

def _retriever() -> CustomRetriever:
    if "ret" not in st.session_state:
        st.session_state.ret = CustomRetriever(
            _embedder(), _vector_store(),
            use_reranker=True, use_cross_encoder=True,
        )
    return st.session_state.ret

def _quiz_gen() -> QuizGenerator:
    if "qgen" not in st.session_state:
        st.session_state.qgen = QuizGenerator(_embedder(), _vector_store(), _llm())
    return st.session_state.qgen

def _history(username: str) -> FileChatMessageHistory:
    k = f"hist_{username}"
    if k not in st.session_state:
        st.session_state[k] = FileChatMessageHistory(username)
    return st.session_state[k]

def _startup_index() -> None:
    if st.session_state.get("_indexed"):
        return
    vs = _vector_store()
    if vs.count(COL_FINANCE) == 0 or vs.count(COL_INFORMATIKE) == 0:
        with st.spinner("Duke indeksuar leksionet baze..."):
            vs.startup_index(_parser(), _chunker(), _embedder())
    st.session_state._indexed = True


# ═══════════════════════════════════════════════════════════
# Admin stats helpers
# ═══════════════════════════════════════════════════════════
from config import USER_DATA_DIR as _USER_DATA_DIR
_STATS_FILE = _USER_DATA_DIR / "admin_stats.json"

def _load_stats() -> dict:
    if _STATS_FILE.exists():
        try:
            return json.loads(_STATS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"questions": []}

def _save_stats(data: dict) -> None:
    _STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def _log_question(question: str, username: str, collections: list[str], role: str = "") -> None:
    data = _load_stats()
    data["questions"].append({
        "text":        question,
        "username":    username,
        "role":        role,
        "collections": collections,
        "ts":          datetime.now().isoformat(),
    })
    data["questions"] = data["questions"][-500:]
    _save_stats(data)

def _top_questions(col_filter: str | None = None, n: int = 5) -> list[tuple[str, int]]:
    data = _load_stats()
    qs   = data.get("questions", [])
    qs = [q for q in qs if q.get("role", "") not in (ROLE_FIN, ROLE_INFO)]
    qs = [q for q in qs if COL_PERSONAL not in q.get("collections", [])]
    if col_filter:
        qs = [q for q in qs if col_filter in q.get("collections", [])]
    freq: dict[str, int] = {}
    for q in qs:
        t = q["text"].strip()
        freq[t] = freq.get(t, 0) + 1
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]

def _unique_active_users(col_filter: str | None = None) -> int:
    data = _load_stats()
    qs   = data.get("questions", [])
    qs = [q for q in qs if q.get("role", "") not in (ROLE_FIN, ROLE_INFO)]
    if col_filter:
        qs = [q for q in qs if col_filter in q.get("collections", [])]
    return len({q.get("username") for q in qs})


# ═══════════════════════════════════════════════════════════
# 3. Faqja e hyrjes (Login)
# ═══════════════════════════════════════════════════════════
def _page_login() -> None:
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        st.markdown("""
        <div style="text-align:center;padding:40px 0 20px">
            <div style="font-family:'Playfair Display',serif;font-size:2.2rem;color:#1a3a5c;">
                Asistent Mësimor
            </div>
            <div style="color:#8a8fa8;font-size:.9rem;margin-top:6px;">
                Sistemi RAG Akademik — Finance &amp; Informatike Biznesi
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            u  = st.text_input("Emri i përdoruesit", placeholder="p.sh. student1")
            p  = st.text_input("Fjalëkalimi", type="password")
            ok = st.form_submit_button("Hyr", use_container_width=True)

        if ok:
            if not u or not p:
                st.error("Plotësoni të dyja fushat.")
            else:
                user = login(u, p)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Kredenciale të gabuara, provo përsëri.")

        # ── Lidhja me regjistrim ──────────────────────────────────────────
        st.markdown(
            "<div style='text-align:center;margin-top:16px;color:#8a8fa8;font-size:.88rem;'>"
            "Nuk keni llogari?</div>",
            unsafe_allow_html=True,
        )
        if st.button("Krijo llogari të re", use_container_width=True, key="go_register"):
            st.session_state._show_register = True
            st.rerun()


# ═══════════════════════════════════════════════════════════
# 4. Faqja e Regjistrimit  ← E RE
# ═══════════════════════════════════════════════════════════
def _page_register() -> None:
    """
    Faqe regjistrimi për studentë.
    Validimet:
      • username: 3-30 karaktere, vetëm [a-zA-Z0-9_]
      • email: duhet të përmbajë @ dhe një pikë pas @
      • password: ≥6 karaktere
      • confirm_password: duhet të përputhet
    """
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        st.markdown("""
        <div style="text-align:center;padding:32px 0 16px">
            <div style="font-family:'Playfair Display',serif;font-size:2rem;color:#1a3a5c;">
                Krijo Llogari
            </div>
            <div style="color:#8a8fa8;font-size:.88rem;margin-top:6px;">
                Regjistrohu si student — falas dhe i menjëhershëm
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("register_form"):
            username = st.text_input(
                "Emri i përdoruesit *",
                placeholder="p.sh. ardit_hoxha",
                help="3–30 karaktere: shkronja, numra, nënvizë (_)",
            )
            email = st.text_input(
                "Email *",
                placeholder="p.sh. ardit@umt.edu.al",
                help="Adresa e email-it institucional ose personal",
            )
            password = st.text_input(
                "Fjalëkalimi *",
                type="password",
                placeholder="Minimum 6 karaktere",
            )
            confirm = st.text_input(
                "Konfirmo fjalëkalimin *",
                type="password",
                placeholder="Përsërit fjalëkalimin",
            )

            submitted = st.form_submit_button(
                "Regjistrohu", use_container_width=True
            )

        # ── Validim UI (para thirrjes backend) ───────────────────────────
        if submitted:
            errors: list[str] = []

            if not username or len(username.strip()) < 3:
                errors.append("Emri i përdoruesit duhet të ketë të paktën 3 karaktere.")
            if not email or "@" not in email or "." not in email.split("@")[-1]:
                errors.append("Adresa e email-it nuk është e vlefshme.")
            if not password or len(password) < 6:
                errors.append("Fjalëkalimi duhet të ketë të paktën 6 karaktere.")
            if password and confirm and password != confirm:
                errors.append("Fjalëkalimet nuk përputhen.")

            if errors:
                for err in errors:
                    st.error(err)
            else:
                ok, msg = register(
                    username=username.strip(),
                    email=email.strip().lower(),
                    password=password,
                    role="student",
                )
                if ok:
                    st.success(
                        f"Llogaria '{username.strip()}' u krijua me sukses! "
                        "Mund të hyni tani."
                    )
                    # Fshi flamurin e regjistrimit → kthehu te login
                    if "_show_register" in st.session_state:
                        del st.session_state["_show_register"]
                    st.balloons()
                    # Auto-login pas regjistrimit
                    user = login(username.strip(), password)
                    if user:
                        st.session_state.user = user
                    st.rerun()
                else:
                    st.error(msg)

        # ── Kthehu te login ──────────────────────────────────────────────
        st.markdown(
            "<div style='text-align:center;margin-top:16px;color:#8a8fa8;font-size:.88rem;'>"
            "Keni tashmë llogari?</div>",
            unsafe_allow_html=True,
        )
        if st.button("← Kthehu te Hyrja", use_container_width=True, key="back_login"):
            if "_show_register" in st.session_state:
                del st.session_state["_show_register"]
            st.rerun()


# ═══════════════════════════════════════════════════════════
# 5. Faqja e Chat-it
# ═══════════════════════════════════════════════════════════
def _esc(t: str) -> str:
    return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")

def _col_label(c: str) -> str:
    return {"finance":"Finance DB","informatike_biznesi":"Info. Biz. DB",
            "personal_docs":"Dokumentet e mia"}.get(c, c)

def _page_chat(username: str, role: str) -> None:
    """Chat page for STUDENTS — all three sources available."""
    vs   = _vector_store()
    ret  = _retriever()
    llm  = _llm()
    hist = _history(username)

    st.markdown("## Bisedo me Asistentin")
    st.caption("Bëni pyetje bazuar në burimet e zgjedhura.")

    st.markdown("#### Zgjidhni burimet")
    c1, c2, c3 = st.columns(3)
    use_fin  = c1.checkbox("Finance DB",          value=True)
    use_info = c2.checkbox("Informatike Biznesi", value=True)
    pers_src = vs.list_sources(COL_PERSONAL, username=username)
    use_pers = c3.checkbox("Dokumentet e mia", value=False,
                           disabled=not pers_src,
                           help="Ngarkoni dokumente fillimisht." if not pers_src
                                else "Kërko në dokumentet tuaja.")

    if not any([use_fin, use_info, use_pers]):
        st.warning("Zgjidhni të paktën një burim.")

    st.divider()
    _render_chat_history(hist, username)
    st.divider()

    ci2, cc = st.columns([5, 1])
    question = ci2.chat_input("Shkruani pyetjen tuaj...")
    if cc.button("New Chat", help="Fillo bisedë të re"):
        hist.clear()
        for k in list(st.session_state.keys()):
            if k.startswith("ck_"):
                del st.session_state[k]
        st.rerun()

    if question:
        if not any([use_fin, use_info, use_pers]):
            st.error("Zgjidhni të paktën një burim."); return

        hist.add_user(question)
        active_cols = (
            (["finance"] if use_fin else []) +
            (["informatike_biznesi"] if use_info else []) +
            (["personal_docs"] if use_pers else [])
        )
        _log_question(question, username, active_cols, role=role)

        with st.spinner("Duke kërkuar dhe gjeneruar..."):
            chunks = ret.retrieve(
                question,
                use_finance=use_fin,
                use_informatike=use_info,
                use_personal=use_pers,
                username=username,
            )
            answer = llm.answer(
                question,
                [c.text for c in chunks],
                chat_history=hist.messages[:-1],
            ) if chunks else "Nuk u gjet asnjë dokument relevant. Provoni burime të tjera."

        hist.add_ai(answer)
        st.session_state[f"ck_{len(hist.messages)}"] = chunks
        st.rerun()


def _page_chat_admin(username: str, role: str) -> None:
    """Restricted chat for ADMINS — only their own base collection."""
    ret  = _retriever()
    llm  = _llm()
    hist = _history(username)

    if role == ROLE_FIN:
        my_col    = COL_FINANCE
        col_label = "Finance DB"
    else:
        my_col    = COL_INFORMATIKE
        col_label = "Informatike Biznesi DB"

    st.markdown("## Bisedo me Asistentin")
    st.caption(f"Si administrator keni akses vetëm te baza juaj: **{col_label}**.")
    st.info(f"Pyetjet tuaja kërkohen ekskluzivisht në **{col_label}**.")

    st.divider()
    _render_chat_history(hist, username)
    st.divider()

    ci2, cc = st.columns([5, 1])
    question = ci2.chat_input("Shkruani pyetjen tuaj...")
    if cc.button("New Chat", help="Fillo bisedë të re"):
        hist.clear()
        for k in list(st.session_state.keys()):
            if k.startswith("ck_"):
                del st.session_state[k]
        st.rerun()

    if question:
        hist.add_user(question)
        _log_question(question, username, [my_col], role=role)

        with st.spinner("Duke kërkuar dhe gjeneruar..."):
            use_fin  = (role == ROLE_FIN)
            use_info = (role == ROLE_INFO)
            chunks = ret.retrieve(
                question,
                use_finance=use_fin,
                use_informatike=use_info,
                use_personal=False,
                username=username,
            )
            if chunks:
                chunks = [c for c in chunks if c.collection == my_col]
            answer = llm.answer(
                question,
                [c.text for c in chunks],
                chat_history=hist.messages[:-1],
            ) if chunks else "Nuk u gjet asnjë dokument relevant në bazën tuaj."

        hist.add_ai(answer)
        st.session_state[f"ck_{len(hist.messages)}"] = chunks
        st.rerun()


def _render_chat_history(hist: FileChatMessageHistory, username: str) -> None:
    msgs = hist.for_display()
    if not msgs:
        st.markdown("""
        <div style="text-align:center;padding:40px;color:#8a8fa8;">
            <div style="font-size:3rem;">*</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.15rem;margin-top:10px;">
                Filloni një bisedë të re
            </div>
        </div>""", unsafe_allow_html=True)
        return

    for i, m in enumerate(msgs):
        if m["role"] == "user":
            st.markdown(
                f'<div class="chat-u"><strong>Ju</strong><br>{_esc(m["content"])}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-a"><strong>Asistenti</strong><br>{_esc(m["content"])}</div>',
                unsafe_allow_html=True,
            )
            chunks: list[Chunk] | None = st.session_state.get(f"ck_{i+1}")
            if chunks:
                with st.expander(f"{min(len(chunks), 3)} burime"):
                    for ci, ch in enumerate(chunks[:3]):
                        ca, cb = st.columns([4, 1])
                        score_pct = (
                            f"{ch.rerank_score:.0%}"
                            if ch.rerank_score > 0
                            else f"{max(0, 1 - ch.distance):.0%}"
                        )
                        ca.markdown(
                            f'<span class="src-tag">{ch.source}</span>'
                            f'<span class="src-tag">{_col_label(ch.collection)}</span>'
                            + (
                                f'<span class="src-tag">f.{ch.metadata.get("page_number","?")}</span>'
                                if ch.metadata.get("page_number") else ""
                            )
                            + f'<div style="font-size:.85rem;background:#f9f9f9;border-left:3px solid'
                              f' #c9973e;padding:8px;border-radius:0 6px 6px 0;margin-top:4px;">'
                              f'{_esc(ch.text[:300])}{"..." if len(ch.text) > 300 else ""}</div>',
                            unsafe_allow_html=True,
                        )
                        cb.metric(f"#{ci+1}", score_pct)


# ═══════════════════════════════════════════════════════════
# 6. Faqja e ngarkimit të dokumenteve personale
# ═══════════════════════════════════════════════════════════
def _page_upload(username: str) -> None:
    vs       = _vector_store()
    existing = vs.list_sources(COL_PERSONAL, username=username)

    st.markdown("## Dokumentet e mia personale")
    st.caption(f"Deri në **{MAX_PERSONAL_DOCS} dokumente** (PDF, DOCX, PPTX). Vetëm ju i shihni.")

    c1, c2 = st.columns(2)
    c1.metric("Të ngarkuara",    len(existing))
    c2.metric("Hapësirë mbetur", MAX_PERSONAL_DOCS - len(existing))

    st.divider()
    st.markdown("#### Ngarko dokument të ri")
    up = st.file_uploader("Zgjidh skedar", type=["pdf","docx","pptx"],
                          disabled=(len(existing) >= MAX_PERSONAL_DOCS))
    if up:
        if st.button("Indekso"):
            dest_dir = PERSONAL_DOCS_DIR / username
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / up.name
            dest.write_bytes(up.read())
            with st.spinner(f"Duke indeksuar '{up.name}'..."):
                ok, msg = vs.index_personal(dest, username, _parser(), _chunker(), _embedder())
            (st.success if ok else st.error)(f"{'OK' if ok else 'ERR'} {msg}")
            st.rerun()

    st.divider()
    st.markdown("#### Dokumentet e ngarkuara")
    if not existing:
        st.info("Nuk keni ngarkuar asnjë dokument ende.")
        return

    for src in existing:
        ca, cb = st.columns([5, 1])
        ca.markdown(f'<div class="card"><strong>{src}</strong></div>', unsafe_allow_html=True)
        if cb.button("Fshi", key=f"d_{src}"):
            vs.delete_source(COL_PERSONAL, src, username=username)
            fp = PERSONAL_DOCS_DIR / username / src
            if fp.exists(): fp.unlink()
            st.success(f"'{src}' u fshi."); st.rerun()


# ═══════════════════════════════════════════════════════════
# 7. Faqja e gjeneratorit të kuizit
# ═══════════════════════════════════════════════════════════
def _page_quiz(username: str) -> None:
    vs = _vector_store()
    st.markdown("## Gjenerator i Kuizit")
    st.caption(f"Zgjidhni një dokument personal → sistemi krijon **{QUIZ_QUESTIONS} pyetje** automatikisht.")

    srcs = vs.list_sources(COL_PERSONAL, username=username)
    if not srcs:
        st.info("Ngarkoni dokumente personale fillimisht (faqja **Dokumentet e mia**).")
        return

    sel = st.selectbox("Zgjidh dokumentin", srcs)
    if st.button("Gjenero Kuizin"):
        with st.spinner("Duke gjeneruar kuizin..."):
            try:
                pdf, text = _quiz_gen().generate(sel, username)
                st.session_state[f"qpdf_{username}"] = pdf
                st.session_state[f"qtxt_{username}"] = text
                st.session_state[f"qsrc_{username}"] = sel
                st.rerun()
            except Exception as e:
                st.error(f"Gabim: {e}")

    if f"qtxt_{username}" in st.session_state:
        st.divider()
        st.markdown(f"#### Kuizi: *{st.session_state[f'qsrc_{username}']}*")
        st.download_button(
            "Shkarko TXT",
            data=st.session_state[f"qpdf_{username}"],
            file_name=f"kuiz_{sel.replace(' ','_')}.txt",
            mime="text/plain",
        )
        with st.expander("Shiko kuizin", expanded=True):
            for line in st.session_state[f"qtxt_{username}"].split("\n"):
                s = line.strip()
                if not s:                             st.write("")
                elif s.startswith("Q") and "." in s[:4]: st.markdown(f"**{s}**")
                elif "ANSWER KEY" in s.upper():       st.markdown("---\n### Answer Key")
                elif s.startswith("Answer:"):         st.markdown(f":green[{s}]")
                else:                                 st.write(s)


# ═══════════════════════════════════════════════════════════
# 8. Faqja admin — Dashboard + Menaxhim + Chat
# ═══════════════════════════════════════════════════════════
def _page_admin(username: str, role: str) -> None:
    vs = _vector_store()
    if role == ROLE_FIN:
        col, label, data_dir, color = COL_FINANCE,     "Finance",             FINANCE_DIR,      "#c9973e"
    elif role == ROLE_INFO:
        col, label, data_dir, color = COL_INFORMATIKE, "Informatike Biznesi", INFORMATIKE_DIR, "#4f77be"
    else:
        st.error("Nuk keni leje admin."); return

    tab_dash, tab_docs, tab_chat = st.tabs([
        "Dashboard",
        "Menaxho Dokumentet",
        "Pyetje mbi të dhënat bazë",
    ])

    with tab_dash:
        st.markdown(f"## Dashboard — <span style='color:{color}'>{label}</span>",
                    unsafe_allow_html=True)
        st.caption("Statistika të sistemit RAG në kohë reale.")
        st.divider()

        existing   = vs.list_sources(col)
        chunk_cnt  = vs.count(col)
        top_qs     = _top_questions(col_filter=col, n=5)
        active_usr = _unique_active_users(col_filter=col)

        total_qs_col = len([
            q for q in _load_stats().get("questions", [])
            if col in q.get("collections", [])
            and q.get("role", "") not in (ROLE_FIN, ROLE_INFO)
            and COL_PERSONAL not in q.get("collections", [])
        ])

        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.markdown(f"""
        <div class="stat-card">
            <h4>Dokumentet</h4>
            <div class="stat-number">{len(existing)}</div>
            <div class="stat-label">skedarë të indeksuar</div>
        </div>""", unsafe_allow_html=True)
        r1c2.markdown(f"""
        <div class="stat-card">
            <h4>Chunk-et</h4>
            <div class="stat-number">{chunk_cnt:,}</div>
            <div class="stat-label">fragmente teksti në bazë</div>
        </div>""", unsafe_allow_html=True)
        r1c3.markdown(f"""
        <div class="stat-card">
            <h4>Pyetje Totale</h4>
            <div class="stat-number">{total_qs_col}</div>
            <div class="stat-label">pyetje studentësh drejtuar kësaj baze</div>
        </div>""", unsafe_allow_html=True)
        r1c4.markdown(f"""
        <div class="stat-card">
            <h4>Studentë Aktivë</h4>
            <div class="stat-number">{active_usr}</div>
            <div class="stat-label">kanë pyetur nga kjo bazë</div>
        </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("### Pyetjet më të shpeshta")
        st.caption("Vetëm pyetjet e studentëve — pyetjet e testimit nga adminët nuk shfaqen.")
        if top_qs:
            for q_text, count in top_qs:
                short = q_text[:90] + ("…" if len(q_text) > 90 else "")
                st.markdown(
                    f'<div class="q-row">'
                    f'<strong>{_esc(short)}</strong>'
                    f'<div class="q-meta">Pyetur {count} herë</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Ende nuk ka pyetje të regjistruara nga studentët për këtë bazë.")
        st.divider()


    with tab_docs:
        existing = vs.list_sources(col)
        st.markdown(f"## Menaxhim Dokumentesh — <span style='color:{color}'>{label}</span>",
                    unsafe_allow_html=True)
        st.caption("Ngarkoni, shikoni dhe fshini dokumentet e bazës suaj.")

        st.markdown("#### Dokumentet e indeksuara")
        if existing:
            for src in existing:
                ca, cb = st.columns([5, 1])
                ca.markdown(
                    f'<div class="card"><strong>{_esc(src)}</strong></div>',
                    unsafe_allow_html=True,
                )
                if cb.button("🗑 Fshi", key=f"adel_{src}"):
                    vs.delete_source(col, src)
                    f = data_dir / src
                    if f.exists():
                        f.unlink()
                    st.success(f"✅ '{src}' u fshi nga baza.")
                    st.rerun()
        else:
            st.info("Nuk ka dokumente të indeksuara ende.")

        st.divider()
        st.markdown("#### Ngarko leksion të ri (PDF)")
        up    = st.file_uploader("Zgjidh PDF", type=["pdf"], key="admin_uploader")
        force = st.checkbox("Ri-indekso nëse ekziston", value=False)
        if up and st.button("Indekso"):
            dest = data_dir / up.name
            dest.write_bytes(up.read())
            with st.spinner(f"Duke indeksuar '{up.name}'..."):
                ok, msg = vs.index_file(dest, col, _parser(), _chunker(), _embedder(), force=force)
            (st.success if ok else st.error)(f"{'✅' if ok else '❌'} {msg}")
            st.rerun()

        st.divider()
        if st.button("Ri-indekso të gjithë direktorinë"):
            with st.spinner("Duke indeksuar..."):
                results = vs.index_directory(data_dir, col, _parser(), _chunker(), _embedder())
            for fname, ok, msg in results:
                (st.success if ok else st.error)(f"{'✅' if ok else '❌'} {msg}")

    with tab_chat:
        _page_chat_admin(username, role)


# ═══════════════════════════════════════════════════════════
# 9. Main — routing + sidebar
# ═══════════════════════════════════════════════════════════
def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    _llm()
    _startup_index()

    # ── Auth routing ─────────────────────────────────────────────────────
    if "user" not in st.session_state:
        if st.session_state.get("_show_register"):
            _page_register()
        else:
            _page_login()
        return

    user     = st.session_state.user
    username = user["username"]
    role     = user["role"]
    email    = user.get("email", "")
    vs       = _vector_store()

    with st.sidebar:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:16px 4px 10px">
            <div class="avatar">{username[0].upper()}</div>
            <div>
                <div style="font-weight:600">{username}</div>
                <div style="font-size:0.72rem;color:rgba(245,240,232,0.65);margin-top:1px;">{email}</div>
                <div class="role-badge rb-{role}">{role.replace('_',' ').title()}</div>
            </div>
        </div>""", unsafe_allow_html=True)
        st.divider()

        if is_admin(role):
            pages = ["Admin"]
        else:
            pages = ["Chat", "Dokumentet e mia", "Quiz Generator"]

        page = st.radio("Nav", pages, label_visibility="collapsed")
        st.divider()

        st.caption("Gjendja e bazave")
        m1, m2 = st.columns(2)
        m1.metric("Finance",    vs.count(COL_FINANCE))
        m2.metric("Info.Biz.", vs.count(COL_INFORMATIKE))
        if not is_admin(role):
            st.metric("Personale",
                      len(vs.list_sources(COL_PERSONAL, username=username)))
        st.divider()

        if st.button("Dil", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # ── Routing ──────────────────────────────────────────────────────────
    if page == "Chat":
        _page_chat(username, role)
    elif page == "Dokumentet e mia":
        _page_upload(username)
    elif page == "Quiz Generator":
        _page_quiz(username)
    elif page == "Admin":
        _page_admin(username, role)


if __name__ == "__main__":
    main()