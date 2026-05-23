"""
core/auth.py — Autentifikim me SQLite, fjalëkalime plaintext.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from config import USER_DATA_DIR, DEFAULT_USERS, ROLE_FIN, ROLE_INFO, UPLOAD_PERMS

DB_PATH = USER_DATA_DIR / "users.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                email      TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                password   TEXT    NOT NULL,
                role       TEXT    NOT NULL DEFAULT 'student',
                created_at TEXT    NOT NULL
            )
        """)
        conn.commit()

        for uname, info in DEFAULT_USERS.items():
            exists = conn.execute(
                "SELECT 1 FROM users WHERE username = ?", (uname,)
            ).fetchone()
            if not exists:
                conn.execute(
                    """INSERT INTO users (username, email, password, role, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        uname,
                        f"{uname}@demo.edu.al",
                        info["password"],
                        info["role"],
                        datetime.now().isoformat(),
                    ),
                )
        conn.commit()


def login(username: str, password: str) -> dict | None:
    _init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT username, email, password, role FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if row and row["password"] == password:
        return {"username": row["username"], "email": row["email"], "role": row["role"]}
    return None


def register(username: str, email: str, password: str, role: str = "student") -> tuple[bool, str]:
    _init_db()
    import re
    u = username.strip()
    e = email.strip().lower()

    if not u or len(u) < 3:
        return False, "Emri duhet të ketë të paktën 3 karaktere."
    if len(u) > 30:
        return False, "Emri nuk mund të kalojë 30 karaktere."
    if not re.match(r'^[a-zA-Z0-9_]+$', u):
        return False, "Emri mund të përmbajë vetëm shkronja, numra dhe '_'."
    if not e or "@" not in e or "." not in e.split("@")[-1]:
        return False, "Email-i nuk është i vlefshëm."
    if len(password) < 6:
        return False, "Fjalëkalimi duhet të ketë të paktën 6 karaktere."
    if role not in ("student",):
        return False, "Nuk lejohet regjistrimi me këtë rol."

    try:
        with _connect() as conn:
            conn.execute(
                """INSERT INTO users (username, email, password, role, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (u, e, password, role, datetime.now().isoformat()),
            )
            conn.commit()
        return True, ""
    except sqlite3.IntegrityError as exc:
        msg = str(exc)
        if "username" in msg:
            return False, f"Emri '{u}' është i zënë."
        if "email" in msg:
            return False, f"Email-i '{e}' është regjistruar tashmë."
        return False, "Gabim gjatë regjistrimit."


def can_upload(role: str, collection: str) -> bool:
    return role in UPLOAD_PERMS.get(collection, [])


def is_admin(role: str) -> bool:
    return role in {ROLE_FIN, ROLE_INFO}


_init_db()