"""
core/chat_history.py — FileChatMessageHistory: historiku i bisedës si skedar JSON.

Çdo përdorues ka skedarin e vet: chat_histories/<username>.json
Struktura: [{"role": "user"|"assistant", "content": "...", "timestamp": "..."}, ...]
"""

import json
from datetime import datetime
from pathlib import Path
from config import CHAT_HISTORY_DIR


class FileChatMessageHistory:
    def __init__(self, username: str):
        self._path     = CHAT_HISTORY_DIR / f"{username}.json"
        self._messages = self._load()

    def _load(self) -> list[dict]:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                return data if isinstance(data, list) else []
            except json.JSONDecodeError:
                return []
        return []

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(self._messages, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def add(self, role: str, content: str) -> None:
        self._messages.append({
            "role": role, "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self._save()

    def add_user(self, content: str)     -> None: self.add("user",      content)
    def add_ai(self,   content: str)     -> None: self.add("assistant", content)

    def clear(self) -> None:
        self._messages = []
        self._save()

    @property
    def messages(self) -> list[dict]:
        return list(self._messages)

    def for_display(self) -> list[dict]:
        """Listë {role, content} për shfaqje në Streamlit."""
        return [{"role": m["role"], "content": m["content"]} for m in self._messages]