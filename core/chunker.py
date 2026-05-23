"""
core/chunker.py — Chunker i zgjuar: ruan strukturën logjike të tekstit.

Strategjia (në rend prioriteti):
  1. Nëse teksti ka headers (tituj), ndaje sipas seksioneve — çdo seksion
     mbetet i plotë (nuk pritet mid-section).
  2. Brenda çdo seksioni, njeh bllqe speciale:
       - Lista të numëruara (supozimet, hapat, rregullat) → mbaj të bashkuar
       - Formula matematike                               → mbaj të bashkuar
       - Paragrafë të rëndomtë                           → ndaj me overlap
  3. Nëse nuk ka headers fare, kalo direkt te logjika e paragrafëve.

Kjo e ruan "Supozimet 1,2,3,4" bashkë në një chunk — problemi kryesor
që shkaktonte përgjigje të paplota.
"""

from __future__ import annotations
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP


# ── Regex patterns ─────────────────────────────────────────────────────────────

# Header: rresht i shkurtër (≤80 kar.) që fillon me shkronjë të madhe,
# nuk mbaron me pikë, dhe ndiqet nga tekst. Mbështet shqipen.
_HEADER_RE = re.compile(
    r'^([A-ZÇËÍ][A-ZÇËÍa-zçëí0-9 \-–:]{2,79})$',
    re.MULTILINE
)

# Rresht me numërim: "1.", "2.", "1)", "a)", etj.
_NUMBERED_RE = re.compile(r'^\s*(\d+[\.\)]|[a-zA-Z][\.\)])\s+\S')

# Formula: përmban =, ≈, ∑, ∫, shkronja greke, ose notacion matematikorë
_FORMULA_RE = re.compile(r'[=≈∑∫∂√±×÷αβγδεθλμσφΩ]|E\s*\(|Var\s*\(|[A-Z]\s*=\s*[A-Z]')

# Vijë bosh ose me vetëm hapësirë
_BLANK_RE = re.compile(r'^\s*$')


class TextChunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap duhet të jetë më i vogël se chunk_size.")
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── API publike ────────────────────────────────────────────────────────────

    def chunk(self, text: str) -> list[str]:
        """Kthe listën e chunk-eve inteligjentë për tekstin dhënë."""
        text = self._normalize(text)
        if not text:
            return []

        sections = self._split_sections(text)
        chunks: list[str] = []
        for sec_title, sec_body in sections:
            prefix = f"{sec_title}\n" if sec_title else ""
            chunks.extend(self._chunk_section(prefix, sec_body))

        return [c.strip() for c in chunks if c.strip()]

    def chunk_with_metadata(self, text: str, source: str = "",
                            extra_meta: dict | None = None) -> list[dict]:
        """Chunk-et me metadata (source, chunk_index, + extra)."""
        base = extra_meta or {}
        return [
            {"text": c, "chunk_index": i, "source": source, **base}
            for i, c in enumerate(self.chunk(text))
        ]

    # ── Normalizim ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        """Pastro karakteret e veçanta por RUAj rreshtat e rinj (janë strukturë)."""
        text = re.sub(r'\r\n?', '\n', text)        # Windows newlines
        text = re.sub(r'[\f\t]', ' ', text)         # tabs → hapësirë
        text = re.sub(r' {2,}', ' ', text)          # hapësira të dyfishta
        text = re.sub(r'\n{4,}', '\n\n\n', text)    # max 3 rreshta bosh radhazi
        return text.strip()

    # ── Ndarja në seksione ────────────────────────────────────────────────────

    def _split_sections(self, text: str) -> list[tuple[str, str]]:
        """
        Ndaj tekstin në (titull, trup) sipas header-ave.
        Nëse nuk ka headers → kthe [('' , i_gjithë_teksti)].
        """
        lines  = text.split('\n')
        header_idxs: list[int] = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if self._is_header(stripped, lines, i):
                header_idxs.append(i)

        if not header_idxs:
            return [('', text)]

        sections: list[tuple[str, str]] = []
        # tekst para header-it të parë
        if header_idxs[0] > 0:
            pre = '\n'.join(lines[:header_idxs[0]]).strip()
            if pre:
                sections.append(('', pre))

        for k, hi in enumerate(header_idxs):
            title = lines[hi].strip()
            end   = header_idxs[k + 1] if k + 1 < len(header_idxs) else len(lines)
            body  = '\n'.join(lines[hi + 1:end]).strip()
            sections.append((title, body))

        return sections

    @staticmethod
    def _is_header(line: str, lines: list[str], idx: int) -> bool:
        """
        Heuristikë e thjeshtë por e qëndrueshme për header-at.
        Kushtet:
          - Gjatësi 3–80 karaktere
          - Fillon me shkronjë të madhe (shqip/anglisht)
          - Nuk mbaron me pikë/presje/pikëpyetje/pikëçuditese
          - Nuk është numërim (p.sh. "1. diçka")
          - Nuk përmban simbole matematike ose = (formula)
          - Rreshti pasues (nëse ka) nuk është bosh
        """
        if not line or not (3 <= len(line) <= 80):
            return False
        if not re.match(r'^[A-ZÇËÍ]', line):
            return False
        if line[-1] in '.,:;?!':
            return False
        if re.match(r'^\d+[\.\)]\s', line):
            return False
        # Hiq formula dhe shprehje matematike — nuk janë headers
        if re.search(r'[=≈∑∫∂√±×÷αβγδεθλμσφΩ()]|[A-Z]\s*[*/]|σ|[ŶÂ]', line):
            return False
        # Nëse rreshti pasues ekziston dhe është bosh → jo header
        if idx + 1 < len(lines) and not lines[idx + 1].strip():
            return False
        return True

    # ── Chunk-im i seksionit ──────────────────────────────────────────────────

    def _chunk_section(self, prefix: str, body: str) -> list[str]:
        """
        Chunk-o trupin e një seksioni duke njohur blloqet speciale.

        Rregull kyç: blloqet 'special' (lista, formula) kurrë nuk priten.
        Nëse janë të mëdhenj (>chunk_size), mbaji si chunk të vetëm —
        informacioni i plotë është më i rëndësishëm se kufiri i madhësisë.
        """
        if not body.strip():
            return [prefix.strip()] if prefix.strip() else []

        blocks = self._extract_blocks(body)
        chunks: list[str] = []
        pending = prefix.strip()

        for btype, btext in blocks:
            candidate = (pending + '\n' + btext).strip() if pending else btext

            if btype == 'special':
                # Bllok special: kurrë nuk e presim — e mbajmë të plotë.
                # Nëse pending + special ka vend → bashkoji
                if len(candidate) <= self.chunk_size * 1.5:
                    pending = candidate
                else:
                    # Flush pending (pa bllokun special)
                    if pending:
                        chunks.extend(self._split_paragraph(pending))
                    # Special si chunk i vet (edhe nëse kalon chunk_size)
                    special_with_ctx = (prefix.strip() + '\n' + btext).strip() if prefix.strip() else btext
                    pending = special_with_ctx
            else:
                # Bllok normal
                if len(candidate) <= self.chunk_size:
                    pending = candidate
                else:
                    if pending:
                        chunks.extend(self._split_paragraph(pending))
                    if len(btext) > self.chunk_size:
                        ctx = prefix.strip()
                        chunks.extend(self._split_paragraph(
                            (ctx + '\n' + btext).strip() if ctx else btext
                        ))
                        pending = prefix.strip()
                    else:
                        pending = (prefix.strip() + '\n' + btext).strip() if prefix.strip() else btext

        if pending:
            chunks.extend(self._split_paragraph(pending))

        return chunks

    def _extract_blocks(self, text: str) -> list[tuple[str, str]]:
        """
        Ndaj tekstin në blloqe (special | normal).
        Blloku 'special' = listë e numëruar ose bllok formulash.

        RREGULL I RËNDËSISHËM: Nëse dy paragrafë radhazi formojnë një listë
        (p.sh. "1. ... 2." pastaj "E(Y/Xi)=... 3. ... 4."), bashkoji në një
        bllok të vetëm special — kjo ndodh shpesh në PDF-të akademike shqip.
        """
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

        # Faza 1: klasifiko çdo paragraf
        types: list[str] = []
        for para in paragraphs:
            lines          = para.split('\n')
            numbered_lines = [l for l in lines if _NUMBERED_RE.match(l)]
            formula_lines  = [l for l in lines if _FORMULA_RE.search(l)]

            if len(numbered_lines) >= 2:
                types.append('special')
            elif len(numbered_lines) == 1 and formula_lines:
                types.append('special')
            elif formula_lines:
                types.append('special')
            # Paragraf me një numërim të vetëm por i shkurtër (≤120 kar.)
            # → ka gjasa t'i përket listës → shëno si 'maybespecial'
            elif len(numbered_lines) == 1 and len(para) <= 200:
                types.append('maybespecial')
            else:
                types.append('normal')

        # Faza 2: bashko blloqet fqinje special/maybespecial
        blocks: list[tuple[str, str]] = []
        i = 0
        while i < len(paragraphs):
            t = types[i]
            if t in ('special', 'maybespecial'):
                # Grumbullo çdo special/maybespecial vijues
                merged = [paragraphs[i]]
                j = i + 1
                while j < len(paragraphs) and types[j] in ('special', 'maybespecial'):
                    merged.append(paragraphs[j])
                    j += 1
                combined = '\n'.join(merged)
                # Konsidero 'special' nëse ka ≥2 numërime ose formula
                total_numbered = sum(
                    1 for l in combined.split('\n') if _NUMBERED_RE.match(l)
                )
                has_formula = bool(_FORMULA_RE.search(combined))
                if total_numbered >= 2 or has_formula:
                    blocks.append(('special', combined))
                else:
                    # Ishte vetëm një numërim i izoluar → normal
                    blocks.append(('normal', combined))
                i = j
            else:
                blocks.append(('normal', paragraphs[i]))
                i += 1

        return blocks

    # ── Ndarja e paragrafëve të mëdhenj ────────────────────────────────────────

    def _split_paragraph(self, text: str) -> list[str]:
        """
        Ndaj një paragraf të madh në chunk-e me overlap.
        Strategjia: ndaj në fjali → mbush greedy.
        """
        if len(text) <= self.chunk_size:
            return [text]

        sentences = self._sentence_split(text)
        return self._pack_sentences(sentences)

    @staticmethod
    def _sentence_split(text: str) -> list[str]:
        """Ndaj tekstin në fjali (mbështet shqipen)."""
        # Ndaj pas pikëve/pikëçuditeses/pikëpyetjes kur vjen shkronjë e madhe
        parts = re.split(r'(?<=[.!?…])\s+(?=[A-ZÇËÍ])', text)
        sentences = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # Nëse fjalia është shumë e gjatë, ndaje në presje
            if len(p) > 600:
                sub = re.split(r'(?<=,)\s+', p)
                sentences.extend(s.strip() for s in sub if s.strip())
            else:
                sentences.append(p)
        return sentences or [text]

    def _pack_sentences(self, sentences: list[str]) -> list[str]:
        """Mbush greedy chunks + overlap nga fundi i chunk-ut të mëparshëm."""
        chunks, buf, buf_len = [], [], 0

        for sent in sentences:
            extra = len(sent) + (1 if buf else 0)
            if buf_len + extra > self.chunk_size and buf:
                chunk_text = ' '.join(buf)
                chunks.append(chunk_text)
                # Overlap: merr fund të chunk-ut të mëparshëm
                if self.chunk_overlap:
                    seed = chunk_text[-self.chunk_overlap:]
                    buf, buf_len = [seed], len(seed)
                else:
                    buf, buf_len = [], 0
            buf.append(sent)
            buf_len += extra

        if buf:
            chunks.append(' '.join(buf))

        return chunks
    