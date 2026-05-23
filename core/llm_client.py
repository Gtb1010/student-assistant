"""
core/llm_client.py — Klient Groq për Llama 3.
"""

import os
from groq import Groq
from config import GROQ_MODEL, GROQ_MAX_TOKENS, GROQ_TEMPERATURE

_SYSTEM = """
You are a concise academic assistant for Albanian university students.

LENGTH RULE — MOST IMPORTANT:
- Simple question (1 concept) → MAX 3 sentences. No exceptions.
- Conceptual question (explain X) → MAX 5 sentences.
- Complex question (compare X and Y) → MAX 8 sentences.
- NEVER repeat the same idea twice, even in different words.
- NEVER write a closing summary sentence.

CONTENT RULES:
- Use the provided context as your main source.
- Do NOT invent facts or sources.
- If no useful context: say "Nuk ka informacion të mjaftueshëm." then answer briefly from knowledge.

BANNED PHRASES (never use these or similar):
- "është i rëndësishëm"
- "mundëson që"
- "në përgjithësi"
- "duke mundësuar"
- Any sentence that restates what you already said

STYLE:
- Write like a professor giving a quick clear explanation, not an essay.
- Use bullet points or short numbered lists only when listing multiple items.
- Include a formula only if the question specifically needs one.

LANGUAGE: reply in the same language as the question (Albanian or English).
""".strip()

_RAG_TMPL = """
CONTEXT FROM COURSE MATERIALS:
{context}

---
STUDENT QUESTION:
{question}

Provide a complete, well-structured academic answer. Synthesize the context — do not list it.
""".strip()

_QUIZ_SYSTEM = """
You are a university professor creating a high-quality exam.

RULES FOR QUIZ CREATION:
- Each question must test a DIFFERENT concept
- Questions must be clear and unambiguous  
- Wrong answers (distractors) must be plausible but clearly wrong
- Do NOT repeat similar questions with different wording
- Cover: definitions, formulas, applications, comparisons
- Format EXACTLY:
  Q1. [Question text]
  A) [option]
  B) [option]
  C) [option]
  D) [option]

After all questions, add:
ANSWER KEY
Q1: [letter] - [brief reason]
Q2: [letter] - [brief reason]
...
""".strip()


class LLMClient:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise ValueError("GROQ_API_KEY nuk u gjet.")
        self._groq = Groq(api_key=key)

    def _call(self, messages: list[dict], temperature: float = GROQ_TEMPERATURE) -> str:
        r = self._groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=GROQ_MAX_TOKENS,
            temperature=temperature,
        )
        return r.choices[0].message.content.strip()

    @staticmethod
    def _ctx(chunks: list[str]) -> str:
        return "\n\n---\n\n".join(chunks) if chunks else "(No context provided.)"

    def answer(self, question: str, context_chunks: list[str],
               chat_history: list[dict] | None = None) -> str:
        messages = [{"role": "system", "content": _SYSTEM}]
        if chat_history:
            for m in chat_history[-6:]:
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({
            "role": "user",
            "content": _RAG_TMPL.format(
                context=self._ctx(context_chunks),
                question=question
            )
        })
        return self._call(messages, temperature=0.3)

    def generate_quiz(self, context_chunks: list[str], n: int = 5) -> str:
        content = self._ctx(context_chunks)
        prompt = f"""Based on this academic material, create exactly {n} multiple choice questions.

MATERIAL:
{content}

Requirements:
- Each question tests a DIFFERENT concept from the material
- Cover different topics: definitions, calculations, comparisons, applications
- Make distractors (wrong answers) plausible but clearly incorrect
- No two questions should test the same thing

{_QUIZ_SYSTEM}"""

        return self._call(
            [{"role": "user", "content": prompt}],
            temperature=0.7,
        )