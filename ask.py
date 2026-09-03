"""
Part 1 — Document Q&A assistant for MGC sales staff.

Loads the three MGC documents from docs/, sends them (whole) to an LLM
together with the user's question, and prints a grounded answer with sources.

Why no vector database: there are only three short documents (~180 lines
total). They fit comfortably in the model's context window, so retrieval
adds complexity without adding accuracy. If the document set grew to
dozens of files, I would add chunking + embedding retrieval in front of
the same prompt.

Why no SDK: Gemini exposes an OpenAI-compatible REST endpoint; one POST with
`requests` is the whole integration, and there is nothing hidden to explain.

Usage:
    export GEMINI_API_KEY=...        # free key from aistudio.google.com/apikey
    python ask.py "What's the transfer fee?"
    python ask.py                    # interactive loop
"""

import os
import sys
import time
from pathlib import Path

import requests

DOCS_DIR = Path(__file__).parent / "docs"
API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
MODEL = "gemini-flash-latest"  # alias: always points at the current Flash model

SYSTEM_PROMPT = """\
You are an internal assistant for the MGC Developments sales team.
You answer questions using ONLY the MGC documents provided below.

Rules — these are strict:

1. GROUNDING. Answer only from the documents. Never use outside knowledge
   about real estate, prices, or MGC. Never estimate or guess a number.

2. SOURCES. End every answer with a "Source:" line naming the document(s)
   and section(s) the answer came from.

3. MISSING INFORMATION. If the documents do not contain the answer, say
   exactly that and tell the salesperson to refer the query to the
   marketing manager. Do not attempt a partial or estimated answer.

4. CONFLICTS. If two documents give different values for the same thing,
   DO NOT pick one. State both values, name both sources, point out that
   they disagree, and advise the salesperson to confirm with management
   before quoting a figure to a customer.

5. UNCONFIRMED ITEMS. If a document says something is unconfirmed, ongoing,
   or not finalised, report it that way. Never present it as a done deal.

6. CALCULATIONS. When a price needs base price plus premiums or discounts,
   show the calculation step by step (base, each premium %, final figure)
   so the salesperson can verify it.

7. It is always better to say "I don't have that information" than to give
   a confident wrong answer. A wrong number costs MGC a sale.

Answer in clear, short English a salesperson can read out on a call.
"""


def load_documents() -> str:
    """Concatenate all docs with clear source markers."""
    blocks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        blocks.append(f"===== DOCUMENT: {path.name} =====\n{text}")
    if not blocks:
        sys.exit(f"No documents found in {DOCS_DIR}")
    return "\n\n".join(blocks)


def ask(question: str, documents: str) -> str:
    payload = {
        "model": MODEL,
        "temperature": 0,  # deterministic — a lookup tool, not a writer
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{documents}\n\n===== QUESTION =====\n{question}",
            },
        ],
    }
    headers = {"Authorization": f"Bearer {os.environ['GEMINI_API_KEY']}"}

    # The free tier throws transient 429/5xx at peak times — retry a few
    # times with a growing pause instead of failing on the first hiccup.
    for attempt in range(4):
        response = requests.post(
            API_URL, headers=headers, json=payload,
            timeout=180,  # free-tier responses can be slow at peak times
        )
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            wait = 5 * (attempt + 1)
            print(f"  (server busy — {response.status_code}, retrying in {wait}s)")
            time.sleep(wait)
            continue
        break
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("Set GEMINI_API_KEY first (free key: https://aistudio.google.com/apikey)")
    documents = load_documents()

    if len(sys.argv) > 1:  # single question mode
        print(ask(" ".join(sys.argv[1:]), documents))
        return

    print("MGC document assistant — type a question, or 'q' to quit.")
    while True:
        question = input("\n? ").strip()
        if question.lower() in {"q", "quit", "exit", ""}:
            break
        print("\n" + ask(question, documents))


if __name__ == "__main__":
    main()
