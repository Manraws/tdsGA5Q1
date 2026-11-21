import json
import math
import os
import re
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
PASSAGES_FILE = Path(os.getenv("PASSAGES_FILE", str(BASE_DIR / "passages.json")))

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.+#-][a-z0-9]+)*", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def load_passages():
    if not PASSAGES_FILE.exists():
        raise RuntimeError(f"Passage file not found: {PASSAGES_FILE}")
    with PASSAGES_FILE.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list) or not rows:
        raise RuntimeError("passages.json must contain a non-empty JSON list.")

    cleaned = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("Each passage must be an object with id, domain, and text.")
        pid = row.get("id")
        domain = row.get("domain", "Unknown")
        text = row.get("text")
        if not isinstance(pid, int) or not isinstance(text, str) or not text.strip():
            raise RuntimeError("Each passage needs an integer id and non-empty text.")
        if pid in seen:
            raise RuntimeError(f"Duplicate passage id: {pid}")
        seen.add(pid)
        cleaned.append({
            "id": pid,
            "domain": str(domain),
            "text": text.strip(),
            "tokens": tokenize(text),
        })
    return cleaned


PASSAGES = load_passages()
N = len(PASSAGES)
AVGDL = sum(len(p["tokens"]) for p in PASSAGES) / max(N, 1)
DOC_FREQ = {}
for p in PASSAGES:
    for term in set(p["tokens"]):
        DOC_FREQ[term] = DOC_FREQ.get(term, 0) + 1


def bm25_score(query_tokens, passage, k1=1.5, b=0.75):
    if not query_tokens:
        return 0.0
    tf = {}
    for term in passage["tokens"]:
        tf[term] = tf.get(term, 0) + 1

    score = 0.0
    dl = len(passage["tokens"])
    for term in set(query_tokens):
        freq = tf.get(term, 0)
        if not freq:
            continue
        df = DOC_FREQ.get(term, 0)
        idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
        denom = freq + k1 * (1.0 - b + b * dl / max(AVGDL, 1.0))
        score += idf * (freq * (k1 + 1.0)) / denom

    # Small exact-phrase boost helps distinguish passages containing the
    # same individual words in a different context.
    normalized_query = " ".join(query_tokens)
    normalized_text = " ".join(passage["tokens"])
    if len(query_tokens) >= 2 and normalized_query in normalized_text:
        score += 1.0

    # Reward broad coverage of the question without allowing it to dominate BM25.
    coverage = len(set(query_tokens) & set(passage["tokens"])) / max(len(set(query_tokens)), 1)
    score += 0.5 * coverage
    return score


def make_answer(question: str, ranked):
    if not ranked:
        return None

    # Select the most relevant sentences from the highest-ranked passages.
    q_terms = set(tokenize(question))
    chosen = []
    seen = set()

    for item in ranked[:2]:
        sentences = [s.strip() for s in SENTENCE_RE.split(item["text"]) if s.strip()]
        scored = []
        for idx, sentence in enumerate(sentences):
            terms = set(tokenize(sentence))
            overlap = len(q_terms & terms)
            if overlap:
                scored.append((overlap, -idx, sentence))
        scored.sort(reverse=True)
        for _, _, sentence in scored[:2]:
            key = sentence.lower()
            if key not in seen:
                seen.add(key)
                chosen.append(sentence)
        if len(chosen) >= 3:
            break

    if not chosen:
        return ranked[0]["text"]
    return " ".join(chosen[:3])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class QueryResponse(BaseModel):
    answer: Optional[str]
    sources: List[int]
    scores: List[float]
    confidence: float


app = FastAPI(
    title="RAG Colosseum API",
    description="Retrieval API over the supplied 50-passage knowledge base.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "RAG Colosseum API",
        "status": "ok",
        "passage_count": len(PASSAGES),
        "docs": "/docs",
        "query_endpoint": "POST /query",
    }


@app.get("/health")
def health():
    return {"status": "ok", "passage_count": len(PASSAGES)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    q_tokens = tokenize(req.question)
    if not q_tokens:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.0)

    ranked_raw = []
    for passage in PASSAGES:
        raw = bm25_score(q_tokens, passage)
        if raw > 0:
            # Smoothly maps the non-negative BM25 score into [0, 1].
            score = raw / (raw + 5.0)
            ranked_raw.append((score, passage))

    ranked_raw.sort(key=lambda x: (-x[0], x[1]["id"]))
    selected = ranked_raw[:req.top_k]

    if not selected:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.0)

    scores = [round(float(s), 6) for s, _ in selected]
    sources = [p["id"] for _, p in selected]
    ranked_passages = [p for _, p in selected]

    # A small floor avoids returning a confident answer for incidental matches.
    top_score = scores[0]
    answer = make_answer(req.question, ranked_passages) if top_score >= 0.08 else None
    confidence = top_score if answer is not None else 0.0

    return QueryResponse(
        answer=answer,
        sources=sources,
        scores=scores,
        confidence=round(float(confidence), 6),
    )
