"""Small answer-generation helpers for the demonstration pipeline."""

from __future__ import annotations

import re

from .models import Document
from .retriever import TfidfRetriever

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.findall(text) if part.strip()]


def extractive_answer(question: str, documents: list[Document]) -> str:
    """Return the retrieved sentence most similar to the question.

    This is intentionally simple: it makes the sample pipeline runnable
    without an LLM while keeping the answer grounded in retrieved text.
    """

    sentences = [sentence for document in documents for sentence in split_sentences(document.text)]
    if not sentences:
        return ""
    scorer = TfidfRetriever(
        [Document(doc_id=str(index), title="", text=sentence) for index, sentence in enumerate(sentences)]
    )
    query = scorer.vectorize(question)
    return max(sentences, key=lambda sentence: scorer.cosine(query, scorer.vectorize(sentence)))
