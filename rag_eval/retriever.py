"""A dependency-free TF-IDF retriever used by the sample RAG pipeline."""

from __future__ import annotations

import math
import re
from collections import Counter

from .models import Document

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "in", "into", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "what", "when", "where", "which", "who", "why", "with",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str, *, remove_stopwords: bool = False) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        return [token for token in tokens if token not in STOPWORDS]
    return tokens


def content_tokens(text: str) -> list[str]:
    return tokenize(text, remove_stopwords=True)


class TfidfRetriever:
    """Cosine-similarity retriever over an in-memory document list."""

    def __init__(self, documents: list[Document]):
        if not documents:
            raise ValueError("At least one document is required.")
        self.documents = documents
        tokenized = [Counter(content_tokens(document.text)) for document in documents]
        document_frequency: Counter[str] = Counter()
        for counts in tokenized:
            document_frequency.update(counts.keys())
        total = len(documents)
        self._idf = {
            token: math.log((1 + total) / (1 + frequency)) + 1.0
            for token, frequency in document_frequency.items()
        }
        self._vectors = [self._vectorize_counts(counts) for counts in tokenized]

    def _vectorize_counts(self, counts: Counter[str]) -> dict[str, float]:
        return {token: count * self._idf.get(token, 1.0) for token, count in counts.items()}

    def vectorize(self, text: str) -> dict[str, float]:
        return self._vectorize_counts(Counter(content_tokens(text)))

    @staticmethod
    def cosine(left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        dot = sum(value * right.get(token, 0.0) for token, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def retrieve(self, question: str, top_k: int = 3) -> list[Document]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        query_vector = self.vectorize(question)
        ranked = sorted(
            range(len(self.documents)),
            key=lambda index: self.cosine(query_vector, self._vectors[index]),
            reverse=True,
        )
        return [self.documents[index] for index in ranked[:top_k]]
