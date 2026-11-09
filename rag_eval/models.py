"""Shared data models for the RAG evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A document available to the sample retriever."""

    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class QARecord:
    """One evaluation question and, optionally, a system response to grade."""

    case_id: str
    question: str
    reference_answer: str
    response: str = ""
    relevant_doc_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationCase:
    """A QA record after retrieval and answer selection."""

    record: QARecord
    answer: str
    retrieved_documents: tuple[Document, ...] = field(default_factory=tuple)

    @property
    def retrieved_contexts(self) -> list[str]:
        return [document.text for document in self.retrieved_documents]

    @property
    def retrieved_doc_ids(self) -> list[str]:
        return [document.doc_id for document in self.retrieved_documents]
