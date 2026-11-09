"""Dataset loading and validation for JSON and CSV inputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .models import Document, QARecord


def _items(payload: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get(key, [])
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"Expected a JSON list of objects or an object containing '{key}'.")
    return payload


def load_documents(path: str | Path) -> list[Document]:
    """Load documents from a JSON list or ``{"documents": [...]}`` file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    documents: list[Document] = []
    seen: set[str] = set()
    for item in _items(payload, "documents"):
        doc_id = str(item.get("id") or item.get("doc_id") or "").strip()
        text = str(item.get("text") or item.get("content") or "").strip()
        if not doc_id or not text:
            raise ValueError("Every document needs a non-empty id/doc_id and text/content.")
        if doc_id in seen:
            raise ValueError(f"Duplicate document id: {doc_id}")
        seen.add(doc_id)
        documents.append(Document(doc_id=doc_id, title=str(item.get("title", "")), text=text))
    if not documents:
        raise ValueError("The document file contains no documents.")
    return documents


def _split_ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return tuple(part.strip() for part in str(value).replace(",", ";").split(";") if part.strip())


def _record_from_mapping(item: dict[str, Any]) -> QARecord:
    case_id = str(item.get("id") or item.get("case_id") or "").strip()
    question = str(item.get("question") or "").strip()
    reference = str(item.get("reference_answer") or item.get("ground_truth") or "").strip()
    if not case_id or not question or not reference:
        raise ValueError("Every QA record needs id/case_id, question, and reference_answer/ground_truth.")
    return QARecord(
        case_id=case_id,
        question=question,
        reference_answer=reference,
        response=str(item.get("response") or item.get("answer") or "").strip(),
        relevant_doc_ids=_split_ids(item.get("relevant_doc_ids")),
    )


def load_qa_records(path: str | Path) -> list[QARecord]:
    """Load QA records from JSON or CSV.

    JSON may be a list or ``{"qa_pairs": [...]}``. CSV columns are
    ``id,question,reference_answer,response,relevant_doc_ids``; separate
    multiple relevance labels with semicolons.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = [_record_from_mapping(item) for item in _items(payload, "qa_pairs")]
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            records = [_record_from_mapping(row) for row in csv.DictReader(handle)]
    else:
        raise ValueError("QA input must be a .json or .csv file.")
    if not records:
        raise ValueError("The QA file contains no records.")
    ids = [record.case_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("QA case ids must be unique.")
    return records


def validate_relevance_labels(records: Iterable[QARecord], documents: Iterable[Document]) -> None:
    known = {document.doc_id for document in documents}
    unknown = sorted({doc_id for record in records for doc_id in record.relevant_doc_ids} - known)
    if unknown:
        raise ValueError(f"Relevance labels reference unknown documents: {', '.join(unknown)}")
