"""End-to-end orchestration for retrieval, answer selection, and scoring."""

from __future__ import annotations

from dataclasses import dataclass

from .backends import SCORE_FIELDS, BackendUnavailable, run_deepeval, run_ragas
from .data import load_documents, load_qa_records, validate_relevance_labels
from .generation import extractive_answer
from .metrics import evaluate_case_locally
from .models import EvaluationCase
from .retriever import TfidfRetriever


@dataclass
class EvaluationOutput:
    rows: list[dict[str, object]]
    backend_requested: str
    backend_used: str
    notes: list[str]


def build_cases(
    documents_path: str,
    qa_path: str,
    *,
    top_k: int = 3,
    answer_source: str = "provided",
) -> list[EvaluationCase]:
    documents = load_documents(documents_path)
    records = load_qa_records(qa_path)
    validate_relevance_labels(records, documents)
    retriever = TfidfRetriever(documents)

    cases: list[EvaluationCase] = []
    for record in records:
        retrieved = retriever.retrieve(record.question, top_k=top_k)
        if answer_source == "extractive":
            answer = extractive_answer(record.question, retrieved)
        elif answer_source == "provided":
            answer = record.response or extractive_answer(record.question, retrieved)
        else:
            raise ValueError("answer_source must be 'provided' or 'extractive'.")
        cases.append(
            EvaluationCase(record=record, answer=answer, retrieved_documents=tuple(retrieved))
        )
    return cases


def evaluate_cases(
    cases: list[EvaluationCase],
    *,
    backend: str = "local",
    require_framework: bool = False,
) -> EvaluationOutput:
    local_scores = {case.record.case_id: evaluate_case_locally(case) for case in cases}
    framework_scores: dict[str, dict[str, float]] = {}
    notes: list[str] = []
    backend_used = "local"

    if backend in {"ragas", "deepeval"}:
        runner = run_ragas if backend == "ragas" else run_deepeval
        try:
            framework_scores = runner(cases)
            backend_used = backend
            notes.append(f"Framework scores supplied by {backend}.")
        except BackendUnavailable as exc:
            if require_framework:
                raise
            notes.append(f"{backend} unavailable: {exc} Local heuristic scores were used instead.")

    rows: list[dict[str, object]] = []
    for case in cases:
        scores = dict(local_scores[case.record.case_id])
        if framework_scores:
            # Framework output replaces the dimensions it actually measures.
            # Answer correctness remains the local token-F1 diagnostic.
            for field in SCORE_FIELDS:
                if field in framework_scores.get(case.record.case_id, {}):
                    scores[field] = round(framework_scores[case.record.case_id][field], 4)
            weighted = (
                0.30 * scores["faithfulness"]
                + 0.20 * scores["answer_relevancy"]
                + 0.20 * scores["context_precision"]
                + 0.15 * scores["context_recall"]
                + 0.15 * scores["answer_correctness_f1"]
            )
            scores["overall"] = round(max(0.0, min(1.0, weighted)), 4)
        rows.append(
            {
                "case_id": case.record.case_id,
                "question": case.record.question,
                "answer": case.answer,
                "reference_answer": case.record.reference_answer,
                "retrieved_doc_ids": ";".join(case.retrieved_doc_ids),
                **scores,
            }
        )
    return EvaluationOutput(
        rows=rows,
        backend_requested=backend,
        backend_used=backend_used,
        notes=notes,
    )
