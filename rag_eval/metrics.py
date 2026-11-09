"""Transparent local metric implementations.

The local backend is deliberately dependency-free. Its scores are
*heuristic proxies* inspired by common RAG evaluation dimensions; they are
not official Ragas or DeepEval scores. Use the framework backends when a
project needs model-graded metrics.
"""

from __future__ import annotations

from collections import Counter

from .generation import split_sentences
from .models import EvaluationCase
from .retriever import TfidfRetriever, content_tokens


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def token_f1(predicted: str, reference: str) -> float:
    predicted_tokens = Counter(content_tokens(predicted))
    reference_tokens = Counter(content_tokens(reference))
    if not predicted_tokens or not reference_tokens:
        return 0.0
    overlap = sum((predicted_tokens & reference_tokens).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(predicted_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    return 2 * precision * recall / (precision + recall)


def context_precision(case: EvaluationCase) -> float:
    """Fraction of retrieved documents carrying a relevance label.

    Without labels, this falls back to the fraction of retrieved documents
    that share at least two content tokens with the question.
    """

    retrieved_ids = case.retrieved_doc_ids
    if not retrieved_ids:
        return 0.0
    relevant = set(case.record.relevant_doc_ids)
    if relevant:
        return sum(doc_id in relevant for doc_id in retrieved_ids) / len(retrieved_ids)
    question_tokens = set(content_tokens(case.record.question))
    useful = sum(
        len(question_tokens & set(content_tokens(document.text))) >= 2
        for document in case.retrieved_documents
    )
    return useful / len(retrieved_ids)


def context_recall(case: EvaluationCase) -> float:
    """Fraction of labelled relevant documents that were retrieved.

    Without labels, reference-answer token coverage in the retrieved context
    is used as a visible fallback proxy.
    """

    relevant = set(case.record.relevant_doc_ids)
    if relevant:
        retrieved = set(case.retrieved_doc_ids)
        return len(relevant & retrieved) / len(relevant)
    reference_tokens = set(content_tokens(case.record.reference_answer))
    if not reference_tokens:
        return 0.0
    context = " ".join(case.retrieved_contexts)
    return len(reference_tokens & set(content_tokens(context))) / len(reference_tokens)


def faithfulness_proxy(case: EvaluationCase) -> float:
    """Fraction of answer sentences lexically supported by the context.

    A sentence is counted as supported when at least 35% of its content
    tokens occur in the retrieved context. This is a fast smoke-test signal,
    not a substitute for entailment- or LLM-based faithfulness grading.
    """

    sentences = split_sentences(case.answer)
    if not sentences:
        return 0.0
    context_token_set = set(content_tokens(" ".join(case.retrieved_contexts)))
    supported = 0
    for sentence in sentences:
        tokens = set(content_tokens(sentence))
        if tokens and len(tokens & context_token_set) / len(tokens) >= 0.35:
            supported += 1
    return supported / len(sentences)


def answer_relevancy_proxy(case: EvaluationCase) -> float:
    """Blend question-token coverage and TF-IDF cosine similarity."""

    question_tokens = set(content_tokens(case.record.question))
    answer_tokens = set(content_tokens(case.answer))
    if not question_tokens or not answer_tokens:
        return 0.0
    coverage = len(question_tokens & answer_tokens) / len(question_tokens)
    scorer = TfidfRetriever([case.retrieved_documents[0]]) if case.retrieved_documents else None
    if scorer is None:
        cosine = 0.0
    else:
        cosine = scorer.cosine(scorer.vectorize(case.record.question), scorer.vectorize(case.answer))
    return _clamp(0.7 * coverage + 0.3 * cosine)


def evaluate_case_locally(case: EvaluationCase) -> dict[str, float]:
    """Compute all local scores for one evaluated case."""

    faithfulness = faithfulness_proxy(case)
    relevancy = answer_relevancy_proxy(case)
    precision = context_precision(case)
    recall = context_recall(case)
    correctness = token_f1(case.answer, case.record.reference_answer)
    overall = (
        0.30 * faithfulness
        + 0.20 * relevancy
        + 0.20 * precision
        + 0.15 * recall
        + 0.15 * correctness
    )
    return {
        "faithfulness": round(_clamp(faithfulness), 4),
        "answer_relevancy": round(_clamp(relevancy), 4),
        "context_precision": round(_clamp(precision), 4),
        "context_recall": round(_clamp(recall), 4),
        "answer_correctness_f1": round(_clamp(correctness), 4),
        "overall": round(_clamp(overall), 4),
    }
