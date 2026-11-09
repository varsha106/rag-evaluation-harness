"""Optional Ragas and DeepEval adapters.

Framework metrics normally need a judge model. These adapters therefore
fail with a clear :class:`BackendUnavailable` message when the package or
model configuration is absent; the CLI can then fall back to local metrics
instead of pretending that a framework score was produced.
"""

from __future__ import annotations

import os
from typing import Any

from .models import EvaluationCase

SCORE_FIELDS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness_f1",
)


class BackendUnavailable(RuntimeError):
    """Raised when an optional framework cannot be run as configured."""


def _result_by_case(values: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for value in values:
        case_id = str(value["case_id"])
        output[case_id] = {
            field: float(value[field]) for field in SCORE_FIELDS if value.get(field) is not None
        }
    return output


def run_ragas(cases: list[EvaluationCase]) -> dict[str, dict[str, float]]:
    """Run Ragas with an Ollama judge model and sentence-transformer embeddings.

    Set ``OLLAMA_MODEL`` (for example ``llama3.1``) and, optionally,
    ``EMBEDDING_MODEL``. A running Ollama service is required.
    """

    try:
        from datasets import Dataset
        from langchain_community.chat_models import ChatOllama
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise BackendUnavailable(
            "Ragas dependencies are not installed. Install requirements.txt and try again."
        ) from exc

    model = os.getenv("OLLAMA_MODEL")
    if not model:
        raise BackendUnavailable("Set OLLAMA_MODEL to a locally available Ollama model for Ragas.")

    dataset = Dataset.from_dict(
        {
            "user_input": [case.record.question for case in cases],
            "response": [case.answer for case in cases],
            "retrieved_contexts": [case.retrieved_contexts for case in cases],
            "reference": [case.record.reference_answer for case in cases],
        }
    )
    llm = LangchainLLMWrapper(ChatOllama(model=model))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    )
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    frame = result.to_pandas()
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        row = frame.iloc[index]
        rows.append(
            {
                "case_id": case.record.case_id,
                "faithfulness": row.get("faithfulness"),
                "answer_relevancy": row.get("answer_relevancy"),
                "context_precision": row.get("context_precision"),
                "context_recall": row.get("context_recall"),
            }
        )
    return _result_by_case(rows)


def run_deepeval(cases: list[EvaluationCase]) -> dict[str, dict[str, float]]:
    """Run DeepEval metrics with an Ollama model when available.

    Set ``OLLAMA_MODEL`` for the local-model path. If it is not set, the
    adapter allows DeepEval's normal hosted-model configuration only when
    ``OPENAI_API_KEY`` is present.
    """

    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        raise BackendUnavailable(
            "DeepEval is not installed. Install requirements.txt and try again."
        ) from exc

    model: Any = None
    ollama_model = os.getenv("OLLAMA_MODEL")
    if ollama_model:
        try:
            from deepeval.models import OllamaModel

            model = OllamaModel(model=ollama_model)
        except Exception as exc:  # pragma: no cover - depends on DeepEval version
            raise BackendUnavailable("This DeepEval version does not expose OllamaModel.") from exc
    elif not os.getenv("OPENAI_API_KEY"):
        raise BackendUnavailable(
            "Set OLLAMA_MODEL for a local DeepEval judge, or configure DeepEval's hosted model."
        )

    metric_types = [
        ("faithfulness", FaithfulnessMetric),
        ("answer_relevancy", AnswerRelevancyMetric),
        ("context_precision", ContextualPrecisionMetric),
        ("context_recall", ContextualRecallMetric),
    ]
    rows: list[dict[str, Any]] = []
    for case in cases:
        test_case = LLMTestCase(
            input=case.record.question,
            actual_output=case.answer,
            expected_output=case.record.reference_answer,
            retrieval_context=case.retrieved_contexts,
        )
        scores: dict[str, Any] = {"case_id": case.record.case_id}
        for field, metric_type in metric_types:
            metric = metric_type(model=model) if model is not None else metric_type()
            measured = metric.measure(test_case)
            scores[field] = measured if measured is not None else getattr(metric, "score", None)
        rows.append(scores)
    return _result_by_case(rows)
