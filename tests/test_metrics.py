import unittest

from rag_eval.metrics import evaluate_case_locally, token_f1
from rag_eval.models import Document, EvaluationCase, QARecord


class MetricTests(unittest.TestCase):
    def test_token_f1_identical_text(self):
        self.assertEqual(token_f1("hybrid retrieval uses BM25", "hybrid retrieval uses BM25"), 1.0)

    def test_supported_answer_scores_above_hallucinated_answer(self):
        document = Document(
            doc_id="d1",
            title="Faithfulness",
            text="Faithfulness measures whether answer claims are supported by retrieved context.",
        )
        supported = EvaluationCase(
            record=QARecord(
                case_id="supported",
                question="What does faithfulness measure?",
                reference_answer="Faithfulness measures whether claims are supported by retrieved context.",
                relevant_doc_ids=("d1",),
            ),
            answer="Faithfulness measures whether claims are supported by retrieved context.",
            retrieved_documents=(document,),
        )
        unsupported = EvaluationCase(
            record=supported.record,
            answer="Faithfulness measures API latency and database throughput.",
            retrieved_documents=(document,),
        )
        self.assertGreater(
            evaluate_case_locally(supported)["faithfulness"],
            evaluate_case_locally(unsupported)["faithfulness"],
        )


if __name__ == "__main__":
    unittest.main()
