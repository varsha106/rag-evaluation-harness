import unittest
from pathlib import Path

from rag_eval.data import load_qa_records
from rag_eval.pipeline import build_cases, evaluate_cases

ROOT = Path(__file__).resolve().parent.parent


class PipelineTests(unittest.TestCase):
    def test_csv_input_loads(self):
        records = load_qa_records(ROOT / "data" / "sample_qa.csv")
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].relevant_doc_ids, ("hybrid-retrieval",))

    def test_sample_pipeline_runs_end_to_end(self):
        cases = build_cases(
            str(ROOT / "data" / "sample_documents.json"),
            str(ROOT / "data" / "sample_qa.json"),
            top_k=3,
        )
        result = evaluate_cases(cases)
        self.assertEqual(len(result.rows), 6)
        self.assertEqual(result.backend_used, "local")
        first = next(row for row in result.rows if row["case_id"] == "q01-hybrid-retrieval")
        self.assertIn("hybrid-retrieval", str(first["retrieved_doc_ids"]))


if __name__ == "__main__":
    unittest.main()
