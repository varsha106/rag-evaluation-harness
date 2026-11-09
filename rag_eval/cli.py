"""Command-line interface for the RAG evaluation harness."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import build_cases, evaluate_cases
from .report import write_csv, write_markdown

ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a small RAG pipeline or supplied RAG answers.")
    parser.add_argument("--documents", default=str(ROOT / "data" / "sample_documents.json"))
    parser.add_argument("--qa", default=str(ROOT / "data" / "sample_qa.json"), help="JSON or CSV QA file")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--report-prefix", default="evaluation")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--answer-source", choices=["provided", "extractive"], default="provided")
    parser.add_argument("--backend", choices=["local", "ragas", "deepeval"], default="local")
    parser.add_argument(
        "--require-framework",
        action="store_true",
        help="Fail instead of falling back when a selected framework is unavailable.",
    )
    parser.add_argument("--threshold", type=float, default=0.70)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = build_cases(
        args.documents,
        args.qa,
        top_k=args.top_k,
        answer_source=args.answer_source,
    )
    result = evaluate_cases(
        cases,
        backend=args.backend,
        require_framework=args.require_framework,
    )
    output_dir = Path(args.output_dir)
    csv_path = write_csv(result.rows, output_dir / f"{args.report_prefix}.csv")
    markdown_path = write_markdown(
        result.rows,
        output_dir / f"{args.report_prefix}.md",
        backend_requested=result.backend_requested,
        backend_used=result.backend_used,
        notes=result.notes,
        threshold=args.threshold,
    )
    print(f"Evaluated {len(result.rows)} cases with backend '{result.backend_used}'.")
    print(f"CSV report: {csv_path}")
    print(f"Markdown report: {markdown_path}")
    for note in result.notes:
        print(f"Note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
