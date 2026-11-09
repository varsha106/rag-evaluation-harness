"""CSV and Markdown report generation."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

SCORE_COLUMNS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness_f1",
    "overall",
]
CSV_COLUMNS = [
    "case_id",
    "question",
    "answer",
    "reference_answer",
    "retrieved_doc_ids",
    *SCORE_COLUMNS,
]


def write_csv(rows: list[dict[str, object]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    return path


def _mean(rows: list[dict[str, object]], column: str) -> float:
    return statistics.fmean(float(row[column]) for row in rows) if rows else 0.0


def write_markdown(
    rows: list[dict[str, object]],
    path: str | Path,
    *,
    backend_requested: str,
    backend_used: str,
    notes: list[str],
    threshold: float = 0.70,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Requested backend: `{backend_requested}`",
        f"- Backend used: `{backend_used}`",
        f"- Cases evaluated: **{len(rows)}**",
        f"- Review threshold: **{threshold:.2f}**",
        "",
        "## Aggregate scores",
        "",
        "| Metric | Mean |",
        "|---|---:|",
    ]
    for column in SCORE_COLUMNS:
        lines.append(f"| {column} | {_mean(rows, column):.4f} |")

    lines.extend(
        [
            "",
            "## Per-case scores",
            "",
            "| Case | Faithfulness | Relevancy | Context precision | Context recall | Answer F1 | Overall |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {case_id} | {faithfulness:.4f} | {answer_relevancy:.4f} | "
            "{context_precision:.4f} | {context_recall:.4f} | "
            "{answer_correctness_f1:.4f} | {overall:.4f} |".format(
                **{key: (float(value) if key in SCORE_COLUMNS else value) for key, value in row.items()}
            )
        )

    flagged = [row for row in rows if float(row["overall"]) < threshold]
    lines.extend(["", "## Cases below threshold", ""])
    if flagged:
        for row in flagged:
            lines.append(
                f"- **{row['case_id']}** — overall {float(row['overall']):.4f}. "
                f"Question: {row['question']}"
            )
    else:
        lines.append("None.")

    lines.extend(
        [
            "",
            "## Method note",
            "",
            "The local backend uses transparent lexical and retrieval-label heuristics. "
            "Those values are useful regression signals, but they are not official "
            "Ragas or DeepEval scores. Framework rows are produced only when the "
            "selected framework and its judge model are actually configured.",
        ]
    )
    if notes:
        lines.extend(["", "## Run notes", ""])
        lines.extend(f"- {note}" for note in notes)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
