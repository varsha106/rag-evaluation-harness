# RAG Evaluation Harness

A reproducible evaluation harness for retrieval-augmented generation (RAG) systems. It runs a small sample RAG pipeline end to end, grades supplied or generated answers, and writes both a machine-readable CSV report and a human-readable Markdown summary.

The project packages patterns used in RAG evaluation work — versioned question/answer sets, retrieval diagnostics, faithfulness checks, and framework integrations with **Ragas** and **DeepEval** — into a public, runnable demo.

> **Honest scope:** The bundled corpus, responses, and scores are demonstration fixtures. They are not production data, production traffic, or production quality results.

## What it does

Given a document set and a QA dataset, the harness:

1. Indexes the documents with a dependency-free TF-IDF retriever.
2. Retrieves the top-*k* contexts for every question.
3. Uses either:
   - the candidate `response` supplied in the QA file, or
   - an extractive answer selected from the retrieved context.
4. Scores retrieval and answer quality for every case.
5. Writes aggregate and per-case results to CSV and Markdown.

This makes it useful as a starting point for regression testing changes to prompts, chunking, embeddings, retrieval settings, or model selection.

## Metrics

All scores are normalized to the range **0.0–1.0**, where higher is better.

| Metric | What it measures | Local implementation |
|---|---|---|
| **Faithfulness** | Whether answer claims are supported by retrieved context | Sentence-level lexical support against the retrieved context |
| **Answer relevancy** | Whether the answer addresses the question | Blend of question-token coverage and TF-IDF cosine similarity |
| **Context precision** | How much of the retrieved context is relevant | Fraction of retrieved documents with relevance labels; lexical fallback when labels are absent |
| **Context recall** | Whether the needed context was retrieved | Fraction of labelled relevant documents retrieved; reference-token coverage fallback |
| **Answer correctness F1** | Lexical agreement with the reference answer | Content-token precision/recall F1 |
| **Overall** | Weighted summary for triage | `0.30 faithfulness + 0.20 relevancy + 0.20 precision + 0.15 recall + 0.15 answer F1` |

### Interpreting the sample precision score

The sample uses `top_k=3`, and each sample question has one labelled gold document. A perfect retriever therefore receives context precision of `1/3 = 0.3333` when the other two returned documents are unlabelled. That is a property of the metric and demonstration data, not a claim that retrieval failed.

### Local scores vs. framework scores

The default local backend is intentionally transparent and dependency-free. Its scores are **heuristic proxies**, not official Ragas or DeepEval results.

Use the optional framework backends when a project needs model-graded evaluation:

- **Ragas:** faithfulness, answer relevancy, context precision, and context recall.
- **DeepEval:** faithfulness, answer relevancy, contextual precision, and contextual recall.

Answer correctness F1 and the weighted overall score remain local diagnostics when a framework backend is used.

## Quickstart

### Requirements

- Python 3.10 or newer
- No third-party packages are required for the default local demo

Clone the repository and run the bundled example:

```bash
git clone <your-repository-url>
cd rag-evaluation-harness
python run_demo.py --report-prefix sample_evaluation
```

Expected console output:

```text
Evaluated 6 cases with backend 'local'.
CSV report: .../outputs/sample_evaluation.csv
Markdown report: .../outputs/sample_evaluation.md
```

You can also invoke the package as a module:

```bash
python -m rag_eval --help
```

### Run the CSV example

The repository includes both JSON and CSV versions of sample QA data:

```bash
python -m rag_eval \
  --qa data/sample_qa.csv \
  --report-prefix csv_evaluation
```

### Generate extractive answers instead of grading supplied answers

```bash
python -m rag_eval \
  --answer-source extractive \
  --report-prefix extractive_evaluation
```

In extractive mode, the harness selects the retrieved sentence most similar to the question. This keeps the demo runnable without an LLM; it is not intended to imitate a production generator.

## Sample output

The bundled local run produced these aggregate results:

| Metric | Mean |
|---|---:|
| Faithfulness | 0.8333 |
| Answer relevancy | 0.5259 |
| Context precision | 0.3333 |
| Context recall | 1.0000 |
| Answer correctness F1 | 0.6862 |
| Overall | 0.6748 |

One deliberately poor sample answer describes DeepEval as only a load-testing tool. The harness flags it clearly:

| Case | Faithfulness | Relevancy | Precision | Recall | Answer F1 | Overall |
|---|---:|---:|---:|---:|---:|---:|
| `q05-deepeval-purpose` | 0.0000 | 0.2946 | 0.3333 | 1.0000 | 0.0833 | 0.2881 |

That separation is the point of the fixture: retrieval found the correct document, while the supplied answer was unsupported and substantially incorrect.

## Dataset formats

### Documents: JSON

```json
{
  "documents": [
    {
      "id": "hybrid-retrieval",
      "title": "Hybrid Retrieval",
      "text": "Hybrid retrieval combines dense vector search with BM25..."
    }
  ]
}
```

A top-level JSON list of document objects is also accepted.

### QA pairs: JSON

```json
{
  "qa_pairs": [
    {
      "id": "q01",
      "question": "How does hybrid retrieval work?",
      "reference_answer": "It combines dense vector search with BM25.",
      "response": "A candidate answer produced by the system being evaluated.",
      "relevant_doc_ids": ["hybrid-retrieval"]
    }
  ]
}
```

### QA pairs: CSV

Required columns:

```csv
id,question,reference_answer,response,relevant_doc_ids
q01,How does hybrid retrieval work?,"It combines dense vector search with BM25.","Candidate answer.",hybrid-retrieval
```

Separate multiple relevance labels with semicolons.

## Optional framework backends

Install the framework dependencies only if you plan to use Ragas or DeepEval:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Framework versions and judge-model APIs evolve quickly. For repeatable production evaluation, pin the versions that pass your own integration tests.

### Ragas with a local Ollama model

Start Ollama, pull a model, and run:

```bash
export OLLAMA_MODEL="llama3.1"
# Optional; this is the default embedding model used by the adapter.
export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

python -m rag_eval --backend ragas --report-prefix ragas_evaluation
```

The Ragas adapter uses Ollama for the judge model and a sentence-transformers embedding model for answer relevancy.

### DeepEval with a local model

```bash
export OLLAMA_MODEL="llama3.1"
python -m rag_eval --backend deepeval --report-prefix deepeval_evaluation
```

The DeepEval adapter uses DeepEval's `OllamaModel` when it is exposed by the installed DeepEval version. Hosted-model configuration is also possible when `OPENAI_API_KEY` is configured for DeepEval.

### Preventing silent fallback

If the selected framework or judge model is unavailable, the default behavior is to:

1. record the reason in the Markdown report, and
2. fall back to local heuristic scores.

To fail instead:

```bash
python -m rag_eval --backend ragas --require-framework
```

## Command-line options

| Option | Default | Purpose |
|---|---|---|
| `--documents` | `data/sample_documents.json` | Document corpus in JSON format |
| `--qa` | `data/sample_qa.json` | QA dataset in JSON or CSV format |
| `--output-dir` | `outputs/` | Report destination |
| `--report-prefix` | `evaluation` | Prefix for the CSV and Markdown filenames |
| `--top-k` | `3` | Number of contexts retrieved per question |
| `--answer-source` | `provided` | Use supplied responses or generate extractive answers |
| `--backend` | `local` | `local`, `ragas`, or `deepeval` |
| `--threshold` | `0.70` | Overall score used to flag cases for review |
| `--require-framework` | off | Fail rather than fall back from a framework backend |

## Using it with a real RAG system

For each evaluation case, export the following from the system under test:

- the user question;
- the system's final response;
- the context passages actually supplied to the generator;
- a reference answer;
- document or passage relevance labels, when available.

The bundled retriever and extractive generator can then be replaced with the real system's trace output while retaining the reporting shape. For serious model-graded evaluation, use the Ragas or DeepEval adapter and preserve the judge model, model version, prompts, dataset version, and retrieval configuration alongside the scores.

## Project structure

```text
rag-evaluation-harness/
├── data/
│   ├── sample_documents.json     # Ten-document demonstration corpus
│   ├── sample_qa.json            # Six labelled QA cases
│   └── sample_qa.csv             # Smaller CSV-format example
├── outputs/
│   ├── sample_evaluation.csv     # Generated machine-readable report
│   └── sample_evaluation.md      # Generated Markdown summary
├── rag_eval/
│   ├── __init__.py
│   ├── __main__.py               # python -m rag_eval entry point
│   ├── backends.py               # Optional Ragas and DeepEval adapters
│   ├── cli.py                    # Argument parsing and execution
│   ├── data.py                   # JSON/CSV loading and validation
│   ├── generation.py             # Extractive demo answer generator
│   ├── metrics.py                # Transparent local metric proxies
│   ├── models.py                 # Shared dataclasses
│   ├── pipeline.py               # Retrieval-to-report orchestration
│   ├── report.py                 # CSV and Markdown writers
│   └── retriever.py              # Dependency-free TF-IDF retriever
├── tests/
│   ├── test_metrics.py
│   └── test_pipeline.py
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── run_demo.py                   # Convenience entry point
```

## Tests

Run the standard-library test suite:

```bash
python -m unittest discover -s tests -v
```

The tests cover token F1, supported vs. unsupported answers, CSV loading, retrieval, and the full sample pipeline.

## Limitations

- The TF-IDF retriever is a teaching and smoke-test component, not a replacement for dense, sparse, or hybrid production retrieval.
- Lexical faithfulness can miss paraphrases and can over-credit answers that reuse context words without preserving meaning.
- Answer relevancy in the local backend is a proxy; model-based methods can better detect indirect or multi-part answers.
- The sample corpus is intentionally small and topically narrow.
- Framework scores depend on the selected judge model and may vary between models, versions, and runs.

## Tech stack

- **Language:** Python 3.10+
- **Default runtime:** Python standard library only
- **Retrieval demo:** TF-IDF cosine similarity
- **Optional evaluation frameworks:** Ragas and DeepEval
- **Optional local judge path:** Ollama
- **Optional embeddings path:** Sentence Transformers
- **Outputs:** CSV and Markdown

## License

MIT
