# RAG Evaluation Report

- Requested backend: `local`
- Backend used: `local`
- Cases evaluated: **6**
- Review threshold: **0.70**

## Aggregate scores

| Metric | Mean |
|---|---:|
| faithfulness | 0.8333 |
| answer_relevancy | 0.5259 |
| context_precision | 0.3333 |
| context_recall | 1.0000 |
| answer_correctness_f1 | 0.6862 |
| overall | 0.6748 |

## Per-case scores

| Case | Faithfulness | Relevancy | Context precision | Context recall | Answer F1 | Overall |
|---|---:|---:|---:|---:|---:|---:|
| q01-hybrid-retrieval | 1.0000 | 0.7984 | 0.3333 | 1.0000 | 0.9333 | 0.8164 |
| q02-faithfulness | 1.0000 | 0.4449 | 0.3333 | 1.0000 | 0.7000 | 0.7106 |
| q03-context-recall | 1.0000 | 0.6167 | 0.3333 | 1.0000 | 0.9524 | 0.7829 |
| q04-answer-relevancy | 1.0000 | 0.5561 | 0.3333 | 1.0000 | 0.6207 | 0.7210 |
| q05-deepeval-purpose | 0.0000 | 0.2946 | 0.3333 | 1.0000 | 0.0833 | 0.2881 |
| q06-evaluation-dataset | 1.0000 | 0.4449 | 0.3333 | 1.0000 | 0.8276 | 0.7298 |

## Cases below threshold

- **q05-deepeval-purpose** — overall 0.2881. Question: What can DeepEval be used to evaluate?

## Method note

The local backend uses transparent lexical and retrieval-label heuristics. Those values are useful regression signals, but they are not official Ragas or DeepEval scores. Framework rows are produced only when the selected framework and its judge model are actually configured.
