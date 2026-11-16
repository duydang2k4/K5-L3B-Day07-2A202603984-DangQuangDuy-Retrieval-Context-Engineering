# Day 7 implementation

This checkout contains a reproducible, dependency-free retrieval pipeline for
the supplied Day 6 export.

## Run

Use Python 3.11 or newer from the repository root:

```bash
python -m unittest discover -s tests -v
python starter/run_retrieval_benchmark.py \
  --chunks data/day6_chunks.json \
  --questions data/day6_questions.json \
  --extra-questions data/day7_abstention_questions.json \
  --split dev --output results/dev
python starter/run_retrieval_benchmark.py \
  --chunks data/day6_chunks.json \
  --questions data/day6_questions.json \
  --extra-questions data/day7_abstention_questions.json \
  --split test --output results/test
```

The benchmark writes one trace file per strategy and a summary containing
Hit@1, Evidence Hit@3, Context Evidence Rate, latency, context size and
zero-cost local execution.

## Retrieval strategies

`starter/retrieval.py` implements BM25 lexical retrieval, a deterministic
hashed dense-vector index, and reciprocal-rank fusion of both. Filters are
exact matches over user-supplied chunk or metadata fields. The hashed index is
an offline reproducibility adapter, not a hosted embedding model; a Day 6
embedding service can be connected through the same `search` contract.

`starter/agentic.py` contains the bounded agent loop, context budget,
deduplication, citation validation and `OpenAIController`. Set
`OPENAI_API_KEY`, optionally `OPENAI_MODEL` and `OPENAI_BASE_URL`, before using
the controller. Secrets are never stored in the repository.

The four `U*` cases intentionally have empty gold evidence and must be
handled as abstentions. Gold is used only by the offline evaluator, never by
the agent or retriever.
