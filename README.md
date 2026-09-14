# Clinical Report Summarization Assistant

A LangGraph-based system that deterministically classifies clinical report
severity, routes critical findings to a human clinician for immediate
acknowledgment, and — for routine/review reports — produces a grounded
summary and guideline-backed recommendation via parallel Summary and
Recommendation agents joined by a rule-based Reconciler.

See `docs/DEMO_AND_EVALUATION.md` for the architecture walkthrough, demo
script, and evaluation approach.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -e .

cp .env.example .env
# then edit .env and set GOOGLE_API_KEY
```

`GOOGLE_API_KEY` is required — the app fails fast at startup if it's unset.
There is no offline/degraded mode.

## Run the demo

```bash
streamlit run app.py
```

Three tabs:
- **Run report** — run any of the four sample reports (`data/reports/*.json`)
  or upload a custom one; critical reports pause for clinician acknowledgment.
- **Monitoring** — recent agent events (timing, success/error) from the local
  SQLite observability store.
- **Evaluation** — displays results from the most recent offline ragas eval
  run (see below); shows a helpful message if none has been run yet.

## Run the evaluation

Ragas evaluation runs offline, not from the Streamlit UI (it makes real LLM
calls and is too slow/costly to trigger from a button click):

```bash
python -m evals.run_ragas_eval
```

This scores the Recommendation agent (faithfulness, answer_relevancy,
context_precision, context_recall) and the Summary agent (faithfulness only)
against the cases in `data/evals/recommendation_eval_cases.json`, prints a
per-case and averaged table with pass/fail verdicts, and writes
`data/runtime/eval_results.json` for the Evaluation tab to read.

## Project layout

```
├── pyproject.toml
├── .env.example
├── data/
│   ├── reports/            sample + eval clinical reports (JSON)
│   ├── evals/              ragas eval case definitions
│   ├── guidelines/         markdown guideline corpus for RAG
│   └── runtime/            generated: Chroma index, observability.db, eval_results.json
├── src/clinical_assistant/ core package (models, services, rag, workflow, observability, config)
├── evals/run_ragas_eval.py offline ragas evaluation script
├── app.py                  Streamlit demo
└── docs/DEMO_AND_EVALUATION.md
```
