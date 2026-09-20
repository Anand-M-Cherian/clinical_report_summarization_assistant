from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

# answer_relevancy's default strictness=3 asks the LLM for 3 candidate
# questions in one call (n=3), which gemini-3.6-flash rejects outright
# ("Multiple candidates is not enabled for this model"). strictness=1 keeps
# the metric working across models that don't support multi-candidate
# generation, at the cost of averaging over only one generated question
# instead of three.
answer_relevancy = AnswerRelevancy(strictness=1)

# Tuned for a paid Gemini tier, not the free tier. The free tier's
# 5 requests/minute cap needed max_workers=1 with a long, patient backoff
# (max_retries=20, max_wait=90) to avoid exhausting retries and returning
# NaN scores; billing removes that per-minute ceiling, so moderate
# concurrency and a normal retry budget are enough here.
RAGAS_RUN_CONFIG = RunConfig(max_workers=4, max_retries=5, max_wait=90, timeout=300)

from clinical_assistant.config import settings
from clinical_assistant.rag import EMBEDDING_MODEL_NAME
from clinical_assistant.workflow import build_workflow

EVAL_CASES_PATH = Path("data/evals/recommendation_eval_cases.json")
RESULTS_PATH = Path("data/runtime/eval_results.json")

THRESHOLDS = {
    "faithfulness": 0.7,
    "context_precision": 0.6,
    "context_recall": 0.6,
    "answer_relevancy": 0.6,
}

# The Gemini free tier caps requests per minute per model. Each eval case
# makes 2 workflow LLM calls (recommendation_agent + summary_agent); pace
# cases far enough apart, and retry on a transient rate-limit error, so a
# 5-case run doesn't burn through the per-minute quota that workflow.py's
# own max_retries=2 (a deliberate fail-fast setting for the live app) can't
# absorb on its own.
SECONDS_BETWEEN_CASES = 15
RATE_LIMIT_RETRY_WAIT_SECONDS = 65
MAX_CASE_ATTEMPTS = 3

# A single ragas faithfulness judge call is not reliable at the per-case
# level, confirmed across two different Gemini models: gemini-3.6-flash
# scored the exact same (question, answer, contexts) tuple 0.25 in one call
# and 1.0 in another, and gemini-3.5-flash scored a near-verbatim-grounded
# response 0.0 on a single call. answer_relevancy/context_precision/
# context_recall showed no such instability across separate full runs, so
# only faithfulness is scored multiple times and averaged per case.
FAITHFULNESS_JUDGE_CALLS = 3


def _build_query(report: dict, clinical_findings: list[dict]) -> str:
    return " ".join(
        [
            report.get("chief_complaint", ""),
            report.get("narrative", ""),
            " ".join(report.get("diagnoses", [])),
            " ".join(f["message"] for f in clinical_findings),
        ]
    )


def _run_case(graph, case: dict) -> dict:
    report = json.loads(Path(case["report_path"]).read_text(encoding="utf-8"))
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = graph.invoke({"report": report, "audit": [], "errors": []}, config)

    clinical_findings = state.get("clinical_findings", [])
    guideline_evidence = state.get("guideline_evidence", [])
    # summary_agent/recommendation_agent now emit structured fields
    # (overview + key_points/action_items), not a single string — the
    # flattened text the eval judges against only exists post-reconciliation.
    reconciled = state["reconciled_output"]

    return {
        "report": report,
        "clinical_findings": clinical_findings,
        "guideline_evidence": guideline_evidence,
        "recommendation_text": reconciled["recommendation"],
        "summary_text": reconciled["report_summary"],
    }


def _build_recommendation_dataset(cases: list[dict], runs: list[dict]) -> Dataset:
    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for case, run in zip(cases, runs):
        rows["question"].append(
            _build_query(run["report"], run["clinical_findings"])
        )
        rows["answer"].append(run["recommendation_text"])
        rows["contexts"].append([e["text"] for e in run["guideline_evidence"]])
        rows["ground_truth"].append(case["ground_truth_recommendation"])
    return Dataset.from_dict(rows)


def _report_context_chunks(report: dict, clinical_findings: list[dict]) -> list[str]:
    """All fields summary_agent's own prompt is grounded in (the full report
    JSON, per SUMMARY_PROMPT in workflow.py), broken into per-field chunks
    rather than the report's narrative/chief_complaint alone — otherwise
    ragas' faithfulness check has no way to see the source for true
    statements like the author's name or a listed medication, and marks
    them unsupported even though they're straight from the report.
    """
    chunks = [
        f"Report ID: {report.get('report_id', '')}",
        f"Patient ID: {report.get('patient_id', '')}",
        f"Report type: {report.get('report_type', '')}",
        f"Encounter date: {report.get('encounter_date', '')}",
        f"Author: {report.get('author', '')}",
        f"Chief complaint: {report.get('chief_complaint', '')}",
        f"History: {report.get('history', '')}",
        f"Narrative: {report.get('narrative', '')}",
        f"Diagnoses: {', '.join(report.get('diagnoses', []))}",
        f"Medications: {', '.join(report.get('medications', []))}",
        f"Allergies: {', '.join(report.get('allergies', []))}",
        f"Follow-up: {report.get('follow_up', '')}",
    ]
    chunks.extend(f["message"] for f in clinical_findings)
    return chunks


def _build_summary_dataset(runs: list[dict]) -> Dataset:
    rows = {"question": [], "answer": [], "contexts": []}
    for run in runs:
        rows["question"].append("Summarize this clinical report factually.")
        rows["answer"].append(run["summary_text"])
        rows["contexts"].append(
            _report_context_chunks(run["report"], run["clinical_findings"])
        )
    return Dataset.from_dict(rows)


def _score_faithfulness_averaged(
    dataset: Dataset,
    llm: LangchainLLMWrapper,
    embeddings: LangchainEmbeddingsWrapper,
    run_config: RunConfig,
    n: int = FAITHFULNESS_JUDGE_CALLS,
):
    """Score faithfulness N times and average per case, instead of trusting a
    single judge call. Returns (averaged_per_case_scores, base_df), where
    base_df carries the non-metric columns (user_input/retrieved_contexts/
    response, etc.) from the first run for building the per-case table.
    """
    base_df = None
    score_runs = []
    for _ in range(n):
        result = evaluate(
            dataset,
            metrics=[faithfulness],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            raise_exceptions=True,
        )
        df = result.to_pandas()
        if base_df is None:
            base_df = df.drop(columns=["faithfulness"])
        score_runs.append(df["faithfulness"])

    averaged = sum(score_runs) / n
    return averaged, base_df


def _print_table(title: str, per_case_scores: list[dict]) -> None:
    print(f"\n=== {title} — per case ===")
    if not per_case_scores:
        print("(no cases)")
        return
    headers = list(per_case_scores[0].keys())
    print(" | ".join(headers))
    for row in per_case_scores:
        print(" | ".join(str(row[h]) for h in headers))


def _print_verdicts(title: str, averages: dict[str, float]) -> None:
    print(f"\n=== {title} — averaged metrics vs. thresholds ===")
    for metric, value in averages.items():
        # "summary_faithfulness" (the Summary agent's own faithfulness score,
        # kept separate from the Recommendation agent's "faithfulness" column)
        # is checked against the same threshold as "faithfulness".
        threshold_key = metric.removeprefix("summary_")
        threshold = THRESHOLDS.get(threshold_key)
        if threshold is None:
            continue
        verdict = "PASS" if value >= threshold else "FAIL"
        print(f"{metric}: {value:.3f} (threshold {threshold}) -> {verdict}")


def _run_case_with_retry(graph, case: dict) -> dict:
    for attempt in range(1, MAX_CASE_ATTEMPTS + 1):
        try:
            return _run_case(graph, case)
        except RuntimeError as exc:
            message = str(exc)
            if "PerDay" in message:
                raise RuntimeError(
                    "Gemini free-tier DAILY quota exhausted for this model — "
                    "retrying won't help until it resets. Original error: "
                    f"{message}"
                ) from exc
            if "RESOURCE_EXHAUSTED" not in message or attempt == MAX_CASE_ATTEMPTS:
                raise
            print(
                f"Rate-limited on {case['report_path']} (attempt {attempt}/"
                f"{MAX_CASE_ATTEMPTS}); waiting {RATE_LIMIT_RETRY_WAIT_SECONDS}s "
                "for the per-minute quota to clear..."
            )
            time.sleep(RATE_LIMIT_RETRY_WAIT_SECONDS)
    raise AssertionError("unreachable")


def main() -> None:
    cases = json.loads(EVAL_CASES_PATH.read_text(encoding="utf-8"))

    graph = build_workflow()
    runs = []
    for i, case in enumerate(cases):
        if i > 0:
            time.sleep(SECONDS_BETWEEN_CASES)
        runs.append(_run_case_with_retry(graph, case))

    ragas_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=settings.model, temperature=0)
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    )

    recommendation_dataset = _build_recommendation_dataset(cases, runs)
    recommendation_result = evaluate(
        recommendation_dataset,
        metrics=[answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RAGAS_RUN_CONFIG,
        raise_exceptions=True,
    )
    recommendation_df = recommendation_result.to_pandas()
    recommendation_faithfulness, _ = _score_faithfulness_averaged(
        recommendation_dataset, ragas_llm, ragas_embeddings, RAGAS_RUN_CONFIG
    )
    recommendation_df["faithfulness"] = recommendation_faithfulness

    summary_dataset = _build_summary_dataset(runs)
    summary_faithfulness, summary_df = _score_faithfulness_averaged(
        summary_dataset, ragas_llm, ragas_embeddings, RAGAS_RUN_CONFIG
    )
    summary_df["faithfulness"] = summary_faithfulness

    recommendation_per_case = recommendation_df.to_dict(orient="records")
    summary_per_case = summary_df.to_dict(orient="records")

    metric_columns = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    averages = {
        col: float(recommendation_df[col].mean())
        for col in metric_columns
        if col in recommendation_df.columns
    }
    if "faithfulness" in summary_df.columns:
        averages["summary_faithfulness"] = float(summary_df["faithfulness"].mean())

    recommendation_averages = {
        k: v for k, v in averages.items() if k != "summary_faithfulness"
    }
    summary_averages = {
        k: v for k, v in averages.items() if k == "summary_faithfulness"
    }

    _print_table("Recommendation agent", recommendation_per_case)
    _print_table("Summary agent", summary_per_case)
    _print_verdicts("Recommendation agent", recommendation_averages)
    _print_verdicts("Summary agent", summary_averages)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "recommendation_per_case": recommendation_per_case,
                "summary_per_case": summary_per_case,
                "averages": averages,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote eval results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
