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
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

# The Gemini free tier caps gemini-2.5-flash at 5 requests/minute, and ragas
# fires many judge calls per row (faithfulness alone decomposes the answer
# into statements, then verifies each one). Force fully sequential judge
# calls with generous retry/backoff so the run fits within that quota
# instead of exhausting retries and returning NaN scores.
RAGAS_RUN_CONFIG = RunConfig(max_workers=1, max_retries=20, max_wait=90, timeout=300)

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
    recommendation = state["recommendation"]
    summary = state["summary"]

    return {
        "report": report,
        "clinical_findings": clinical_findings,
        "guideline_evidence": guideline_evidence,
        "recommendation": recommendation,
        "summary": summary,
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
        rows["answer"].append(run["recommendation"]["recommendation"])
        rows["contexts"].append([e["text"] for e in run["guideline_evidence"]])
        rows["ground_truth"].append(case["ground_truth_recommendation"])
    return Dataset.from_dict(rows)


def _build_summary_dataset(runs: list[dict]) -> Dataset:
    rows = {"question": [], "answer": [], "contexts": []}
    for run in runs:
        report = run["report"]
        rows["question"].append("Summarize this clinical report factually.")
        rows["answer"].append(run["summary"]["report_summary"])
        rows["contexts"].append(
            [report.get("narrative", ""), report.get("chief_complaint", "")]
            + [f["message"] for f in run["clinical_findings"]]
        )
    return Dataset.from_dict(rows)


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
        threshold = THRESHOLDS.get(metric)
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
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RAGAS_RUN_CONFIG,
    )
    recommendation_df = recommendation_result.to_pandas()

    summary_dataset = _build_summary_dataset(runs)
    summary_result = evaluate(
        summary_dataset,
        metrics=[faithfulness],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RAGAS_RUN_CONFIG,
    )
    summary_df = summary_result.to_pandas()

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

    _print_table("Recommendation agent", recommendation_per_case)
    _print_table("Summary agent", summary_per_case)
    _print_verdicts("Recommendation agent", averages)
    if "summary_faithfulness" in averages:
        print(
            f"\nsummary agent faithfulness: {averages['summary_faithfulness']:.3f} "
            f"(threshold {THRESHOLDS['faithfulness']}) -> "
            f"{'PASS' if averages['summary_faithfulness'] >= THRESHOLDS['faithfulness'] else 'FAIL'}"
        )

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
