from __future__ import annotations

import json
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from clinical_assistant.config import settings
from clinical_assistant.models import (
    ClinicalReport,
    ClinicianAck,
    ClinicalState,
    Finding,
    RecommendationOutput,
    SummaryOutput,
)
from clinical_assistant.observability import observer
from clinical_assistant.rag import GuidelineRetriever
from clinical_assistant.services import (
    check_completeness,
    interpret_labs,
    reconcile_outputs,
    safety_level,
)

SUMMARY_PROMPT = """You are a clinical report summarization assistant for licensed clinicians.
Summarize only the facts present in the supplied report and the deterministic
findings below. Do not diagnose, prescribe, recommend follow-up actions, or add
any fact not present in the source data. Preserve numeric values and units
exactly as given. List every abnormal or critical finding provided below in the
abnormal_findings field, using the finding's own message and evidence text.

REPORT:
{report_json}

DETERMINISTIC FINDINGS:
{clinical_findings_json}
"""

RECOMMENDATION_PROMPT = """You are a clinical follow-up recommendation assistant for licensed clinicians.
Using only the deterministic findings and the retrieved guideline passages
below, suggest reasonable next-step actions for the treating clinician to
consider. Do not diagnose. Do not invent guidance not supported by the
retrieved passages. Cite the guideline source name for every recommendation you
make, in the guideline_citations field.

DETERMINISTIC FINDINGS:
{clinical_findings_json}

GUIDELINE EVIDENCE:
{guideline_evidence_json}
"""


def _report_id(state: ClinicalState) -> str:
    return state["report"]["report_id"]


def make_intake_agent():
    def intake_agent(state: ClinicalState) -> dict:
        with observer.trace(_report_id(state), "intake_agent"):
            validated = ClinicalReport.model_validate(state["report"])
            return {"report": validated.model_dump()}

    return intake_agent


def make_completeness_agent():
    def completeness_agent(state: ClinicalState) -> dict:
        with observer.trace(_report_id(state), "completeness_agent"):
            report = ClinicalReport.model_validate(state["report"])
            findings = check_completeness(report)
            return {
                "completeness_findings": [f.model_dump() for f in findings]
            }

    return completeness_agent


def make_clinical_finding_agent():
    def clinical_finding_agent(state: ClinicalState) -> dict:
        with observer.trace(_report_id(state), "clinical_finding_agent"):
            report = ClinicalReport.model_validate(state["report"])
            findings = interpret_labs(report)
            return {"clinical_findings": [f.model_dump() for f in findings]}

    return clinical_finding_agent


def make_router():
    def router(state: ClinicalState) -> dict:
        with observer.trace(_report_id(state), "router"):
            completeness = [
                Finding.model_validate(f)
                for f in state.get("completeness_findings", [])
            ]
            clinical = [
                Finding.model_validate(f)
                for f in state.get("clinical_findings", [])
            ]
            level = safety_level(completeness, clinical)
            return {"safety_level": level}

    return router


def choose_route(
    state: ClinicalState,
) -> Literal["request_information", "human_interrupt", "rag_agent"]:
    level = state["safety_level"]
    if level == "incomplete":
        return "request_information"
    if level == "critical":
        return "human_interrupt"
    return "rag_agent"


def make_request_information():
    def request_information(state: ClinicalState) -> dict:
        with observer.trace(_report_id(state), "request_information"):
            return {"route": "request_information", "status": "needs_information"}

    return request_information


def make_human_interrupt():
    def human_interrupt(state: ClinicalState) -> dict:
        with observer.trace(_report_id(state), "human_interrupt"):
            payload = {
                "report_id": _report_id(state),
                "safety_level": "critical",
                "clinical_findings": state["clinical_findings"],
                "instruction": (
                    "Review the flagged critical result immediately and "
                    "acknowledge with one of: escalate_to_er, "
                    "acknowledge_pending_action, false_alarm, plus a rationale."
                ),
            }
            ack_payload = interrupt(payload)
            ack = ClinicianAck.model_validate(ack_payload)
            return {
                "clinician_ack": ack.model_dump(),
                "route": "human_interrupt",
                "status": "critical_alert_acknowledged",
            }

    return human_interrupt


def make_rag_agent(retriever: GuidelineRetriever):
    def rag_agent(state: ClinicalState) -> dict:
        with observer.trace(_report_id(state), "rag_agent"):
            report = state["report"]
            clinical_findings = state.get("clinical_findings", [])
            query = " ".join(
                [
                    report.get("chief_complaint", ""),
                    report.get("narrative", ""),
                    " ".join(report.get("diagnoses", [])),
                    " ".join(f["message"] for f in clinical_findings),
                ]
            )
            evidence = retriever.search(query, top_k=3)
            return {"guideline_evidence": [e.model_dump() for e in evidence]}

    return rag_agent


def make_recommendation_agent():
    llm = ChatGoogleGenerativeAI(
        model=settings.model, temperature=0, max_retries=2
    ).with_structured_output(RecommendationOutput)

    def recommendation_agent(state: ClinicalState) -> dict:
        with observer.trace(_report_id(state), "recommendation_agent"):
            prompt = RECOMMENDATION_PROMPT.format(
                clinical_findings_json=json.dumps(state.get("clinical_findings", [])),
                guideline_evidence_json=json.dumps(
                    state.get("guideline_evidence", [])
                ),
            )
            try:
                result: RecommendationOutput = llm.invoke(prompt)
            except Exception as exc:
                msg = (
                    f"recommendation_agent: LLM call failed after retries: {exc}"
                )
                raise RuntimeError(msg) from exc
            return {"recommendation": result.model_dump()}

    return recommendation_agent


def make_summary_agent():
    llm = ChatGoogleGenerativeAI(
        model=settings.model, temperature=0, max_retries=2
    ).with_structured_output(SummaryOutput)

    def summary_agent(state: ClinicalState) -> dict:
        with observer.trace(_report_id(state), "summary_agent"):
            prompt = SUMMARY_PROMPT.format(
                report_json=json.dumps(state["report"]),
                clinical_findings_json=json.dumps(state.get("clinical_findings", [])),
            )
            try:
                result: SummaryOutput = llm.invoke(prompt)
            except Exception as exc:
                msg = f"summary_agent: LLM call failed after retries: {exc}"
                raise RuntimeError(msg) from exc
            return {"summary": result.model_dump()}

    return summary_agent


def make_reconciler():
    def reconciler(state: ClinicalState) -> dict:
        with observer.trace(_report_id(state), "reconciler"):
            clinical_findings = [
                Finding.model_validate(f) for f in state.get("clinical_findings", [])
            ]
            summary = SummaryOutput.model_validate(state["summary"])
            recommendation = RecommendationOutput.model_validate(
                state["recommendation"]
            )
            reconciled = reconcile_outputs(
                clinical_findings, summary, recommendation
            )
            return {"reconciled_output": reconciled.model_dump()}

    return reconciler


def make_output_agent():
    def output_agent(state: ClinicalState) -> dict:
        with observer.trace(_report_id(state), "output_agent"):
            route = state.get("route")

            if route == "request_information":
                missing_sections = [
                    f["message"] for f in state.get("completeness_findings", [])
                ]
                final_output = {
                    "status": "needs_information",
                    "missing_sections": missing_sections,
                }
            elif route == "human_interrupt":
                critical_messages = [
                    f["message"]
                    for f in state.get("clinical_findings", [])
                    if f["severity"] == "critical"
                ]
                final_output = {
                    "status": "critical_alert_acknowledged",
                    "abnormal_findings": critical_messages,
                    "clinician_ack": state["clinician_ack"],
                }
            else:
                from clinical_assistant.models import ReconciledOutput

                reconciled = ReconciledOutput.model_validate(
                    state["reconciled_output"]
                )
                final_output = reconciled.to_contract()

            return {"final_output": final_output}

    return output_agent


def build_workflow():
    retriever = GuidelineRetriever()

    graph = StateGraph(ClinicalState)

    graph.add_node("intake_agent", make_intake_agent())
    graph.add_node("completeness_agent", make_completeness_agent())
    graph.add_node("clinical_finding_agent", make_clinical_finding_agent())
    graph.add_node("router", make_router())
    graph.add_node("request_information", make_request_information())
    graph.add_node("human_interrupt", make_human_interrupt())
    graph.add_node("rag_agent", make_rag_agent(retriever))
    graph.add_node("recommendation_agent", make_recommendation_agent())
    graph.add_node("summary_agent", make_summary_agent())
    graph.add_node("reconciler", make_reconciler())
    graph.add_node("output_agent", make_output_agent())

    graph.add_edge(START, "intake_agent")
    graph.add_edge("intake_agent", "completeness_agent")
    graph.add_edge("intake_agent", "clinical_finding_agent")
    graph.add_edge("completeness_agent", "router")
    graph.add_edge("clinical_finding_agent", "router")

    graph.add_conditional_edges("router", choose_route)

    graph.add_edge("request_information", "output_agent")
    graph.add_edge("human_interrupt", "output_agent")

    graph.add_edge("rag_agent", "recommendation_agent")
    graph.add_edge("rag_agent", "summary_agent")
    graph.add_edge("recommendation_agent", "reconciler")
    graph.add_edge("summary_agent", "reconciler")

    graph.add_edge("reconciler", "output_agent")
    graph.add_edge("output_agent", END)

    return graph.compile(checkpointer=MemorySaver())
