from __future__ import annotations

import json
import uuid
from pathlib import Path

import streamlit as st
from langgraph.types import Command

from clinical_assistant.observability import observer
from clinical_assistant.workflow import build_workflow

REPORTS_DIR = Path("data/reports")
EVAL_RESULTS_PATH = Path("data/runtime/eval_results.json")

THRESHOLDS = {
    "faithfulness": 0.7,
    "context_precision": 0.6,
    "context_recall": 0.6,
    "answer_relevancy": 0.6,
}

st.set_page_config(page_title="Clinical Report Summarization Assistant", layout="wide")


@st.cache_resource
def get_workflow():
    return build_workflow()


def _sample_reports() -> list[Path]:
    return sorted(REPORTS_DIR.glob("*.json"))


def _render_final_output(final_output: dict) -> None:
    if final_output.get("status") == "needs_information":
        st.warning("Report is incomplete — additional information is required.")
        st.markdown("**Missing sections:**")
        for item in final_output.get("missing_sections", []):
            st.markdown(f"- {item}")
    elif final_output.get("status") == "critical_alert_acknowledged":
        st.error("Critical alert acknowledged.")
        st.markdown("**Critical findings:**")
        for item in final_output.get("abnormal_findings", []):
            st.markdown(f"- {item}")
        ack = final_output.get("clinician_ack", {})
        st.markdown("**Clinician acknowledgment:**")
        st.json(ack)
    else:
        st.subheader("Report summary")
        st.write(final_output.get("Report summary", ""))
        st.subheader("Abnormal findings")
        for item in final_output.get("Abnormal findings", []):
            st.markdown(f"- {item}")
        st.subheader("Recommendation")
        st.write(final_output.get("Recommendation", ""))


def render_run_report_tab() -> None:
    st.header("Run report")

    sample_paths = _sample_reports()
    options = [str(p) for p in sample_paths] + ["Upload custom JSON"]
    choice = st.selectbox("Select a report", options)

    report_dict = None
    if choice == "Upload custom JSON":
        uploaded = st.file_uploader("Upload a report JSON file", type=["json"])
        if uploaded is not None:
            report_dict = json.loads(uploaded.read().decode("utf-8"))
    else:
        report_dict = json.loads(Path(choice).read_text(encoding="utf-8"))

    thread_key = "thread_id"
    if thread_key not in st.session_state:
        st.session_state[thread_key] = str(uuid.uuid4())

    if st.button("Run", disabled=report_dict is None):
        st.session_state["thread_id"] = str(uuid.uuid4())
        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
        graph = get_workflow()
        try:
            result = graph.invoke(
                {"report": report_dict, "audit": [], "errors": []}, config
            )
        except Exception as exc:
            st.error(
                "Could not generate summary — LLM call failed after 3 attempts. "
                f"See error details below.\n\n{exc}"
            )
            st.session_state.pop("pending_interrupt", None)
        else:
            if "__interrupt__" in result:
                st.session_state["pending_interrupt"] = {
                    "payload": result["__interrupt__"][0].value,
                    "config": config,
                }
            else:
                st.session_state.pop("pending_interrupt", None)
                _render_final_output(result["final_output"])

    pending = st.session_state.get("pending_interrupt")
    if pending:
        st.subheader("Critical alert — clinician acknowledgment required")
        st.json(pending["payload"])

        with st.form("ack_form"):
            action = st.selectbox(
                "Action",
                ["escalate_to_er", "acknowledge_pending_action", "false_alarm"],
            )
            acknowledged_by = st.text_input("Acknowledged by")
            rationale = st.text_area("Rationale")
            submitted = st.form_submit_button("Submit acknowledgment")

        if submitted:
            ack_dict = {
                "acknowledged_by": acknowledged_by,
                "action": action,
                "rationale": rationale,
            }
            graph = get_workflow()
            try:
                result = graph.invoke(Command(resume=ack_dict), pending["config"])
            except Exception as exc:
                st.error(f"Could not process acknowledgment: {exc}")
            else:
                st.session_state.pop("pending_interrupt", None)
                _render_final_output(result["final_output"])


def render_monitoring_tab() -> None:
    st.header("Monitoring")

    events = observer.recent_events(limit=100)
    if not events:
        st.info("No agent events recorded yet — run a report first.")
        return

    st.dataframe(events, use_container_width=True)

    durations: dict[str, list[float]] = {}
    for event in events:
        durations.setdefault(event["agent"], []).append(event["duration_ms"])
    averages = {agent: sum(vals) / len(vals) for agent, vals in durations.items()}
    st.subheader("Average duration by agent (ms)")
    st.bar_chart(averages)


def render_evaluation_tab() -> None:
    st.header("Evaluation")

    if not EVAL_RESULTS_PATH.exists():
        st.info(
            "No eval results yet — run 'python -m evals.run_ragas_eval' first."
        )
        return

    data = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))

    st.subheader("Recommendation agent — per-case scores")
    st.dataframe(data.get("recommendation_per_case", []), use_container_width=True)

    st.subheader("Summary agent — per-case scores")
    st.dataframe(data.get("summary_per_case", []), use_container_width=True)

    st.subheader("Averaged metrics")
    averages = data.get("averages", {})
    cols = st.columns(len(averages) or 1)
    for col, (metric, value) in zip(cols, averages.items()):
        threshold = THRESHOLDS.get(metric)
        passed = threshold is not None and value >= threshold
        badge = "✅ PASS" if passed else "❌ FAIL"
        col.metric(label=f"{metric} ({badge})", value=f"{value:.2f}")


def main() -> None:
    st.title("Clinical Report Summarization Assistant")

    tab1, tab2, tab3 = st.tabs(["Run report", "Monitoring", "Evaluation"])
    with tab1:
        render_run_report_tab()
    with tab2:
        render_monitoring_tab()
    with tab3:
        render_evaluation_tab()


if __name__ == "__main__":
    main()
