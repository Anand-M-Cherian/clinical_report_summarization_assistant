from __future__ import annotations

from clinical_assistant.models import (
    ClinicalReport,
    Finding,
    ReconciledOutput,
    RecommendationOutput,
    SummaryOutput,
)

REQUIRED_TEXT_FIELDS = ("author", "chief_complaint", "narrative", "follow_up")


def check_completeness(report: ClinicalReport) -> list[Finding]:
    findings: list[Finding] = []

    for field in REQUIRED_TEXT_FIELDS:
        value = getattr(report, field)
        if not value or not value.strip():
            findings.append(
                Finding(
                    code=f"MISSING_{field.upper()}",
                    severity="abnormal",
                    message=f"Required field '{field}' is missing or empty.",
                    evidence=f"report.{field} = {value!r}",
                )
            )

    if not report.diagnoses:
        findings.append(
            Finding(
                code="MISSING_DIAGNOSES",
                severity="abnormal",
                message="Required field 'diagnoses' is missing or empty.",
                evidence="report.diagnoses = []",
            )
        )

    return findings


def interpret_labs(report: ClinicalReport) -> list[Finding]:
    findings: list[Finding] = []

    for lab in report.labs:
        is_critical = (
            lab.critical_low is not None and lab.value <= lab.critical_low
        ) or (lab.critical_high is not None and lab.value >= lab.critical_high)
        is_abnormal = (
            lab.reference_low is not None and lab.value < lab.reference_low
        ) or (lab.reference_high is not None and lab.value > lab.reference_high)

        evidence = f"{lab.name} = {lab.value} {lab.unit}"

        if is_critical:
            findings.append(
                Finding(
                    code=f"CRITICAL_{lab.name.upper()}",
                    severity="critical",
                    message=(
                        f"{lab.name} is at a critical level: {lab.value} {lab.unit}."
                    ),
                    evidence=evidence,
                )
            )
        elif is_abnormal:
            findings.append(
                Finding(
                    code=f"ABNORMAL_{lab.name.upper()}",
                    severity="abnormal",
                    message=(
                        f"{lab.name} is outside the reference range: "
                        f"{lab.value} {lab.unit}."
                    ),
                    evidence=evidence,
                )
            )

    return findings


def safety_level(
    completeness: list[Finding], clinical: list[Finding]
) -> str:
    if any(f.severity == "critical" for f in clinical):
        return "critical"
    if completeness:
        return "incomplete"
    if clinical:
        return "review"
    return "routine"


def reconcile_outputs(
    clinical_findings: list[Finding],
    summary: SummaryOutput,
    recommendation: RecommendationOutput,
) -> ReconciledOutput:
    abnormal_findings = list(summary.abnormal_findings)
    dropped_findings_recovered: list[str] = []

    for finding in clinical_findings:
        if finding.severity not in ("abnormal", "critical"):
            continue

        already_referenced = any(
            finding.code in entry or finding.evidence in entry
            for entry in abnormal_findings
        )

        if not already_referenced:
            abnormal_findings.append(
                f"{finding.message} Evidence: {finding.evidence}"
            )
            dropped_findings_recovered.append(finding.code)

    report_summary_text = summary.overview + "\n\n" + "\n".join(
        f"- {p}" for p in summary.key_points
    )
    recommendation_text = recommendation.overview + "\n\n" + "\n".join(
        f"- {a}" for a in recommendation.action_items
    )

    return ReconciledOutput(
        report_summary=report_summary_text,
        abnormal_findings=abnormal_findings,
        recommendation=recommendation_text,
        dropped_findings_recovered=dropped_findings_recovered,
    )
