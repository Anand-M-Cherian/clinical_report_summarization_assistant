from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator


class LabResult(BaseModel):
    name: str
    value: float
    unit: str
    reference_low: float | None = None
    reference_high: float | None = None
    critical_low: float | None = None
    critical_high: float | None = None

    @model_validator(mode="after")
    def validate_ranges(self) -> "LabResult":
        if (
            self.reference_low is not None
            and self.reference_high is not None
            and self.reference_low > self.reference_high
        ):
            raise ValueError("reference_low cannot exceed reference_high")
        return self


class ClinicalReport(BaseModel):
    report_id: str
    patient_id: str
    report_type: str
    encounter_date: str
    author: str = ""
    chief_complaint: str = ""
    history: str = ""
    narrative: str = ""
    diagnoses: list[str] = Field(default_factory=list)
    labs: list[LabResult] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    follow_up: str = ""


class Finding(BaseModel):
    code: str
    severity: Literal["info", "abnormal", "critical"]
    message: str
    evidence: str


class GuidelineEvidence(BaseModel):
    source: str
    section: str
    text: str
    score: float = Field(
        description="A fused, rank-based relevance score (1/(rank+1) after RRF "
        "fusion of dense and lexical retrieval) — not a raw cosine similarity or "
        "BM25 score. Larger is more relevant; the absolute scale is not "
        "comparable to pre-hybrid-search score values."
    )


class SummaryOutput(BaseModel):
    overview: str
    key_points: list[str]
    abnormal_findings: list[str]


class RecommendationOutput(BaseModel):
    overview: str
    action_items: list[str]
    guideline_citations: list[str]


class ReconciledOutput(BaseModel):
    report_summary: str
    abnormal_findings: list[str]
    recommendation: str
    dropped_findings_recovered: list[str] = Field(default_factory=list)

    def to_contract(self) -> dict[str, Any]:
        return {
            "Report summary": self.report_summary,
            "Abnormal findings": self.abnormal_findings,
            "Recommendation": self.recommendation,
        }


class ClinicianAck(BaseModel):
    acknowledged_by: str = Field(min_length=2)
    action: Literal["escalate_to_er", "acknowledge_pending_action", "false_alarm"]
    rationale: str = Field(min_length=5)


class ClinicalState(TypedDict, total=False):
    report: dict[str, Any]
    completeness_findings: list[dict[str, Any]]
    clinical_findings: list[dict[str, Any]]
    safety_level: Literal["routine", "review", "incomplete", "critical"]
    route: str
    guideline_evidence: list[dict[str, Any]]
    recommendation: dict[str, Any]
    summary: dict[str, Any]
    reconciled_output: dict[str, Any]
    clinician_ack: dict[str, Any]
    status: str
    final_output: dict[str, Any]
    audit: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
