# Responsible AI Use in Clinical Summarization

## Scope and Limitations

This system assists licensed clinicians by summarizing structured clinical
reports and suggesting possible follow-up actions grounded in retrieved
guideline text. It does not diagnose conditions, does not prescribe or adjust
medications, and does not replace clinical judgment. All severity
classification is performed by deterministic, non-AI logic comparing lab
values to reference and critical ranges supplied with the report — no
generative model participates in that decision.

## Human Oversight Requirements

Every output produced by this system is intended for review by a licensed
clinician before any action is taken. Critical findings are routed directly to
a human for acknowledgment and are never summarized or narrated by a
generative model prior to that acknowledgment, since generating prose about an
unreviewed critical result could delay the clinician's attention to it.
Non-critical outputs (summaries and recommendations) must still be reviewed
before being acted upon; they are decision support, not decision replacement.

## Disclaimers for Generated Content

Any recommendation text produced by this system should be treated as a
starting point for clinician review, not a final instruction. Recommendations
are grounded only in the deterministic findings for the report and the
retrieved guideline passages provided to the model — if a recommendation cites
a guideline source, that citation should be checked against the original
guideline text before the recommendation is followed. Summaries are grounded
only in the source report and deterministic findings; they should never
introduce facts, diagnoses, or interpretations beyond what was supplied.
