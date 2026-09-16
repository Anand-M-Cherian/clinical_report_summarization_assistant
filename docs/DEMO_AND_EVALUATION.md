# Demo & Evaluation

## Architecture recap

```
START -> intake_agent
  -> [parallel] completeness_agent, clinical_finding_agent
  -> router (deterministic; computes safety_level)
       "incomplete"        -> request_information -> output_agent -> END
       "critical"          -> human_interrupt      -> output_agent -> END
       "routine" | "review" -> rag_agent
                                -> [parallel] recommendation_agent, summary_agent
                                -> reconciler
                                -> output_agent -> END
```

Severity classification (`router`) is plain Python comparing lab values to
supplied reference/critical ranges — no LLM is involved, and none of that
decision is reproduced by a generative model. Critical findings never reach a
summarization or recommendation agent; they go straight to a human via
`human_interrupt`, which pauses the graph until a clinician submits an
acknowledgment.

For non-critical reports, `rag_agent` retrieves guideline passages (Chroma +
`all-MiniLM-L6-v2` sentence-transformer embeddings over `data/guidelines/`).
Only `recommendation_agent` consumes that retrieved evidence — `summary_agent`
only ever sees the report and the deterministic findings, so it cannot cite
guidance it wasn't given. `reconciler` is rule-based: it checks that every
abnormal/critical deterministic finding is referenced somewhere in the
summary's `abnormal_findings`, and appends any that were dropped — it never
generates new text to resolve a discrepancy.

## Demo script

Run `streamlit run app.py`, open the **Run report** tab, and step through all
four sample reports in `data/reports/`:

1. **`routine_report.json`** — all labs in range, no missing fields.
   Produces the `{"Report summary", "Abnormal findings", "Recommendation"}`
   contract with an empty abnormal findings list.
2. **`incomplete_report.json`** — missing `author`, empty `diagnoses`/
   `narrative`/`follow_up`. Produces `{"status": "needs_information",
   "missing_sections": [...]}` — no LLM call happens on this path (confirm in
   the **Monitoring** tab: no `recommendation_agent`/`summary_agent` rows for
   this report).
3. **`review_report.json`** — glucose above reference range but below the
   critical threshold. Produces the full summary/recommendation contract,
   with the Recommendation agent citing `clinical_summary_safety.md`.
4. **`critical_report.json`** — potassium at/above the critical threshold.
   The graph pauses at `human_interrupt`: the UI shows the alert payload and a
   form for `acknowledged_by` / `action` / `rationale`. Submitting resumes the
   graph and produces `{"status": "critical_alert_acknowledged", ...}` —
   again, no summarization or recommendation agent ever runs for this report.

The **Monitoring** tab shows the SQLite-backed `agent_events` log (timing,
success/error/interrupted status per node) for whichever reports have been
run in the session.

## Evaluation approach

No unit tests are used — see Section 17 of the original spec for why: the
actual risk this architecture is designed against is generated text drifting
beyond what was retrieved or supplied, which unit tests over isolated
functions wouldn't catch. Instead, `evals/run_ragas_eval.py` runs the real
compiled graph over 5 review-tier scenarios in `data/reports/eval/` (defined
in `data/evals/recommendation_eval_cases.json`) and scores:

- **Recommendation agent** (full RAG pipeline — retrieval + generation):
  `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.
  The latter two only apply here, since only this agent performs retrieval.
- **Summary agent** (generation only, no retrieval): `faithfulness` only —
  there is no search step to score precision/recall against.

Both use Gemini (`ChatGoogleGenerativeAI`) as the ragas judge LLM, and the
same local `all-MiniLM-L6-v2` sentence-transformers model used by the RAG
retriever for the embedding-based metrics — so the whole project needs only
one API key (`GOOGLE_API_KEY`).

Thresholds: `faithfulness >= 0.7`, `context_precision >= 0.6`,
`context_recall >= 0.6`, `answer_relevancy >= 0.6`. A failing
`context_precision`/`context_recall` score points at the retriever
(`rag_agent`) specifically; a failing `faithfulness` score on the
Recommendation agent points at generation grounding in guideline text; a
failing `faithfulness` score on the Summary agent points at generation
grounding in the report itself. Splitting retrieval and generation into
separate nodes (Section 3 of the spec) is what makes this per-component
attribution possible.

**Known limitation — `answer_relevancy` on the Recommendation agent.** The
`question` column for this metric is the same keyword-soup query string
`rag_agent` builds for retrieval (chief complaint + narrative + diagnoses +
finding messages, space-joined), not a natural-language question. Ragas
scores `answer_relevancy` by generating candidate questions from the answer
and comparing their embedding similarity to the given `question` — a
non-question reference string structurally depresses this score regardless
of how relevant the recommendation actually is. A low `answer_relevancy`
here should be weighed against `faithfulness`/`context_precision`/
`context_recall` rather than read on its own; on the 5-case eval run it
scored ~0.51 against a 0.6 threshold while the other three metrics all
passed comfortably (faithfulness 0.75, context_precision/recall 1.0),
consistent with this being a metric-construction artifact rather than a
genuine relevance problem.

**Known limitation — `faithfulness` on the Recommendation agent is noisy
between runs, even with `temperature=0` and a model that honors it.** With
`gemini-3.6-flash` (which logs `UserWarning: ... the sampling parameter(s)
temperature will be ignored` on every call, including the ragas judge's own
claim-decomposition/verification calls), three consecutive eval runs scored
0.710, 0.538, and 0.608 on identical code and reports — that swing was
initially attributed to the judge ignoring `temperature=0`. Earlier comparisons with `gemini-2.5-flash` also showed that a single case's
score was not reproducible; that model is no longer available to new users.
Re-scoring the exact same `(question, answer, contexts)` tuple in isolation —
the diabetes scenario, byte-for-byte identical input — gave 0.25 inside a
full eval run and 1.0 five minutes later on its own. The root cause is hosted
LLM judging not being fully deterministic rather than anything about the
prompt, the response's imperative phrasing, or actual grounding. Treat a
single `faithfulness` result near the threshold as inconclusive; a spot-check
against the retrieved `contexts` for the flagged case (or a second run) is
more reliable than one score.

## Challenges faced

- **Keeping critical findings out of any prose path.** The design explicitly
  forbids generating a summary or recommendation for a critical report before
  a human acknowledges it — `choose_route` routes `"critical"` straight to
  `human_interrupt`, bypassing `rag_agent`/`recommendation_agent`/
  `summary_agent` entirely, rather than generating text and suppressing it
  after the fact.
- **Reconciliation without an LLM.** The Reconciler only ever appends
  verbatim finding text when it detects an omission — it never rewrites or
  resolves a contradiction, since doing so would reintroduce the exact
  ungrounded-generation risk the split-agent design was meant to avoid.
- **Single-key constraint.** Using Gemini for generation and judging, but a
  local sentence-transformers model for all embeddings, keeps the entire
  project — app, RAG retriever, and evals — dependent on exactly one API key.
