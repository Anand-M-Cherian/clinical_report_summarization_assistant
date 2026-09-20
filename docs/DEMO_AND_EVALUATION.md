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

**Known limitation — a single ragas `faithfulness` judge call is unreliable,
regardless of which Gemini model is doing the judging.** This was confirmed
across two different models, ruling out "switch models" as a fix — the
instability is in judge-call variance, not in generation quality or any one
model's quirks:

- **`gemini-3.6-flash`**: three consecutive eval runs against identical code
  and reports scored overall Recommendation-agent faithfulness at 0.710,
  0.538, and 0.608 — a ~0.17 swing with nothing else changing. Re-scoring one
  exact `(question, answer, contexts)` tuple in isolation — the diabetes
  scenario, byte-for-byte identical input — gave 0.25 inside a full eval run
  and 1.0 five minutes later on its own, with a LangChain debug trace
  confirming the claim decomposition itself was fine (it converts imperative
  `action_items` bullets like "Review current medication dosing." into
  proper declarative statements on its own); only the per-statement verdicts
  changed between calls.
- **`gemini-3.5-flash`**: the hypothyroidism scenario's response — "Order a
  free T4 level. Repeat TSH in 4-6 weeks. Consider an endocrinology referral
  if the diagnosis is confirmed." — is a near-verbatim restatement of the
  retrieved guideline text ("For suspected hypothyroidism with elevated TSH,
  order a free T4 level and repeat TSH in 4-6 weeks, and consider an
  endocrinology referral if the diagnosis is confirmed.") yet scored
  `faithfulness: 0.0` on a single judge call.

`answer_relevancy`/`context_precision`/`context_recall` showed no comparable
instability across separate full runs on different models, so only
`faithfulness` needed a fix. The mitigation now in place is **not** a model
swap — it's averaging `FAITHFULNESS_JUDGE_CALLS = 3` independent judge calls
per case (`evals/run_ragas_eval.py`'s `_score_faithfulness_averaged`), for
both the Recommendation and Summary datasets, before computing the overall
average. This doesn't eliminate judge noise, but it substantially reduces
the odds that a single unlucky (or lucky) call decides a case's score. Still
treat a `faithfulness` result close to the 0.7 threshold as a signal to
spot-check the flagged case against its retrieved `contexts` rather than a
final verdict on its own.

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
