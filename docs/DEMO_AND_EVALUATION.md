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

For non-critical reports, `rag_agent` retrieves guideline passages from
`data/guidelines/` using hybrid search: dense retrieval (Chroma +
`all-MiniLM-L6-v2` sentence-transformer embeddings) and lexical BM25, fused by
reciprocal rank fusion (see [Hybrid retrieval](#hybrid-retrieval) below).
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

An earlier version of this section said `answer_relevancy`/
`context_precision`/`context_recall` showed no comparable instability, so
only `faithfulness` needed a fix. **That is not true for `context_precision`.**
During the hybrid-retrieval work, case 2 (hypokalemia) scored
`context_precision` 1.0 in one run and 0.583 in three others, with the same
three chunks retrieved in the same order every time. `context_precision` is
also judged by the LLM (it asks whether each retrieved chunk was useful), so
it carries the same kind of judge noise, and per-case values should not be
read as exact. The faithfulness mitigation below has not been applied to it.

The faithfulness mitigation now in place is **not** a model swap — it's
averaging `FAITHFULNESS_JUDGE_CALLS = 3` independent judge calls
per case (`evals/run_ragas_eval.py`'s `_score_faithfulness_averaged`), for
both the Recommendation and Summary datasets, before computing the overall
average. This reduces judge noise but does not remove it. Four eval runs on
identical code, with 3-call averaging in place, scored overall
Recommendation-agent faithfulness at 0.612, 0.725, 0.623 and 0.677, a 0.113
spread. Single cases varied far more: case 2 ranged from 0.374 to 0.768 and
case 5 from 0.333 to 0.714. Treat a `faithfulness` result close to the 0.7
threshold as a signal to spot-check the flagged case against its retrieved
`contexts` rather than a final verdict on its own. Compare changes across
several runs, not one.

## Hybrid retrieval

`GuidelineRetriever` (`src/clinical_assistant/rag.py`) was moved from
dense-only search to hybrid search. Its interface did not change:
`rag_agent` still calls `retriever.search(query, top_k=3)`.

- **Chunking.** Guideline files are split on markdown headings, so each
  chunk keeps its section name. Any section longer than 800 characters is
  split again with LangChain's `RecursiveCharacterTextSplitter` (90-character
  overlap), and every piece keeps the parent heading.
- **Two retrievers.** Dense retrieval (Chroma embeddings) and lexical BM25
  (`rank-bm25`, over lowercased alphanumeric tokens) each return their top 15
  candidates. Both index the same chunk list, built once per construction.
- **Fusion.** LangChain's `EnsembleRetriever` merges the two lists with
  reciprocal rank fusion at equal 0.5/0.5 weights. `GuidelineEvidence.score`
  is now `1/(rank+1)` of the fused order, so it can't be compared with the
  older cosine-based scores.
- **Embedding cache.** The Chroma collection stores a SHA-256 hash of every
  guideline file's name and content, plus the embedding model name and the
  chunking settings. If the hash matches on startup, the stored embeddings
  are reused; otherwise they are rebuilt. BM25 is cheap and is rebuilt every
  time.

**Verified:**

- **Cache hit/miss.** Two back-to-back constructions printed `rebuilding
  embeddings` and then `reusing cached embeddings` with the same hash.
  Editing a guideline file changed the hash and triggered a rebuild.
- **Chunking.** On the original guidelines, only "Condition-Specific
  Follow-Up Actions" (1,124 characters) exceeded the limit, and it was split
  into two pieces (709 and 413 characters). That split put the diabetes and
  hypokalemia guidance in one chunk. Per-condition `###` sub-headings were
  then added, so each of the five conditions is its own chunk (11 chunks in
  total), and the size fallback no longer fires on the current files.
- **Fusion.** A paraphrased diabetes query still returned the relevant
  sections. An exact-term query for "hydrochlorothiazide" ranked the
  hypokalemia guidance first, but dense-only search already did. On a corpus
  this small, BM25 showed no measurable gain.
- **No retrieval regression.** Average `context_precision` stayed at 0.917
  and `context_recall` at 1.0. Cases 1, 3, 4 and 5 kept `context_precision`
  1.0; case 2 stayed at 0.583 (apart from the one noisy 1.0 described
  above). Faithfulness stayed within judge noise on cases 1, 2, 4 and 5;
  case 3 is covered below.

**Why case 2 did not improve.** For this report, all three retrievers
(dense, BM25 and fused) rank the hypokalemia chunk third, behind the two
general sections on abnormal lab values and escalation. The query `rag_agent`
builds uses the chief complaint, narrative, diagnoses and finding messages.
It never includes `medications` or `history`, which is where
"hydrochlorothiazide" appears, so BM25 has no drug name to match. Widening
the query was left for later. `evals/run_ragas_eval.py` has its own copy of
that query logic (`_build_query`), so both places would need the change.

**Known limitation.** Each retriever's candidate pool is sized for the
default `top_k=3`. A caller asking for more results would not get a larger
pool (marked `TODO` in `rag.py`).

### Case 3: suspected regression, not confirmed

Case 3 (suspected iron-deficiency anemia) scored Recommendation faithfulness
0.800 in the one run before the per-condition sub-headings were added, and
0.567, 0.600, 0.381 and 0.611 in four runs after. None of the four reaches
the earlier score. A plausible cause is that the anemia guidance is now a
short standalone chunk (258 characters), leaving the judge less supporting
text for some claims.

It is reported as suspected rather than confirmed because the "before" side
is a single run, and single-case faithfulness has been seen to swing by
about 0.4 on identical input (case 2: 0.374 to 0.768). One high draw of
0.800 could produce this gap by itself. Confirming it would need several
runs with the sub-headings reverted, which was not done within this
project's budget. Retrieval for case 3 was unaffected: `context_precision`
and `context_recall` stayed at 1.0 in every run.

### Hypokalemia guideline content fix

The hypokalemia sub-section of `clinical_summary_safety.md` was rewritten to
be clinically specific rather than generic. It now defines hypokalemia
(serum potassium below 3.5 mmol/L; mild is typically 3.0-3.4 mmol/L). It
names thiazide diuretics such as hydrochlorothiazide, commonly prescribed for
hypertension, as a frequent cause, and lists typical symptoms (leg cramps,
fatigue, generalized weakness). The follow-up actions are unchanged, and it
adds when to seek prompt clinician review. No other section was edited.

Result from one eval run:

- **Retrieval order changed.** For case 2, the hypokalemia chunk is now
  retrieved first, ahead of the two general sections, where it was third
  before. Retrieval is deterministic, so this change is real, not judge
  noise. The query now shares specific terms with the chunk: potassium,
  mmol/L, hypertension, weakness.
- **Scores did not clearly improve.** Case 2 scored `context_precision` 1.0
  and `faithfulness` 0.667. Both fall inside the ranges already seen on
  identical input (0.583 to 1.0 and 0.374 to 0.768), so the metrics alone
  can't show an improvement. `context_precision` 1.0 is the expected value
  when the most relevant chunk ranks first, but a noisy run has also
  produced 1.0 before.
- **Overall.** Averages for this run were `context_precision` 1.0,
  `context_recall` 1.0, Recommendation faithfulness 0.669 and Summary
  faithfulness 1.0. Case 3 scored faithfulness 0.683, still below its
  single pre-sub-heading score of 0.800. This run used a changed corpus, so
  it isn't counted among the four comparable runs above.

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
