# Helios RFP Responder

Starter repo for the **Applied AI Agent Engineering Challenge** — an agent that
takes an RFP questionnaire and produces a grounded, cited, structured draft
response in minutes instead of hours.

```
┌──────────┐   ┌──────────────┐   ┌───────────────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────┐
│ RFP json │──▶│ Parse + Tag  │──▶│  Retrieve             │──▶│ Draft + Cite │──▶│ Self-Review  │──▶│  JSON  │
│ (Q list) │   │ (in-prompt)  │   │  tool: search_kb()    │   │ + confidence │   │ (2nd submit) │   │ export │
└──────────┘   └──────────────┘   └──────────┬────────────┘   └──────────────┘   └──────────────┘   └────────┘
                                             │
                                  ┌──────────▼──────────┐
                                  │ kb.py — mock corpus │
                                  │ 15 docs, keyword    │
                                  │ scorer, top-3       │
                                  └─────────────────────┘
```

## Architecture & design rationale

The brief frames this as a *consistency* problem more than a *generation*
problem — eight hours of copy-paste produces answers that contradict each
other. Every structural choice below optimizes for **groundedness and
cross-answer consistency** first, throughput second.

### 1. Single agent, two tools — not an orchestrator/worker swarm

**What we did.** One Claude context owns the whole RFP end-to-end. It has
exactly two tools: `search_kb` (read) and `submit_answers` (write).

**Why.** The headline failure mode in the brief is "answers contradict each
other because no one reviews the response holistically." A multi-agent design
(one worker per question, an orchestrator to merge) *recreates* that failure
mode — each worker drafts in isolation, and you need a separate reconciliation
step that has to re-read everything anyway. With a single context, the model
that drafts Q5 has already seen what it wrote for Q1. Consistency is ambient,
not bolted on.

**Trade-off.** Doesn't scale to a 200-question RFP in one context window. The
fix is sharding into ~20-question batches that share a small "facts cache"
(numbers/dates extracted from earlier batches) — clear next step, didn't need
it at 5 questions. We chose **correct at small N** over **fast at large N**
for a 1h45 build.

### 2. Structured output via tool schema — not "respond in JSON"

**What we did.** `submit_answers` is a tool whose `input_schema` *is* the
output contract (`question_id`, `answer`, `confidence` enum, `sources[]`,
`flags[]`). The agent's only way to return results is to call it.

**Why.** Three things for free: (a) the API validates shape, so we never
regex a malformed JSON string out of prose; (b) `tool_choice` can force the
call, so the agent can't "forget" to emit structured output; (c) the schema
doubles as live documentation — the field descriptions are in the prompt
because they're in the tool definition.

**Alternative considered.** A `response_format: json` style constraint or
post-hoc parsing. Both work, but tool-use is the idiomatic Anthropic pattern,
keeps the agent loop uniform (everything is a tool call), and lets us
intercept the *first* draft for the review pass below.

### 3. Review pass = re-prompt after first `submit_answers`

**What we did.** After the model's first `submit_answers` call, we don't
stop. We append a tool_result that says "here's your draft — review for
contradictions, uncited claims, missing sub-parts; call `submit_answers`
again with the final." Same context, one extra model call.

**Why.** This is the cheapest implementation of the brief's "self-check
answers for consistency" requirement that actually works. The model is a
better reviewer than first-drafter when it can see the whole artifact at
once. Because output is a tool call (see #2), we can capture the draft
deterministically and feed it back without parsing.

**Trade-off.** ~2x token cost per RFP. For 200-Q RFPs you'd review in
chunks. We also keep the *first* submit in `out/` for eval purposes
(`review_pass: true` flag) so you can diff what the review actually fixed —
useful demo moment.

### 4. Confidence is categorical (high / medium / low) + free-text `flags[]`

**What we did.** Three buckets, defined operationally in the system prompt
(high = KB directly answers every part; medium = partial coverage or minor
extrapolation; low = placeholder for human review). `low` ⇒ must carry a
flag naming the SME to route to.

**Why.** A 0–1 float is uncalibrated noise without a training set; the SE
reading the output just wants to know *which answers to look at*. Three
buckets map directly to a triage workflow (ship / skim / rewrite). It's also
**testable** — `test_calibration_*` can assert exact values instead of
thresholding a float. The `flags[]` array is the human-readable *why*, which
is what the SE actually needs to act.

**Trade-off.** `flags[]` is free-text, so downstream tooling can't route on
it automatically. With another hour we'd make it an enum
(`needs_sme:security`, `no_kb_match`, `stale_source`) — called out in the
retro.

### 5. Mock KB is hand-seeded with *specific* facts — and one deliberate conflict

**What we did.** 15 docs in `kb.py`, each with concrete numbers (1.8s
latency, $42/endpoint, SOC 2 dated 2026-02-14, 87 FS customers). One doc,
`DOC-PR-01-2024`, is a **stale pricing sheet** ($48/endpoint) marked
"superseded" — it conflicts with the current `DOC-PR-01` on purpose.
Retrieval is naive keyword scoring, top-3.

**Why.** The brief says mock-data richness counts under Agent Quality.
Specific facts make grounding evals *bite* — `FaithfulnessMetric` and the
deterministic spot-checks have something to verify against. The stale doc is
a planted regression: real KBs are full of outdated proposals, and "agent
parrots whichever doc ranked first" is the most realistic silent failure.
`test_stale_pricing_not_used` guards it.

**Trade-off.** Keyword search is the weakest link here (intentionally — the
brief says don't build real integrations). It happens to rank the current
sheet above the stale one; if you want a *genuinely* hard test, retitle the
stale doc to tie on relevance and see whether the agent reads the
"ARCHIVED" note.

### 6. Eval suite is layered: deterministic → faithfulness → rubric

**What we did.** `test_evals.py` has three tiers, fastest/cheapest first:

1. **Deterministic** (no LLM): structure, source-ids-exist, calibration on
   known-coverage Qs, pricing arithmetic, stale-doc check, false-premise
   correction. These run in <100ms off cached `out/*.json` and need no API
   key — teammates can iterate without burning quota.
2. **Faithfulness** (`FaithfulnessMetric`): for each high-confidence answer,
   feed the judge *only the docs the agent cited* and ask whether the claims
   are supported. Catches hallucination that slipped past citations.
3. **Rubric** (`GEval`): cross-answer consistency and multi-part
   completeness — the fuzzy properties code can't easily check.

**Why this order.** It mirrors a test pyramid: lots of fast deterministic
checks, fewer expensive LLM-judge checks. It also means CI can run tier 1 on
every commit for free. Judge is **Claude** (`AnthropicModel`) so one
`ANTHROPIC_API_KEY` covers agent *and* evals, and we're not grading an
Anthropic agent with an OpenAI model.

**Why DeepEval.** Open-source (Apache 2.0), pytest-native (no separate
runner DSL), first-class Anthropic judge support. promptfoo was the
brief's suggestion but was acquired by OpenAI in March — we'd rather not
take that dependency.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
python run.py rfps/rfp_01.json          # full trace + JSON to stdout, also writes out/rfp_01.json
python run.py rfps/rfp_02.json --quiet  # JSON only
```

## Evals

```bash
deepeval test run test_evals.py    # full suite (19 tests), Claude-as-judge
pytest test_evals.py -v            # same thing, plain pytest
pytest test_evals.py -v -k "not grounding and not consistency and not multipart"
                                   # deterministic-only — runs with NO api key off cached out/
pytest test_agent_unit.py -v       # plumbing tests — parallel dispatch + prompt caching
```

Results cache to `out/` (committed) so re-running evals doesn't re-run the
agent. Set `HELIOS_FORCE_RERUN=1` to bust the cache.

**Current: 19/19 passing** across 4 RFPs. The adversarial set (`rfp_04`)
covers the four ways RFP agents typically fail: sycophancy on a false
premise, hallucinating an adjacent cert (HIPAA), parroting stale data, and
answering off-topic trivia.

## Performance: parallel retrieval + prompt caching

Two latency/cost optimizations on the agent loop:

- **Parallel tool dispatch.** `search_kb` calls within a single assistant
  turn fan out via a thread pool (`MAX_PARALLEL_TOOLS = 8`). The system
  prompt asks the model to issue all initial searches in one batched turn,
  so a 5-question RFP retrieves in ~one round-trip instead of five. The
  mock keyword scorer is in-process and won't see speedup, but the design
  is right for a real I/O-bound retriever (HTTP / vector store).
- **Prompt caching.** A `cache_control: ephemeral` breakpoint sits on the
  last tool definition, caching the system prompt + tools array as one
  prefix. Every turn after the first reads from cache (~10× cheaper input
  tokens). The agent's return dict now includes a `usage` block with
  `cache_creation_input_tokens` and `cache_read_input_tokens` so you can
  verify it from `out/*.json`. To check live: `HELIOS_FORCE_RERUN=1 python
  run.py rfps/rfp_01.json` — `cache_read_input_tokens` should be ~0 on the
  first turn and grow on each subsequent turn.

Both behaviors are covered by `test_agent_unit.py` (mocked Anthropic
client, no API key needed).

## Files

```
.
├── agent.py             # the agent loop — read this first (~200 lines, heavily commented)
├── kb.py                # 15-doc mock KB + search_kb(); includes stale DOC-PR-01-2024
├── run.py               # CLI wrapper
├── rfps/
│   ├── rfp_01.json      # the 5 canonical Qs from the brief
│   ├── rfp_02.json      # edge: no-KB-match, ambiguous, 3-part compound, computed pricing
│   ├── rfp_03.json      # short FS-flavored RFP (consistency cross-check vs rfp_01)
│   └── rfp_04.json      # adversarial: false premise, HIPAA bait, stale-data trap, off-topic
├── test_evals.py        # DeepEval/pytest suite — 19 tests, 3 tiers
├── test_agent_unit.py   # plumbing unit tests (parallel dispatch, caching)
├── out/                 # cached agent outputs (committed, so evals run keyless)
└── requirements.txt
```

## 5-minute demo script

| | |
|---|---|
| **~1m Architecture** | Diagram above. Hit three points: single agent / two tools (consistency is ambient), `submit_answers` as the structured-output contract, review pass = re-prompt after first submit. Mention DeepEval + Claude-as-judge in one breath. |
| **~2m Live run** | `python run.py rfps/rfp_01.json` — narrate the `[tool_use] search_kb(...)` lines as they stream, then scroll the JSON: point at `sources`, `confidence`, inline `[DOC-*]` cites. Then `python run.py rfps/rfp_02.json` — show Q1 (quantum crypto) coming back `low` + flagged with empty sources. "That's the knows-what-it-doesn't-know behavior." |
| **~1m Evals** | `pytest test_evals.py -v` (pre-run, terminal already open). Walk the three tiers. Land on the adversarial block: false-premise corrected, HIPAA not hallucinated, stale $48 not used, off-topic refused. "We built traps for the four common failure modes — all green, but they're regression guards now." |
| **~1m Retro** | What worked: single-context = zero consistency bugs; tool-schema output = zero JSON parsing. What's next (below). |

## Where to spend the next hour

1. **Harden the stale-data test** — retitle `DOC-PR-01-2024` so it ties the
   current sheet on relevance. Right now it passes partly because keyword
   ranking does the work; we want to prove the *agent* reads "superseded."
2. **Consistency as code, not rubric** — regex every number/date out of
   `answers[]`, assert they match across questions. Replaces the GEval
   consistency rubric with something deterministic and free.
3. **Calibration curve** — 15 Qs with known KB-coverage labels, compute
   precision/recall of `low`. Turns "knows what it doesn't know" into a
   number you can put on a slide.
4. **Shard for scale** — batch a 50-Q RFP into 3 passes with a shared
   extracted-facts cache between them. Proves the architecture scales
   without giving up the consistency property.

## Retro talking points (pre-baked honesty)

- Keyword search is the weakest link — a real Helios needs embeddings/BM25.
  We mocked it on purpose (brief says so), but it does mask retrieval
  failures the agent would have to handle in prod.
- Review pass doubles token cost. Fine at 5 Qs, needs chunking at 200.
- `flags[]` should be an enum so downstream tooling can route automatically.
- All-green evals are slightly suspicious — the adversarial set is a start,
  but a real suite would include LLM-generated paraphrases of every KB fact
  to stress retrieval recall.
