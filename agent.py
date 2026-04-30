"""
Helios RFP Responder — agent loop on the Anthropic Messages API.

Design notes (the stuff worth saying in the demo):

  * Single agent, two tools. We resisted the urge to build a multi-agent
    swarm. One model, one context, two tools (search_kb, submit_answers)
    is enough — and it makes the consistency-review step trivial because
    the model already has every draft answer in its own context.

  * Tool-forced output. The final structured JSON comes back via the
    submit_answers tool, not by asking the model to "respond in JSON".
    Tool schemas are validated by the API, so we get guaranteed shape
    without regex-parsing the assistant text.

  * Separate review pass, same context. After the model calls
    submit_answers the first time, we feed it back its own output and
    ask it to self-review for contradictions / missing citations, then
    call submit_answers again with the final version. Cheap, and it
    catches the "Q1 says 2s latency, Q5 says 5s" class of bug the brief
    explicitly calls out.

  * Parallel tool dispatch + prompt caching. search_kb calls within one
    assistant turn fan out via ThreadPoolExecutor (up to MAX_PARALLEL_TOOLS).
    The system prompt + tools array carry a cache_control: ephemeral
    breakpoint so every turn after the first reads from cache. Both are
    transparent to the model; usage telemetry is returned in the result
    dict so a caller can verify caching is active.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import anthropic

from kb import search_kb

MODEL = "claude-sonnet-4-5"
MAX_PARALLEL_TOOLS = 8

SYSTEM = """\
You are a senior Solutions Engineer at Helios Security (endpoint protection,
SIEM, and MDR vendor). You are completing an RFP response on behalf of Helios.

Process — follow it exactly:

1. PARSE the RFP into individual questions. For each, assign a category:
   one of technical | compliance | pricing | company-info. A question may
   have a secondary category.

2. RETRIEVE. For each question, call search_kb at least once with a focused
   query (and category filter when obvious). After parsing all questions,
   issue your initial searches as multiple parallel tool calls in a single
   assistant turn — the runtime fans them out concurrently, so one search
   per question in one turn is much faster than one search per turn. Use
   follow-up turns only when a result is ambiguous and a refined query is
   needed. Never answer from memory — every factual claim must be traceable
   to a returned doc id.

3. DRAFT an answer for each question. 2–5 sentences, customer-facing tone,
   specific numbers where the KB provides them. Cite source doc ids inline
   like [DOC-TD-01].

4. CONFIDENCE. Rate each answer high / medium / low:
     high   — KB directly answers every part of the question
     medium — KB covers most of it; minor extrapolation or stale data
     low    — KB has little or nothing; answer is a placeholder for human review
   If low, add a flag explaining what's missing.

5. SUBMIT by calling submit_answers with the full list. Do not emit the
   JSON as plain text — use the tool.

6. REVIEW. After your first submit you will be shown your own output and
   asked to self-review. Check for: contradictions across answers,
   uncited claims, missing sub-parts, tone. Fix and call submit_answers
   again with the final version.
"""

TOOLS = [
    {
        "name": "search_kb",
        "description": (
            "Search the Helios knowledge base (past proposals, product docs, "
            "compliance records, pricing sheets). Returns up to 3 relevant "
            "snippets with doc ids you must cite."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Focused keyword query, e.g. 'detection latency benchmark'",
                },
                "category": {
                    "type": "string",
                    "enum": ["technical", "compliance", "pricing", "company-info"],
                    "description": "Optional filter to a single doc category",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "submit_answers",
        "description": (
            "Submit the structured RFP response. Call this exactly once after "
            "drafting all answers, and once more after the review pass."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_id": {"type": "string"},
                            "question": {"type": "string"},
                            "category": {"type": "string"},
                            "answer": {"type": "string"},
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                            "sources": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Doc ids cited, e.g. ['DOC-TD-01']",
                            },
                            "flags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Human-review flags, e.g. 'no KB match — needs SME input'",
                            },
                        },
                        "required": [
                            "question_id", "question", "category",
                            "answer", "confidence", "sources", "flags",
                        ],
                    },
                }
            },
            "required": ["answers"],
        },
        # Cache breakpoint: caches the system prompt + the whole tools array
        # as one prefix. Every turn after the first reads from cache.
        "cache_control": {"type": "ephemeral"},
    },
]


# `system` param wrapped as a list of content blocks so we can attach a cache
# breakpoint. The bytes here must be identical across every turn — that's the
# cache key.
SYSTEM_BLOCKS = [
    {"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}},
]


def _dispatch(name: str, args: dict) -> str:
    """Route tool calls. Only search_kb actually does work; submit_answers
    is a sink — the loop intercepts it."""
    if name == "search_kb":
        hits = search_kb(args["query"], args.get("category"))
        return json.dumps(hits, indent=2)
    raise ValueError(f"unexpected tool dispatch: {name}")


def _accumulate_usage(total: dict, usage) -> None:
    """Sum token counters from a Messages response onto a running total.
    Missing fields default to 0 (older SDKs / cache-disabled responses)."""
    for k in ("input_tokens", "output_tokens",
              "cache_creation_input_tokens", "cache_read_input_tokens"):
        total[k] += getattr(usage, k, 0) or 0


def _run_external_tools(tool_uses) -> dict[str, str]:
    """Fan out every non-submit_answers tool call in parallel, return a
    {tool_use_id: tool_result_content} map. We dispatch concurrently because
    real retrieval is I/O-bound (HTTP / vector store) — the mock keyword
    scorer doesn't benefit, but the architecture is right for production."""
    external = [tu for tu in tool_uses if tu.name != "submit_answers"]
    if not external:
        return {}
    workers = min(MAX_PARALLEL_TOOLS, len(external))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outputs = list(pool.map(lambda tu: _dispatch(tu.name, tu.input), external))
    return {tu.id: out for tu, out in zip(external, outputs)}


def run_agent(rfp: dict, verbose: bool = True) -> dict:
    """Run the full parse→retrieve→draft→review→export loop on one RFP."""
    client = anthropic.Anthropic()

    user_turn = (
        f"RFP from prospect: {rfp.get('prospect','(unnamed)')}\n"
        f"Context: {rfp.get('context','')}\n\n"
        "Questions:\n"
        + "\n".join(f"{q['id']}. {q['text']}" for q in rfp["questions"])
    )

    messages = [{"role": "user", "content": user_turn}]
    first_submit = None
    final_submit = None
    total_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }

    # Hard cap so a runaway loop can't burn the demo.
    for step in range(40):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_BLOCKS,
            tools=TOOLS,
            messages=messages,
        )
        _accumulate_usage(total_usage, resp.usage)

        # Collect any tool_use blocks from this turn.
        tool_uses = [b for b in resp.content if b.type == "tool_use"]

        if verbose:
            for b in resp.content:
                if b.type == "text" and b.text.strip():
                    print(f"\n[assistant] {b.text.strip()[:500]}")
                if b.type == "tool_use":
                    print(f"[tool_use] {b.name}({json.dumps(b.input)[:120]})")

        # No tool calls → model is done talking. If we already have a final
        # submit, great; otherwise this is a failure mode worth surfacing.
        if not tool_uses:
            break

        # Append the assistant turn verbatim (required for tool_result pairing).
        messages.append({"role": "assistant", "content": resp.content})

        # Dispatch external tools (search_kb, ...) in parallel; submit_answers
        # is handled inline because it controls loop state.
        external_results = _run_external_tools(tool_uses)

        results = []
        for tu in tool_uses:
            if tu.name == "submit_answers":
                if first_submit is None:
                    first_submit = tu.input
                    # Kick off the review pass: feed its own output back.
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": (
                            "Draft received. Now REVIEW your own answers for: "
                            "contradictions across questions, uncited factual "
                            "claims, unanswered sub-parts, and low-confidence "
                            "items missing a flag. Then call submit_answers "
                            "again with the corrected final version."
                        ),
                    })
                else:
                    final_submit = tu.input
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": "Final response accepted.",
                    })
            else:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": external_results[tu.id],
                })

        messages.append({"role": "user", "content": results})

        if final_submit is not None:
            break
        # else keep looping (more searches, or the review pass)

    out = final_submit or first_submit or {"answers": []}
    return {
        "prospect": rfp.get("prospect"),
        "model": MODEL,
        "review_pass": final_submit is not None,
        "usage": total_usage,
        **out,
    }


if __name__ == "__main__":
    import sys
    with open(sys.argv[1]) as f:
        rfp = json.load(f)
    print(json.dumps(run_agent(rfp), indent=2))
