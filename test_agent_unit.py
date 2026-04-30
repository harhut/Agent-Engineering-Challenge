"""
Unit tests for agent.py — the plumbing for parallel tool dispatch and
prompt caching. test_evals.py covers behavioral correctness on cached
agent outputs; this file mocks the Anthropic client and pokes at the
loop's wiring directly so we can iterate without an API key.
"""
import copy
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import agent


# ----------------------------------------------------------------------------
# Tiny fakes for the Anthropic SDK response shapes.
# ----------------------------------------------------------------------------
class FakeBlock:
    def __init__(self, type_, **kw):
        self.type = type_
        for k, v in kw.items():
            setattr(self, k, v)


def _tu(id_, name, input_):
    return FakeBlock("tool_use", id=id_, name=name, input=input_)


def _text(s):
    return FakeBlock("text", text=s)


def _usage(**kw):
    base = dict(input_tokens=0, output_tokens=0,
                cache_creation_input_tokens=0, cache_read_input_tokens=0)
    base.update(kw)
    return MagicMock(**base)


class FakeResp:
    def __init__(self, content, usage=None):
        self.content = content
        self.usage = usage or _usage()


_ANSWERS_OK = {"answers": [{
    "question_id": "Q1", "question": "q", "category": "technical",
    "answer": "a [DOC-X]", "confidence": "high",
    "sources": ["DOC-X"], "flags": [],
}]}

_RFP = {"prospect": "X", "questions": [{"id": "Q1", "text": "q"}]}


def _make_client(responses):
    """A fake Anthropic client whose messages.create() yields canned responses."""
    fc = MagicMock()
    fc.messages.create.side_effect = list(responses)
    return fc


# =============================================================================
# 1. Parallel dispatch
# =============================================================================
def test_search_kb_calls_dispatch_in_parallel(monkeypatch):
    """Two search_kb tool_uses in one assistant turn must run concurrently.
    A 2-party Barrier with a short timeout proves overlap: if dispatch is
    sequential the first thread waits alone and the barrier breaks."""
    barrier = threading.Barrier(2, timeout=2.0)
    completions = []

    def slow_search(query, category=None):
        barrier.wait()  # raises BrokenBarrierError on timeout
        completions.append(time.monotonic())
        return [{"id": "DOC-X", "title": "x", "category": "technical",
                 "snippet": "x", "score": 1}]

    monkeypatch.setattr(agent, "search_kb", slow_search)

    responses = [
        FakeResp([
            _tu("a", "search_kb", {"query": "alpha"}),
            _tu("b", "search_kb", {"query": "bravo"}),
        ]),
        FakeResp([_tu("s1", "submit_answers", _ANSWERS_OK)]),  # draft
        FakeResp([_tu("s2", "submit_answers", _ANSWERS_OK)]),  # final
        FakeResp([_text("done")]),
    ]
    fake_client = _make_client(responses)

    with patch.object(agent.anthropic, "Anthropic", return_value=fake_client):
        result = agent.run_agent(_RFP, verbose=False)

    assert len(completions) == 2, "both search_kb calls must complete"
    assert result["answers"][0]["question_id"] == "Q1"


def test_dispatch_preserves_tool_use_id_pairing(monkeypatch):
    """Even when calls finish out of order, every tool_result must pair with
    the originating tool_use_id."""
    def search(query, category=None):
        # 'slow' takes longer; if we naively zip by completion order we'd
        # mis-pair the ids.
        time.sleep(0.05 if query == "slow" else 0.0)
        return [{"id": f"DOC-{query}", "title": "t", "category": "technical",
                 "snippet": query, "score": 1}]
    monkeypatch.setattr(agent, "search_kb", search)

    captured = []
    responses = [
        FakeResp([
            _tu("id-slow", "search_kb", {"query": "slow"}),
            _tu("id-fast", "search_kb", {"query": "fast"}),
        ]),
        FakeResp([_tu("s1", "submit_answers", _ANSWERS_OK)]),
        FakeResp([_tu("s2", "submit_answers", _ANSWERS_OK)]),
        FakeResp([_text("done")]),
    ]

    def capture(**kwargs):
        # messages is a live list mutated by the agent loop — snapshot it.
        captured.append(copy.deepcopy(kwargs.get("messages", [])))
        return responses.pop(0)

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = capture

    with patch.object(agent.anthropic, "Anthropic", return_value=fake_client):
        agent.run_agent(_RFP, verbose=False)

    # Second messages.create payload's last user turn carries the tool_results
    # produced by parallel dispatch.
    user_results = captured[1][-1]["content"]
    pairing = {r["tool_use_id"]: r["content"] for r in user_results}
    assert set(pairing.keys()) == {"id-slow", "id-fast"}
    assert "slow" in pairing["id-slow"]
    assert "fast" in pairing["id-fast"]


# =============================================================================
# 2. Prompt caching — request shape
# =============================================================================
def test_system_prompt_uses_cache_control(monkeypatch):
    """The system param must be a list of content blocks, with cache_control
    set on the trailing block. (Caching the system prefix is the whole point.)"""
    monkeypatch.setattr(agent, "search_kb", lambda *a, **k: [])
    fake_client = _make_client([
        FakeResp([_tu("s1", "submit_answers", _ANSWERS_OK)]),
        FakeResp([_tu("s2", "submit_answers", _ANSWERS_OK)]),
        FakeResp([_text("done")]),
    ])

    with patch.object(agent.anthropic, "Anthropic", return_value=fake_client):
        agent.run_agent(_RFP, verbose=False)

    kwargs = fake_client.messages.create.call_args_list[0].kwargs
    system = kwargs["system"]
    assert isinstance(system, list), "system must be a list of content blocks"
    assert system[-1].get("cache_control") == {"type": "ephemeral"}, (
        "last system block must mark a cache breakpoint"
    )
    # And it actually contains the system prompt text.
    assert any(b.get("type") == "text" and "Solutions Engineer" in b.get("text", "")
               for b in system)


def test_tools_use_cache_control_on_last_block(monkeypatch):
    """cache_control on the last tool definition extends the cached prefix
    through the tools list, so retrieval-loop turns hit the cache."""
    monkeypatch.setattr(agent, "search_kb", lambda *a, **k: [])
    fake_client = _make_client([
        FakeResp([_tu("s1", "submit_answers", _ANSWERS_OK)]),
        FakeResp([_tu("s2", "submit_answers", _ANSWERS_OK)]),
        FakeResp([_text("done")]),
    ])

    with patch.object(agent.anthropic, "Anthropic", return_value=fake_client):
        agent.run_agent(_RFP, verbose=False)

    kwargs = fake_client.messages.create.call_args_list[0].kwargs
    tools = kwargs["tools"]
    assert tools, "tools must be sent"
    assert tools[-1].get("cache_control") == {"type": "ephemeral"}


def test_request_shape_is_consistent_across_turns(monkeypatch):
    """system + tools (the cached prefix) must be byte-identical across every
    turn, otherwise the cache breakpoint moves and we miss."""
    monkeypatch.setattr(agent, "search_kb", lambda *a, **k: [
        {"id": "DOC-X", "title": "x", "category": "technical",
         "snippet": "x", "score": 1}])
    fake_client = _make_client([
        FakeResp([_tu("a", "search_kb", {"query": "x"})]),
        FakeResp([_tu("s1", "submit_answers", _ANSWERS_OK)]),
        FakeResp([_tu("s2", "submit_answers", _ANSWERS_OK)]),
        FakeResp([_text("done")]),
    ])

    with patch.object(agent.anthropic, "Anthropic", return_value=fake_client):
        agent.run_agent(_RFP, verbose=False)

    calls = fake_client.messages.create.call_args_list
    assert len(calls) >= 2
    first_system = calls[0].kwargs["system"]
    first_tools = calls[0].kwargs["tools"]
    for c in calls[1:]:
        assert c.kwargs["system"] == first_system
        assert c.kwargs["tools"] == first_tools


# =============================================================================
# 3. Usage aggregation — observability for verifying caching at runtime
# =============================================================================
def test_usage_is_aggregated_across_turns(monkeypatch):
    """run_agent should sum input/output/cache_creation/cache_read tokens so
    a caller can verify caching is working from the returned dict alone.
    Three turns: search → first submit → final submit. The cache_creation
    happens on turn 1 and cache_read shows up on every subsequent turn."""
    monkeypatch.setattr(agent, "search_kb", lambda *a, **k: [
        {"id": "DOC-X", "title": "x", "category": "technical",
         "snippet": "x", "score": 1}])

    fake_client = _make_client([
        # Turn 1: parallel search calls. Cache is being created.
        FakeResp([_tu("a", "search_kb", {"query": "alpha"})],
                 usage=_usage(input_tokens=100, output_tokens=20,
                              cache_creation_input_tokens=80,
                              cache_read_input_tokens=0)),
        # Turn 2: first submit. Cache hit on prefix.
        FakeResp([_tu("s1", "submit_answers", _ANSWERS_OK)],
                 usage=_usage(input_tokens=10, output_tokens=15,
                              cache_creation_input_tokens=0,
                              cache_read_input_tokens=80)),
        # Turn 3: review pass + final submit. Cache hit again.
        FakeResp([_tu("s2", "submit_answers", _ANSWERS_OK)],
                 usage=_usage(input_tokens=5, output_tokens=2,
                              cache_creation_input_tokens=0,
                              cache_read_input_tokens=80)),
    ])

    with patch.object(agent.anthropic, "Anthropic", return_value=fake_client):
        result = agent.run_agent(_RFP, verbose=False)

    u = result["usage"]
    assert u["input_tokens"] == 115
    assert u["output_tokens"] == 37
    assert u["cache_creation_input_tokens"] == 80
    assert u["cache_read_input_tokens"] == 160


# =============================================================================
# 4. Smoke — the existing single-tool sequential path still works
# =============================================================================
def test_single_search_kb_call_still_works(monkeypatch):
    """Belt-and-suspenders: parallelization must not regress the 1-call case."""
    seen = []
    def search(query, category=None):
        seen.append(query)
        return [{"id": "DOC-X", "title": "x", "category": "technical",
                 "snippet": "x", "score": 1}]
    monkeypatch.setattr(agent, "search_kb", search)

    fake_client = _make_client([
        FakeResp([_tu("a", "search_kb", {"query": "only"})]),
        FakeResp([_tu("s1", "submit_answers", _ANSWERS_OK)]),
        FakeResp([_tu("s2", "submit_answers", _ANSWERS_OK)]),
        FakeResp([_text("done")]),
    ])

    with patch.object(agent.anthropic, "Anthropic", return_value=fake_client):
        result = agent.run_agent(_RFP, verbose=False)

    assert seen == ["only"]
    assert result["review_pass"] is True
    assert result["answers"][0]["question_id"] == "Q1"
