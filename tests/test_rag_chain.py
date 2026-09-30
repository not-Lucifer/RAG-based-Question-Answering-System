"""RAG chain + memory with a fake LLM."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.chains.memory import condense, format_history
from app.chains.rag_chain import answer, format_context, stream_answer
from app.core.exceptions import LLMUnavailableError
from app.llm import factory as llm_factory
from app.llm.prompts import NOT_FOUND_MESSAGE, QA_PROMPT, SYSTEM_PROMPT


class BrokenChatModel(FakeListChatModel):
    """Simulates a provider outage."""

    def _call(self, *args: Any, **kwargs: Any) -> str:
        raise RuntimeError("connection refused")


def test_prompt_matches_blueprint() -> None:
    assert "ONLY" in SYSTEM_PROMPT and NOT_FOUND_MESSAGE in SYSTEM_PROMPT
    msgs = QA_PROMPT.format_messages(context="[1] (a.pdf, p.1)\ntext", question="Q?")
    assert msgs[0].type == "system" and "Question: Q?" in msgs[1].content


def test_in_material_question_is_answered_with_sources(indexed_docs, fake_llm) -> None:
    llm = fake_llm("Dijkstra runs in O(V^2) with an array [1].")
    result = answer("What is the time complexity of Dijkstra?", mode="similarity", top_k=3)
    assert result.used_context is True
    assert result.answer.startswith("Dijkstra runs")
    assert [s.n for s in result.sources] == list(range(1, len(result.sources) + 1))
    assert result.sources[0].source == "cn.pdf" and result.sources[0].page == 2
    assert llm.calls == 1  # exactly one LLM call (no condense without history)


def test_out_of_material_question_refused_without_llm(indexed_docs, fake_llm) -> None:
    llm = fake_llm("SHOULD NOT BE USED")
    result = answer("Who won IPL 2020 cricket tournament?")
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.used_context is False and result.sources == []
    assert llm.calls == 0


def test_empty_index_refuses(fake_llm) -> None:
    fake_llm("unused")
    assert answer("What is normalization?").answer == NOT_FOUND_MESSAGE


def test_filters_passed_through(indexed_docs, fake_llm) -> None:
    fake_llm("answer [1]")
    result = answer("normalization B-tree advantages", filters={"subject": "DBMS"})
    assert result.sources and all(s.subject == "DBMS" for s in result.sources)


def test_format_context_numbering() -> None:
    from langchain_core.documents import Document

    ctx = format_context([Document(page_content="hello", metadata={"source": "DBMS_Unit3.pdf", "page": 12})])
    assert ctx == "[1] (DBMS_Unit3.pdf, p.12)\nhello"


def test_llm_failure_maps_to_unavailable(indexed_docs) -> None:
    llm_factory.set_llm_override(BrokenChatModel(responses=["x"]))
    with pytest.raises(LLMUnavailableError):
        answer("Dijkstra time complexity")


def test_missing_openai_key_is_unavailable(indexed_docs) -> None:
    llm_factory.set_llm_override(None)
    with pytest.raises(LLMUnavailableError, match="OPENAI_API_KEY"):
        answer("Dijkstra time complexity")


def test_streaming_yields_tokens_then_result(indexed_docs, fake_llm) -> None:
    fake_llm("Handshake uses SYN [1].")
    events = list(stream_answer("TCP three-way handshake SYN ACK"))
    kinds = [k for k, _ in events]
    assert kinds[-1] == "result" and kinds.count("token") >= 1
    result = events[-1][1]
    assert "".join(p for k, p in events if k == "token").strip() == result.answer
    assert result.sources


# ------------------------------------------------------------------ memory


def test_format_history_keeps_last_turns() -> None:
    msgs = []
    for i in range(6):
        msgs += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    text = format_history(msgs, turns=2)
    assert text.splitlines() == ["Student: q4", "Assistant: a4", "Student: q5", "Assistant: a5"]


def test_condense_skipped_without_history(fake_llm) -> None:
    llm = fake_llm("unused")
    assert condense("explain its advantages", "") == "explain its advantages"
    assert llm.calls == 0


def test_follow_up_is_rewritten_and_retrieves_btree(indexed_docs, fake_llm) -> None:
    llm = fake_llm("What are the advantages of B-trees?", "B-trees stay balanced [1].")
    history = [
        {"role": "user", "content": "What is a B-tree?"},
        {"role": "assistant", "content": "A B-tree is a balanced search tree [1]."},
    ]
    result = answer("explain its advantages", history=history, mode="similarity", top_k=2)
    assert result.standalone_question == "What are the advantages of B-trees?"
    assert result.sources[0].source == "dbms.pdf" and result.sources[0].page == 2
    assert llm.calls == 2  # condense + answer


def test_refusal_needs_no_llm_provider(indexed_docs) -> None:
    llm_factory.set_llm_override(None)  # no OPENAI_API_KEY configured in tests
    assert answer("Who won IPL 2020 cricket tournament?").answer == NOT_FOUND_MESSAGE


def test_ollama_model_kept_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import get_settings

    llm_factory.set_llm_override(None)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "45m")
    get_settings.cache_clear()
    llm = llm_factory.get_llm()
    assert type(llm).__name__ == "ChatOllama" and llm.keep_alive == "45m"


def test_warm_up_never_raises_when_ollama_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import get_settings
    from app.main import _warm_up

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")  # nothing listens here
    get_settings.cache_clear()
    _warm_up()  # must only log a warning


# ------------------------------------------------------------------ condensing guards

REFUSED_3NF = [
    {"role": "user", "content": "what is 3nf"},
    {"role": "assistant", "content": NOT_FOUND_MESSAGE},
]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("what is resume?", False),
        ("What is the difference between a resume and a CV?", False),
        ("explain its advantages", True),
        ("What about the time complexity?", True),
        ("why?", True),
        ("and B+ trees?", True),
        ("Give an example", True),
        ("explain the last one in detail", True),
        ("tell me more about the second point", True),
    ],
)
def test_needs_condensing(question: str, expected: bool) -> None:
    from app.chains.memory import needs_condensing

    assert needs_condensing(question) is expected


def test_standalone_question_mid_chat_is_not_rewritten(fake_llm) -> None:
    llm = fake_llm("What is the meaning of 3NF in the context of database normalization?")
    assert condense("what is resume?", format_history(REFUSED_3NF)) == "what is resume?"
    assert llm.calls == 0  # also saves an LLM round-trip


def test_rewrite_that_drops_topic_is_rejected(fake_llm) -> None:
    # A small model "rewrote" the follow-up into the previous question (real bug seen with llama3.2:3b).
    fake_llm("What is the meaning of 3NF in the context of database normalization?")
    question = "what does it say about resume formats?"
    assert condense(question, format_history(REFUSED_3NF)) == question


def test_standalone_question_after_refusal_is_answered(indexed_docs, fake_llm) -> None:
    fake_llm("A B-tree is a balanced search tree [1].")
    result = answer("What is a B-tree?", history=REFUSED_3NF, mode="similarity")
    assert result.used_context and result.standalone_question == "What is a B-tree?"


def test_echoed_context_label_is_stripped() -> None:
    from app.chains.rag_chain import clean_answer

    raw = "[1] (Unit 2 communication skills for Career building.pdf, p.1)\n\nA resume is a summary [1]."
    assert clean_answer(raw) == "A resume is a summary [1]."
    assert clean_answer("A resume is a summary [1].") == "A resume is a summary [1]."


@pytest.mark.parametrize(
    "raw",
    [
        "How should I prepare for a group discussion?",
        '"How should I prepare for a group discussion?"',
        "Standalone question: How should I prepare for a group discussion?",
        "Here is the rewritten question:\n\nHow should I prepare for a group discussion?",
        "**How should I prepare for a group discussion?**\n\nThis keeps the context of the conversation.",
    ],
)
def test_extract_question_from_chatty_output(raw: str) -> None:
    from app.chains.memory import extract_question

    assert extract_question(raw) == "How should I prepare for a group discussion?"
