"""Streamlit UI smoke tests with an in-memory fake backend (no servers needed)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
sys.path.insert(0, str(FRONTEND))

import api_client  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

ANSWER = {
    "answer": "3NF removes transitive dependencies [1].",
    "sources": [
        {"n": 1, "source": "DBMS_Notes.pdf", "page": 2, "snippet": "Third Normal Form...", "score": 0.61}
    ],
    "session_id": "s1",
    "latency_ms": 1200,
    "message_id": 2,
}


class FakeClient(api_client.ApiClient):
    """Backend stand-in recording calls."""

    calls: list[tuple[str, Any]] = []

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "llm_provider": "openai", "llm_model": "gpt-4o-mini", "docs_indexed": 1}

    def list_documents(self) -> list[dict[str, Any]]:
        return [
            {"id": "d1", "filename": "DBMS_Notes.pdf", "subject": "DBMS", "num_pages": 5, "num_chunks": 8,
             "status": "READY", "error_message": None},
        ]  # fmt: skip

    def list_sessions(self) -> list[dict[str, Any]]:
        return [{"id": "s1", "title": "What is 3NF?", "created_at": "2026-01-01T00:00:00"}]

    def ask(self, *args: Any) -> dict[str, Any]:
        self.calls.append(("ask", args))
        return ANSWER

    def stream_ask(self, *args: Any, result: dict[str, Any]):
        self.calls.append(("stream", args))
        yield "3NF removes "
        yield "transitive dependencies [1]."
        result.update(ANSWER)


class DownClient(api_client.ApiClient):
    """Backend that is not running."""

    def _request(self, *args: Any, **kwargs: Any) -> Any:
        raise api_client.ApiError("BACKEND_DOWN", "down")


def _app(monkeypatch: pytest.MonkeyPatch, client_cls: type) -> AppTest:
    monkeypatch.setattr(api_client, "ApiClient", client_cls)
    FakeClient.calls = []
    at = AppTest.from_file(str(FRONTEND / "streamlit_app.py"), default_timeout=30)
    at.run()
    return at


@pytest.fixture(autouse=True)
def _clear_streamlit_cache() -> None:
    import streamlit as st

    st.cache_resource.clear()


def test_page_renders_status_and_library(monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(monkeypatch, FakeClient)
    assert not at.exception
    assert "Academic Question Answering" in at.title[0].value
    assert any("documents indexed" in c.value for c in at.caption)
    assert any("DBMS_Notes.pdf" in m.value for m in at.sidebar.markdown)


@pytest.mark.parametrize("stream", [False, True])
def test_ask_shows_answer_and_sources(monkeypatch: pytest.MonkeyPatch, stream: bool) -> None:
    at = _app(monkeypatch, FakeClient)
    at.sidebar.toggle[0].set_value(stream).run()
    at.chat_input[0].set_value("What is 3NF?").run()
    assert not at.exception
    assert FakeClient.calls and FakeClient.calls[0][0] == ("stream" if stream else "ask")
    texts = " ".join(m.value for m in at.markdown)
    assert "transitive dependencies" in texts
    assert at.expander and "Sources (1)" in at.expander[0].label
    assert at.session_state.session_id == "s1"


def test_backend_down_shows_friendly_error(monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(monkeypatch, DownClient)
    assert not at.exception
    assert any("not reachable" in e.value for e in at.error)


def test_latex_delimiters_converted() -> None:
    sys.path.insert(0, str(FRONTEND))
    from components.chat_view import to_streamlit_markdown

    assert to_streamlit_markdown(r"Area \(\pi r^2\)") == r"Area $\pi r^2$"
    assert to_streamlit_markdown(r"\[E = mc^2\]") == "$$\nE = mc^2\n$$"


def test_interrupted_answer_that_arrived_is_kept() -> None:
    from components.chat_view import recover_interrupted

    state = {"session_id": None, "messages": [{"role": "user", "content": "magnetism"}]}
    recover_interrupted(
        state, "magnetism", {"answer": "Magnetism is ...", "session_id": "s9", "message_id": 28}
    )
    assert state["session_id"] == "s9"  # next question continues the same chat
    assert [m["role"] for m in state["messages"]] == ["user", "assistant"]
    assert "interrupted" not in state


def test_interrupted_answer_that_never_arrived_is_reported() -> None:
    from components.chat_view import recover_interrupted

    state = {"session_id": "s1", "messages": [{"role": "user", "content": "magnetism"}]}
    recover_interrupted(state, "magnetism", {})
    assert state["messages"] == [] and state["interrupted"] == "magnetism"


def test_interrupted_warning_is_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(monkeypatch, FakeClient)
    at.session_state["interrupted"] = "magnetism"
    at.run()
    assert any("magnetism" in w.value and "interrupted" in w.value for w in at.warning)
