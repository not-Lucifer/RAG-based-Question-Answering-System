"""Main chat area: history, input, answers with sources and feedback."""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from typing import Any

import streamlit as st
from api_client import ApiClient, ApiError
from components.source_card import render_sources

_BLOCK_MATH = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH = re.compile(r"\\\((.+?)\\\)", re.DOTALL)


def to_streamlit_markdown(text: str) -> str:
    """Convert LaTeX delimiters \\( \\) and \\[ \\] to the $ / $$ that Streamlit renders."""
    text = _BLOCK_MATH.sub(lambda m: f"$$\n{m.group(1).strip()}\n$$", text)
    return _INLINE_MATH.sub(lambda m: f"${m.group(1).strip()}$", text)


def _feedback(client: ApiClient, msg: dict[str, Any], idx: int) -> None:
    message_id = msg.get("message_id")
    if not message_id:
        return
    choice = st.feedback("thumbs", key=f"fb-{message_id}-{idx}")
    sent: set[int] = st.session_state.setdefault("feedback_sent", set())
    if choice is not None and message_id not in sent:
        try:
            client.feedback(message_id, 1 if choice == 1 else -1)
            sent.add(message_id)
            st.toast("Thanks for the feedback!")
        except ApiError as err:
            st.error(err.message)


def _history(client: ApiClient) -> None:
    messages = st.session_state.messages
    for idx, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            st.markdown(to_streamlit_markdown(msg["content"]))
            if msg["role"] == "assistant":
                render_sources(msg.get("sources", []))
                _feedback(client, msg, idx)
            elif idx + 1 < len(messages) and messages[idx + 1]["role"] == "user":
                st.caption("⚠️ No answer was received for this question.")


def record_answer(state: MutableMapping[str, Any], data: dict[str, Any]) -> None:
    """Store a finished answer and the session it belongs to in the UI state."""
    state["session_id"] = data.get("session_id") or state.get("session_id")
    state["messages"].append(
        {
            "role": "assistant",
            "content": data.get("answer", ""),
            "sources": data.get("sources", []),
            "message_id": data.get("message_id"),
        }
    )


def recover_interrupted(state: MutableMapping[str, Any], question: str, final: dict[str, Any]) -> None:
    """Repair the UI state when a rerun (user clicked or typed) interrupted an answer.

    If the backend already returned the full answer it is kept, together with its
    session id, so the next question stays in the same chat. Otherwise the orphaned
    question is removed and the user is told to ask it again.
    """
    if final.get("answer"):
        record_answer(state, final)
        return
    messages = state["messages"]
    if messages and messages[-1]["role"] == "user" and messages[-1]["content"] == question:
        messages.pop()
    state["interrupted"] = question


def _ask(client: ApiClient, question: str, filters: dict[str, Any]) -> None:
    args = (
        question,
        st.session_state.session_id,
        filters["subjects"],
        filters["doc_ids"],
        filters["top_k"],
    )
    final: dict[str, Any] = {}  # filled as soon as the backend's full answer arrives
    handled = False
    try:
        with st.chat_message("assistant"):
            try:
                if filters["stream"]:
                    with st.spinner("Thinking…"):
                        tokens = client.stream_ask(*args, result=final)
                        first = next(tokens, None)
                    if first is None:
                        raise ApiError("ERROR", "The backend returned an empty answer.")
                    answer = st.write_stream(_chain(first, tokens))
                    data = final or {"answer": answer, "sources": []}
                else:
                    with st.spinner("Thinking…"):
                        final.update(client.ask(*args))
                    data = final
                    st.markdown(to_streamlit_markdown(data["answer"]))
            except ApiError as err:
                handled = True
                st.error(err.message)
                st.session_state.messages.pop()  # drop the unanswered question
                return
            render_sources(data.get("sources", []))
            if data.get("latency_ms"):
                st.caption(f"⏱ {data['latency_ms'] / 1000:.1f}s")
        record_answer(st.session_state, data)
        handled = True
    finally:
        # A Streamlit rerun (the user typed or clicked while the answer was arriving)
        # raises inside the calls above; without this the answer and session id were lost.
        if not handled:
            recover_interrupted(st.session_state, question, final)
    st.rerun()  # redraw with feedback buttons attached to the new answer


def _chain(first: str, rest):  # noqa: ANN001, ANN202 - generator helper
    yield first
    yield from rest


def render(client: ApiClient, filters: dict[str, Any], ready_docs: int) -> None:
    """Draw the conversation and handle a new question."""
    if not st.session_state.messages:
        st.info(
            "Upload your study PDFs in the sidebar, then ask a question. "
            "Answers cite the file and page they came from."
            if ready_docs == 0
            else "Ask anything about your uploaded material, e.g. *“What is 3NF?”*"
        )
    _history(client)
    if interrupted := st.session_state.pop("interrupted", None):
        st.warning(
            f"Your question “{interrupted}” was interrupted before the answer arrived "
            "(the page was used while it was loading). Please ask it again."
        )
    if question := st.chat_input("Ask a question about your notes…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        _ask(client, question, filters)
