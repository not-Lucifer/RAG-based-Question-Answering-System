"""Main chat area: history, input, answers with sources and feedback."""

from __future__ import annotations

import re
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
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(to_streamlit_markdown(msg["content"]))
            if msg["role"] == "assistant":
                render_sources(msg.get("sources", []))
                _feedback(client, msg, idx)


def _ask(client: ApiClient, question: str, filters: dict[str, Any]) -> None:
    args = (
        question,
        st.session_state.session_id,
        filters["subjects"],
        filters["doc_ids"],
        filters["top_k"],
    )
    with st.chat_message("assistant"):
        try:
            if filters["stream"]:
                final: dict[str, Any] = {}
                with st.spinner("Thinking…"):
                    tokens = client.stream_ask(*args, result=final)
                    first = next(tokens, None)
                if first is None:
                    raise ApiError("ERROR", "The backend returned an empty answer.")
                answer = st.write_stream(_chain(first, tokens))
                data = final or {"answer": answer, "sources": []}
            else:
                with st.spinner("Thinking…"):
                    data = client.ask(*args)
                st.markdown(to_streamlit_markdown(data["answer"]))
        except ApiError as err:
            st.error(err.message)
            st.session_state.messages.pop()  # drop the unanswered question
            return
        render_sources(data.get("sources", []))
        if data.get("latency_ms"):
            st.caption(f"⏱ {data['latency_ms'] / 1000:.1f}s")
    st.session_state.session_id = data.get("session_id") or st.session_state.session_id
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": data.get("answer", ""),
            "sources": data.get("sources", []),
            "message_id": data.get("message_id"),
        }
    )
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
    if question := st.chat_input("Ask a question about your notes…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        _ask(client, question, filters)
