"""Sidebar: library (upload / list / delete), filters and chat sessions."""

from __future__ import annotations

from typing import Any

import streamlit as st
from api_client import ApiClient, ApiError

STATUS_ICON = {"READY": "🟢", "PROCESSING": "🟡", "FAILED": "🔴"}


def _library(client: ApiClient, docs: list[dict[str, Any]]) -> None:
    st.subheader("📚 Library")
    with st.form("upload", clear_on_submit=True):
        files = st.file_uploader("PDF notes", type=["pdf"], accept_multiple_files=True)
        subject = st.text_input("Subject tag", placeholder="e.g. DBMS")
        submitted = st.form_submit_button("Upload", width="stretch")
    if submitted and files:
        for f in files:
            with st.spinner(f"Indexing {f.name}…"):
                try:
                    res = client.upload(f.name, f.getvalue(), subject.strip() or None)
                    st.toast(f"✅ {res['filename']}: {res['num_pages']} pages, {res['num_chunks']} chunks")
                except ApiError as err:
                    st.error(f"{f.name}: {err.message}")
        st.rerun()

    if not docs:
        st.caption("No documents yet. Upload a text-based PDF to start.")
        return
    for doc in docs:
        col_info, col_del = st.columns([5, 1])
        icon = STATUS_ICON.get(doc["status"], "⚪")
        col_info.markdown(f"{icon} **{doc['filename']}**")
        col_info.caption(f"{doc['subject']} · {doc['num_pages']} pages · {doc['num_chunks']} chunks")
        if doc["status"] == "FAILED" and doc.get("error_message"):
            col_info.caption(f"⚠️ {doc['error_message']}")
        if col_del.button("🗑", key=f"del-{doc['id']}", help=f"Delete {doc['filename']}"):
            try:
                client.delete_document(doc["id"])
                st.rerun()
            except ApiError as err:
                st.error(err.message)


def _filters(docs: list[dict[str, Any]]) -> dict[str, Any]:
    st.subheader("🔎 Filters")
    ready = [d for d in docs if d["status"] == "READY"]
    subjects = sorted({d["subject"] for d in ready})
    chosen_subjects = st.multiselect("Subjects", subjects)
    visible = [d for d in ready if not chosen_subjects or d["subject"] in chosen_subjects]
    names = {d["id"]: d["filename"] for d in visible}
    chosen_docs = st.multiselect("Documents", list(names), format_func=names.get)
    top_k = st.slider("Passages per answer (top-k)", 3, 8, 5)
    stream = st.toggle("Stream answers", value=True)
    return {"subjects": chosen_subjects, "doc_ids": chosen_docs, "top_k": top_k, "stream": stream}


def _sessions(client: ApiClient) -> None:
    st.subheader("💬 Chats")
    if st.button("➕ New chat", width="stretch"):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()
    try:
        sessions = client.list_sessions()
    except ApiError as err:
        st.caption(err.message)
        return
    for sess in sessions[:30]:
        col_open, col_del = st.columns([5, 1])
        active = sess["id"] == st.session_state.get("session_id")
        label = ("▶ " if active else "") + sess["title"]
        if col_open.button(label, key=f"open-{sess['id']}", width="stretch"):
            try:
                st.session_state.messages = [
                    {
                        "role": m["role"],
                        "content": m["content"],
                        "sources": m.get("sources", []),
                        "message_id": m["id"] if m["role"] == "assistant" else None,
                    }
                    for m in client.get_messages(sess["id"])
                ]
                st.session_state.session_id = sess["id"]
                st.rerun()
            except ApiError as err:
                st.error(err.message)
        if col_del.button("🗑", key=f"delsess-{sess['id']}", help="Delete chat"):
            try:
                client.delete_session(sess["id"])
                if active:
                    st.session_state.session_id = None
                    st.session_state.messages = []
                st.rerun()
            except ApiError as err:
                st.error(err.message)


def render(client: ApiClient) -> dict[str, Any]:
    """Draw the sidebar and return the active filters."""
    with st.sidebar:
        try:
            docs = client.list_documents()
        except ApiError as err:
            st.error(err.message)
            docs = []
        _library(client, docs)
        st.divider()
        filters = _filters(docs)
        st.divider()
        _sessions(client)
    return filters
