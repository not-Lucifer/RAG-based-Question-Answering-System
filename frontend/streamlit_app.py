"""Streamlit entry point: `streamlit run frontend/streamlit_app.py`."""

from __future__ import annotations

import streamlit as st
from api_client import ApiClient, ApiError
from components import chat_view, sidebar

st.set_page_config(page_title="Academic RAG QA", page_icon="🎓", layout="wide")


@st.cache_resource
def get_client() -> ApiClient:
    """One HTTP client per server process."""
    return ApiClient()


def status_bar(client: ApiClient) -> int:
    """Show backend health, model and document count. Returns READY document count."""
    try:
        h = client.health()
    except ApiError as err:
        st.error(f"🔌 {err.message}")
        return 0
    index = "" if h.get("index_ok", True) else " · ⚠️ index built with another embedding model"
    st.caption(
        f"🟢 Backend online · LLM: **{h['llm_provider']} / {h.get('llm_model', '?')}** · "
        f"retrieval: {h.get('retrieval_mode', '?')} · documents indexed: **{h['docs_indexed']}**{index}"
    )
    return int(h["docs_indexed"])


def main() -> None:
    st.session_state.setdefault("session_id", None)
    st.session_state.setdefault("messages", [])
    client = get_client()

    st.title("🎓 Academic Question Answering")
    ready = status_bar(client)
    filters = sidebar.render(client)
    chat_view.render(client, filters, ready)


main()
