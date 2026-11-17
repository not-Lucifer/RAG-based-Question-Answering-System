"""Streamlit entry point: `streamlit run frontend/streamlit_app.py`."""

from __future__ import annotations

import streamlit as st
import theme
from api_client import ApiClient, ApiError
from components import chat_view, sidebar

st.set_page_config(page_title="Academic RAG QA", page_icon=":material/menu_book:", layout="wide")


@st.cache_resource
def get_client() -> ApiClient:
    """One HTTP client per server process."""
    return ApiClient()


def status_bar(client: ApiClient) -> int:
    """Show backend health, model and document count. Returns READY document count."""
    try:
        h = client.health()
    except ApiError as err:
        st.error(err.message)
        return 0
    index = "" if h.get("index_ok", True) else " · index built with another embedding model: re-index"
    st.caption(
        f":green[●] Backend online · LLM: **{h['llm_provider']} / {h.get('llm_model', '?')}** · "
        f"retrieval: {h.get('retrieval_mode', '?')} · documents indexed: **{h['docs_indexed']}**{index}"
    )
    return int(h["docs_indexed"])


def main() -> None:
    theme.apply()
    st.session_state.setdefault("session_id", None)
    st.session_state.setdefault("messages", [])
    client = get_client()

    st.title("Academic Question Answering")
    ready = status_bar(client)
    filters = sidebar.render(client)
    chat_view.render(client, filters, ready)


main()
