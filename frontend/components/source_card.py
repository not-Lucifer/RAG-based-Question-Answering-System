"""Citation expander shown under each answer."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Expander "Sources (n)" with one card per cited passage."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for src in sources:
            score = src.get("score")
            score_txt = f" · relevance {score:.2f}" if isinstance(score, int | float) else ""
            st.markdown(f"**[{src['n']}] {src['source']} · page {src['page']}**{score_txt}")
            st.caption(src.get("snippet", ""))
