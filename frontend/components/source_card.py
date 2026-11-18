"""Citation expander shown under each answer."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st


def _slip(src: dict[str, Any]) -> str:
    """One cited passage as a page slip: page number block, file name, snippet."""
    score = src.get("score")
    score_txt = (
        f'<span class="slip-score">relevance {score:.2f}</span>' if isinstance(score, int | float) else ""
    )
    return (
        '<div class="slip">'
        f'<div class="slip-page"><span>page</span><b>{escape(str(src["page"]))}</b></div>'
        '<div class="slip-body">'
        f'<div class="slip-title"><mark>{escape(str(src["n"]))}</mark> {escape(str(src["source"]))}'
        f"{score_txt}</div>"
        f'<div class="slip-snippet">{escape(str(src.get("snippet", "")))}</div>'
        "</div></div>"
    )


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Expander "Sources (n)" with one page slip per cited passage."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        st.markdown("".join(_slip(src) for src in sources), unsafe_allow_html=True)
