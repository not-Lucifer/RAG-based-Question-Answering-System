"""Visual identity for the Streamlit app: a marked-up study notebook.

Cool graph paper, ballpoint-blue ink, highlighter-yellow citations and a single
pink margin rule on each answer. Fonts load from Google Fonts and fall back to
system fonts offline.
"""

# ruff: noqa: E501 - long selector lines are clearer unwrapped
from __future__ import annotations

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400..800&family=Literata:ital,opsz,wght@0,7..72,400..700;1,7..72,400..700&display=swap');

:root {
  --page: #F4F7FB;
  --sheet: #FFFFFF;
  --ink: #17213A;
  --muted: #5B6783;
  --ballpoint: #2B3FA6;
  --highlighter: #FFE15A;
  --margin: #D9506F;
  --line: #D5DCEA;
  --grid: #E8EDF5;
  --ui: 'Bricolage Grotesque', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --read: 'Literata', Georgia, 'Times New Roman', serif;
}

.stApp {
  background-color: var(--page);
  background-image:
    linear-gradient(var(--grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size: 28px 28px;
  color: var(--ink);
  font-family: var(--ui);
}
[data-testid="stHeader"] { background: transparent; }

/* Reading width: answers stay under ~70 characters per line. */
.block-container, [data-testid="stBottomBlockContainer"] { max-width: 52rem; }
.block-container { padding-top: 2.5rem; }

h1 {
  font-family: var(--ui);
  font-weight: 800;
  font-size: 2.5rem;
  letter-spacing: -0.025em;
  line-height: 1.05;
  color: var(--ink);
  padding-bottom: 0.25rem;
}
h2, h3 { font-family: var(--ui); font-weight: 700; color: var(--ink); letter-spacing: -0.01em; }

/* Status line under the title */
[data-testid="stCaptionContainer"] { color: var(--muted); font-family: var(--ui); }

/* Sidebar: the binder */
[data-testid="stSidebar"] { background: #E7EDF6; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] h3 { font-size: 1.05rem; margin-top: 0.25rem; }
[data-testid="stSidebar"] hr { border-color: var(--line); }

/* Chat: avatars off, typography and the margin rule carry the roles. */
[data-testid="stChatMessage"] { background: transparent; padding: 0.4rem 0; gap: 0; }
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] { display: none; }

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p {
  font-family: var(--ui);
  font-weight: 650;
  font-size: 1.2rem;
  letter-spacing: -0.01em;
  color: var(--ink);
  margin-top: 1rem;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
  background: var(--sheet);
  border: 1px solid var(--line);
  border-left: 3px solid var(--margin);
  border-radius: 2px 10px 10px 2px;
  padding: 1rem 1.4rem 0.9rem 1.5rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) li,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) li {
  font-family: var(--read);
  font-size: 1.02rem;
  line-height: 1.7;
}

/* Citation numbers [n] arrive as coloured-background spans: make them highlighter marks. */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span[style*="background"] {
  background: var(--highlighter) !important;
  color: var(--ink) !important;
  font-family: var(--ui);
  font-weight: 700;
  font-size: 0.78em;
  padding: 0.05em 0.4em;
  border-radius: 3px;
  vertical-align: 0.12em;
}

/* Sources: page slips */
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 8px; background: transparent; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { font-family: var(--ui); font-weight: 600; color: var(--ballpoint); }
.slip {
  display: grid;
  grid-template-columns: 4.2rem 1fr;
  margin: 0.55rem 0;
  background: #FBFCFE;
  border: 1px solid var(--line);
  border-radius: 6px;
  overflow: hidden;
}
.slip-page {
  background: var(--ballpoint);
  color: #fff;
  font-family: var(--ui);
  text-align: center;
  padding: 0.55rem 0.2rem;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.slip-page b { font-size: 1.55rem; line-height: 1; font-weight: 800; }
.slip-page span { font-size: 0.72rem; opacity: 0.85; }
.slip-body { padding: 0.6rem 0.9rem; min-width: 0; }
.slip-title { font-family: var(--ui); font-weight: 650; font-size: 0.92rem; color: var(--ink); }
.slip-title mark { background: var(--highlighter); color: var(--ink); padding: 0 0.3em; border-radius: 3px; font-size: 0.8rem; }
.slip-score { display: block; color: var(--muted); font-size: 0.78rem; font-weight: 400; }
.slip-snippet { font-family: var(--read); font-size: 0.86rem; line-height: 1.55; color: var(--muted); margin-top: 0.25rem; }

/* Controls */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
  font-family: var(--ui);
  font-weight: 600;
  border-radius: 6px;
}
[data-testid="stChatInput"] { border-radius: 10px; border: 1px solid var(--line); background: var(--sheet); }
[data-testid="stChatInput"]:focus-within { border-color: var(--ballpoint); box-shadow: 0 0 0 2px rgba(43, 63, 166, 0.18); }
:focus-visible { outline: 2px solid var(--ballpoint); outline-offset: 2px; }

@media (max-width: 640px) {
  h1 { font-size: 1.9rem; }
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .slip { grid-template-columns: 3.4rem 1fr; }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { padding: 0.8rem 0.9rem 0.7rem 1rem; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""


def apply() -> None:
    """Inject the stylesheet. Call once per run, right after set_page_config."""
    st.markdown(CSS, unsafe_allow_html=True)
