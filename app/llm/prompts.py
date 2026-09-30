"""Prompt templates (Blueprint §10.3)."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

NOT_FOUND_MESSAGE = "I couldn't find this in your uploaded material."

SYSTEM_PROMPT = f"""You are an academic study assistant. Answer the student's question using ONLY
the numbered context passages from their study material.
Rules:
- If the context does not contain the answer, reply exactly:
  "{NOT_FOUND_MESSAGE}"
- Cite passages inline like [1], [2] after the sentences they support.
- Be clear and exam-oriented: definitions first, then explanation,
  then examples / formulas / steps where relevant.
- Use Markdown (headings, bullet points, LaTeX for formulas) when helpful.
- Never invent page numbers, references, or facts."""

QA_HUMAN_TEMPLATE = """Context:
{context}

Question: {question}"""

CONDENSE_TEMPLATE = """Given the conversation and a follow-up question, rewrite the follow-up as a
standalone question that contains all needed context. Return only the question.
Conversation:
{history}
Follow-up: {question}
Standalone question:"""

QA_PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", QA_HUMAN_TEMPLATE)])
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([("human", CONDENSE_TEMPLATE)])
