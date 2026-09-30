# CLAUDE.md: Academic RAG QA (P_102)

Answers academic questions from uploaded study PDFs with cited (file + page) answers. The spec is the P_102 Project Blueprint (keep a copy at `docs/P_102_Academic_RAG_QA_Project_Blueprint.pdf`). The build workflow is the P_102 Claude Code Build Playbook.

**Current phase:** Phases 0–9 done; evaluation set drafted (30 questions over the 3 uploaded notes, awaiting student review). Next: review the dataset, re-run `run_eval.py --llm`, take screenshots, push, optionally deploy.

## Commands (Windows: `.\make.ps1 <target>`)

- `make test`: offline pytest suite (fake embeddings + fake LLM). Must stay green.
- `make lint`: ruff + black (line length 110).
- `make api` / `make ui`: FastAPI on :8000, Streamlit on :8501.
- `python scripts/run_eval.py [--llm]`, `python scripts/ask_cli.py "..."`, `python scripts/reset_index.py --reindex`.

## Rules

- Follow the blueprint's directory structure and the public signatures in §8. Only add optional keyword arguments, and record any deviation in `docs/architecture.md`.
- Layering: routes call services; services call the RAG core and DB; the RAG core never imports FastAPI; the frontend never imports `app.*`.
- All tunables come from `Settings` (`.env` + `config/settings.yaml`). No hard-coded paths or model names.
- **Never commit secrets.** Keys live only in `.env` (gitignored). `.env.example` has placeholders. The key is a `SecretStr`, and logs pass through `app.core.logging.redact`.
- Wrap external calls (LLM, embeddings, file IO, Chroma) and raise `AppError` subclasses from `app/core/exceptions.py`.
- Tests must run offline: use `set_embeddings_override` / `set_llm_override` and the fixtures in `tests/conftest.py`.
- Use LangChain 1.x split packages and LCEL. No legacy `LLMChain` / `RetrievalQA`.
- `evaluation/qa_dataset.json` was AI-drafted at the student's request; items stay `"verified": false` until the student checks them. Never set `verified` yourself, and do not write report conclusions.
