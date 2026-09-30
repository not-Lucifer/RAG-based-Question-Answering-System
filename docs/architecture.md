# Architecture

## Layers

| Layer | Code | Rule |
|---|---|---|
| Presentation | `frontend/` | Talks to the backend over HTTP only (`BACKEND_URL`). Imports no `app.*` code, so it deploys separately. |
| API | `app/api/routes/*` | Validation, routing and error mapping. No business logic. |
| Service | `app/services/*` | Owns DB transactions. The only layer that touches both SQLite and the RAG core. |
| RAG core | `app/ingestion`, `embeddings`, `vectorstore`, `retrieval`, `llm`, `chains` | Pure pipeline logic. Never imports FastAPI, and is testable with fakes. |
| Data | ChromaDB (`data/vectorstore`), SQLite (`data/app.db`), PDFs (`data/raw`) | |

`app/core` (config, logging, exceptions) is shared by all layers. `exceptions.py` imports FastAPI lazily, inside `register_exception_handlers`, so the RAG core stays FastAPI-free.

## Ingestion flow

1. `DocumentService.upload` sanitises the filename and validates size, extension, MIME type and `%PDF-` magic bytes.
2. It computes the SHA-256 hash. A READY or PROCESSING duplicate returns `DUPLICATE_DOCUMENT`; a FAILED duplicate is retried.
3. The file is saved as `data/raw/<uuid>.pdf` and a `documents` row is inserted with status `PROCESSING`.
4. `pipeline.ingest_file`:
   - `load_pdf`: one Document per page, 1-based `page`, `source` = original filename. Raises `EMPTY_PDF` when no text is found.
   - `clean_pages`: drops lines repeated on more than 50% of pages (digits normalised, so "Page 3" equals "Page 4"), bare page numbers and pages under 30 characters. Also fixes `algo-\nrithm` and collapses whitespace.
   - `split_documents`: RecursiveCharacterTextSplitter (1000/150) per page, so chunks never span pages. Adds `doc_id`, `subject` and `chunk_index`.
   - `delete_doc(doc_id)` then `add_chunks`: embeds in batches of 32 with ids `doc_id:chunk_index`.
5. The row is updated to READY with page and chunk counts. On any error it becomes FAILED with a user-safe message, and any partial vectors are removed.

## Query flow

1. `ChatService.ask` loads the last `2 × history_turns` messages of the session.
2. `chains.memory.condense` rewrites a follow-up into a standalone question. It is skipped when there is no history **or the question is already standalone**, i.e. it has no back-references such as "it", "its", "the last one" or "what about…". The rewrite is rejected when it drops the question's topic words. Small local models (llama3.2:3b) otherwise tend to replace a new question with the previous one.
3. `retrieval.build_retriever(mode, top_k, filters)` runs one of three modes:
   - **similarity**: nearest neighbours from Chroma.
   - **mmr**: fetch `fetch_k` candidates with their embeddings, then Maximal Marginal Relevance (λ = 0.6).
   - **hybrid**: BM25 (`rank_bm25`, cached per filter, invalidated by an index-version counter on every write) fused with vector results using weighted Reciprocal Rank Fusion (0.4 / 0.6).

   Every result carries `metadata.score`, the cosine similarity between the query and the stored chunk embedding.
4. The optional cross-encoder reranker is lazy-loaded.
5. **Score gate:** if there are no chunks or the best score is below `score_threshold`, the answer is "I couldn't find this in your uploaded material." The LLM is never built or called in that case.
6. The LCEL chain runs `QA_PROMPT | llm | StrOutputParser()` over numbered context blocks, `[n] (file.pdf, p.12)\n<text>`.
7. Both messages are saved, with sources as JSON and the latency. The session is created on the first question and titled from it.

## Deviations from the blueprint (and why)

| Blueprint | Implemented | Reason |
|---|---|---|
| Python 3.11 | Runs on 3.11+; developed and tested on 3.13 | Only 3.13 was installed. |
| `EnsembleRetriever` in `hybrid.py` | Local weighted RRF (`HybridRetriever`), the same algorithm | LangChain 1.x moved `EnsembleRetriever` to `langchain-classic`; a local version avoids that dependency. |
| `load_pdf(path)`, `ingest_file(path, doc_id, subject)`, `split_documents(docs, doc_id, subject)`, `answer(question, history, filters, top_k)` | Same signatures plus **optional** keyword arguments: `source_name`, `chunk_size`, `chunk_overlap`, `mode` | Uploads are stored as `<uuid>.pdf`, so the original filename is passed for citations. Evaluation needs chunk-size and mode overrides. Existing calls are unaffected. |
| `subject` optional | Missing subject is stored as `"General"` | Chroma metadata values cannot be null. |
| Relevance scores via LangChain | Cosine similarity computed from stored embeddings | MMR in LangChain returns no scores, and the threshold needs one in every mode. |
| Error codes in §9 | All of them, plus `EMBEDDING_UNAVAILABLE` (503), `VECTORSTORE_ERROR` / `INGESTION_FAILED` / `INTERNAL_ERROR` (500) and `VALIDATION_ERROR` (422) | Every failure maps to the same `{error, detail}` shape. |
| `AskRequest.subject: str` | `str` or `list[str]`, plus optional `mode` | The UI subject filter is a multiselect. |
| `AskResponse` | Also returns `message_id`, `used_context`, `standalone_question` | Needed for feedback buttons and debugging. |
| `/feedback` route file | `app/api/routes/sessions.py` | Feedback belongs to stored messages. |
| `import fitz` | `import pymupdf` | `fitz` is deprecated in current PyMuPDF. |
| `docker-compose.yml` (optional) | Not included | Render native Python runtime is used instead, as the playbook suggests. |
