"""FastAPI application factory."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, documents, health, sessions
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.db.database import init_db
from app.embeddings.factory import get_embeddings
from app.vectorstore.store import index_status

log = get_logger(__name__)


def _warm_up() -> None:
    """Preload slow models so the first question is not delayed (runs in a background thread).

    Loads the embedding model (~30 s for MiniLM on first use) and, in Ollama mode, asks
    Ollama to load the chat model into memory. Failures are logged, never raised.
    """
    settings = get_settings()
    started = time.perf_counter()
    try:
        get_embeddings().embed_query("warm up")
    except Exception as exc:
        log.warning("embedding warm-up failed", extra={"fields": {"error": type(exc).__name__}})
    if settings.llm_provider == "ollama":
        body = json.dumps({"model": settings.ollama_model, "keep_alive": settings.ollama_keep_alive}).encode()
        req = urllib.request.Request(
            settings.ollama_base_url.rstrip("/") + "/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as res:
                res.read()
        except Exception as exc:
            log.warning(
                "ollama warm-up failed (is Ollama running?)", extra={"fields": {"error": type(exc).__name__}}
            )
    log.info("warm-up done", extra={"fields": {"s": round(time.perf_counter() - started, 1)}})


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Create tables and warn early about an incompatible vector index."""
    settings = get_settings()
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    status = index_status()
    if not status["ok"]:
        log.error(
            "vector index built with a different embedding model; queries will be refused until "
            "`python scripts/reset_index.py --reindex` is run",
            extra={"fields": {"stored": status["stored_model"], "current": status["current_model"]}},
        )
    log.info(
        "api started",
        extra={"fields": {"llm": settings.llm_provider, "embeddings": settings.active_embedding_model}},
    )
    if settings.warmup_on_startup and status["ok"]:
        threading.Thread(target=_warm_up, name="warm-up", daemon=True).start()
    yield


def create_app() -> FastAPI:
    """Build the FastAPI app with routers, CORS and error handlers."""
    settings = get_settings()
    setup_logging(settings.log_level)
    app = FastAPI(
        title="Academic RAG QA",
        version="1.0.0",
        description="Ask questions about your study PDFs and get cited answers.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    for module in (health, documents, chat, sessions):
        app.include_router(module.router)
    return app


app = create_app()
