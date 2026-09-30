"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, documents, health, sessions
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.db.database import init_db
from app.vectorstore.store import index_status

log = get_logger(__name__)


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
