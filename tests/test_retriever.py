"""Vector store + retrievers: filters, scores, MMR, hybrid, cache invalidation, model mismatch."""

from __future__ import annotations

import pytest

from app.core.exceptions import IndexMismatchError
from app.embeddings import factory as emb_factory
from app.ingestion.pipeline import ingest_file
from app.retrieval.hybrid import weighted_rrf
from app.retrieval.retriever import build_retriever, build_where, mmr_select
from app.vectorstore import store
from tests.conftest import FakeEmbeddings


def test_build_where_mapping() -> None:
    assert build_where(None) is None
    assert build_where({"subject": "DBMS"}) == {"subject": "DBMS"}
    assert build_where({"subject": ["DBMS", "OS"]}) == {"subject": {"$in": ["DBMS", "OS"]}}
    assert build_where({"doc_ids": ["a", "b"]}) == {"doc_id": {"$in": ["a", "b"]}}
    assert build_where({"subject": "DBMS", "doc_ids": ["a"]}) == {
        "$and": [{"subject": "DBMS"}, {"doc_id": "a"}]
    }


def test_deterministic_ids_and_count(indexed_docs) -> None:
    n = store.count()
    assert n >= 4
    ingest_file(indexed_docs["dbms"], "doc-dbms", "DBMS")  # re-ingest overwrites, no duplicates
    assert store.count() == n
    assert store.list_doc_ids() == {"doc-dbms", "doc-cn"}


def test_delete_doc(indexed_docs) -> None:
    removed = store.delete_doc("doc-cn")
    assert removed > 0
    assert store.list_doc_ids() == {"doc-dbms"}


@pytest.mark.parametrize("mode", ["similarity", "mmr", "hybrid"])
def test_modes_return_relevant_scored_chunks(indexed_docs, mode: str) -> None:
    docs = build_retriever(mode=mode, top_k=2).invoke("Dijkstra time complexity")
    assert docs and len(docs) <= 2
    assert docs[0].metadata["doc_id"] == "doc-cn" and docs[0].metadata["page"] == 2
    assert all(isinstance(d.metadata["score"], float) for d in docs)


def test_subject_filter(indexed_docs) -> None:
    docs = build_retriever("similarity", 5, {"subject": "DBMS"}).invoke("Dijkstra shortest path handshake")
    assert docs and all(d.metadata["subject"] == "DBMS" for d in docs)


def test_doc_id_filter(indexed_docs) -> None:
    docs = build_retriever("mmr", 5, {"doc_ids": ["doc-cn"]}).invoke("normalization")
    assert docs and {d.metadata["doc_id"] for d in docs} == {"doc-cn"}


def test_hybrid_finds_exact_term(indexed_docs) -> None:
    docs = build_retriever("hybrid", 3).invoke("3NF")
    assert docs[0].metadata["doc_id"] == "doc-dbms"


def test_bm25_cache_invalidated_on_ingest(indexed_docs, make_pdf) -> None:
    assert not any(d.metadata["doc_id"] == "doc-os" for d in build_retriever("hybrid", 5).invoke("semaphore"))
    ingest_file(make_pdf(["A semaphore supports wait and signal operations."], "os.pdf"), "doc-os", "OS")
    docs = build_retriever("hybrid", 5).invoke("semaphore")
    assert docs[0].metadata["doc_id"] == "doc-os"


def test_mmr_prefers_diversity() -> None:
    query = [1.0, 0.0, 0.0]
    vectors = [[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.7, 0.7, 0.0]]
    assert mmr_select(query, vectors, 2, lambda_mult=0.3) == [0, 2]
    assert mmr_select(query, vectors, 2, lambda_mult=1.0) == [0, 1]


def test_weighted_rrf_merges_and_dedupes() -> None:
    from langchain_core.documents import Document

    a = Document(page_content="a", metadata={"chunk_id": "a"})
    b = Document(page_content="b", metadata={"chunk_id": "b"})
    c = Document(page_content="c", metadata={"chunk_id": "c"})
    fused = weighted_rrf([[a, b], [b, c]], [0.4, 0.6])
    assert [d.metadata["chunk_id"] for d in fused] == ["b", "c", "a"]


def test_embedding_model_mismatch_refused(indexed_docs) -> None:
    class OtherEmbeddings(FakeEmbeddings):
        model_id = "fake:other-model"

    emb_factory.set_embeddings_override(OtherEmbeddings())
    store.reset_caches()
    assert store.index_status()["ok"] is False
    with pytest.raises(IndexMismatchError):
        build_retriever("similarity", 3).invoke("normalization")
