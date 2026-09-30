"""API flow with FastAPI TestClient: upload -> list -> ask -> follow-up -> feedback -> delete."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app
from tests.conftest import CN_PAGES, DBMS_PAGES


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as c:
        yield c


def _upload(client: TestClient, path: Path, subject: str | None = None, name: str | None = None):
    data = {"subject": subject} if subject else {}
    with path.open("rb") as fh:
        return client.post(
            "/documents/upload", files={"file": (name or path.name, fh, "application/pdf")}, data=data
        )


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["llm_provider"] == "openai" and body["docs_indexed"] == 0


def test_full_flow(client: TestClient, make_pdf, fake_llm) -> None:
    res = _upload(client, make_pdf(DBMS_PAGES, "DBMS_Unit3.pdf"), "DBMS")
    assert res.status_code == 201, res.text
    up = res.json()
    assert up["status"] == "READY" and up["num_pages"] == 2 and up["num_chunks"] >= 2
    assert _upload(client, make_pdf(CN_PAGES, "cn.pdf"), "CN").status_code == 201

    docs = client.get("/documents").json()
    assert {d["filename"] for d in docs} == {"DBMS_Unit3.pdf", "cn.pdf"}
    assert client.get("/health").json()["docs_indexed"] == 2

    fake_llm("A B-tree is a balanced search tree [1].")
    ask = client.post("/chat/ask", json={"question": "What is a B-tree?", "subject": "DBMS"}).json()
    assert ask["answer"].startswith("A B-tree") and ask["sources"]
    assert ask["sources"][0]["source"] == "DBMS_Unit3.pdf" and ask["sources"][0]["n"] == 1
    session_id = ask["session_id"]

    fake_llm("What are the advantages of B-trees?", "Balanced height and few disk reads [1].")
    follow = client.post("/chat/ask", json={"question": "explain its advantages", "session_id": session_id})
    assert follow.status_code == 200
    assert follow.json()["standalone_question"] == "What are the advantages of B-trees?"

    sessions = client.get("/sessions").json()
    assert sessions[0]["id"] == session_id and sessions[0]["title"] == "What is a B-tree?"
    msgs = client.get(f"/sessions/{session_id}/messages").json()
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[1]["sources"][0]["page"] >= 1

    fb = client.post("/feedback", json={"message_id": msgs[1]["id"], "rating": 1, "comment": "clear"})
    assert fb.json() == {"ok": True}
    assert client.post("/feedback", json={"message_id": msgs[0]["id"], "rating": 1}).status_code == 404

    doc_id = up["doc_id"]
    reindexed = client.post(f"/documents/{doc_id}/reindex").json()
    assert reindexed["status"] == "READY" and reindexed["num_chunks"] == up["num_chunks"]

    assert client.delete(f"/documents/{doc_id}").json() == {"deleted": True}
    assert not (get_settings().raw_dir / f"{doc_id}.pdf").exists()
    assert client.delete(f"/documents/{doc_id}").json()["error"] == "NOT_FOUND"

    assert client.delete(f"/sessions/{session_id}").json() == {"deleted": True}
    assert client.get(f"/sessions/{session_id}/messages").status_code == 404


def test_out_of_scope_question_refused(client: TestClient, make_pdf, fake_llm) -> None:
    _upload(client, make_pdf(DBMS_PAGES, "dbms.pdf"))
    llm = fake_llm("unused")
    body = client.post("/chat/ask", json={"question": "Who won IPL 2020?"}).json()
    assert body["answer"] == "I couldn't find this in your uploaded material."
    assert body["sources"] == [] and body["used_context"] is False and llm.calls == 0


def test_duplicate_upload_409(client: TestClient, make_pdf) -> None:
    path = make_pdf(DBMS_PAGES, "a.pdf")
    assert _upload(client, path).status_code == 201
    res = _upload(client, path, name="renamed.pdf")
    assert res.status_code == 409 and res.json()["error"] == "DUPLICATE_DOCUMENT"


def test_invalid_file_400(client: TestClient) -> None:
    res = client.post("/documents/upload", files={"file": ("notes.pdf", b"hello world", "application/pdf")})
    assert res.status_code == 400 and res.json()["error"] == "INVALID_FILE"
    res = client.post("/documents/upload", files={"file": ("notes.txt", b"%PDF-1.4 x", "text/plain")})
    assert res.status_code == 400


def test_path_traversal_filename_is_sanitised(client: TestClient, make_pdf, isolated_env: Path) -> None:
    res = _upload(client, make_pdf(DBMS_PAGES, "x.pdf"), name="../../evil.pdf")
    assert res.status_code == 201
    assert res.json()["filename"] == "evil.pdf"
    assert not (isolated_env / "evil.pdf").exists()


def test_file_too_large_413(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()
    big = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 10)
    res = client.post("/documents/upload", files={"file": ("big.pdf", big, "application/pdf")})
    assert res.status_code == 413 and res.json()["error"] == "FILE_TOO_LARGE"


def test_empty_pdf_422_and_marked_failed(client: TestClient, make_pdf) -> None:
    res = _upload(client, make_pdf([None, None], "scan.pdf"))
    assert res.status_code == 422 and res.json()["error"] == "EMPTY_PDF"
    docs = client.get("/documents").json()
    assert docs[0]["status"] == "FAILED" and docs[0]["error_message"]


def test_llm_unavailable_503(client: TestClient, make_pdf) -> None:
    _upload(client, make_pdf(CN_PAGES, "cn.pdf"))
    res = client.post("/chat/ask", json={"question": "Dijkstra time complexity"})
    assert res.status_code == 503 and res.json()["error"] == "LLM_UNAVAILABLE"
    assert "sk-" not in res.text


def test_unknown_session_404(client: TestClient) -> None:
    res = client.post("/chat/ask", json={"question": "hi", "session_id": "nope"})
    assert res.status_code == 404 and res.json()["error"] == "NOT_FOUND"


def test_validation_error_shape(client: TestClient) -> None:
    res = client.post("/chat/ask", json={"question": "   "})
    assert res.status_code == 422 and res.json()["error"] == "VALIDATION_ERROR"


def test_stream_endpoint(client: TestClient, make_pdf, fake_llm) -> None:
    _upload(client, make_pdf(CN_PAGES, "cn.pdf"))
    fake_llm("SYN, SYN-ACK, ACK [1].")
    with client.stream("POST", "/chat/stream", json={"question": "TCP three-way handshake"}) as res:
        body = "".join(res.iter_text())
    events = [block for block in body.split("\n\n") if block.strip()]
    assert events[0].startswith("event: token")
    final = events[-1]
    assert final.startswith("event: sources")
    payload = json.loads(final.split("data: ", 1)[1])
    assert payload["answer"] == "SYN, SYN-ACK, ACK [1]." and payload["sources"] and payload["session_id"]
    assert len(client.get(f"/sessions/{payload['session_id']}/messages").json()) == 2
