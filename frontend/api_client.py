"""Thin HTTP client for the backend. The UI never imports backend code."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import requests

try:  # load .env for local runs; on hosting platforms env vars are set directly
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

DEFAULT_BACKEND = "http://127.0.0.1:8000"
FRIENDLY = {
    "BACKEND_DOWN": "The backend is not reachable. Start it with `make api` (or `.\\make.ps1 api`).",
    "TIMEOUT": "The backend took too long to respond. Try again, or use a smaller PDF.",
    "LLM_UNAVAILABLE": "The language model is unavailable. Check OPENAI_API_KEY or that Ollama is running.",
    "INDEX_MISMATCH": "The index was built with a different embedding model. Run "
    "`python scripts/reset_index.py --reindex`.",
    "EMBEDDING_UNAVAILABLE": "The embedding model could not be loaded. Check the backend logs.",
}


class ApiError(Exception):
    """A backend error with a user-friendly message."""

    def __init__(self, code: str, detail: str, status: int | None = None) -> None:
        self.code, self.detail, self.status = code, detail, status
        super().__init__(self.message)

    @property
    def message(self) -> str:
        """Text to show in the UI."""
        return FRIENDLY.get(self.code, self.detail)


class ApiClient:
    """All HTTP calls from the Streamlit UI."""

    def __init__(self, base_url: str | None = None, timeout: float = 15.0) -> None:
        url = (base_url or os.getenv("BACKEND_URL") or DEFAULT_BACKEND).rstrip("/")
        # Render's private-network wiring (fromService: hostport) gives a bare "host:port".
        self.base_url = url if "://" in url else f"http://{url}"
        self.timeout = timeout
        self.http = requests.Session()

    # ------------------------------------------------------------------ core
    def _request(self, method: str, path: str, timeout: float | None = None, **kwargs: Any) -> Any:
        try:
            res = self.http.request(method, self.base_url + path, timeout=timeout or self.timeout, **kwargs)
        except requests.ConnectionError as exc:
            raise ApiError("BACKEND_DOWN", f"Cannot reach {self.base_url}") from exc
        except requests.Timeout as exc:
            raise ApiError("TIMEOUT", "Request timed out") from exc
        if res.status_code >= 400:
            raise self._error(res)
        return res.json() if res.content else None

    @staticmethod
    def _error(res: requests.Response) -> ApiError:
        try:
            body = res.json()
            return ApiError(body.get("error", "ERROR"), body.get("detail", res.reason), res.status_code)
        except ValueError:
            return ApiError("ERROR", f"Backend error ({res.status_code})", res.status_code)

    # ------------------------------------------------------------------ endpoints
    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health", timeout=5)

    def list_documents(self) -> list[dict[str, Any]]:
        return self._request("GET", "/documents")

    def upload(self, filename: str, data: bytes, subject: str | None = None) -> dict[str, Any]:
        files = {"file": (filename, data, "application/pdf")}
        form = {"subject": subject} if subject else {}
        return self._request("POST", "/documents/upload", timeout=600, files=files, data=form)

    def delete_document(self, doc_id: str) -> None:
        self._request("DELETE", f"/documents/{doc_id}")

    def reindex(self, doc_id: str) -> dict[str, Any]:
        return self._request("POST", f"/documents/{doc_id}/reindex", timeout=600)

    @staticmethod
    def _ask_body(question: str, session_id: str | None, subjects: list[str], doc_ids: list[str], top_k: int):
        return {
            "question": question,
            "session_id": session_id,
            "subject": subjects or None,
            "doc_ids": doc_ids or None,
            "top_k": top_k,
        }

    def ask(
        self, question: str, session_id: str | None, subjects: list[str], doc_ids: list[str], top_k: int
    ) -> dict[str, Any]:
        body = self._ask_body(question, session_id, subjects, doc_ids, top_k)
        return self._request("POST", "/chat/ask", timeout=180, json=body)

    def stream_ask(
        self,
        question: str,
        session_id: str | None,
        subjects: list[str],
        doc_ids: list[str],
        top_k: int,
        result: dict[str, Any],
    ) -> Iterator[str]:
        """Yield answer tokens; the final payload (sources, ids) is stored into ``result``."""
        body = self._ask_body(question, session_id, subjects, doc_ids, top_k)
        try:
            res = self.http.post(self.base_url + "/chat/stream", json=body, stream=True, timeout=180)
        except requests.ConnectionError as exc:
            raise ApiError("BACKEND_DOWN", f"Cannot reach {self.base_url}") from exc
        except requests.Timeout as exc:
            raise ApiError("TIMEOUT", "Request timed out") from exc
        if res.status_code >= 400:
            raise self._error(res)
        event = None
        for line in res.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1])
                if event == "token":
                    yield data["t"]
                elif event == "sources":
                    result.update(data)
                elif event == "error":
                    raise ApiError(data.get("error", "ERROR"), data.get("detail", "Streaming failed"))

    def list_sessions(self) -> list[dict[str, Any]]:
        return self._request("GET", "/sessions")

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/sessions/{session_id}/messages")

    def delete_session(self, session_id: str) -> None:
        self._request("DELETE", f"/sessions/{session_id}")

    def feedback(self, message_id: int, rating: int, comment: str | None = None) -> None:
        self._request(
            "POST", "/feedback", json={"message_id": message_id, "rating": rating, "comment": comment}
        )
