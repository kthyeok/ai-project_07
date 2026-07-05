from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import traceback
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
HTML_PATH = PROJECT_DIR / "rag_chatbot.html"
ENV_PATH = PROJECT_DIR / ".env"

TABLE_NAME = "documents_policy_2026"
MATCH_FUNCTION = "match_documents_policy_2026"
SOURCE_NAME = "policy_fund_2026_pdf"
EMBEDDING_DIMENSIONS = 384
OPENROUTER_MODEL_DEFAULT = "google/gemma-4-31b-it:free"
OPENROUTER_FALLBACK_MODEL_DEFAULT = "openrouter/free"
COHERE_RERANK_MODEL_DEFAULT = "rerank-v3.5"


def load_env(path: Path) -> dict[str, str]:
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({key: value for key, value in os.environ.items() if value})
    return values


ENV = load_env(ENV_PATH)


def require_env(name: str) -> str:
    value = ENV.get(name)
    if not value:
        raise RuntimeError(f"Missing {name}. Add it to {ENV_PATH}")
    return value


SUPABASE_URL = require_env("SUPABASE_URL").rstrip("/")
SUPABASE_KEY = ENV.get("SUPABASE_SERVICE_ROLE_KEY") or ENV.get("SUPABASE_KEY")
if not SUPABASE_KEY:
    raise RuntimeError(f"Missing SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY in {ENV_PATH}")

OPENROUTER_API_KEY = require_env("OPENROUTER_API_KEY")
COHERE_API_KEY = ENV.get("COHERE_API_KEY", "")
OPENROUTER_MODEL = ENV.get("OPENROUTER_MODEL", OPENROUTER_MODEL_DEFAULT)
OPENROUTER_FALLBACK_MODEL = ENV.get("OPENROUTER_FALLBACK_MODEL", OPENROUTER_FALLBACK_MODEL_DEFAULT)
COHERE_RERANK_MODEL = ENV.get("COHERE_RERANK_MODEL", COHERE_RERANK_MODEL_DEFAULT)
USE_COHERE_RERANK = ENV.get("USE_COHERE_RERANK", "true").lower() not in {"0", "false", "no"}


def hash_embedding(text: str, dim: int = EMBEDDING_DIMENSIONS) -> list[float]:
    vec = [0.0] * dim
    text = " ".join(str(text).lower().split())
    for n in (2, 3, 4, 5):
        if len(text) < n:
            continue
        for index in range(len(text) - n + 1):
            gram = text[index : index + n]
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).hexdigest()
            h = int(digest, 16)
            slot = h % dim
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vec[slot] += sign
    norm = math.sqrt(sum(item * item for item in vec)) or 1.0
    return [round(item / norm, 8) for item in vec]


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(item) for item in vector) + "]"


def request_json(url: str, payload: dict, headers: dict[str, str], timeout: int = 60):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def retrieve_documents(query: str, vector_k: int) -> list[dict]:
    url = f"{SUPABASE_URL}/rest/v1/rpc/{MATCH_FUNCTION}"
    payload = {
        "query_embedding": vector_literal(hash_embedding(query)),
        "match_count": vector_k,
        "filter": {"source": SOURCE_NAME},
    }
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    result = request_json(url, payload, headers, timeout=60)
    return result or []


def rerank_documents(query: str, docs: list[dict], final_k: int) -> list[dict]:
    if not docs:
        return []
    if not USE_COHERE_RERANK or not COHERE_API_KEY:
        return docs[:final_k]

    url = "https://api.cohere.com/v2/rerank"
    payload = {
        "model": COHERE_RERANK_MODEL,
        "query": query,
        "documents": [doc.get("content", "") for doc in docs],
        "top_n": min(final_k, len(docs)),
    }
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json",
    }
    result = request_json(url, payload, headers, timeout=60)
    reranked = []
    for item in result.get("results", []):
        index = item["index"]
        doc = dict(docs[index])
        doc["rerank_score"] = item.get("relevance_score")
        reranked.append(doc)
    return reranked


def compact_sources(docs: list[dict]) -> list[dict]:
    sources = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.get("metadata") or {}
        content = doc.get("content") or ""
        sources.append(
            {
                "rank": index,
                "id": doc.get("id"),
                "page": metadata.get("page_start"),
                "section": metadata.get("section_title"),
                "similarity": doc.get("similarity"),
                "rerank_score": doc.get("rerank_score"),
                "preview": content[:260].replace("\n", " "),
            }
        )
    return sources


def format_context(docs: list[dict]) -> str:
    blocks = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.get("metadata") or {}
        score_bits = []
        if isinstance(doc.get("similarity"), (int, float)):
            score_bits.append(f"similarity={doc['similarity']:.4f}")
        if isinstance(doc.get("rerank_score"), (int, float)):
            score_bits.append(f"rerank={doc['rerank_score']:.4f}")
        score_text = ", ".join(score_bits)
        blocks.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"page: {metadata.get('page_start')}",
                    f"section: {metadata.get('section_title')}",
                    f"scores: {score_text}",
                    "content:",
                    doc.get("content", ""),
                ]
            )
        )
    return "\n\n".join(blocks)


def openrouter_stream_for_model(model: str, question: str, docs: list[dict]):
    context = format_context(docs)
    system_prompt = (
        "You are a Korean RAG assistant. Answer only from the provided context. "
        "If the context is insufficient, say so clearly. Cite sources as [Source 1], [Source 2]. "
        "Keep answers practical and concise."
    )
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer in Korean. Use bullets or a short table when helpful."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8787",
        "X-Title": "Policy PDF RAG Chatbot",
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=data,
        method="POST",
        headers=headers,
    )

    with urllib.request.urlopen(request, timeout=35) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                payload = json.loads(chunk)
                delta = payload["choices"][0].get("delta", {})
                text = delta.get("content")
                if text:
                    yield text
            except Exception:
                continue


def openrouter_stream(question: str, docs: list[dict]):
    models = [OPENROUTER_MODEL]
    if OPENROUTER_FALLBACK_MODEL and OPENROUTER_FALLBACK_MODEL not in models:
        models.append(OPENROUTER_FALLBACK_MODEL)

    last_error = None
    for model in models:
        try:
            if model != OPENROUTER_MODEL:
                yield f"\n\n[알림] 기본 무료 Gemma 모델이 제한되어 무료 라우터({model})로 전환합니다.\n\n"
            yield from openrouter_stream_for_model(model, question, docs)
            return
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"OpenRouter model {model} failed with HTTP {exc.code}: {detail}")
            if exc.code in {429, 500, 502, 503, 504}:
                continue
            raise last_error
    if last_error:
        raise last_error


def sse_line(event: dict) -> bytes:
    return ("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "PolicyRAGChatbot/1.0"

    def log_message(self, fmt, *args):
        sys.stdout.write("%s - %s\n" % (self.address_string(), fmt % args))

    def send_bytes(self, body: bytes, content_type: str = "text/plain; charset=utf-8", status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in {"/", "/rag_chatbot.html"}:
            self.send_bytes(HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/health":
            body = {
                "ok": True,
                "table": TABLE_NAME,
                "match_function": MATCH_FUNCTION,
                "source": SOURCE_NAME,
                "openrouter_model": OPENROUTER_MODEL,
                "openrouter_fallback_model": OPENROUTER_FALLBACK_MODEL,
                "cohere_rerank": USE_COHERE_RERANK and bool(COHERE_API_KEY),
            }
            self.send_bytes(json.dumps(body, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            return
        self.send_bytes(b"Not found", status=404)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path != "/api/chat":
            self.send_bytes(b"Not found", status=404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            question = (payload.get("message") or "").strip()
            vector_k = max(1, min(int(payload.get("vector_k", 20)), 50))
            final_k = max(1, min(int(payload.get("final_k", 5)), 10))
            if not question:
                raise ValueError("질문을 입력하세요.")

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            docs = retrieve_documents(question, vector_k)
            selected_docs = rerank_documents(question, docs, final_k)
            self.wfile.write(sse_line({"type": "sources", "sources": compact_sources(selected_docs)}))
            self.wfile.flush()

            if not selected_docs:
                self.wfile.write(sse_line({"type": "delta", "text": "검색된 근거 문서가 없습니다."}))
                self.wfile.write(sse_line({"type": "done"}))
                self.wfile.flush()
                return

            for text in openrouter_stream(question, selected_docs):
                self.wfile.write(sse_line({"type": "delta", "text": text}))
                self.wfile.flush()
            self.wfile.write(sse_line({"type": "done"}))
            self.wfile.flush()
        except Exception as exc:
            traceback.print_exc()
            try:
                if not self.wfile.closed:
                    self.wfile.write(sse_line({"type": "error", "message": str(exc)}))
                    self.wfile.flush()
            except Exception:
                error_body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
                self.send_bytes(error_body, "application/json; charset=utf-8", status=500)


def main():
    port = int(os.environ.get("PORT", "8787"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"RAG chatbot server running: http://127.0.0.1:{port}")
    print(f"Using Supabase table: {TABLE_NAME}")
    print(f"Using OpenRouter model: {OPENROUTER_MODEL}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
