"""Pytest configuration and lightweight ASGI TestClient fallback when httpx2 is not installed."""

import asyncio
import json as std_json
import sys
import types
import urllib.parse
from typing import Any, Dict, Optional


class _ASGIMockResponse:
    def __init__(self, status_code: int, headers: Dict[str, str], body: bytes):
        self.status_code = status_code
        self.headers = headers
        self.content = body

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return std_json.loads(self.content)


class _ASGITestClient:
    """Zero-dependency synchronous ASGI test client for FastAPI/Starlette."""

    def __init__(self, app: Any, base_url: str = "http://testserver"):
        self.app = app
        self.base_url = base_url

    def request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json: Any = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> _ASGIMockResponse:
        parsed = urllib.parse.urlsplit(url)
        path = parsed.path or "/"
        query_string = parsed.query
        if params:
            extra_q = urllib.parse.urlencode(params, doseq=True)
            query_string = f"{query_string}&{extra_q}" if query_string else extra_q

        req_headers = {
            "host": "testserver",
            "user-agent": "testclient",
        }
        if headers:
            for k, v in headers.items():
                req_headers[k.lower()] = v

        body_bytes = b""
        if json is not None:
            body_bytes = std_json.dumps(json).encode("utf-8")
            req_headers.setdefault("content-type", "application/json")
        elif files is not None:
            boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
            parts = []
            if data:
                for k, v in data.items():
                    parts.append(
                        f"--{boundary}\r\n"
                        f'Content-Disposition: form-data; name="{k}"\r\n\r\n'
                        f"{v}\r\n".encode("utf-8")
                    )
            for field_name, file_info in files.items():
                if isinstance(file_info, (tuple, list)):
                    filename = file_info[0]
                    content = file_info[1]
                    ctype = (
                        file_info[2]
                        if len(file_info) > 2
                        else "application/octet-stream"
                    )
                else:
                    filename = field_name
                    content = file_info
                    ctype = "application/octet-stream"
                if hasattr(content, "read"):
                    content = content.read()
                if isinstance(content, str):
                    content = content.encode("utf-8")
                header_part = (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
                    f"Content-Type: {ctype}\r\n\r\n"
                ).encode("utf-8")
                parts.append(header_part + content + b"\r\n")
            parts.append(f"--{boundary}--\r\n".encode("utf-8"))
            body_bytes = b"".join(parts)
            req_headers["content-type"] = f"multipart/form-data; boundary={boundary}"
        elif data is not None:
            body_bytes = urllib.parse.urlencode(data, doseq=True).encode("utf-8")
            req_headers.setdefault(
                "content-type", "application/x-www-form-urlencoded"
            )

        req_headers["content-length"] = str(len(body_bytes))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method.upper(),
            "scheme": "http",
            "path": urllib.parse.unquote(path),
            "raw_path": path.encode("ascii", errors="ignore"),
            "query_string": query_string.encode("ascii", errors="ignore"),
            "headers": [
                (k.encode("latin-1"), v.encode("latin-1"))
                for k, v in req_headers.items()
            ],
            "client": ("127.0.0.1", 50000),
            "server": ("testserver", 80),
        }

        status_code = 500
        resp_headers: Dict[str, str] = {}
        resp_chunks = []
        request_sent = False

        async def receive():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {
                    "type": "http.request",
                    "body": body_bytes,
                    "more_body": False,
                }
            return {"type": "http.disconnect"}

        async def send(message):
            nonlocal status_code, resp_headers
            if message["type"] == "http.response.start":
                status_code = message["status"]
                resp_headers = {
                    k.decode("latin-1").lower(): v.decode("latin-1")
                    for k, v in message.get("headers", [])
                }
            elif message["type"] == "http.response.body":
                resp_chunks.append(message.get("body", b""))

        asyncio.run(self.app(scope, receive, send))
        return _ASGIMockResponse(status_code, resp_headers, b"".join(resp_chunks))

    def get(self, url: str, **kwargs) -> _ASGIMockResponse:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> _ASGIMockResponse:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs) -> _ASGIMockResponse:
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs) -> _ASGIMockResponse:
        return self.request("DELETE", url, **kwargs)

    def patch(self, url: str, **kwargs) -> _ASGIMockResponse:
        return self.request("PATCH", url, **kwargs)


try:
    from fastapi.testclient import TestClient  # noqa: F401
except RuntimeError:
    tc_mod = types.ModuleType("fastapi.testclient")
    tc_mod.TestClient = _ASGITestClient
    sys.modules["fastapi.testclient"] = tc_mod
    sys.modules["starlette.testclient"] = tc_mod

