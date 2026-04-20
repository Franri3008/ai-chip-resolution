"""Lightweight HTTP + SSE server for the ai-chip-resolution live dashboard.

Runs in a background daemon thread while main.py executes.
StatusTracker calls broadcast() on every update, pushing the full state
snapshot to all connected EventSource clients.
"""
from __future__ import annotations

import json
import mimetypes
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PORT = 8770
DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "ui" / "live"

_sse_clients: list[queue.Queue] = []
_sse_lock = threading.Lock()
_last_msg: bytes | None = None


def broadcast(state: dict) -> None:
    """Push the current state snapshot to every connected SSE client."""
    global _last_msg
    msg = ("data: " + json.dumps(state, ensure_ascii=False, default=str) + "\n\n").encode()
    with _sse_lock:
        _last_msg = msg
        dead: list[queue.Queue] = []
        for q in _sse_clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)


class VizHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args) -> None:
        pass

    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/") or "/"

        if path in ("/", "/index.html"):
            self._serve_file(DASHBOARD_DIR / "index.html")
        elif path == "/api/status/stream":
            self._sse_stream()
        elif path.startswith(("/css/", "/js/")):
            self._serve_file(DASHBOARD_DIR / path.lstrip("/"))
        else:
            self.send_response(404)
            self.end_headers()

    def _sse_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        client_q: queue.Queue = queue.Queue(maxsize=64)
        with _sse_lock:
            _sse_clients.append(client_q)
            if _last_msg is not None:
                client_q.put_nowait(_last_msg)

        try:
            while True:
                try:
                    msg = client_q.get(timeout=20)
                    self.wfile.write(msg)
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with _sse_lock:
                if client_q in _sse_clients:
                    _sse_clients.remove(client_q)

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_response(404)
            self.end_headers()
            return
        mime, _ = mimetypes.guess_type(str(path))
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start_viz_server(port: int = PORT) -> bool:
    """Start the viz server in a background daemon thread and open the browser.

    Returns True on success, False if the port is unavailable.
    """
    try:
        server = HTTPServer(("localhost", port), VizHandler)
    except OSError:
        return False

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://localhost:{port}"
    print(f"\nLive Dashboard: {url}\n")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    return True
