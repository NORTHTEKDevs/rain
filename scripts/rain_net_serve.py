"""Simple HTTP server exposing RAIN-Net over POST /query.

Single-file stdlib http.server -- no extra deps. Returns AuditReport
as JSON. Optionally Bearer-token gated via RAIN_AUTH_TOKEN env var.

Run:
    python scripts/rain_net_serve.py --port 8080
    rain-serve --port 8080                      # if pip-installed

Query:
    curl -X POST http://localhost:8080/query \\
        -H 'Content-Type: application/json' \\
        -d '{"query": "what is 47 * 38"}'

Health:
    curl http://localhost:8080/health
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Lock

from rain.core.rain_net import RainNet, RainNetConfig


_NET_LOCK = Lock()
_NET: RainNet | None = None
_TOKEN: str = ""


def _make_handler() -> type[BaseHTTPRequestHandler]:
    """Closure factory so we can capture per-instance state in the handler."""

    class Handler(BaseHTTPRequestHandler):
        # Silence default per-request logging; use stderr only on error.
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        def _send_json(self, status: int, body: dict) -> None:
            data = json.dumps(body, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _auth_ok(self) -> bool:
            if not _TOKEN:
                return True
            header = self.headers.get("Authorization", "")
            return header == f"Bearer {_TOKEN}"

        def do_GET(self) -> None:
            if self.path == "/health":
                assert _NET is not None
                self._send_json(200, {"ok": True, "stats": _NET.stats()})
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._auth_ok():
                self._send_json(401, {"error": "unauthorized"})
                return
            if self.path == "/query":
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length <= 0:
                    self._send_json(400, {"error": "missing body"})
                    return
                raw = self.rfile.read(length).decode("utf-8")
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError as e:
                    self._send_json(400, {"error": f"invalid JSON: {e}"})
                    return
                query = body.get("query", "").strip()
                modality = body.get("modality", "text")
                n_candidates = body.get("n_candidates")
                if not query:
                    self._send_json(400, {"error": "missing 'query'"})
                    return
                assert _NET is not None
                with _NET_LOCK:
                    try:
                        report = _NET.answer(
                            query, modality=modality, n_candidates=n_candidates
                        )
                    except Exception as e:  # noqa: BLE001
                        self._send_json(500, {"error": str(e)})
                        return
                self._send_json(200, report.to_dict())
                return
            if self.path == "/ingest":
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length).decode("utf-8")
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError as e:
                    self._send_json(400, {"error": f"invalid JSON: {e}"})
                    return
                text = body.get("text", "").strip()
                source = body.get("source", "")
                if not text:
                    self._send_json(400, {"error": "missing 'text'"})
                    return
                assert _NET is not None
                with _NET_LOCK:
                    fid = _NET.ingest_fact(text=text, source=source)
                self._send_json(200, {"fact_id": fid, "kb_size": len(_NET.memory.semantic)})
                return
            self._send_json(404, {"error": "not found"})

    return Handler


def main(argv: list[str]) -> int:
    global _NET, _TOKEN
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--dim", type=int, default=10_000)
    p.add_argument("--candidates", type=int, default=2)
    p.add_argument(
        "--skills", default="skills", help="path to skills/ dir (empty to disable)"
    )
    p.add_argument(
        "--verifier",
        default="data/checkpoints/verifier_head_v0.npz",
        help="trained verifier head path; not loaded if missing",
    )
    p.add_argument("--hymn", default=None, help="path to trained HYMN-Plus .npz")
    args = p.parse_args(argv)

    cfg = RainNetConfig(
        dim=args.dim,
        n_candidates=args.candidates,
        skills_dir=args.skills or None,
        verifier_checkpoint_path=args.verifier or None,
        hymn_checkpoint_path=args.hymn,
        semantic_top_k=4,
    )
    _NET = RainNet(config=cfg)
    _TOKEN = os.environ.get("RAIN_AUTH_TOKEN", "")
    auth_note = f"(auth: Bearer {_TOKEN[:6]}...)" if _TOKEN else "(auth: disabled)"
    print(f"rain-net serve: dim={args.dim} skills={len(_NET.skill_registry)} {auth_note}")
    print(f"  POST {args.host}:{args.port}/query   {{\"query\": \"...\"}}")
    print(f"  POST {args.host}:{args.port}/ingest  {{\"text\": \"...\", \"source\": \"...\"}}")
    print(f"  GET  {args.host}:{args.port}/health")
    server = HTTPServer((args.host, args.port), _make_handler())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
