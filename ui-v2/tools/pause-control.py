#!/usr/bin/env python3
"""Tiny host-side control endpoint so the browser can pause the engine.

EngineCore runs in its own process inside the container and exposes no RPC for
pause_scheduler. The run directory is bind-mounted, so the overlay's pause
watcher reads a flag file from it; this server is the only thing that writes
that file, and it exists purely so a page served on another port can reach it.

    python3 ui-v2/tools/pause-control.py <run-dir> [port]

    POST /pause    -> write "1"   (scheduler PAUSED_ALL, generation stops)
    POST /resume   -> write "0"   (scheduler UNPAUSED, same request continues)
    GET  /state    -> {"paused": bool}
"""
import json
import pathlib
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RUN_DIR = pathlib.Path(sys.argv[1]).resolve()
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8091
FLAG = RUN_DIR / "semantic-shm" / "pause.flag"


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # The page is served from another port, so the browser needs this.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _paused(self):
        try:
            return FLAG.read_text().strip() == "1"
        except FileNotFoundError:
            return False

    def do_OPTIONS(self):
        self._send({})

    def do_GET(self):
        if self.path.rstrip("/") == "/state":
            self._send({"paused": self._paused(), "flag": str(FLAG)})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.rstrip("/")
        if path == "/pause":
            FLAG.write_text("1\n")
        elif path == "/resume":
            FLAG.write_text("0\n")
        else:
            self._send({"error": "not found"}, 404)
            return
        self._send({"paused": self._paused()})

    def log_message(self, fmt, *args):  # quieter than the default access log
        sys.stderr.write("pause-control: " + (fmt % args) + "\n")


if __name__ == "__main__":
    if not (RUN_DIR / "semantic-shm").is_dir():
        sys.exit(f"no semantic-shm directory under {RUN_DIR}")
    print(f"pause-control: writing {FLAG}, listening on 127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
