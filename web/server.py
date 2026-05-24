#!/usr/bin/env python3
"""Local batch-run dashboard API + static UI (stdlib only)."""

from __future__ import annotations

import json
import mimetypes
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

WEB_ROOT = Path(__file__).resolve().parent
WORKBENCH_ROOT = WEB_ROOT.parent
sys.path.insert(0, str(WORKBENCH_ROOT))
sys.path.insert(0, str(WORKBENCH_ROOT / "tools"))

from hyw_paths import OUTPUT_DIR, WORKBENCH_ROOT  # noqa: E402

from batch_run_scenarios import (  # noqa: E402
    CPP_MODES,
    DEFAULT_GRADING_BIN,
    DEFAULT_SIM_RUNNER_HINT,
    LOG_LEVELS,
    METRIC_CATALOG,
    PLANNERS,
    REFERENCE_SOURCES,
    BatchConfig,
    list_scenarios,
    run_batch,
)

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
DEFAULT_PORT = 8765


def _job_log_append(job_id: str, line: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["log"].append(line)
        if len(job["log"]) > 8000:
            job["log"] = job["log"][-6000:]


class Handler(BaseHTTPRequestHandler):
    server_version = "HywBatchUI/1.0"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, code: int, obj: Any) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Any:
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._serve_file(WEB_ROOT / "index.html")
        if path.startswith("/api/jobs/"):
            job_id = path.split("/")[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                return self._send_json(404, {"error": "job not found"})
            return self._send_json(200, job)
        if path == "/api/meta":
            grading_bin = Path(DEFAULT_GRADING_BIN)
            return self._send_json(
                200,
                {
                    "planners": PLANNERS,
                    "metrics": [{"name": k} for k in METRIC_CATALOG],
                    "scenarios": list_scenarios(),
                    "log_levels": LOG_LEVELS,
                    "cpp_modes": CPP_MODES,
                    "reference_sources": REFERENCE_SOURCES,
                    "defaults": {
                        "planner": "local_dwa",
                        "metrics": list(METRIC_CATALOG.keys()),
                        "dt": 0.1,
                        "desired_speed": 13.9,
                        "reference_source": "map",
                        "reference_step": 1.0,
                        "cpp_mode": "both",
                        "log_level": "info",
                        "gif_fps": 120,
                        "gif_dpi": 100,
                        "make_gif": True,
                        "run_grading": True,
                    },
                    "paths": {
                        "workbench_root": str(WORKBENCH_ROOT),
                        "grading_bin": str(grading_bin),
                        "grading_bin_exists": grading_bin.is_file(),
                        "sim_runner_hint": DEFAULT_SIM_RUNNER_HINT,
                    },
                },
            )
        return self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/jobs":
            return self._send_json(404, {"error": "not found"})
        try:
            body = self._read_json()
        except json.JSONDecodeError as e:
            return self._send_json(400, {"error": f"invalid json: {e}"})

        scenarios = body.get("scenarios") or []
        if not scenarios:
            return self._send_json(400, {"error": "select at least one scenario"})

        job_id = uuid.uuid4().hex[:12]
        cfg = BatchConfig(
            scenario_names=list(scenarios),
            planner=body.get("planner", "local_dwa"),
            metrics=body.get("metrics") or list(METRIC_CATALOG.keys()),
            reference_source=body.get("reference_source", "map"),
            reference_step=float(body.get("reference_step", 1.0)),
            dt=float(body.get("dt", 0.1)),
            max_seconds=float(body.get("max_seconds", 0.0)),
            desired_speed=float(body.get("desired_speed", 13.9)),
            cpp_mode=body.get("cpp_mode", "both"),
            run_grading=bool(body.get("run_grading", True)),
            grading_bin=body.get("grading_bin", ""),
            log_level=body.get("log_level", "info"),
            log_dir=body.get("log_dir", ""),
            make_gif=bool(body.get("make_gif", True)),
            gif_fps=int(body.get("gif_fps", 120)),
            gif_dpi=int(body.get("gif_dpi", 100)),
            gif_reference_step=float(body.get("gif_reference_step", 1.0)),
            output_log_dir=body.get("output_log_dir", ""),
            output_report_dir=body.get("output_report_dir", ""),
            output_viz_dir=body.get("output_viz_dir", ""),
        )

        with JOBS_LOCK:
            JOBS[job_id] = {
                "id": job_id,
                "status": "running",
                "config": body,
                "log": [],
                "result": None,
                "error": None,
                "created_at": time.time(),
            }

        def worker() -> None:
            def log_cb(line: str) -> None:
                _job_log_append(job_id, line)

            try:
                metrics_path = OUTPUT_DIR / "batch" / f"metrics_{job_id}.json"
                summary = run_batch(cfg, log=log_cb, metrics_config_path=metrics_path)
                with JOBS_LOCK:
                    JOBS[job_id]["status"] = "done"
                    JOBS[job_id]["result"] = summary
            except Exception as e:
                _job_log_append(job_id, traceback.format_exc())
                with JOBS_LOCK:
                    JOBS[job_id]["status"] = "error"
                    JOBS[job_id]["error"] = str(e)

        threading.Thread(target=worker, daemon=True).start()
        return self._send_json(202, {"job_id": job_id})

    def _serve_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = p.parse_args()

    (OUTPUT_DIR / "batch").mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[web] http://{args.host}:{args.port}/")
    print(f"[web] workbench: {WORKBENCH_ROOT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] stopped")


if __name__ == "__main__":
    main()
