"""Start and monitor hyw-planner ``planner_server`` for batch sim / dashboard."""

from __future__ import annotations

import atexit
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

WORKBENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKBENCH_ROOT))

from hyw_paths import HYW_PLANNER, HYW_ROOT  # noqa: E402

DEFAULT_PLANNER_HOST = "127.0.0.1"
DEFAULT_PLANNER_PORT = 50051
DEFAULT_PLANNER_SERVER_BIN = HYW_PLANNER / "bazel-bin" / "cpp" / "planner_server"

_START_TIMEOUT_S = 20.0
_POLL_INTERVAL_S = 0.25

_planner_proc: Optional[subprocess.Popen] = None
_managed_port: Optional[int] = None
_planner_binary_mtime_at_start: Optional[float] = None


def _log(cb: Optional[Callable[[str], None]], msg: str) -> None:
    line = msg.rstrip("\n")
    if cb:
        cb(line)
    else:
        print(line, flush=True)


def is_planner_port_open(
    host: str = DEFAULT_PLANNER_HOST, port: int = DEFAULT_PLANNER_PORT
) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def _binary_mtime() -> Optional[float]:
    if not DEFAULT_PLANNER_SERVER_BIN.is_file():
        return None
    return DEFAULT_PLANNER_SERVER_BIN.stat().st_mtime


def _server_is_stale(
    host: str = DEFAULT_PLANNER_HOST, port: int = DEFAULT_PLANNER_PORT
) -> bool:
    """True when a listener exists but binary was rebuilt or process is external."""
    if not is_planner_port_open(host, port):
        return False
    bin_mtime = _binary_mtime()
    if bin_mtime is None:
        return False
    if _planner_proc is None or _planner_proc.poll() is not None:
        return True
    if _managed_port != port:
        return True
    if _planner_binary_mtime_at_start is None:
        return True
    return bin_mtime > _planner_binary_mtime_at_start + 1e-6


def _kill_listener_on_port(port: int) -> None:
    try:
        subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True,
            timeout=5.0,
            check=False,
        )
    except FileNotFoundError:
        pass
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not is_planner_port_open(port=port):
            return
        time.sleep(0.2)


def _reserve_planner_port(preferred: int = DEFAULT_PLANNER_PORT) -> int:
    if not is_planner_port_open(port=preferred):
        return preferred
    _kill_listener_on_port(preferred)
    if not is_planner_port_open(port=preferred):
        return preferred
    for port in range(preferred + 1, preferred + 50):
        if not is_planner_port_open(port=port):
            return port
    raise RuntimeError(f"no free planner port near {preferred}")


def planner_server_status(
    host: str = DEFAULT_PLANNER_HOST, port: Optional[int] = None
) -> dict:
    if port is None:
        port = _managed_port if _managed_port is not None else DEFAULT_PLANNER_PORT
    bin_path = DEFAULT_PLANNER_SERVER_BIN
    return {
        "host": host,
        "port": port,
        "address": f"{host}:{port}",
        "running": is_planner_port_open(host, port),
        "binary": str(bin_path),
        "binary_exists": bin_path.is_file(),
        "managed_by_workbench": _planner_proc is not None
        and _planner_proc.poll() is None,
    }


def _build_planner_server(log: Optional[Callable[[str], None]]) -> None:
    _log(log, "[planner] building //cpp:planner_server …")
    rc = subprocess.call(
        ["bazel", "build", "//cpp:planner_server"],
        cwd=str(HYW_PLANNER),
    )
    if rc != 0:
        raise RuntimeError(
            f"bazel build //cpp:planner_server failed (rc={rc}); "
            f"cd {HYW_PLANNER} && bazel build //cpp:planner_server"
        )


def _start_process(
    port: int, log: Optional[Callable[[str], None]]
) -> subprocess.Popen:
    global _planner_proc, _managed_port, _planner_binary_mtime_at_start
    bin_path = DEFAULT_PLANNER_SERVER_BIN
    if not bin_path.is_file():
        _build_planner_server(log)
    if not bin_path.is_file():
        raise FileNotFoundError(f"planner_server binary missing: {bin_path}")

    _log(log, f"[planner] starting {bin_path.name} on port {port}")
    proc = subprocess.Popen(
        [str(bin_path), "--port", str(port)],
        cwd=str(HYW_PLANNER),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    _planner_proc = proc
    _managed_port = port
    _planner_binary_mtime_at_start = _binary_mtime()

    def _drain() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            _log(log, "[planner] " + line.rstrip("\n"))

    import threading

    threading.Thread(target=_drain, daemon=True).start()
    return proc


def ensure_planner_server(
    host: str = DEFAULT_PLANNER_HOST,
    port: int = DEFAULT_PLANNER_PORT,
    log: Optional[Callable[[str], None]] = None,
    auto_build: bool = True,
) -> int:
    """Ensure gRPC planner_server is listening; start a managed process if needed."""
    global _planner_proc

    if is_planner_port_open(host, port) and not _server_is_stale(host, port):
        _log(log, f"[planner] already up at {host}:{port}")
        return port

    if is_planner_port_open(host, port):
        _log(
            log,
            f"[planner] stale planner_server on {host}:{port} "
            "(binary rebuilt or foreign process); restarting …",
        )
        stop_planner_server(log=log)
        _kill_listener_on_port(port)

    if is_planner_port_open(host, port):
        alt = _reserve_planner_port(port)
        if alt != port:
            _log(
                log,
                f"[planner] warning: could not free {host}:{port}; "
                f"using {host}:{alt} instead",
            )
            port = alt

    if _planner_proc is not None and _planner_proc.poll() is None:
        if _managed_port != port:
            stop_planner_server(log=log)
        else:
            _log(log, "[planner] restarting managed planner_server …")
            stop_planner_server(log=log)

    if not DEFAULT_PLANNER_SERVER_BIN.is_file():
        if auto_build:
            _build_planner_server(log)
        else:
            raise FileNotFoundError(
                f"planner_server not found: {DEFAULT_PLANNER_SERVER_BIN}\n"
                f"Build: cd {HYW_PLANNER} && bazel build //cpp:planner_server"
            )

    proc = _start_process(port, log)
    deadline = time.time() + _START_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"planner_server exited early (rc={proc.returncode}); "
                "see log lines above"
            )
        if is_planner_port_open(host, port):
            _log(log, f"[planner] ready at {host}:{port}")
            return port
        time.sleep(_POLL_INTERVAL_S)

    raise RuntimeError(
        f"planner_server did not open {host}:{port} within {_START_TIMEOUT_S}s"
    )


def stop_planner_server(log: Optional[Callable[[str], None]] = None) -> None:
    global _planner_proc, _managed_port, _planner_binary_mtime_at_start
    if _planner_proc is None:
        return
    if _planner_proc.poll() is None:
        _log(log, "[planner] stopping managed planner_server …")
        _planner_proc.terminate()
        try:
            _planner_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            _planner_proc.kill()
            _planner_proc.wait(timeout=2.0)
    _planner_proc = None
    _managed_port = None
    _planner_binary_mtime_at_start = None


atexit.register(lambda: stop_planner_server())
