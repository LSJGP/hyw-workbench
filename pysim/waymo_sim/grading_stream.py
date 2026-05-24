"""Streaming grading: in-process Python grader + offline SimLog writer + live C++ pipe.

* `OnlineGrader` mirrors the three C++ metrics (planning_limit_checker,
  speed_checker, regulatory_collision_checker) so the user sees PASS/FAIL each
  frame without rebuilding C++.
* `SimLogWriter` flushes a SimLog JSON to disk that the C++ `grading_main`
  binary can re-score offline (batch mode).
* `CppOnlineGrader` spawns `grading_main --stream` once and feeds it one
  MetricFrameInput JSON per stdin line, so the C++ scorer is **also** online
  and writes its `grading_report.json` when stdin closes.

All three paths agree by construction (same thresholds, same liability rules).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, TextIO

from .world import CollisionInfo, FrameHook, FrameRecord


# Defaults match the C++ side
DEFAULT_MAX_SPEED_MPS = 33.3
DEFAULT_MAX_DESIRED_SPEED_MPS = 33.3


@dataclass
class MetricSummary:
    name: str
    passed: bool
    detail: str


@dataclass
class _SpeedState:
    violations: int = 0
    total: int = 0


@dataclass
class _LimitState:
    violations: int = 0
    total: int = 0


@dataclass
class _CollisionState:
    total: int = 0
    collisions: int = 0
    exempt: int = 0
    non_exempt: int = 0


class OnlineGrader(FrameHook):
    """Per-frame Python grader; PASS/FAIL printed every `print_every` frames."""

    def __init__(
        self,
        max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
        max_desired_speed_mps: float = DEFAULT_MAX_DESIRED_SPEED_MPS,
        sink: Optional[TextIO] = None,
        print_every: int = 5,
    ):
        self.max_speed_mps = max_speed_mps
        self.max_desired_speed_mps = max_desired_speed_mps
        self.sink: TextIO = sink or sys.stdout
        self.print_every = max(1, int(print_every))

        self._speed = _SpeedState()
        self._limit = _LimitState()
        self._coll = _CollisionState()
        self._first_collision_frame: Optional[int] = None

    # ---- FrameHook ----

    def on_frame(self, rec: FrameRecord) -> None:
        speed_ok = self._tick_speed(rec)
        limit_ok = self._tick_limit(rec)
        coll_ok = self._tick_collision(rec)
        if rec.frame_id % self.print_every == 0 or rec.collision.collided or not (speed_ok and limit_ok):
            tag = "PASS" if (speed_ok and limit_ok and coll_ok) else "FAIL"
            print(
                f"[grader] f={rec.frame_id:>4d} t={rec.timestamp_us/1e6:7.2f}s "
                f"v={rec.ego.speed:5.2f}m/s acc={rec.ego.acceleration:+5.2f} "
                f"des={rec.command.desired_speed_mps:5.2f} npcs={rec.num_npcs:>3d} "
                f"coll={'Y' if rec.collision.collided else 'n'} -> {tag}",
                file=self.sink,
                flush=True,
            )
            if rec.collision.collided:
                ev = rec.collision
                print(
                    f"[grader]   collision kind={ev.kind} other={ev.other_id} "
                    f"exempt={ev.exempt}({ev.exempt_reason or '-'}) "
                    f"approach={ev.approach_angle_deg:.1f}deg rel_v={ev.relative_speed_mps:.2f}",
                    file=self.sink,
                    flush=True,
                )

    def on_finish(self, records: List[FrameRecord]) -> None:
        for s in self.summaries():
            print(f"[grader] {s.name}: {'PASS' if s.passed else 'FAIL'} ({s.detail})", file=self.sink, flush=True)
        overall = all(s.passed for s in self.summaries())
        print(f"[grader] OVERALL: {'PASS' if overall else 'FAIL'}", file=self.sink, flush=True)

    # ---- Public API ----

    def summaries(self) -> List[MetricSummary]:
        return [
            MetricSummary(
                name="planning_limit_checker",
                passed=self._limit.violations == 0,
                detail=f"bad_frames={self._limit.violations}/{self._limit.total}",
            ),
            MetricSummary(
                name="speed_checker",
                passed=self._speed.violations == 0,
                detail=f"violations={self._speed.violations}/{self._speed.total}",
            ),
            MetricSummary(
                name="regulatory_collision_checker",
                passed=self._coll.non_exempt == 0,
                detail=(
                    f"non_exempt={self._coll.non_exempt} "
                    f"exempt={self._coll.exempt} "
                    f"total_collision_frames={self._coll.collisions}/{self._coll.total}"
                ),
            ),
        ]

    @property
    def overall_passed(self) -> bool:
        return all(s.passed for s in self.summaries())

    # ---- per-frame tick ----

    def _tick_speed(self, rec: FrameRecord) -> bool:
        self._speed.total += 1
        ok = rec.ego.speed <= self.max_speed_mps + 1e-6
        if not ok:
            self._speed.violations += 1
        return ok

    def _tick_limit(self, rec: FrameRecord) -> bool:
        self._limit.total += 1
        ok = rec.command.desired_speed_mps <= self.max_desired_speed_mps + 1e-6
        if not ok:
            self._limit.violations += 1
        return ok

    def _tick_collision(self, rec: FrameRecord) -> bool:
        self._coll.total += 1
        if not rec.collision.collided:
            return True
        self._coll.collisions += 1
        if rec.collision.exempt:
            self._coll.exempt += 1
            return True
        if self._first_collision_frame is None:
            self._first_collision_frame = rec.frame_id
        self._coll.non_exempt += 1
        return False


class SimLogWriter(FrameHook):
    """Stream frames into a SimLog JSON file consumable by grading_main.

    Layout:
      { "source": "<src>", "frames": [ MetricFrameInput, ... ] }

    The file is rewritten after each frame so the user can `tail -f` mid-run
    or re-score with the C++ binary at any time.
    """

    def __init__(self, output_path: Path, source: str = "waymo_sim", flush_every: int = 1):
        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.flush_every = max(1, int(flush_every))
        self._frames: List[Dict] = []
        self._dirty_since_flush = 0

    def on_frame(self, rec: FrameRecord) -> None:
        self._frames.append(_frame_to_dict(rec))
        self._dirty_since_flush += 1
        if self._dirty_since_flush >= self.flush_every:
            self._flush()
            self._dirty_since_flush = 0

    def on_finish(self, records: List[FrameRecord]) -> None:
        self._flush()

    def _flush(self) -> None:
        doc = {"source": self.source, "frames": self._frames}
        tmp = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        os.replace(tmp, self.output_path)


class CppOnlineGrader(FrameHook):
    """Pipe each frame to a long-lived `grading_main --stream` subprocess.

    Lifecycle:
      * `__init__` spawns the binary with stdin piped + stdout/stderr forwarded.
      * `on_frame` writes a single MetricFrameInput JSON line to stdin.
      * `on_finish` closes stdin and waits — the binary then calls `Finish()`
        and writes the GradingReport JSON to `report_path`.

    The binary's tick lines (`[cpp] frame=K t=Ts ...`) appear on the same TTY
    as the Python `[grader]` output, so the user sees both interleaved live.
    """

    def __init__(self, binary_path: Path, report_path: Path, sink: Optional[TextIO] = None):
        self.binary_path = Path(binary_path).expanduser().resolve()
        self.report_path = Path(report_path).expanduser().resolve()
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.sink = sink or sys.stdout
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._closed = False
        self._start()

    def _start(self) -> None:
        if not self.binary_path.is_file():
            raise FileNotFoundError(f"grading_main not found: {self.binary_path}")
        # stdout: line-buffered text passthrough so [cpp] tick lines appear immediately.
        # stderr: forwarded on a daemon thread (spdlog goes there).
        self._proc = subprocess.Popen(
            [str(self.binary_path), "--stream", str(self.report_path)],
            stdin=subprocess.PIPE,
            stdout=None,        # inherit -> shows up on user's terminal
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stderr_thread = threading.Thread(
            target=self._pump_stderr, args=(self._proc.stderr,), daemon=True
        )
        self._stderr_thread.start()

    @staticmethod
    def _pump_stderr(stream) -> None:
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                # spdlog goes to stderr; surface it but tag for clarity
                sys.stderr.write(line)
                sys.stderr.flush()
        except Exception:
            pass

    def on_frame(self, rec: FrameRecord) -> None:
        if self._proc is None or self._proc.stdin is None or self._closed:
            return
        line = json.dumps(_frame_to_dict(rec), separators=(",", ":"))
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            self._closed = True
            print("[sim] WARNING: cpp grader subprocess closed early", file=sys.stderr)

    def on_finish(self, records: List[FrameRecord]) -> None:
        self.close()

    def close(self) -> None:
        if self._proc is None or self._closed:
            return
        self._closed = True
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2)


def _frame_to_dict(rec: FrameRecord) -> Dict:
    out: Dict = {
        "frame_id": int(rec.frame_id),
        "timestamp_us": int(rec.timestamp_us),
        "vehicle_state": {
            "x": float(rec.ego.x),
            "y": float(rec.ego.y),
            "heading": float(rec.ego.heading),
            "speed": float(rec.ego.speed),
            "acceleration": float(rec.ego.acceleration),
        },
        "planning_command": {
            "desired_speed_mps": float(rec.command.desired_speed_mps),
        },
    }
    ev: CollisionInfo = rec.collision
    if ev.collided:
        out["collision_event"] = {
            "collided": True,
            "other_id": int(ev.other_id),
            "kind": ev.kind,
            "ego_at_fault": bool(ev.ego_at_fault),
            "exempt": bool(ev.exempt),
            "exempt_reason": ev.exempt_reason,
            "relative_speed_mps": float(ev.relative_speed_mps),
            "ego_speed_mps": float(ev.ego_speed_mps),
            "approach_angle_deg": float(ev.approach_angle_deg),
        }
    return out
