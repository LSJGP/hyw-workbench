"""Streaming grading: in-process Python grader + offline SimLog writer + live C++ pipe.

* `OnlineGrader` mirrors the C++ metrics (planning_limit_checker, speed_checker,
  regulatory_collision_checker, collision_risk_checker) so the user sees
  PASS/FAIL each frame without rebuilding C++.
* `SimLogWriter` flushes a SimLog JSON to disk that the C++ `grading_main`
  binary can re-score offline (batch mode).
* `CppOnlineGrader` spawns `grading_main --stream` once and feeds it one
  MetricFrameInput JSON per stdin line, so the C++ scorer is **also** online
  and writes its `grading_report.json` when stdin closes.

All paths agree by construction (same thresholds, same liability rules).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, TextIO

from .planners.base import NPCSnapshot
from .world import CollisionInfo, FrameHook, FrameRecord


# Defaults match the C++ side
DEFAULT_MAX_SPEED_MPS = 33.3
DEFAULT_MAX_DESIRED_SPEED_MPS = 33.3
DEFAULT_EGO_LENGTH_M = 4.5
DEFAULT_EGO_WIDTH_M = 1.85

WARN_TTC_S = 3.0
CRITICAL_TTC_S = 1.5
NEAR_CLEARANCE_M = 0.5


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


@dataclass
class _CollisionRiskState:
    total: int = 0
    risky_frames: int = 0
    overlap_frames: int = 0
    imminent_frames: int = 0
    min_ttc_s: float = float("inf")
    min_clearance_m: float = float("inf")


@dataclass
class _PairRisk:
    overlap: bool = False
    clearance_m: float = float("inf")
    ttc_s: float = float("inf")
    closing_speed_mps: float = 0.0


def _evaluate_collision_risk(
    ego_x: float,
    ego_y: float,
    ego_heading: float,
    ego_speed: float,
    npcs: List[NPCSnapshot],
    ego_length: float = DEFAULT_EGO_LENGTH_M,
    ego_width: float = DEFAULT_EGO_WIDTH_M,
) -> _PairRisk:
    """Mirror hyw-grading collision_risk_checker.cc (AABB gap + TTC)."""
    worst = _PairRisk()
    if not npcs:
        return worst

    ego_hl = ego_length * 0.5
    ego_hw = ego_width * 0.5
    c = math.cos(-ego_heading)
    s = math.sin(-ego_heading)
    ego_vx = ego_speed * math.cos(ego_heading)
    ego_vy = ego_speed * math.sin(ego_heading)

    for n in npcs:
        npc_hl = max(0.5, n.length * 0.5)
        npc_hw = max(0.3, n.width * 0.5)

        dx_w = n.x - ego_x
        dy_w = n.y - ego_y
        dx = c * dx_w - s * dy_w
        dy = s * dx_w + c * dy_w

        x_sep = abs(dx) - (ego_hl + npc_hl)
        y_sep = abs(dy) - (ego_hw + npc_hw)
        x_gap = max(0.0, x_sep)
        y_gap = max(0.0, y_sep)
        clearance_m = math.hypot(x_gap, y_gap)
        overlap = x_sep <= 0.0 and y_sep <= 0.0

        if overlap:
            risk = _PairRisk(
                overlap=True,
                clearance_m=clearance_m,
                ttc_s=0.0,
            )
        else:
            rvx_w = n.vx - ego_vx
            rvy_w = n.vy - ego_vy
            rvx = c * rvx_w - s * rvy_w
            rvy = s * rvx_w + c * rvy_w
            center_dist = math.hypot(dx, dy)
            if center_dist < 1e-6:
                risk = _PairRisk(
                    clearance_m=clearance_m,
                    ttc_s=0.0,
                    closing_speed_mps=math.hypot(rvx, rvy),
                )
            else:
                closing = -((dx / center_dist) * rvx + (dy / center_dist) * rvy)
                closing_speed_mps = max(0.0, closing)
                ttc_s = float("inf")
                if closing_speed_mps > 1e-3:
                    ttc_s = clearance_m / closing_speed_mps
                risk = _PairRisk(
                    clearance_m=clearance_m,
                    ttc_s=ttc_s,
                    closing_speed_mps=closing_speed_mps,
                )

        if risk.ttc_s < worst.ttc_s or (
            abs(risk.ttc_s - worst.ttc_s) < 1e-6
            and risk.clearance_m < worst.clearance_m
        ):
            worst = risk
    return worst


def _is_dangerous_risk(risk: _PairRisk) -> bool:
    if risk.overlap:
        return True
    if risk.ttc_s < CRITICAL_TTC_S:
        return True
    if risk.ttc_s < WARN_TTC_S and risk.clearance_m < 1.0:
        return True
    if risk.clearance_m < NEAR_CLEARANCE_M and risk.closing_speed_mps > 0.5:
        return True
    return False


class OnlineGrader(FrameHook):
    """Per-frame Python grader; PASS/FAIL printed every `print_every` frames."""

    def __init__(
        self,
        max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
        max_desired_speed_mps: float = DEFAULT_MAX_DESIRED_SPEED_MPS,
        ego_length_m: float = DEFAULT_EGO_LENGTH_M,
        ego_width_m: float = DEFAULT_EGO_WIDTH_M,
        sink: Optional[TextIO] = None,
        print_every: int = 5,
    ):
        self.max_speed_mps = max_speed_mps
        self.max_desired_speed_mps = max_desired_speed_mps
        self.ego_length_m = ego_length_m
        self.ego_width_m = ego_width_m
        self.sink: TextIO = sink or sys.stdout
        self.print_every = max(1, int(print_every))

        self._speed = _SpeedState()
        self._limit = _LimitState()
        self._coll = _CollisionState()
        self._risk = _CollisionRiskState()
        self._first_collision_frame: Optional[int] = None

    # ---- FrameHook ----

    def on_frame(self, rec: FrameRecord) -> None:
        speed_ok = self._tick_speed(rec)
        limit_ok = self._tick_limit(rec)
        coll_ok = self._tick_collision(rec)
        risk_ok = self._tick_collision_risk(rec)
        all_ok = speed_ok and limit_ok and coll_ok and risk_ok
        if (
            rec.frame_id % self.print_every == 0
            or rec.collision.collided
            or not all_ok
        ):
            tag = "PASS" if all_ok else "FAIL"
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
        min_ttc = self._risk.min_ttc_s
        min_clearance = self._risk.min_clearance_m
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
            MetricSummary(
                name="collision_risk_checker",
                passed=self._risk.risky_frames == 0,
                detail=(
                    f"risky_frames={self._risk.risky_frames}/{self._risk.total} "
                    f"overlap_frames={self._risk.overlap_frames} "
                    f"imminent_frames={self._risk.imminent_frames} "
                    f"min_ttc_s={min_ttc if math.isfinite(min_ttc) else 'inf'} "
                    f"min_clearance_m={min_clearance if math.isfinite(min_clearance) else 'inf'}"
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

    def _tick_collision_risk(self, rec: FrameRecord) -> bool:
        self._risk.total += 1
        if not rec.npcs:
            return True

        worst = _evaluate_collision_risk(
            rec.ego.x,
            rec.ego.y,
            rec.ego.heading,
            rec.ego.speed,
            rec.npcs,
            self.ego_length_m,
            self.ego_width_m,
        )
        self._risk.min_ttc_s = min(self._risk.min_ttc_s, worst.ttc_s)
        self._risk.min_clearance_m = min(self._risk.min_clearance_m, worst.clearance_m)

        dangerous = _is_dangerous_risk(worst)
        if worst.overlap:
            self._risk.overlap_frames += 1
        if worst.ttc_s < WARN_TTC_S:
            self._risk.imminent_frames += 1
        if dangerous:
            self._risk.risky_frames += 1
        return not dangerous


class SimLogWriter(FrameHook):
    """Stream frames into a SimLog JSON file consumable by grading_main.

    Layout:
      { "source": "<src>", "frames": [ MetricFrameInput, ... ] }

    The file is rewritten after each frame so the user can `tail -f` mid-run
    or re-score with the C++ binary at any time.
    """

    def __init__(
        self,
        output_path: Path,
        source: str = "waymo_sim",
        flush_every: int = 1,
        ego_length_m: float = DEFAULT_EGO_LENGTH_M,
        ego_width_m: float = DEFAULT_EGO_WIDTH_M,
        ego_wheelbase_m: float = 2.7,
        ego_rear_overhang_m: float = 0.95,
    ):
        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.flush_every = max(1, int(flush_every))
        self.ego_length_m = ego_length_m
        self.ego_width_m = ego_width_m
        self.ego_wheelbase_m = ego_wheelbase_m
        self.ego_rear_overhang_m = ego_rear_overhang_m
        self._frames: List[Dict] = []
        self._dirty_since_flush = 0

    def on_frame(self, rec: FrameRecord) -> None:
        self._frames.append(
            _frame_to_dict(
                rec,
                ego_length_m=self.ego_length_m,
                ego_width_m=self.ego_width_m,
                ego_wheelbase_m=self.ego_wheelbase_m,
                ego_rear_overhang_m=self.ego_rear_overhang_m,
            )
        )
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

    def __init__(
        self,
        binary_path: Path,
        report_path: Path,
        metrics_config_path: str = "",
        sink: Optional[TextIO] = None,
        ego_length_m: float = DEFAULT_EGO_LENGTH_M,
        ego_width_m: float = DEFAULT_EGO_WIDTH_M,
        ego_wheelbase_m: float = 2.7,
        ego_rear_overhang_m: float = 0.95,
    ):
        self.binary_path = Path(binary_path).expanduser().resolve()
        self.report_path = Path(report_path).expanduser().resolve()
        self.metrics_config_path = metrics_config_path
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.sink = sink or sys.stdout
        self.ego_length_m = ego_length_m
        self.ego_width_m = ego_width_m
        self.ego_wheelbase_m = ego_wheelbase_m
        self.ego_rear_overhang_m = ego_rear_overhang_m
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._closed = False
        self._start()

    def _start(self) -> None:
        if not self.binary_path.is_file():
            raise FileNotFoundError(f"grading_main not found: {self.binary_path}")
        cmd = [str(self.binary_path), "--stream"]
        if self.metrics_config_path:
            cmd.extend(["--metrics-config", self.metrics_config_path])
        cmd.append(str(self.report_path))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=None,
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
                sys.stderr.write(line)
                sys.stderr.flush()
        except Exception:
            pass

    def on_frame(self, rec: FrameRecord) -> None:
        if self._proc is None or self._proc.stdin is None or self._closed:
            return
        line = json.dumps(
            _frame_to_dict(
                rec,
                ego_length_m=self.ego_length_m,
                ego_width_m=self.ego_width_m,
                ego_wheelbase_m=self.ego_wheelbase_m,
                ego_rear_overhang_m=self.ego_rear_overhang_m,
            ),
            separators=(",", ":"),
        )
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


def _frame_to_dict(
    rec: FrameRecord,
    *,
    ego_length_m: float = DEFAULT_EGO_LENGTH_M,
    ego_width_m: float = DEFAULT_EGO_WIDTH_M,
    ego_wheelbase_m: float = 2.7,
    ego_rear_overhang_m: float = 0.95,
) -> Dict:
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
        "ego_vehicle": {
            "length": float(ego_length_m),
            "width": float(ego_width_m),
            "wheelbase": float(ego_wheelbase_m),
            "rear_overhang": float(ego_rear_overhang_m),
        },
        "npcs": [
            {
                "id": int(n.id),
                "object_type": n.object_type,
                "x": float(n.x),
                "y": float(n.y),
                "z": float(n.z),
                "heading": float(n.heading),
                "vx": float(n.vx),
                "vy": float(n.vy),
                "length": float(n.length),
                "width": float(n.width),
                "height": float(n.height),
            }
            for n in rec.npcs
        ],
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
