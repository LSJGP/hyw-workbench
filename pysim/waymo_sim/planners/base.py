"""Planner abstract interface.

Every planner consumes the ego state + a snapshot of NPCs and emits a
PlanCommand: target longitudinal acceleration, steering angle, and the
desired-speed scalar that the grading limit-checker reads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..vehicle import VehicleState


@dataclass
class NPCSnapshot:
    """Read-only snapshot of an NPC at the current sim tick."""

    id: int
    object_type: str
    x: float
    y: float
    z: float
    heading: float
    vx: float
    vy: float
    length: float
    width: float
    height: float


@dataclass
class PlanCommand:
    target_acceleration: float = 0.0   # m/s^2
    steering_angle: float = 0.0        # rad
    desired_speed_mps: float = 0.0
    leader_id: Optional[int] = None
    leader_gap: float = float("inf")
    debug: Dict[str, Any] = field(default_factory=dict)


class Planner(ABC):
    """Subclasses must keep `name` set and implement `plan`."""

    name: str = "abstract"

    @abstractmethod
    def plan(
        self,
        ego_state: VehicleState,
        npcs: List[NPCSnapshot],
        t: float,
        dt: float,
    ) -> PlanCommand:
        ...
