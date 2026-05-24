"""Kinematic bicycle vehicle with bounded controls + bounding-box geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .geometry import OBB


@dataclass
class VehicleParams:
    length: float = 4.5            # bumper-to-bumper, meters
    width: float = 1.85
    height: float = 1.6
    wheelbase: float = 2.7
    rear_overhang: float = 0.95    # rear axle to rear bumper
    max_speed: float = 33.3        # m/s (~120 km/h)
    max_accel: float = 2.5         # m/s^2
    max_decel: float = 6.0
    max_steer: float = math.radians(35)
    max_steer_rate: float = math.radians(180)  # rad/s


@dataclass
class VehicleState:
    """Pose at the rear-axle plus longitudinal speed.

    `heading` is the body yaw in the world XY plane (rad).
    """

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    acceleration: float = 0.0
    steer: float = 0.0


@dataclass
class BicycleVehicle:
    """Discrete-time kinematic bicycle around the rear axle."""

    state: VehicleState
    params: VehicleParams = field(default_factory=VehicleParams)

    def step(self, accel_cmd: float, steer_cmd: float, dt: float) -> None:
        p = self.params
        accel = max(-p.max_decel, min(p.max_accel, accel_cmd))
        steer = max(-p.max_steer, min(p.max_steer, steer_cmd))
        ds_max = p.max_steer_rate * dt
        steer = max(self.state.steer - ds_max, min(self.state.steer + ds_max, steer))

        s = self.state
        new_speed = max(0.0, min(p.max_speed, s.speed + accel * dt))
        # mid-step heading propagation
        wb = max(0.5, p.wheelbase)
        new_heading = s.heading + (new_speed / wb) * math.tan(steer) * dt
        new_x = s.x + new_speed * math.cos(new_heading) * dt
        new_y = s.y + new_speed * math.sin(new_heading) * dt
        self.state = VehicleState(
            x=new_x,
            y=new_y,
            heading=new_heading,
            speed=new_speed,
            acceleration=accel,
            steer=steer,
        )

    def bbox(self) -> OBB:
        """OBB centered at the geometric center of the body (not the rear axle)."""
        p = self.params
        s = self.state
        # rear-axle to body-center offset (meters)
        d = p.length / 2.0 - p.rear_overhang
        cx = s.x + d * math.cos(s.heading)
        cy = s.y + d * math.sin(s.heading)
        return OBB(cx=cx, cy=cy, heading=s.heading, half_length=p.length / 2.0, half_width=p.width / 2.0)
