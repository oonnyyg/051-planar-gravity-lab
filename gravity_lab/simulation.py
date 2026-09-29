"""Fixed-step velocity Verlet integration and per-step snapshots."""

import math

from .model import Experiment
from .physics import (
    accelerations,
    all_finite_diagnostics,
    all_finite_state,
    diagnostics,
)


class NonFiniteStateError(RuntimeError):
    """Raised when positions, velocities or diagnostics become non-finite."""

    def __init__(self, step: int, detail: str):
        super().__init__(f"non-finite value at step {step}: {detail}")
        self.step = step


def run_simulation(experiment: Experiment):
    """Run the experiment.

    Returns ``(ids, snapshots)`` where each snapshot is a dict with keys
    ``step``, ``time``, ``positions``, ``velocities``, ``diagnostics``.
    Snapshot 0 is the initial state; snapshot k follows k completed steps.
    """
    ids = experiment.ids
    masses = [p.mass for p in experiment.particles]
    positions = [[p.x, p.y] for p in experiment.particles]
    velocities = [[p.vx, p.vy] for p in experiment.particles]
    g = experiment.gravitational_constant
    dt = experiment.time_step
    softening = experiment.softening

    def snapshot(step: int):
        if not all_finite_state(positions, velocities):
            raise NonFiniteStateError(step, "position or velocity")
        diag = diagnostics(positions, velocities, masses, g, softening)
        if not all_finite_diagnostics(diag):
            raise NonFiniteStateError(step, "energy, momentum or angular momentum")
        return {
            "step": step,
            "time": step * dt,
            "positions": [(pos[0], pos[1]) for pos in positions],
            "velocities": [(vel[0], vel[1]) for vel in velocities],
            "diagnostics": diag,
        }

    snapshots = [snapshot(0)]
    acc = accelerations(positions, masses, g, softening)

    for step in range(1, experiment.steps + 1):
        half_dt_sq = 0.5 * dt * dt
        new_positions = [
            [
                positions[i][0] + velocities[i][0] * dt + acc[i][0] * half_dt_sq,
                positions[i][1] + velocities[i][1] * dt + acc[i][1] * half_dt_sq,
            ]
            for i in range(len(positions))
        ]
        new_acc = accelerations(new_positions, masses, g, softening)
        new_velocities = [
            [
                velocities[i][0] + 0.5 * (acc[i][0] + new_acc[i][0]) * dt,
                velocities[i][1] + 0.5 * (acc[i][1] + new_acc[i][1]) * dt,
            ]
            for i in range(len(positions))
        ]
        positions = new_positions
        velocities = new_velocities
        acc = new_acc
        snapshots.append(snapshot(step))

    return ids, snapshots


def summarize(snapshots) -> dict:
    """Maximum drift relative to the initial-state diagnostics."""
    initial = snapshots[0]["diagnostics"]
    max_abs_energy_change = 0.0
    max_momentum_change = 0.0
    max_abs_angular_change = 0.0
    for snap in snapshots[1:]:
        diag = snap["diagnostics"]
        energy_change = abs(diag.total_energy - initial.total_energy)
        dp_x = diag.momentum_x - initial.momentum_x
        dp_y = diag.momentum_y - initial.momentum_y
        momentum_change = math.hypot(dp_x, dp_y)
        angular_change = abs(diag.angular_momentum - initial.angular_momentum)
        max_abs_energy_change = max(max_abs_energy_change, energy_change)
        max_momentum_change = max(max_momentum_change, momentum_change)
        max_abs_angular_change = max(max_abs_angular_change, angular_change)
    return {
        "max_abs_energy_change": max_abs_energy_change,
        "max_momentum_change_magnitude": max_momentum_change,
        "max_abs_angular_momentum_change": max_abs_angular_change,
    }
