"""Fixed-step velocity Verlet integration and per-instant snapshots."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import RunError
from .experiment import Experiment
from .physics import (
    Vec2,
    angular_momentum,
    compute_accelerations,
    kinetic_energy,
    potential_energy,
    total_momentum,
)


@dataclass(frozen=True)
class ParticleState:
    id: str
    mass: float
    position: Vec2
    velocity: Vec2


@dataclass(frozen=True)
class Diagnostics:
    step: int
    time: float
    kinetic_energy: float
    potential_energy: float
    total_energy: float
    momentum_x: float
    momentum_y: float
    angular_momentum: float


@dataclass(frozen=True)
class Snapshot:
    diagnostics: Diagnostics
    particles: tuple[ParticleState, ...]


@dataclass(frozen=True)
class SimulationResult:
    gravity_constant: float
    time_step: float
    softening_length: float
    snapshots: tuple[Snapshot, ...]
    max_abs_energy_change: float
    max_momentum_drift: float
    max_abs_angular_momentum_change: float


def _diagnostics(
    step: int,
    time: float,
    masses: list[float],
    positions: list[Vec2],
    velocities: list[Vec2],
    gravity_constant: float,
    softening: float,
) -> Diagnostics:
    ke = kinetic_energy(masses, velocities)
    pe = potential_energy(masses, positions, gravity_constant, softening)
    px, py = total_momentum(masses, velocities)
    ang = angular_momentum(masses, positions, velocities)
    return Diagnostics(
        step=step,
        time=time,
        kinetic_energy=ke,
        potential_energy=pe,
        total_energy=ke + pe,
        momentum_x=px,
        momentum_y=py,
        angular_momentum=ang,
    )


def _assert_finite_snapshot(snapshot: Snapshot) -> None:
    d = snapshot.diagnostics
    scalar_values = (
        d.kinetic_energy,
        d.potential_energy,
        d.total_energy,
        d.momentum_x,
        d.momentum_y,
        d.angular_momentum,
    )
    for value in scalar_values:
        if not math.isfinite(value):
            raise RunError(f"non-finite diagnostic at step {d.step}")
    for particle in snapshot.particles:
        x, y = particle.position
        vx, vy = particle.velocity
        if not all(math.isfinite(v) for v in (x, y, vx, vy)):
            raise RunError(
                f"non-finite state for particle {particle.id!r} at step {d.step}"
            )


def run_experiment(experiment: Experiment) -> SimulationResult:
    """Integrate the experiment and return snapshots for steps 0..steps."""
    g = experiment.gravity_constant
    dt = experiment.time_step
    softening = experiment.softening_length

    # A canonical order by id guarantees that input order never affects the
    # numbers computed for a given particle.
    ordered = sorted(enumerate(experiment.particles), key=lambda item: item[1].id)
    ids = [p.id for _, p in ordered]
    masses = [p.mass for _, p in ordered]
    positions: list[Vec2] = [p.position for _, p in ordered]
    velocities: list[Vec2] = [p.velocity for _, p in ordered]

    def make_snapshot(step: int) -> Snapshot:
        diag = _diagnostics(
            step, step * dt, masses, positions, velocities, g, softening
        )
        particles = tuple(
            ParticleState(
                id=ids[i],
                mass=masses[i],
                position=positions[i],
                velocity=velocities[i],
            )
            for i in range(len(ids))
        )
        return Snapshot(diagnostics=diag, particles=particles)

    initial = make_snapshot(0)
    _assert_finite_snapshot(initial)
    snapshots = [initial]

    acc = compute_accelerations(positions, masses, g, softening)
    for step in range(1, experiment.steps + 1):
        # Stage 1: advance every position using old velocities/accelerations.
        new_positions: list[Vec2] = [
            (
                positions[i][0]
                + velocities[i][0] * dt
                + 0.5 * acc[i][0] * dt * dt,
                positions[i][1]
                + velocities[i][1] * dt
                + 0.5 * acc[i][1] * dt * dt,
            )
            for i in range(len(ids))
        ]
        # Stage 2: recompute accelerations from ALL new positions together.
        new_acc = compute_accelerations(new_positions, masses, g, softening)
        # Stage 3: average old/new accelerations to finish the velocity update.
        new_velocities: list[Vec2] = [
            (
                velocities[i][0]
                + 0.5 * (acc[i][0] + new_acc[i][0]) * dt,
                velocities[i][1]
                + 0.5 * (acc[i][1] + new_acc[i][1]) * dt,
            )
            for i in range(len(ids))
        ]
        positions = new_positions
        velocities = new_velocities
        acc = new_acc
        snapshot = make_snapshot(step)
        _assert_finite_snapshot(snapshot)
        snapshots.append(snapshot)

    d0 = snapshots[0].diagnostics
    max_energy = 0.0
    max_momentum = 0.0
    max_angular = 0.0
    e0 = d0.total_energy
    p0 = (d0.momentum_x, d0.momentum_y)
    l0 = d0.angular_momentum
    for snapshot in snapshots[1:]:
        d = snapshot.diagnostics
        max_energy = max(max_energy, abs(d.total_energy - e0))
        momentum_drift = math.hypot(d.momentum_x - p0[0], d.momentum_y - p0[1])
        max_momentum = max(max_momentum, momentum_drift)
        max_angular = max(max_angular, abs(d.angular_momentum - l0))

    return SimulationResult(
        gravity_constant=g,
        time_step=dt,
        softening_length=softening,
        snapshots=tuple(snapshots),
        max_abs_energy_change=max_energy,
        max_momentum_drift=max_momentum,
        max_abs_angular_momentum_change=max_angular,
    )
