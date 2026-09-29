"""Force evaluation and conserved-quantity diagnostics for the 2D N-body.

Softened distance for a pair is ``sqrt(dx^2 + dy^2 + softening^2)``.
The force on particle i due to particle j is
``G * m_i * m_j * (r_j - r_i) / soft_distance^3`` (a vector), so the
acceleration of i is ``G * m_j * (r_j - r_i) / soft_distance^3``.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Diagnostics:
    kinetic_energy: float
    potential_energy: float
    total_energy: float
    momentum_x: float
    momentum_y: float
    angular_momentum: float


def accelerations(positions, masses, g: float, softening: float):
    """Acceleration of every particle from all other particles.

    positions: list of (x, y) pairs. Returns list of (ax, ay) pairs.
    All positions are used consistently (no mixed old/new states).
    """
    n = len(positions)
    acc = [[0.0, 0.0] for _ in range(n)]
    soft_sq = softening * softening
    for i in range(n):
        xi, yi = positions[i]
        for j in range(i + 1, n):
            xj, yj = positions[j]
            dx = xj - xi
            dy = yj - yi
            distance = math.sqrt(dx * dx + dy * dy + soft_sq)
            inv_d3 = 1.0 / (distance * distance * distance)
            factor_i = g * masses[j] * inv_d3
            factor_j = g * masses[i] * inv_d3
            acc[i][0] += factor_i * dx
            acc[i][1] += factor_i * dy
            acc[j][0] -= factor_j * dx
            acc[j][1] -= factor_j * dy
    return [(ax, ay) for ax, ay in acc]


def diagnostics(positions, velocities, masses, g: float, softening: float) -> Diagnostics:
    """Kinetic/potential energy, momentum and angular momentum at one instant."""
    n = len(positions)
    kinetic = 0.0
    momentum_x = 0.0
    momentum_y = 0.0
    angular = 0.0
    for i in range(n):
        m = masses[i]
        vx, vy = velocities[i]
        kinetic += 0.5 * m * (vx * vx + vy * vy)
        momentum_x += m * vx
        momentum_y += m * vy
        x, y = positions[i]
        angular += m * (x * vy - y * vx)

    potential = 0.0
    soft_sq = softening * softening
    for i in range(n):
        xi, yi = positions[i]
        for j in range(i + 1, n):
            xj, yj = positions[j]
            dx = xj - xi
            dy = yj - yi
            distance = math.sqrt(dx * dx + dy * dy + soft_sq)
            potential -= g * masses[i] * masses[j] / distance

    return Diagnostics(
        kinetic_energy=kinetic,
        potential_energy=potential,
        total_energy=kinetic + potential,
        momentum_x=momentum_x,
        momentum_y=momentum_y,
        angular_momentum=angular,
    )


def all_finite_state(positions, velocities) -> bool:
    for pair in positions:
        if not math.isfinite(pair[0]) or not math.isfinite(pair[1]):
            return False
    for pair in velocities:
        if not math.isfinite(pair[0]) or not math.isfinite(pair[1]):
            return False
    return True


def all_finite_diagnostics(diag: Diagnostics) -> bool:
    return all(
        math.isfinite(value)
        for value in (
            diag.kinetic_energy,
            diag.potential_energy,
            diag.total_energy,
            diag.momentum_x,
            diag.momentum_y,
            diag.angular_momentum,
        )
    )
