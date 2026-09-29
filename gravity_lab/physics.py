"""Core physics: softened pairwise gravity and conserved-quantity diagnostics.

Units are abstract and self-consistent: the numbers in the experiment file
fix the unit system (e.g. masses in kg, distances in m, times in s gives G in
m^3 kg^-1 s^-2). No unit conversion is performed.

The softened distance between particles i and j is

    d_ij = sqrt(|r_i - r_j|^2 + softening^2)

so coincident positions (|r_i - r_j| = 0) remain well defined.
"""

from __future__ import annotations

import math

Vec2 = tuple[float, float]


def soft_distance(ri: Vec2, rj: Vec2, softening: float) -> float:
    dx = rj[0] - ri[0]
    dy = rj[1] - ri[1]
    return math.sqrt(dx * dx + dy * dy + softening * softening)


def compute_accelerations(
    positions: list[Vec2],
    masses: list[float],
    gravity_constant: float,
    softening: float,
) -> list[Vec2]:
    """Acceleration of every particle from the superposition of all others.

    a_i = sum_{j != i} G * m_j * (r_j - r_i) / d_ij^3

    All positions must describe the same instant; callers must never mix
    positions of different particles from different integration stages.
    """
    n = len(positions)
    acc: list[Vec2] = [(0.0, 0.0)] * n
    soft2 = softening * softening
    for i in range(n):
        ax = 0.0
        ay = 0.0
        xi, yi = positions[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = positions[j]
            dx = xj - xi
            dy = yj - yi
            d3 = (dx * dx + dy * dy + soft2) ** 1.5
            factor = gravity_constant * masses[j] / d3
            ax += factor * dx
            ay += factor * dy
        acc[i] = (ax, ay)
    return acc


def kinetic_energy(masses: list[float], velocities: list[Vec2]) -> float:
    total = 0.0
    for mass, (vx, vy) in zip(masses, velocities):
        total += 0.5 * mass * (vx * vx + vy * vy)
    return total


def potential_energy(
    masses: list[float],
    positions: list[Vec2],
    gravity_constant: float,
    softening: float,
) -> float:
    """Sum over each unordered pair exactly once: -G m_i m_j / d_ij."""
    total = 0.0
    n = len(masses)
    for i in range(n):
        for j in range(i + 1, n):
            d = soft_distance(positions[i], positions[j], softening)
            total -= gravity_constant * masses[i] * masses[j] / d
    return total


def total_momentum(masses: list[float], velocities: list[Vec2]) -> Vec2:
    px = 0.0
    py = 0.0
    for mass, (vx, vy) in zip(masses, velocities):
        px += mass * vx
        py += mass * vy
    return px, py


def angular_momentum(
    masses: list[float], positions: list[Vec2], velocities: list[Vec2]
) -> float:
    """Total scalar angular momentum about the origin.

    L = sum_i m_i (x_i * v_iy - y_i * v_ix); positive is counterclockwise.
    """
    total = 0.0
    for mass, (x, y), (vx, vy) in zip(masses, positions, velocities):
        total += mass * (x * vy - y * vx)
    return total
