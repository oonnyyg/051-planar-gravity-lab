"""Data types for gravity experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Particle:
    particle_id: str
    mass: float
    x: float
    y: float
    vx: float
    vy: float


@dataclass(frozen=True)
class Experiment:
    gravitational_constant: float
    time_step: float
    softening: float
    steps: int
    particles: tuple[Particle, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(p.particle_id for p in self.particles)
