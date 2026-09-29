"""Experiment model and JSON input validation.

Input schema (JSON object):

{
  "gravity_constant": <positive finite number>,
  "time_step": <positive finite number>,
  "softening_length": <positive finite number>,
  "steps": <non-negative integer>,
  "particles": [
    {
      "id": <non-empty string>,
      "mass": <positive finite number>,
      "position": [<finite number>, <finite number>],
      "velocity": [<finite number>, <finite number>]
    },
    ...
  ]
}
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .errors import ExperimentError


@dataclass(frozen=True)
class Particle:
    id: str
    mass: float
    position: tuple[float, float]
    velocity: tuple[float, float]


@dataclass(frozen=True)
class Experiment:
    gravity_constant: float
    time_step: float
    softening_length: float
    steps: int
    particles: tuple[Particle, ...]


def _is_real_number(value: object) -> bool:
    # bool is a subclass of int; reject it explicitly.
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _require_object(value: object, what: str) -> dict:
    if not isinstance(value, dict):
        raise ExperimentError(f"{what} must be a JSON object")
    return value


def _require_positive(obj: dict, key: str) -> float:
    if key not in obj:
        raise ExperimentError(f"missing required field: {key!r}")
    value = obj[key]
    if not _is_real_number(value):
        raise ExperimentError(f"{key!r} must be a finite real number")
    result = float(value)
    if result <= 0.0:
        raise ExperimentError(f"{key!r} must be positive, got {result}")
    return result


def _require_vector(obj: dict, key: str, label: str) -> tuple[float, float]:
    if key not in obj:
        raise ExperimentError(f"missing required field: {label!r}")
    value = obj[key]
    if not isinstance(value, list) or len(value) != 2:
        raise ExperimentError(f"{label!r} must be a list of exactly two numbers")
    x, y = value
    if not _is_real_number(x) or not _is_real_number(y):
        raise ExperimentError(f"{label!r} components must be finite real numbers")
    return float(x), float(y)


def _require_steps(obj: dict) -> int:
    if "steps" not in obj:
        raise ExperimentError("missing required field: 'steps'")
    value = obj["steps"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExperimentError("'steps' must be a non-negative integer")
    if value < 0:
        raise ExperimentError(f"'steps' must be a non-negative integer, got {value}")
    return value


def _parse_particle(raw: object, index: int) -> Particle:
    what = f"particles[{index}]"
    obj = _require_object(raw, what)
    if "id" not in obj:
        raise ExperimentError(f"{what} missing required field: 'id'")
    pid = obj["id"]
    if not isinstance(pid, str) or pid == "":
        raise ExperimentError(f"{what}.id must be a non-empty string")
    if "mass" not in obj:
        raise ExperimentError(f"{what} missing required field: 'mass'")
    mass_value = obj["mass"]
    if not _is_real_number(mass_value) or float(mass_value) <= 0.0:
        raise ExperimentError(f"{what}.mass must be a positive finite number")
    position = _require_vector(obj, "position", f"{what}.position")
    velocity = _require_vector(obj, "velocity", f"{what}.velocity")
    return Particle(
        id=pid,
        mass=float(mass_value),
        position=(float(position[0]), float(position[1])),
        velocity=(float(velocity[0]), float(velocity[1])),
    )


def parse_experiment(raw: object) -> Experiment:
    """Validate a decoded JSON value and build an :class:`Experiment`."""
    root = _require_object(raw, "experiment")
    g = _require_positive(root, "gravity_constant")
    dt = _require_positive(root, "time_step")
    softening = _require_positive(root, "softening_length")
    steps = _require_steps(root)
    if "particles" not in root:
        raise ExperimentError("missing required field: 'particles'")
    raw_particles = root["particles"]
    if not isinstance(raw_particles, list):
        raise ExperimentError("'particles' must be a list")
    if len(raw_particles) == 0:
        raise ExperimentError("'particles' must contain at least one particle")
    particles = [_parse_particle(item, i) for i, item in enumerate(raw_particles)]
    seen: set[str] = set()
    for particle in particles:
        if particle.id in seen:
            raise ExperimentError(f"duplicate particle id: {particle.id!r}")
        seen.add(particle.id)
    return Experiment(
        gravity_constant=g,
        time_step=dt,
        softening_length=softening,
        steps=steps,
        particles=tuple(particles),
    )


def load_experiment(path: str | Path) -> Experiment:
    """Read and validate an experiment JSON file."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExperimentError(f"cannot read experiment file {p}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ExperimentError(f"experiment file {p} is not valid UTF-8 text") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExperimentError(f"invalid JSON in {p}: {exc}") from exc
    return parse_experiment(raw)
