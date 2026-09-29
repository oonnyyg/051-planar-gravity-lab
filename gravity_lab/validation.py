"""Parsing and strict validation of experiment JSON files."""

import json
import math

from .model import Experiment, Particle


class ValidationError(ValueError):
    """Raised when an experiment file is missing required data or invalid."""


def _reject_non_finite(value):
    raise ValidationError(f"non-finite numeric literal is not allowed: {value}")


def load_experiment(path) -> Experiment:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=_reject_non_finite)
    except FileNotFoundError as exc:
        raise ValidationError(f"experiment file not found: {path}") from exc
    except OSError as exc:
        raise ValidationError(f"cannot read experiment file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON in {path}: {exc}") from exc
    return parse_experiment(data)


def _is_real_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value, field: str) -> float:
    if not _is_real_number(value):
        raise ValidationError(f"{field} must be a finite real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{field} must be finite, got {result}")
    return result


def _positive(value, field: str) -> float:
    result = _finite_number(value, field)
    if result <= 0.0:
        raise ValidationError(f"{field} must be positive, got {result}")
    return result


def _vector2(value, field: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValidationError(f"{field} must be a list of exactly two numbers")
    return (
        _finite_number(value[0], f"{field}[0]"),
        _finite_number(value[1], f"{field}[1]"),
    )


def parse_experiment(data) -> Experiment:
    if not isinstance(data, dict):
        raise ValidationError("top-level experiment must be a JSON object")

    g = _positive(data.get("gravitational_constant"), "gravitational_constant") \
        if "gravitational_constant" in data else _missing("gravitational_constant")
    dt = _positive(data.get("time_step"), "time_step") \
        if "time_step" in data else _missing("time_step")
    softening = _positive(data.get("softening"), "softening") \
        if "softening" in data else _missing("softening")

    if "steps" not in data:
        _missing("steps")
    steps_raw = data["steps"]
    if isinstance(steps_raw, bool) or not isinstance(steps_raw, int) or steps_raw < 0:
        raise ValidationError(f"steps must be a non-negative integer, got {steps_raw!r}")

    if "particles" not in data:
        _missing("particles")
    raw_particles = data["particles"]
    if not isinstance(raw_particles, list) or not raw_particles:
        raise ValidationError("particles must be a non-empty list")

    particles = []
    seen_ids = set()
    for index, item in enumerate(raw_particles):
        where = f"particles[{index}]"
        if not isinstance(item, dict):
            raise ValidationError(f"{where} must be an object")
        particle_id = item.get("id")
        if not isinstance(particle_id, str) or not particle_id:
            raise ValidationError(f"{where}.id must be a non-empty string")
        if particle_id in seen_ids:
            raise ValidationError(f"duplicate particle id: {particle_id!r}")
        seen_ids.add(particle_id)
        mass = _positive(item.get("mass"), f"{where}.mass")
        pos = _vector2(item.get("position"), f"{where}.position")
        vel = _vector2(item.get("velocity"), f"{where}.velocity")
        particles.append(
            Particle(particle_id, mass, pos[0], pos[1], vel[0], vel[1])
        )

    return Experiment(g, dt, softening, steps_raw, tuple(particles))


def _missing(field: str):
    raise ValidationError(f"missing required field: {field}")
