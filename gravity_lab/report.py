"""CSV/JSON report writers.

Reports are written only after the whole simulation has finished without
non-finite values, so a failed run never leaves a forged "normal" report
behind.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .experiment import Experiment
from .runner import SimulationResult

TRAJECTORY_FIELDS = (
    "step",
    "time",
    "particle_id",
    "mass",
    "x",
    "y",
    "vx",
    "vy",
)

DIAGNOSTICS_FIELDS = (
    "step",
    "time",
    "kinetic_energy",
    "potential_energy",
    "total_energy",
    "momentum_x",
    "momentum_y",
    "angular_momentum",
)


def _fmt(value: float) -> str:
    # repr round-trips IEEE-754 doubles and stays human readable.
    return repr(float(value))


def trajectories_csv(result: SimulationResult) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(TRAJECTORY_FIELDS)
    for snapshot in result.snapshots:
        d = snapshot.diagnostics
        for particle in snapshot.particles:
            writer.writerow(
                (
                    d.step,
                    _fmt(d.time),
                    particle.id,
                    _fmt(particle.mass),
                    _fmt(particle.position[0]),
                    _fmt(particle.position[1]),
                    _fmt(particle.velocity[0]),
                    _fmt(particle.velocity[1]),
                )
            )
    return buf.getvalue()


def diagnostics_csv(result: SimulationResult) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(DIAGNOSTICS_FIELDS)
    for snapshot in result.snapshots:
        d = snapshot.diagnostics
        writer.writerow(
            (
                d.step,
                _fmt(d.time),
                _fmt(d.kinetic_energy),
                _fmt(d.potential_energy),
                _fmt(d.total_energy),
                _fmt(d.momentum_x),
                _fmt(d.momentum_y),
                _fmt(d.angular_momentum),
            )
        )
    return buf.getvalue()


def summary_json(experiment: Experiment, result: SimulationResult) -> str:
    initial = result.snapshots[0].diagnostics
    final = result.snapshots[-1].diagnostics
    summary = {
        "steps": experiment.steps,
        "particle_count": len(experiment.particles),
        "gravity_constant": experiment.gravity_constant,
        "time_step": experiment.time_step,
        "softening_length": experiment.softening_length,
        "initial_time": initial.time,
        "final_time": final.time,
        "initial_total_energy": initial.total_energy,
        "final_total_energy": final.total_energy,
        "initial_momentum": [initial.momentum_x, initial.momentum_y],
        "final_momentum": [final.momentum_x, final.momentum_y],
        "initial_angular_momentum": initial.angular_momentum,
        "final_angular_momentum": final.angular_momentum,
        "max_abs_energy_change": result.max_abs_energy_change,
        "max_momentum_drift": result.max_momentum_drift,
        "max_abs_angular_momentum_change": result.max_abs_angular_momentum_change,
    }
    return json.dumps(summary, indent=2, allow_nan=False, sort_keys=True) + "\n"


def write_reports(
    experiment: Experiment,
    result: SimulationResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Create output_dir (if needed) and write the three report files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "trajectories": out / "trajectories.csv",
        "diagnostics": out / "diagnostics.csv",
        "summary": out / "summary.json",
    }
    contents = {
        "trajectories": trajectories_csv(result),
        "diagnostics": diagnostics_csv(result),
        "summary": summary_json(experiment, result),
    }
    for name, path in paths.items():
        # newline="" lets csv manage line endings; utf-8 keeps ids portable.
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write(contents[name])
    return paths
