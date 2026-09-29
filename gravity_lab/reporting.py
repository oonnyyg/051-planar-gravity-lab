"""CSV / JSON report writers.

Reports are written into a freshly created output directory. If the
directory already exists no report is produced, so a failed or re-run
experiment can never silently overwrite previous results.
"""

import csv
import json
import os

TRAJECTORY_FILE = "trajectories.csv"
DIAGNOSTICS_FILE = "diagnostics.csv"
SUMMARY_FILE = "summary.json"


class OutputError(RuntimeError):
    pass


def _format(value: float) -> str:
    return repr(float(value))


def write_reports(output_dir, experiment, ids, snapshots, summary) -> None:
    try:
        os.makedirs(output_dir)
    except FileExistsError as exc:
        raise OutputError(
            f"output directory already exists, refusing to overwrite: {output_dir}"
        ) from exc
    except OSError as exc:
        raise OutputError(f"cannot create output directory {output_dir}: {exc}") from exc

    try:
        _write_trajectories(output_dir, ids, snapshots)
        _write_diagnostics(output_dir, snapshots)
        _write_summary(output_dir, experiment, ids, snapshots, summary)
    except OSError as exc:
        raise OutputError(f"failed while writing reports: {exc}") from exc


def _write_trajectories(output_dir, ids, snapshots) -> None:
    path = os.path.join(output_dir, TRAJECTORY_FILE)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "time", "particle_id", "x", "y", "vx", "vy"])
        ordered_ids = sorted(ids)
        id_index = {particle_id: i for i, particle_id in enumerate(ids)}
        for snap in snapshots:
            for particle_id in ordered_ids:
                i = id_index[particle_id]
                x, y = snap["positions"][i]
                vx, vy = snap["velocities"][i]
                writer.writerow(
                    [snap["step"], _format(snap["time"]), particle_id,
                     _format(x), _format(y), _format(vx), _format(vy)]
                )


def _write_diagnostics(output_dir, snapshots) -> None:
    path = os.path.join(output_dir, DIAGNOSTICS_FILE)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "step", "time", "kinetic_energy", "potential_energy",
            "total_energy", "momentum_x", "momentum_y", "angular_momentum",
        ])
        for snap in snapshots:
            diag = snap["diagnostics"]
            writer.writerow([
                snap["step"], _format(snap["time"]),
                _format(diag.kinetic_energy), _format(diag.potential_energy),
                _format(diag.total_energy), _format(diag.momentum_x),
                _format(diag.momentum_y), _format(diag.angular_momentum),
            ])


def _write_summary(output_dir, experiment, ids, snapshots, summary) -> None:
    initial = snapshots[0]["diagnostics"]
    final = snapshots[-1]["diagnostics"]
    payload = {
        "gravitational_constant": experiment.gravitational_constant,
        "time_step": experiment.time_step,
        "softening": experiment.softening,
        "steps": experiment.steps,
        "particle_ids": sorted(ids),
        "snapshots": len(snapshots),
        "initial_time": 0.0,
        "final_time": snapshots[-1]["time"],
        "initial_total_energy": initial.total_energy,
        "final_total_energy": final.total_energy,
        "initial_momentum": [initial.momentum_x, initial.momentum_y],
        "final_momentum": [final.momentum_x, final.momentum_y],
        "initial_angular_momentum": initial.angular_momentum,
        "final_angular_momentum": final.angular_momentum,
        **summary,
    }
    path = os.path.join(output_dir, SUMMARY_FILE)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False, sort_keys=True)
        handle.write("\n")
