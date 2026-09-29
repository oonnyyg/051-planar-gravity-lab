import csv
import json
import tempfile
import unittest
from pathlib import Path

from gravity_lab.__main__ import main
from gravity_lab.model import Experiment, Particle
from gravity_lab.simulation import NonFiniteStateError, run_simulation, summarize


BINARY = {
    "gravitational_constant": 1.0,
    "time_step": 0.005,
    "softening": 0.05,
    "steps": 1000,
    "particles": [
        {"id": "alpha", "mass": 1.0, "position": [-1.0, 0.0], "velocity": [0.0, 0.5]},
        {"id": "beta", "mass": 1.0, "position": [1.0, 0.0], "velocity": [0.0, -0.5]},
    ],
}


def write_json(directory, payload, name="exp.json"):
    path = Path(directory) / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class CliTests(unittest.TestCase):
    def test_end_to_end_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_path = write_json(tmp, BINARY)
            out = Path(tmp) / "out"
            code = main([str(exp_path), "-o", str(out)])
            self.assertEqual(code, 0)
            rows = read_csv(out / "trajectories.csv")
            self.assertEqual(len(rows), (1000 + 1) * 2)
            self.assertEqual(rows[0]["step"], "0")
            self.assertEqual(rows[0]["particle_id"], "alpha")
            self.assertEqual(rows[1]["particle_id"], "beta")
            diag_rows = read_csv(out / "diagnostics.csv")
            self.assertEqual(len(diag_rows), 1001)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["steps"], 1000)
            self.assertEqual(summary["particle_ids"], ["alpha", "beta"])
            for key in (
                "max_abs_energy_change",
                "max_momentum_change_magnitude",
                "max_abs_angular_momentum_change",
            ):
                self.assertIn(key, summary)
                self.assertGreaterEqual(summary[key], 0.0)

    def test_zero_steps_emits_only_initial_state(self):
        payload = json.loads(json.dumps(BINARY))
        payload["steps"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            exp_path = write_json(tmp, payload)
            out = Path(tmp) / "out"
            self.assertEqual(main([str(exp_path), "-o", str(out)]), 0)
            rows = read_csv(out / "trajectories.csv")
            self.assertEqual(len(rows), 2)
            self.assertTrue({row["step"] for row in rows} == {"0"})
            diag_rows = read_csv(out / "diagnostics.csv")
            self.assertEqual(len(diag_rows), 1)
            self.assertEqual(float(diag_rows[0]["time"]), 0.0)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["max_abs_energy_change"], 0.0)
            self.assertEqual(summary["max_momentum_change_magnitude"], 0.0)
            self.assertEqual(summary["max_abs_angular_momentum_change"], 0.0)

    def test_particle_order_does_not_change_per_particle_result(self):
        payload = json.loads(json.dumps(BINARY))
        reversed_payload = json.loads(json.dumps(BINARY))
        reversed_payload["particles"] = list(reversed(payload["particles"]))

        def trajectories(data):
            particles = [
                Particle(p["id"], p["mass"], *p["position"], *p["velocity"])
                for p in data["particles"]
            ]
            exp = Experiment(
                data["gravitational_constant"], data["time_step"],
                data["softening"], data["steps"], tuple(particles),
            )
            ids, snapshots = run_simulation(exp)
            index = {pid: i for i, pid in enumerate(ids)}
            return {
                pid: [
                    (snap["positions"][index[pid]], snap["velocities"][index[pid]])
                    for snap in snapshots
                ]
                for pid in ids
            }

        self.assertEqual(trajectories(payload), trajectories(reversed_payload))

    def test_smaller_step_has_smaller_or_comparable_energy_drift(self):
        def drift(dt, steps):
            particles = [
                Particle("alpha", 1.0, -1.0, 0.0, 0.0, 0.5),
                Particle("beta", 1.0, 1.0, 0.0, 0.0, -0.5),
            ]
            exp = Experiment(1.0, dt, 0.05, steps, tuple(particles))
            _, snapshots = run_simulation(exp)
            return summarize(snapshots)["max_abs_energy_change"]

        coarse = drift(0.02, 250)
        fine = drift(0.005, 1000)
        self.assertLess(fine, coarse)

    def test_illegal_input_fails_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_path = Path(tmp) / "bad.json"
            exp_path.write_text('{"steps": -1}', encoding="utf-8")
            out = Path(tmp) / "out"
            self.assertEqual(main([str(exp_path), "-o", str(out)]), 1)
            self.assertFalse(out.exists())

    def test_existing_output_dir_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_path = write_json(tmp, BINARY)
            out = Path(tmp) / "out"
            out.mkdir()
            marker = out / "keep.txt"
            marker.write_text("original", encoding="utf-8")
            self.assertEqual(main([str(exp_path), "-o", str(out)]), 1)
            self.assertEqual(marker.read_text(encoding="utf-8"), "original")
            self.assertFalse((out / "trajectories.csv").exists())

    def test_non_finite_state_reports_step(self):
        particles = (
            Particle("a", 1.0, 0.0, 0.0, 1e150, 1e150),
            Particle("b", 1.0, 1.0, 0.0, 0.0, 0.0),
        )
        exp = Experiment(1.0, 1e110, 1.0, 2, particles)
        with self.assertRaises(NonFiniteStateError) as context:
            run_simulation(exp)
        self.assertEqual(context.exception.step, 1)


if __name__ == "__main__":
    unittest.main()
