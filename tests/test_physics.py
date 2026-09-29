import csv
import io
import json
import math
import unittest

from gravity_lab.experiment import parse_experiment
from gravity_lab.physics import (
    compute_accelerations,
    potential_energy,
    soft_distance,
)
from gravity_lab.report import diagnostics_csv, trajectories_csv
from gravity_lab.runner import run_experiment


def _single(steps=5, dt=0.1):
    return parse_experiment(
        {
            "gravity_constant": 1.0,
            "time_step": dt,
            "softening_length": 0.01,
            "steps": steps,
            "particles": [
                {
                    "id": "solo",
                    "mass": 2.0,
                    "position": [1.0, -2.0],
                    "velocity": [3.0, 4.0],
                }
            ],
        }
    )


class SingleParticleTests(unittest.TestCase):
    def test_uniform_motion_exact(self):
        exp = _single(steps=10, dt=0.1)
        result = run_experiment(exp)
        self.assertEqual(len(result.snapshots), 11)
        for snapshot in result.snapshots:
            t = snapshot.diagnostics.time
            p = snapshot.particles[0]
            self.assertAlmostEqual(p.position[0], 1.0 + 3.0 * t, places=12)
            self.assertAlmostEqual(p.position[1], -2.0 + 4.0 * t, places=12)
            self.assertEqual(p.velocity, (3.0, 4.0))

        # A lone particle has zero potential energy and constant KE.
        ke = 0.5 * 2.0 * (3.0**2 + 4.0**2)
        for snapshot in result.snapshots:
            d = snapshot.diagnostics
            self.assertEqual(d.potential_energy, 0.0)
            self.assertAlmostEqual(d.kinetic_energy, ke, places=12)
            self.assertAlmostEqual(d.total_energy, ke, places=12)
            self.assertAlmostEqual(d.momentum_x, 6.0, places=12)
            self.assertAlmostEqual(d.momentum_y, 8.0, places=12)

        self.assertEqual(result.max_abs_energy_change, 0.0)
        self.assertEqual(result.max_momentum_drift, 0.0)


class SymmetricBinaryTests(unittest.TestCase):
    def _experiment(self, dt, steps):
        return parse_experiment(
            {
                "gravity_constant": 1.0,
                "time_step": dt,
                "softening_length": 0.05,
                "steps": steps,
                "particles": [
                    {
                        "id": "alpha",
                        "mass": 1.0,
                        "position": [-1.0, 0.0],
                        "velocity": [0.0, -0.5],
                    },
                    {
                        "id": "beta",
                        "mass": 1.0,
                        "position": [1.0, 0.0],
                        "velocity": [0.0, 0.5],
                    },
                ],
            }
        )

    def test_symmetry_and_conservation(self):
        result = run_experiment(self._experiment(0.01, 100))
        initial_energy = result.snapshots[0].diagnostics.total_energy
        for snapshot in result.snapshots:
            d = snapshot.diagnostics
            alpha, beta = snapshot.particles
            # Mirror symmetry about the origin is preserved.
            self.assertAlmostEqual(alpha.position[0], -beta.position[0], places=12)
            self.assertAlmostEqual(alpha.position[1], -beta.position[1], places=12)
            self.assertAlmostEqual(d.momentum_x, 0.0, places=12)
            self.assertAlmostEqual(d.momentum_y, 0.0, places=12)
        self.assertLess(abs(result.snapshots[-1].diagnostics.total_energy -
                            initial_energy), 1e-6)

    def test_smaller_step_reduces_error(self):
        horizon = 1.0
        coarse = run_experiment(self._experiment(0.02, int(horizon / 0.02)))
        fine = run_experiment(self._experiment(0.005, int(horizon / 0.005)))
        # Velocity Verlet is second order: 4x finer step -> markedly smaller drift.
        self.assertLess(
            fine.max_abs_energy_change, coarse.max_abs_energy_change / 4.0
        )


class CoincidentPositionsTests(unittest.TestCase):
    def test_coincident_force_and_energy_are_finite(self):
        g, soft, ma, mb = 2.0, 0.25, 3.0, 1.0
        positions = [(0.7, -0.4), (0.7, -0.4)]
        masses = [ma, mb]
        acc = compute_accelerations(positions, masses, g, soft)
        # At zero separation the force points nowhere and must vanish, but the
        # softened potential must remain finite (-G m_a m_b / softening).
        self.assertEqual(acc[0], (0.0, 0.0))
        self.assertEqual(acc[1], (0.0, 0.0))
        self.assertTrue(math.isfinite(soft_distance(positions[0], positions[1], soft)))
        pe = potential_energy(masses, positions, g, soft)
        self.assertAlmostEqual(pe, -g * ma * mb / soft)

        exp = parse_experiment(
            {
                "gravity_constant": g,
                "time_step": 0.05,
                "softening_length": soft,
                "steps": 10,
                "particles": [
                    {
                        "id": "a",
                        "mass": ma,
                        "position": [0.0, 0.0],
                        "velocity": [0.0, 0.0],
                    },
                    {
                        "id": "b",
                        "mass": mb,
                        "position": [0.0, 0.0],
                        "velocity": [0.1, -0.2],
                    },
                ],
            }
        )
        result = run_experiment(exp)
        self.assertEqual(len(result.snapshots), 11)
        for snapshot in result.snapshots:
            self.assertTrue(
                math.isfinite(snapshot.diagnostics.total_energy)
            )


class ZeroStepsTests(unittest.TestCase):
    def test_only_initial_snapshot(self):
        exp = _single(steps=0, dt=0.1)
        result = run_experiment(exp)
        self.assertEqual(len(result.snapshots), 1)
        d = result.snapshots[0].diagnostics
        self.assertEqual(d.step, 0)
        self.assertEqual(d.time, 0.0)
        self.assertEqual(result.max_abs_energy_change, 0.0)
        self.assertEqual(result.max_momentum_drift, 0.0)
        self.assertEqual(result.max_abs_angular_momentum_change, 0.0)


class OrderIndependenceTests(unittest.TestCase):
    BASE = {
        "gravity_constant": 1.3,
        "time_step": 0.02,
        "softening_length": 0.1,
        "steps": 25,
        "particles": [
            {"id": "p1", "mass": 1.0, "position": [0.0, 0.0],
             "velocity": [0.1, 0.0]},
            {"id": "p2", "mass": 2.0, "position": [1.0, 1.0],
             "velocity": [-0.2, 0.3]},
            {"id": "p3", "mass": 0.5, "position": [-1.0, 0.5],
             "velocity": [0.0, -0.4]},
        ],
    }

    def test_shuffled_input_gives_identical_outputs(self):
        raw_a = json.loads(json.dumps(self.BASE))
        raw_b = json.loads(json.dumps(self.BASE))
        raw_b["particles"] = list(reversed(raw_b["particles"]))
        result_a = run_experiment(parse_experiment(raw_a))
        result_b = run_experiment(parse_experiment(raw_b))
        self.assertEqual(
            trajectories_csv(result_a), trajectories_csv(result_b)
        )
        self.assertEqual(
            diagnostics_csv(result_a), diagnostics_csv(result_b)
        )


class ReportShapeTests(unittest.TestCase):
    def test_trajectories_sorted_and_consistent(self):
        raw = json.loads(json.dumps(OrderIndependenceTests.BASE))
        raw["steps"] = 2
        result = run_experiment(parse_experiment(raw))
        rows = list(csv.DictReader(io.StringIO(trajectories_csv(result))))
        self.assertEqual(len(rows), 3 * 3)
        keys = [(int(r["step"]), r["particle_id"]) for r in rows]
        self.assertEqual(keys, sorted(keys))
        diag_rows = list(csv.DictReader(io.StringIO(diagnostics_csv(result))))
        self.assertEqual(len(diag_rows), 3)
        # Positions and velocities in a trajectory row match the same instant's
        # diagnostics indirectly via time = step * dt.
        for row, drow in zip(rows[::3], diag_rows):
            self.assertEqual(row["step"], drow["step"])
            self.assertEqual(row["time"], drow["time"])


if __name__ == "__main__":
    unittest.main()
