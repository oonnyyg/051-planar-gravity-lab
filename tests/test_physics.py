import math
import unittest

from gravity_lab.model import Experiment, Particle
from gravity_lab.physics import accelerations, diagnostics
from gravity_lab.simulation import run_simulation


def make_experiment(particles, steps=100, dt=0.01, g=1.0, softening=0.1):
    return Experiment(g, dt, softening, steps, tuple(particles))


class PhysicsTests(unittest.TestCase):
    def test_single_particle_has_zero_acceleration(self):
        exp = make_experiment([Particle("a", 1.0, 0.0, 0.0, 1.0, 0.5)])
        acc = accelerations([[0.0, 0.0]], [1.0], 1.0, 0.1)
        self.assertEqual(acc, [(0.0, 0.0)])
        ids, snapshots = run_simulation(exp)
        self.assertEqual(ids, ("a",))
        for snap in snapshots:
            t = snap["time"]
            x, y = snap["positions"][0]
            vx, vy = snap["velocities"][0]
            self.assertAlmostEqual(x, t, places=12)
            self.assertAlmostEqual(y, 0.5 * t, places=12)
            self.assertAlmostEqual(vx, 1.0, places=12)
            self.assertAlmostEqual(vy, 0.5, places=12)

    def test_single_particle_diagnostics_constant(self):
        exp = make_experiment([Particle("a", 2.0, 1.0, -1.0, 3.0, 4.0)])
        _, snapshots = run_simulation(exp)
        first = snapshots[0]["diagnostics"]
        self.assertAlmostEqual(first.kinetic_energy, 25.0, places=12)
        self.assertEqual(first.potential_energy, 0.0)
        self.assertAlmostEqual(first.momentum_x, 6.0, places=12)
        self.assertAlmostEqual(first.momentum_y, 8.0, places=12)
        self.assertAlmostEqual(first.angular_momentum, 2.0 * (1.0 * 4.0 - (-1.0) * 3.0), places=12)
        for snap in snapshots:
            diag = snap["diagnostics"]
            self.assertAlmostEqual(diag.total_energy, first.total_energy, places=10)
            self.assertAlmostEqual(diag.angular_momentum, first.angular_momentum, places=10)

    def test_coincident_particles_computable_and_symmetric(self):
        g, soft, m1, m2 = 2.0, 0.25, 1.0, 3.0
        positions = [[0.0, 0.0], [0.0, 0.0]]
        acc = accelerations(positions, [m1, m2], g, soft)
        self.assertEqual(acc[0], (0.0, 0.0))
        self.assertEqual(acc[1], (0.0, 0.0))
        particles = [
            Particle("p", m1, 0.0, 0.0, 0.0, 0.0),
            Particle("q", m2, 0.0, 0.0, 0.0, 0.0),
        ]
        exp = make_experiment(particles, steps=10, g=g, softening=soft)
        ids, snapshots = run_simulation(exp)
        for snap in snapshots:
            self.assertEqual(snap["positions"][0], (0.0, 0.0))
            self.assertEqual(snap["positions"][1], (0.0, 0.0))

    def test_force_direction_and_newton_third_law(self):
        g, soft = 1.0, 0.1
        masses = [2.0, 3.0]
        positions = [[0.0, 0.0], [3.0, 4.0]]
        acc = accelerations(positions, masses, g, soft)
        dx, dy = 3.0, 4.0
        distance = math.sqrt(dx * dx + dy * dy + soft * soft)
        a1 = g * masses[1] / distance ** 3
        self.assertAlmostEqual(acc[0][0], a1 * dx, places=12)
        self.assertAlmostEqual(acc[0][1], a1 * dy, places=12)
        a2 = g * masses[0] / distance ** 3
        self.assertAlmostEqual(acc[1][0], -a2 * dx, places=12)
        self.assertAlmostEqual(acc[1][1], -a2 * dy, places=12)
        force_ratio_x = acc[0][0] * masses[0]
        self.assertAlmostEqual(force_ratio_x, -acc[1][0] * masses[1], places=12)

    def test_potential_energy_pair_formula(self):
        g, soft = 1.0, 0.1
        masses = [2.0, 3.0]
        positions = [[0.0, 0.0], [3.0, 4.0]]
        diag = diagnostics(positions, [[0, 0], [0, 0]], masses, g, soft)
        distance = math.sqrt(25.0 + soft * soft)
        self.assertAlmostEqual(diag.potential_energy, -g * 6.0 / distance, places=12)

    def test_symmetric_binary_conservation_drift_is_small(self):
        particles = [
            Particle("alpha", 1.0, -1.0, 0.0, 0.0, 0.5),
            Particle("beta", 1.0, 1.0, 0.0, 0.0, -0.5),
        ]
        exp = make_experiment(particles, steps=1000, dt=0.005, softening=0.05)
        _, snapshots = run_simulation(exp)
        first = snapshots[0]["diagnostics"]
        last = snapshots[-1]["diagnostics"]
        self.assertAlmostEqual(last.total_energy - first.total_energy, 0.0, places=4)
        self.assertAlmostEqual(last.momentum_x, 0.0, places=10)
        self.assertAlmostEqual(last.momentum_y, 0.0, places=10)
        self.assertAlmostEqual(last.angular_momentum, first.angular_momentum, places=5)
        final_a = snapshots[-1]["positions"][0]
        final_b = snapshots[-1]["positions"][1]
        self.assertAlmostEqual(final_a[0], -final_b[0], places=10)
        self.assertAlmostEqual(final_a[1], -final_b[1], places=10)


if __name__ == "__main__":
    unittest.main()
