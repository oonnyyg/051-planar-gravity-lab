import json
import math
import tempfile
import unittest
from pathlib import Path

from gravity_lab.validation import ValidationError, load_experiment, parse_experiment


VALID = {
    "gravitational_constant": 1.0,
    "time_step": 0.01,
    "softening": 0.1,
    "steps": 10,
    "particles": [
        {"id": "a", "mass": 1.0, "position": [0.0, 0.0], "velocity": [0.0, 0.0]}
    ],
}


class ValidationTests(unittest.TestCase):
    def test_valid_experiment(self):
        experiment = parse_experiment(VALID)
        self.assertEqual(experiment.steps, 10)
        self.assertEqual(experiment.ids, ("a",))

    def test_missing_fields_rejected(self):
        for field in ("gravitational_constant", "time_step", "softening", "steps", "particles"):
            with self.subTest(field=field):
                data = dict(VALID)
                del data[field]
                with self.assertRaises(ValidationError):
                    parse_experiment(data)

    def test_non_positive_scalars_rejected(self):
        for field in ("gravitational_constant", "time_step", "softening"):
            for value in (0.0, -1.0):
                with self.subTest(field=field, value=value):
                    data = json.loads(json.dumps(VALID))
                    data[field] = value
                    with self.assertRaises(ValidationError):
                        parse_experiment(data)

    def test_negative_and_fractional_steps_rejected(self):
        for value in (-1, 1.5, True, "3"):
            with self.subTest(value=value):
                data = json.loads(json.dumps(VALID))
                data["steps"] = value
                with self.assertRaises(ValidationError):
                    parse_experiment(data)

    def test_zero_steps_allowed(self):
        data = json.loads(json.dumps(VALID))
        data["steps"] = 0
        self.assertEqual(parse_experiment(data).steps, 0)

    def test_particle_rules(self):
        bad_versions = [
            [],
            [{"id": "", "mass": 1.0, "position": [0, 0], "velocity": [0, 0]}],
            [{"id": "a", "mass": 0.0, "position": [0, 0], "velocity": [0, 0]}],
            [{"id": "a", "mass": 1.0, "position": [0], "velocity": [0, 0]}],
            [
                {"id": "a", "mass": 1.0, "position": [0, 0], "velocity": [0, 0]},
                {"id": "a", "mass": 1.0, "position": [1, 0], "velocity": [0, 0]},
            ],
        ]
        for particles in bad_versions:
            with self.subTest(particles=particles):
                data = json.loads(json.dumps(VALID))
                data["particles"] = particles
                with self.assertRaises(ValidationError):
                    parse_experiment(data)

    def test_non_finite_literals_rejected(self):
        for literal in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(literal=literal):
                text = json.dumps(VALID).replace("1.0", literal, 1)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "exp.json"
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValidationError):
                        load_experiment(path)

    def test_missing_file_rejected(self):
        with self.assertRaises(ValidationError):
            load_experiment(Path(tempfile.gettempdir()) / "no-such-file-xyz.json")

    def test_inf_float_rejected(self):
        data = json.loads(json.dumps(VALID))
        data["gravitational_constant"] = math.inf
        with self.assertRaises(ValidationError):
            parse_experiment(data)


if __name__ == "__main__":
    unittest.main()
