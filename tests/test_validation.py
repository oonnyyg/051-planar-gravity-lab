import json
import subprocess
import sys
import unittest
from pathlib import Path

from gravity_lab.__main__ import main
from gravity_lab.errors import ExperimentError
from gravity_lab.experiment import parse_experiment


class ValidationTests(unittest.TestCase):
    def _valid(self):
        return {
            "gravity_constant": 1.0,
            "time_step": 0.1,
            "softening_length": 0.01,
            "steps": 3,
            "particles": [
                {
                    "id": "a",
                    "mass": 1.0,
                    "position": [0.0, 0.0],
                    "velocity": [0.0, 0.0],
                }
            ],
        }

    def test_valid_input_parses(self):
        experiment = parse_experiment(self._valid())
        self.assertEqual(experiment.steps, 3)
        self.assertEqual(experiment.particles[0].id, "a")

    def test_missing_field_rejected(self):
        raw = self._valid()
        del raw["time_step"]
        with self.assertRaises(ExperimentError):
            parse_experiment(raw)

    def test_non_positive_scalars_rejected(self):
        for key in ("gravity_constant", "time_step", "softening_length"):
            raw = self._valid()
            raw[key] = 0.0
            with self.assertRaises(ExperimentError):
                parse_experiment(raw)
            raw[key] = -1.0
            with self.assertRaises(ExperimentError):
                parse_experiment(raw)

    def test_negative_or_non_integer_steps_rejected(self):
        raw = self._valid()
        raw["steps"] = -1
        with self.assertRaises(ExperimentError):
            parse_experiment(raw)
        raw["steps"] = 1.5
        with self.assertRaises(ExperimentError):
            parse_experiment(raw)
        raw["steps"] = True
        with self.assertRaises(ExperimentError):
            parse_experiment(raw)

    def test_empty_or_duplicate_ids_rejected(self):
        raw = self._valid()
        raw["particles"][0]["id"] = ""
        with self.assertRaises(ExperimentError):
            parse_experiment(raw)
        raw = self._valid()
        raw["particles"].append(dict(raw["particles"][0]))
        with self.assertRaises(ExperimentError):
            parse_experiment(raw)

    def test_no_particles_rejected(self):
        raw = self._valid()
        raw["particles"] = []
        with self.assertRaises(ExperimentError):
            parse_experiment(raw)

    def test_non_finite_and_wrong_shape_rejected(self):
        raw = self._valid()
        raw["particles"][0]["position"] = [1.0, float("nan")]
        with self.assertRaises(ExperimentError):
            parse_experiment(raw)
        raw = self._valid()
        raw["particles"][0]["velocity"] = [1.0]
        with self.assertRaises(ExperimentError):
            parse_experiment(raw)
        raw = self._valid()
        raw["particles"][0]["mass"] = float("inf")
        with self.assertRaises(ExperimentError):
            parse_experiment(raw)


class CliFailureTests(unittest.TestCase):
    def test_invalid_file_exits_nonzero_without_outputs(self, tmp=None):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "bad.json"
            bad.write_text(json.dumps({"steps": 0}), encoding="utf-8")
            out_dir = Path(tmpdir) / "out"
            code = main([str(bad), "-o", str(out_dir)])
            self.assertEqual(code, 1)
            self.assertFalse(out_dir.exists())

    def test_module_entrypoint_nonzero_on_bad_json(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "gravity_lab",
                    str(bad),
                    "-o",
                    str(Path(tmpdir) / "out"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("error", proc.stderr.lower())


if __name__ == "__main__":
    unittest.main()
