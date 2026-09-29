"""Command line entry point: ``python -m gravity_lab``."""

from __future__ import annotations

import argparse
import sys

from .errors import ExperimentError, RunError
from .experiment import load_experiment
from .report import write_reports
from .runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gravity_lab",
        description=(
            "Run a fixed-step velocity Verlet 2D N-body gravity experiment "
            "and write trajectory, diagnostic and summary reports."
        ),
    )
    parser.add_argument(
        "experiment",
        help="path to the experiment JSON file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="directory for trajectories.csv, diagnostics.csv and summary.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        experiment = load_experiment(args.experiment)
        result = run_experiment(experiment)
        paths = write_reports(experiment, result, args.output_dir)
    except (ExperimentError, RunError) as exc:
        print(f"gravity_lab: error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"gravity_lab: error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {paths['trajectories']}")
    print(f"wrote {paths['diagnostics']}")
    print(f"wrote {paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
