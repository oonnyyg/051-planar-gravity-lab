"""Command-line entry point: ``python -m gravity_lab``."""

import argparse
import sys

from .reporting import OutputError, write_reports
from .simulation import NonFiniteStateError, run_simulation, summarize
from .validation import ValidationError, load_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gravity_lab",
        description="Run a fixed-step 2D N-body gravity experiment.",
    )
    parser.add_argument("experiment", help="path to the experiment JSON file")
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="directory for trajectories.csv, diagnostics.csv and summary.json",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        experiment = load_experiment(args.experiment)
        ids, snapshots = run_simulation(experiment)
        summary = summarize(snapshots)
        write_reports(args.output_dir, experiment, ids, snapshots, summary)
    except (ValidationError, NonFiniteStateError, OutputError) as exc:
        print(f"gravity_lab: error: {exc}", file=sys.stderr)
        return 1
    print(
        f"gravity_lab: wrote {len(snapshots)} snapshots "
        f"({len(ids)} particles) to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
