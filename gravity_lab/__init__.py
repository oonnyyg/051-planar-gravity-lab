"""Planar gravity lab: reproducible 2D N-body experiments with velocity Verlet."""

from .experiment import Experiment, Particle, load_experiment
from .runner import RunError, run_experiment

__all__ = ["Experiment", "Particle", "RunError", "load_experiment", "run_experiment"]
