"""Exception types for gravity_lab."""


class ExperimentError(ValueError):
    """The experiment file is missing fields or contains illegal values."""


class RunError(RuntimeError):
    """A simulation step produced a non-finite value."""
