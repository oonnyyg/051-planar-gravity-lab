# planar-gravity-lab

A reproducible two-dimensional N-body gravity experiment. It reads a JSON
experiment file, integrates every particle with a **fixed-step velocity
Verlet** scheme, and writes trajectories, per-step diagnostics and a summary
report. Python 3.12 standard library only; no third-party packages and no
network access are required.

## Running

```sh
python -m gravity_lab examples/symmetric_binary.json -o out/binary
```

The output directory is created if it does not exist and receives:

- `trajectories.csv` — every particle's position and velocity at step 0 and
  after each completed step.
- `diagnostics.csv` — system-wide energy, momentum and angular momentum for
  the same instants.
- `summary.json` — configuration echo plus maximum drifts relative to the
  initial state.

On invalid input or a non-finite computed value, the program prints
`gravity_lab: error: ...` to stderr and exits with status 1 **before** any
report is written, so an existing report is never replaced by failed output.

## Input file

```json
{
  "gravity_constant": 1.0,
  "time_step": 0.01,
  "softening_length": 0.05,
  "steps": 200,
  "particles": [
    {"id": "alpha", "mass": 1.0, "position": [-1.0, 0.0], "velocity": [0.0, -0.5]},
    {"id": "beta",  "mass": 1.0, "position": [ 1.0, 0.0], "velocity": [0.0,  0.5]}
  ]
}
```

Rules:

- `gravity_constant`, `time_step`, `softening_length` are positive finite
  numbers.
- `steps` is a non-negative integer. `0` records only the initial state.
- At least one particle is required; `id` is a non-empty unique string and
  `mass` a positive finite number.
- `position` and `velocity` are lists of exactly two finite numbers.
- There are no boundaries, collisions or mergers. Particle order in the file
  does not affect results: states are canonicalized by `id` internally and
  reports are sorted by step then particle id.

## Physics

For particles `i` and `j`, with `r_ij = r_j - r_i`, the softened distance is

```
d_ij = sqrt(|r_ij|^2 + softening^2)
```

which stays well defined for exactly coincident positions. The pairwise
contributions are

```
F_ij = G m_i m_j r_ij / d_ij^3        (force on i toward j)
U_ij = -G m_i m_j / d_ij               (counted once per unordered pair)
```

Each particle feels the superposition of all other particles, never itself.

### Velocity Verlet

With fixed step `dt`, each step performs three stages using consistent
global state:

1. `r' = r + v dt + 1/2 a dt^2` for **all** particles.
2. Recompute accelerations `a'` from **all** new positions together.
3. `v' = v + 1/2 (a + a') dt`.

The step size is never adapted and velocities are never corrected to force
energy conservation.

### Units

Units are abstract and self-consistent: whatever units the inputs use, the
outputs use consistently. For example kilograms, metres and seconds imply
`gravity_constant` in m^3 kg^-1 s^-2 (use 6.6743e-11 for SI). The softening
length uses the same length unit as positions; time in the reports is
`step * time_step`.

## Diagnostics

Positions and velocities in every diagnostics row belong to the same instant
as the matching trajectory rows:

- Kinetic energy `K = sum 1/2 m_i |v_i|^2`.
- Potential energy `U = sum_{i<j} -G m_i m_j / d_ij` (each pair once).
- Total energy `E = K + U`.
- Total momentum vector `(sum m_i v_ix, sum m_i v_iy)`.
- Scalar angular momentum about the origin
  `L = sum m_i (x_i v_iy - y_i v_ix)`; positive is counterclockwise.

`summary.json` reports, relative to the initial state:

- `max_abs_energy_change`
- `max_momentum_drift` (maximum length of the momentum-vector difference)
- `max_abs_angular_momentum_change`

These are **absolute** differences (never divided by the initial quantity),
so experiments starting with zero energy/momentum are still meaningful.

## Step size and numerical error

Velocity Verlet is second order: when the trajectory is smooth, halving the
step reduces local truncation error by roughly a factor of four and the
accumulated drift over a fixed time by roughly a factor of four. The test
suite checks this on the symmetric binary example. Momentum stays conserved
to machine precision for pairwise forces regardless of step size, while
energy drift grows visibly when the step is too coarse for the closest
encounters. The softening length sets a smallest resolved scale: steps fine
enough to resolve motion over that scale give the cleanest diagnostics.
Exact conservation is not expected and no plotting is provided — load the
CSVs with any tool you like.

## Examples

- `examples/single_particle.json` — force-free uniform motion.
- `examples/symmetric_binary.json` — mirror-symmetric two-body orbit.
- `examples/coincident_particles.json` — two particles starting at the exact
  same position (exercises the softening).
- `examples/zero_steps.json` — initial-state-only report.

## Tests

```sh
python -m unittest discover -s tests
```

The suite covers uniform single-particle motion, the symmetric binary (
symmetry, momentum conservation and second-order step convergence),
coincident positions, zero-step runs, input-order independence, report
shape/sorting and illegal-input rejection with non-zero exit status.
