# Planar Gravity Lab

A reproducible two-dimensional N-body gravity experiment. It integrates the
motion of point particles with a **fixed-step velocity Verlet** scheme and
records trajectories together with energy, momentum and angular-momentum
diagnostics so that physical motion can be distinguished from numerical error.

Only the Python 3.12 standard library is used — no third-party packages and no
online services.

## Running

```sh
python -m gravity_lab examples/symmetric_binary.json --output-dir results/binary
```

The output directory is created by the program. If it already exists the run
fails with exit code 1 and nothing is written, so existing reports are never
overwritten. Invalid input or a non-finite computation likewise fails with a
clear message and exit code 1, and produces no report files.

## Experiment file

The experiment is a single JSON object:

```json
{
  "gravitational_constant": 1.0,
  "time_step": 0.005,
  "softening": 0.05,
  "steps": 1000,
  "particles": [
    {"id": "alpha", "mass": 1.0, "position": [-1.0, 0.0], "velocity": [0.0, 0.5]},
    {"id": "beta",  "mass": 1.0, "position": [ 1.0, 0.0], "velocity": [0.0, -0.5]}
  ]
}
```

Validation rules (any violation rejects the run):

- `gravitational_constant`, `time_step`, `softening` are positive finite reals.
- `steps` is a non-negative integer (`0` records only the initial state).
- `particles` is a non-empty list. Each `id` is a non-empty, unique string;
  `mass` is a positive finite real; `position` and `velocity` are lists of
  exactly two finite reals.
- Literal `NaN` / `Infinity` are rejected by the JSON parser.

The order of particles in the file never affects any particle's result.

## Physics

For an unordered pair of particles `i`, `j` with position difference
`d = r_j - r_i`, the softened distance is

```
s = sqrt(|d|^2 + softening^2)
```

and the force on `i` due to `j` is the vector

```
F_ij = G * m_i * m_j * d / s^3
```

with `F_ji = -F_ij`. A particle never acts on itself, and there are no walls,
collisions or mergers. Because the softening term is added inside the square
root, forces are well defined even when two particles occupy exactly the same
position (the force is then simply the zero vector).

### Velocity Verlet step

With `h = time_step`, positions of **all** particles are updated first from
the old state:

```
r_new = r_old + v_old * h + a_old * h^2 / 2
```

Accelerations are then recomputed from **all** new positions together, and
velocities are updated last:

```
v_new = v_old + (a_old + a_new) * h / 2
```

No partial old/new state mixing occurs, the step size is never adapted, and
no velocity correction is applied to force energy conservation.

## Outputs

### `trajectories.csv`

One row per particle per snapshot: `step,time,particle_id,x,y,vx,vy`.
`time = step * time_step`. Step `0` is the initial state. Rows are sorted by
step and then by particle id. Positions and velocities in a row belong to the
same instant.

### `diagnostics.csv`

One row per snapshot:

- `kinetic_energy` — sum of `1/2 m |v|^2`.
- `potential_energy` — each unordered pair counted once: `-G m_i m_j / s`.
- `total_energy` — kinetic plus potential.
- `momentum_x`, `momentum_y` — sums of `m v`.
- `angular_momentum` — sum about the origin of `m (x v_y - y v_x)`; positive
  values mean counterclockwise motion.

All quantities in a diagnostics row describe the same instant as the matching
trajectory snapshot.

### `summary.json`

Initial/final totals plus, relative to the initial state:

- `max_abs_energy_change` — maximum `|E(t) - E(0)|` over completed steps.
- `max_momentum_change_magnitude` — maximum length of the vector difference
  `|P(t) - P(0)|`.
- `max_abs_angular_momentum_change` — maximum `|L(t) - L(0)|`.

These are absolute drifts; nothing is divided by an initial quantity that
could be zero. If any position, velocity or diagnostic becomes non-finite, the
run names the offending step number and fails without writing fake results.

## Units

The program is unit-agnostic: all input is treated as plain numbers in one
consistent system. If positions are in metres and masses in kilograms, choose
`gravitational_constant` in m^3 kg^-1 s^-2 and give `time_step` in seconds;
then energies are in joules, momenta in kg m/s and angular momentum in
kg m^2/s. Dimensionless units (e.g. `G = 1`) are equally valid.

## Step size and error

Velocity Verlet is second-order accurate: per-step truncation error scales as
`O(h^3)` and accumulated trajectory error as `O(h^2)`. Halving the step size
(and doubling the step count to cover the same simulated time) typically
reduces energy drift by roughly a factor of four. The summary diagnostics let
you check this directly — re-run an experiment with a smaller `time_step` and
compare `max_abs_energy_change`. Exact conservation is neither expected nor
enforced; persistent, shrinking-with-`h` drift is discretization error, while
diverging energies indicate an unstable or under-resolved simulation.

## Examples

- `examples/single_particle.json` — one particle in uniform motion.
- `examples/symmetric_binary.json` — symmetric two-body orbit.
- `examples/coincident_particles.json` — two particles at the same point.
- `examples/zero_steps.json` — zero steps, initial state only.

## Tests

```sh
python -m unittest discover -s tests
```

The suite covers uniform single-particle motion, the symmetric binary,
coincident particles, zero steps, particle reordering invariance, step-size
effect on energy drift, non-finite failure with step reporting, and rejection
of illegal input without writing reports.
