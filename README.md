# Shocks-and-WENO

Structured implementation of WENO and WENO-NN shock-capturing methods, based on the code accompanying Ben Stevens and Tim Colonius, [“Enhancement of shock-capturing methods via machine learning”](https://doi.org/10.1007/s00162-020-00531-1).

The repository currently ships the original five-point training data and focuses on the WENO5-NN advection experiment. Classical ENO3, WENO3, WENO5, and WENO7 reconstruction modules are also present, but the included dataset is specifically a five-point dataset.

## Environment

The original model and training behavior rely on the legacy standalone Keras stack:

- Python 3.7
- TensorFlow 1.15.0
- Keras 2.2.4

The supported interpreter version is declared in `.python-version` as Python `3.7.12`. Other Python versions are not supported because TensorFlow 1.15 and Keras 2.2.4 are legacy dependencies.

Using modern `tensorflow.keras` changes training behavior and can produce substantially different numerical results.

Create and activate a Python 3.7 environment, then install the dependencies:

```bash
conda create -n weno-nn python=3.7
conda activate weno-nn
pip install -r requirements.txt
```

If the TensorFlow 1.15 wheel is unavailable for your platform, install the pinned packages through Conda instead. CUDA is optional; TensorFlow falls back to the CPU.

## Shared configuration

Training and inference use [src/config.py](src/config.py):

```python
WENO_ORDER = 5
USE_SCALING = True
MODEL_PATH = 'trained_weno5.h5'
INPUT_DATA_PATH = 'data/2ndNewAvgs.csv'
OUTPUT_DATA_PATH = 'data/2ndNewFlux.csv'
RANDOM_SEED = 42
BOUNDARY_CONDITION = 'periodic'
MAX_CFL = 1.0
```

Keep `USE_SCALING` unchanged between training and inference. When enabled, each stencil and its target are normalized using the stencil’s local minimum and range. The prediction is converted back to physical units during simulation.

## Training

Run:

```bash
conda activate weno-nn
python Script_TrainNetworksFixLeak.py --seed 42
```

The dataset is divided into contiguous training, validation, and untouched test sets. Each attempt creates and trains a freshly initialized model. Early stopping monitors validation loss.

After every attempt, the model is tested on periodic step advection using:

- `max_tv`: maximum total variation; required to be below `2.016` by default.
- `shock_width`: maximum number of cells whose error exceeds the tolerance; required to be at most `20` by default.

Training makes at most 10 attempts by default (`--max-attempts`). It ranks candidates lexicographically by total-variation-limit violation, shock-width-limit violation, and validation loss. Every improvement is checkpointed to `<model-path>.best.h5`; if no candidate passes, the script prints the best candidate and exits unsuccessfully. A successful model is saved to `MODEL_PATH`, normally:

```text
trained_weno5.h5
```

This is the one canonical trained-model filename used by training, evaluation, and the basic comparison. The included `SeemsGood3.h5` is retained only as an upstream reference artifact; new training does not overwrite or select it automatically.

Training options can be inspected with:

```bash
python Script_TrainNetworksFixLeak.py --help
```

## Running the comparison

After training:

```bash
python Script_Basic.py
```

This compares the scaled or unscaled WENO-NN model—according to `USE_SCALING`—against classical WENO5.

To run the original supplied model instead:

```bash
python Script_Basic.py --model-path SeemsGood3.h5
```

The scalar solver currently supports periodic boundary conditions. CFL validation rejects configurations exceeding the configured limit.

The Euler solver uses transmissive boundaries by default. It constructs the full WENO3, WENO5, or WENO7 stencil at every interface by constant-extrapolating the first and last conservative states into ghost cells. This replaces the former WENO5-specific one-sided closure. Periodic Euler boundaries are also available by passing `boundary='periodic'` to `FiniteVolumeMethodEuler`.

Characteristic Euler reconstruction uses one Roe-averaged eigenbasis at each interface. The same left eigenvector matrix projects both positive and negative Lax–Friedrichs flux stencils, and the matching right matrix transforms their combined reconstruction back to conservative variables.

Euler benchmarks use adaptive time steps computed from the current maximum `|velocity| + sound_speed`, never just the initial state. Every accepted step therefore satisfies the requested CFL, with the final step shortened to land exactly on the requested final time. Density and pressure are validated after every SSPRK stage. Flux splitting uses a local interface spectral radius, the maximum of the adjacent cell wave speeds.

## Numerical verification

Run the smooth periodic-advection spatial convergence study with:

```bash
python Script_Convergence.py
```

The study initializes true cell averages using Gauss–Legendre quadrature and compares the flux divergence with the exact evolution of those averages. It evaluates the semi-discrete derivative, isolating spatial accuracy from the third-order SSPRK time integrator. The current final-grid rates are approximately `3.97`, `5.05`, and `7.12` for nominally third-, fifth-, and seventh-order WENO. The WENO3 number is an asymptotic measurement for this profile, fixed epsilon, and grid sequence; it is not a claim that WENO3 is generally fourth order.

Run the physically timed Euler regressions and print all measured quantities as JSON with:

```bash
python Script_EulerRegression.py
```

Sod is run to `t=0.2` and compared with the exact Riemann solution. Shu–Osher is run on the standard translated `[-5,5]` problem to `t=1.8`. By default it uses a 320-cell WENO7 result only as an internal regression baseline, not as independent validation. Supply independently validated or published data with `--shu-reference reference.csv`, using columns `x,density,momentum,energy`, for scientific validation. Each scheme reports L1 density, momentum, and energy errors; minimum density and pressure; density and pressure overshoots; shock and contact locations and location errors; and component-wise conservation-balance errors.

The WENO-NN correction is passed through a bounded hyperbolic tangent before the consistency projection. This limits learned departures while retaining the exact coefficient-sum condition. Final pointwise polynomial coefficients are intentionally not forced nonnegative: classical high-order reconstruction itself requires some negative coefficients. A future TENO-NN classifier should instead put nonnegative normalized probabilities on stencil-admissibility decisions.

## Canonical evaluation

Evaluate the canonical trained model without plots:

```bash
conda activate weno-nn
python Script_Evaluate.py --model-path trained_weno5.h5
```

The command prints a JSON result containing the numerical parameters, total variation, shock width, solution range, and acceptance status.

The canonical benchmark uses 100 periodic cells, 451 time levels, final time `6`, SSPRK3, Lax–Friedrichs splitting with unit wave speed, and the step initial condition. A successfully trained model must produce:

```text
max_tv < 2.016
shock_width <= 20
accepted = true
```

Because optimization is stochastic, the exact accepted values vary. As a verified reference, the included upstream `SeemsGood3.h5` evaluated on July 19, 2026 with this environment produced:

```text
max_tv       = 2.0049578981686045
shock_width  = 22
solution_min = -0.001185931837874278
solution_max = 1.0011859063902506
accepted     = false
```

That reference model is not expected to satisfy the stricter current `shock_width <= 20` training condition.

## Training-data format

Both files are headerless, comma-separated, decimal floating-point matrices:

- `data/2ndNewAvgs.csv`: shape `(5, 75241)`. Rows are the five ordered stencil positions `[u_(i-2), u_(i-1), u_i, u_(i+1), u_(i+2)]`; each column is one training sample.
- `data/2ndNewFlux.csv`: shape `(75241, 1)` when loaded by the project CSV reader. Row `j` is the scalar target interface value for column `j` of the input file.

The input values currently range from `0` to `1`; target values range approximately from `-0.5` to `1.5`. Input and target files must contain the same number of samples. With `USE_SCALING=True`, each input column and its target are transformed using that stencil’s local minimum and range before splitting.

## TENO and NN-TENO status

The repository includes deterministic TENO3, TENO5, and TENO7 scalar and
Euler reconstructions.  It also includes NENO-style classifier architectures
and scalar/Euler NN-TENO wrappers for all three orders.  If every candidate is
rejected, the wrappers use the center/upwind value as a first-order fallback.

NN-TENO input stencils are normalized independently as

```text
X / max(max(abs(X)), 1)
```

Each sigmoid output classifies one candidate stencil: `0` means smooth/keep
and `1` means troubled/discard.  The network predicts candidate selection,
not the final flux, candidate polynomial, or reconstruction weights.

NN-TENO training datasets, trained classifier models, classifier training
scripts, and extensive trained-model PDE benchmarks are not implemented yet.

## Tests

Run the regression suite with:

```bash
python -m unittest discover -s tests -v
```

The tests cover constant-state reconstruction, finite periodic advection, CFL rejection, boundary validation, scaled-data handling, and dataset separation. They verify the Roe matrices are mutual inverses and establish smooth periodic-advection convergence for WENO3, WENO5, and WENO7. The Euler end-to-end matrix runs all three schemes on constant stationary and uniformly moving states as well as the physically timed Sod and Shu–Osher regressions described above.

Numerical tests depend only on `requirements-numerical.txt`; they do not import TensorFlow. GitHub Actions runs them under modern Python and runs the neural smoke test separately inside a TensorFlow 1.15 container.

## Repository layout

```text
data/                   Five-point training stencils and target fluxes
src/core/               Simulation, time stepping, flux splitting, equations
src/initial_conditions/ Initial conditions for scalar and Euler problems
src/networks/           WENO-NN model definitions and data loading
src/schemes/            ENO and WENO reconstruction schemes
src/viz/                Plotting, metrics, and analysis helpers
tests/                  Numerical and data-pipeline regression tests
```

## Attribution

The numerical method and original WENO-NN implementation are derived from the work of Ben Stevens and Tim Colonius. See the linked paper and retain the relevant attribution when redistributing derived code.
