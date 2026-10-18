# Apsis Vision

**Relative spacecraft navigation in Rust.**

Apsis estimates a known target's position, velocity, attitude, and angular rate
from camera images. Its navigation core is allocation-free and `no_std`.
The desktop demo makes tracking—and its failures—visible.

Research prototype. No thruster commands, docking autonomy, or flight qualification.

## Run the demo

With stable [Rust](https://www.rust-lang.org/tools/install) and Python 3.11+:

```sh
cargo run --release --locked -p apsis-cli
python scripts/serve.py
```

Open **[localhost:18765](http://127.0.0.1:18765)**.

On Windows, build natively or in WSL. The Windows server can use the Linux executable
from the same checkout in `Ubuntu-24.04`; see the [demo guide](docs/demo.md).

The three panes show:

1. A shaded, tumbling **3U CubeSat**, directional sunlight, and specular glare.
2. Grayscale corner detections, docking-plate bounds, and an estimated 3D wireframe.
3. Position, angular rate, covariance uncertainty, and navigation health.

Press **Cut camera · 3 s** to recompute a blackout from the current playback time.
The EKF continues predicting; uncertainty grows; optical updates resume when images
return. The estimate becomes GNC-invalid after 0.5 s without an accepted measurement.

Noise, occlusion, false image features, calibration bias, delivery delay, and glare
controls rerun the Rust pipeline. The viewer replays the resulting recording;
it does not invent estimator output in JavaScript.

Open **Scenario lab** for failure presets, run evidence, and exports. Inspect the
health timeline or step between recorded acquisition, outage, and validity changes.
Expand the covariance row for all 12 error-state uncertainties. The three open views
fill the desktop workspace; controls stay in a drawer.
Orbital curves, warm light, and an age gauge give the console its instrument style.

## Repeatable failure tests

```sh
cargo build --release --locked -p apsis-cli
python scripts/scenario_suite.py
```

Seven scenarios × three seeds produce CSV/JSON reports in `output/suite`.
The report distinguishes **contract checks** from **tracking accuracy guards**;
a stress run can obey health rules and still track badly. Current trials expose
acquisition failure under occlusion and calibration sensitivity. The new boundary
guard reduces the seeded clutter false-valid error from 86.6 cm to 10.8 cm;
this is a regression result, not a reliability guarantee.
See the [scenario evidence](docs/scenarios.md) for results and limits.

## Pipeline

```text
Shaded camera raster → Harris corners → association → PnP → boundary verification
                    → multiplicative error-state EKF → health + telemetry
```

The camera runs at a simulated 10 Hz; navigation propagates at 50 Hz.
The default recording lasts 20 seconds, including an 8–11 s blackout.
Simulation truth is used for rendering and scoring, never estimator updates.

| Crate | Purpose |
| --- | --- |
| `apsis-core` | Navigation API, timestamps, health, and validity rules |
| `apsis-math` | Calibrated pinhole projection, geometry, quaternions |
| `apsis-vision` | Allocation-free corners, correspondence, robust pose fitting |
| `apsis-ekf` | 13-value nominal state, 12-dimensional error covariance |
| `apsis-sim` | Desktop image formation and controlled failures |
| `apsis-cli` | Recording, three-pane viewer, CSV/JSON export |
| `apsis-bench` | Calibrated image evaluation; truth isolated to scoring |

## Evidence and limits

- **Known geometry and a warm start.** Matching uses an approximate pose prior.
  This is not global acquisition or an unknown-satellite tracker.
- **Simplified environment.** Local prescribed motion, not orbital propagation;
  lighting and camera noise are not radiometrically calibrated.
- **Explicit failure behavior.** Acquiring, tracking, coasting, and lost states;
  stale and out-of-order measurements are rejected. Unsupported body boundaries
  latch an integrity hold; recovery requires repeated accepted poses. Default validity thresholds
  are demo policy, not mission acceptance criteria.
- **Embedded build portability.** CI cross-builds the core for
  `aarch64-unknown-linux-gnu`, `aarch64-unknown-none`, and
  `thumbv7em-none-eabihf`. Board support, hardware timing, and radiation tolerance
  remain unverified.

The [benchmark guide](docs/benchmarking.md) covers SPEED+/SHIRT import and evaluation.
The [embedded guide](docs/embedded.md) covers the consumer example, storage sizes,
stack evidence, and timing scope. No public-dataset accuracy result is claimed yet.

## Development

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --release --locked -- -D warnings
cargo test --workspace --release --locked
python -m pip install -r scripts/requirements-dev.txt
python -m black --check scripts
python -m ruff check scripts
python -m unittest discover -s scripts -p 'test_*.py'
npm ci
npm run check
```

CI tests tracking, blackouts, timestamp handling, outliers, geometry degeneracy,
numerical guards, importer conventions, and ground-truth isolation. It uploads
the demo, seeded scenario report, and embedded evidence. Node is only needed for
viewer formatting checks. Features and fixes land on `development`;
merge into `main` after checks and review.

See the [validation roadmap](docs/validation-roadmap.md) for remaining integration gates.
