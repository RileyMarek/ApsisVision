# Apsis Vision

**Relative spacecraft navigation, built in Rust.**

Apsis explores how camera images can estimate a nearby spacecraft's position,
orientation, velocity, and rotation rate. The current demo tracks a known synthetic
target and produces an interactive offline viewer and CSV telemetry.

This is a research prototype, not flight software. It does not command thrusters or dock a spacecraft.

## Run the demo

Install stable [Rust](https://www.rust-lang.org/tools/install), then:

```sh
cargo run --release --locked -p apsis-cli
```

Open `output/index.html` in your browser. No server, JavaScript packages, GPU, or
network connection is needed at runtime. Building requires downloading Rust crates.

```sh
cargo run --release --locked -p apsis-cli -- --frames 1500 --seed 42 --out output
```

The default run simulates 30 seconds at 50 Hz. Playback shows the target and estimated
wireframes, detected markers, tracking errors, and projected position uncertainty.
`output/telemetry.csv` includes truth, estimated position, position and attitude
errors, position uncertainty, marker counts, accepted updates, and processing time.
Runs are repeatable for a given seed; wall-clock timings vary by machine.
Frame count is limited to 1–3000 for the supplied scenario.

## How it works

```text
Synthetic camera → marker centroids → iterative PnP → error-state EKF → viewer + CSV
```

Eight markers on a known 3D target encode their identities as grayscale values.
The renderer adds pixel noise and periodically removes all markers for half a second.
The vision pipeline detects marker centers from the raster image and solves for pose.
The filter estimates motion, rejects large innovations, and predicts through missing images.
Simulation truth is used only for rendering and scoring, never as an estimator measurement.

| Crate | Responsibility |
| --- | --- |
| `apsis-math` | Camera projection, target geometry, quaternion pose operations |
| `apsis-sim` | Local target motion and noisy marker images |
| `apsis-vision` | Coded marker detection and iterative pose fitting |
| `apsis-ekf` | Pose, velocity, and angular-rate estimation |
| `apsis-cli` | Demo runner, offline playback, CSV export |

Coordinates are right-handed camera coordinates: X right, Y down, Z forward.
Pose maps target-body coordinates into the camera frame; distances are meters,
time is seconds, and internal angles are radians. Quaternions rotate body to camera.
The EKF uses a 13-value nominal state and a 12-dimensional multiplicative error state.

## Scope and limits

- **Known target only.** Marker identity comes from artificial intensity codes.
  This is not a general feature detector or unknown-satellite tracker.
- **Local pose solver.** Iterative PnP starts at an assumed 12 m range and identity
  attitude, then uses the previous prediction. Global acquisition and robust
  initialization remain future work; this is not EPnP.
- **Simplified simulation.** Point emitters have no surface occlusion, shadows,
  glare, lens distortion, or motion blur. Motion is prescribed in camera coordinates,
  not propagated with orbital dynamics.
- **Approximate uncertainty.** Measurement noise is tuned and diagonal. Covariance
  is not derived from the pose fit or calibrated across viewing geometries. The
  displayed ellipse approximates transverse position uncertainty, omitting depth
  correlation and attitude uncertainty.
- **Desktop implementation.** Uses `std` and heap allocation. Rust provides memory
  safety, but hard real-time timing, `no_std`, and flight qualification require
  separate engineering and evidence. A simulated 50 Hz cadence is not a measured
  real-time performance guarantee.

## Development

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

Tests cover projection, quaternion normalization, image-to-pose recovery,
outlier rejection, uncertainty growth during dropouts, and full-pipeline tracking
across three noise seeds. CI runs these checks and uploads the generated demo.

Features and fixes land as focused commits on `development`. Merge into `main`
after checks pass and the demo is reviewed.

Next: robust failure handling, difficult lighting and occlusion, calibrated
uncertainty, and recorded/live camera benchmarks. Zynq UltraScale+ MPSoC is the
intended hardware target; deployment and radiation tolerance are unverified.
See the [validation roadmap](docs/validation-roadmap.md) for the integration gates.
Relative guidance comes after navigation has earned its confidence.
