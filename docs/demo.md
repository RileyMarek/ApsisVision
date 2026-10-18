# Demo and navigation contract

## Run and inspect

Build/run `cargo run --release --locked -p apsis-cli`, then launch
`python scripts/serve.py`. The loopback server defaults to port 18765; override
with `--port`. It serves only `output/` and accepts bounded scenario parameters.
On Windows it uses a native `.exe` if available, otherwise the shared-checkout
Linux binary through WSL distribution `Ubuntu-24.04`.

Each scenario creates a separate `output/runs/<id>/` recording. The server serializes
scenario runs, executes the binary without a shell, and rejects non-local hosts and
foreign origins. Close the server when finished. Run directories are retained for
comparison; they are ignored by Git. No GPU, npm packages, or hosted service is used.

`output/index.html` also supports offline playback. Keep its `frames/` directory
alongside it. Interactive scenario controls require the local server.

The simulation pane deliberately shows truth during outages. The camera pane
shows no optical data while unavailable; its dashed wireframe is the EKF prediction.
The blue box is estimated. White crosses are corners; green circles are candidate
matches (pose-fit inliers when fitting succeeds). The green docking outline requires
all four observed plate correspondences; the amber outline is predicted geometry.

CSV contains capture and navigation timestamps, health/validity/rejection reason,
position, velocity, angular rate, quaternion, all twelve marginal standard deviations,
errors, and host vision/filter times. JSON also contains overlays and run statistics.
Missing estimates/errors are blank in CSV and null in JSON, never zero-error successes.

## Failure controls

| Control | Actual effect |
| --- | --- |
| Pixel noise | Seeded intensity noise in the rendered sensor image, in ADU |
| Occlusion | Black mask covering a fraction of the target's projected width |
| False image features | Bright, uncoded patches injected near the target |
| Focal-length bias | Incorrect focal lengths in the estimator calibration |
| Frame delay | Capture-time pose packets delivered later to the navigator |
| Glare | Directional specular reflection and a simplified sweeping flare |
| Cut camera | No optical measurement during a recomputed three-second interval |

The default pixel-noise setting is one approximate ADU standard deviation from a
uniform distribution; it is not Gaussian. Values below three ADU are clipped by a
simplified black level. Lighting is a stress visualization, not a calibrated orbital
radiance/exposure model. Visible faces use depth-tested rasterization, not a full
ray tracer or cast-shadow simulation. The synthetic docking plate is an assumption
of this cooperative target, not a CubeSat hardware standard.

## GNC interface

`apsis_core::navigation::Navigator` owns the filter. Call `predict_to(t)` with
finite, nonnegative, strictly increasing seconds. An error means the consumer must
not continue as though prediction succeeded. `observe(captured, pose, reason)`
uses camera capture time, not delivery time. No simulator state is part of this API.

Default policy:

| State | Condition | GNC-valid? |
| --- | --- | --- |
| Acquiring | No pose yet, fewer than three accepted poses, or 0.5 s reacquisition ramp | No |
| Tracking | Last accepted pose age <= 0.15 s, bounded position covariance | Yes |
| Coasting | Age > 0.15 s and <= 3.5 s, bounded position covariance | Only while age <= 0.5 s |
| Lost | Age > 3.5 s, numerical fault, or any position sigma > 0.5 m | No |

Measurements older than 25 ms at delivery, duplicates, out-of-order capture times,
future timestamps, nonfinite poses, failed fits, and large innovations are rejected.
Up to 25 ms lateness is treated as tolerated timing jitter; there is no historical
rewind/replay or exact out-of-sequence measurement fusion. Rejected observations do
not refresh the accepted-measurement clock. The filter may keep a finite prediction
after loss, but that prediction remains invalid for guidance.

After a gap beyond the valid-coast limit, measurement covariance is initially
inflated by 4x and returns to baseline over 0.5 s. This weighting affects the real
filter; there is no cosmetic pose smoothing in the viewer. Optical-return time,
first optical correction, and time until GNC-valid again are reported separately.
These tuned choices and covariance limits require mission-specific validation.

Position/velocity are target-relative quantities in camera coordinates: X right,
Y down, Z forward. The scalar-first quaternion `(w,x,y,z)` actively rotates target
body vectors into the camera frame. Angular rate is expressed in camera coordinates.
Meters, seconds, and radians are used internally. The covariance order is
`[p(3), v(3), attitude_error(3), angular_rate(3)]`.

## Current acquisition limits

The demo starts near 1.6 m with a fixed approximate attitude prior. Harris corners
are associated using that prior, an estimated visibility mask, and a pixel gate.
Bounded RANSAC rejects some mismatches; it cannot fix an arbitrary incorrect
correspondence model. Planar ambiguities, symmetry, feature switching, and inaccurate
calibration remain failure modes. A tracking label does not guarantee true accuracy
when the covariance/noise model is wrong; the synthetic truth plots expose that gap.
