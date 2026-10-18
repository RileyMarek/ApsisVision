# Integration and validation roadmap

The first intended evaluator is a spacecraft GNC engineer. Priorities are reliable,
accurate relative navigation under difficult space imaging conditions, followed by
recorded/live camera integration and AMD Xilinx Zynq UltraScale+ MPSoC deployment.

## What the demo establishes

The desktop pipeline detects synthetic coded markers, estimates pose, and filters
motion without reading simulation truth. Regression tests exercise pixel noise,
missing measurements, duplicate marker IDs, pose outliers, covariance growth, and
finite exported states. These tests establish a baseline, not mission reliability.

## Next gates, in order

| Gate | Evidence needed before calling it complete |
| --- | --- |
| Integration contract | Timestamped calibrated observations; frame/unit conventions; explicit acquiring, tracking, coasting, and lost states; stale/out-of-order input handling; reset and validity rules |
| Estimator integrity | Independent Jacobian checks; covariance consistency across Monte Carlo trials; degenerate-geometry tests; bounded rejection/reacquisition behavior; numerical fault handling |
| Image robustness | Labeled sweeps of saturation, deep shadows, glare, blur, occlusion, false detections, calibration errors, and partial target visibility; report failures as well as RMS errors |
| Recorded camera data | Public benchmark with verified calibration, coordinate conversion, pose ground truth, dataset provenance, and separate tuning/evaluation sequences |
| Live camera replay | Timestamp and latency accounting; exposure changes; dropped/delayed frames; repeatable recording/replay; independent pose reference |
| Zynq deployment | Exact part/board, OS, camera link, processor allocation, power budget, memory budget, and compiler target; measured timing distributions and deadline misses on hardware |
| Mission fault tolerance | Hardware-specific radiation assessment and fault model; memory/data integrity mechanisms; detection and recovery tests; system-level hazard analysis and qualification evidence |

Numerical accuracy limits, maximum coasting time, acquisition range, tolerated
lighting envelope, and acceptable deadline-miss rate must come from the intended
mission. They are not yet specified. Current CI error limits are regression guards,
not flight acceptance criteria.

## Hardware boundary

The requested development target is Zynq UltraScale+ MPSoC; no particular board or
radiation-qualified part has been selected. Begin by measuring the CPU implementation
on the selected hardware. Add programmable-logic acceleration only for measured
bottlenecks, after establishing a trusted CPU reference.

Rust memory safety does not prevent radiation-induced memory corruption, sensor
upsets, hardware faults, or algorithmic errors. Software fault injection can test
recovery logic; it cannot substitute for radiation qualification. Device grade,
mission environment, mitigation, and qualification need explicit evidence.
AMD distinguishes its defense and space portfolios and publishes device-specific
radiation information: [aerospace and defense](https://www.amd.com/en/solutions/aerospace-and-defense.html),
[space products and radiation resources](https://www.amd.com/en/solutions/aerospace-and-defense/space.html).

Guidance and actuation stay outside the current scope. A future integration must
define how the consuming GNC system responds when the navigation solution is stale,
uncertain, inconsistent, or unavailable.
