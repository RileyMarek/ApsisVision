# Embedded integration evidence

`apsis-core` and its math, vision, and filter dependencies use `no_std` and fixed-size
storage, without an allocator. Desktop rendering, imports, and recording use `std`.

## Consumer example

```sh
cargo run --release --locked -p apsis-core --example consumer
```

The host example demonstrates timestamped observations, a three-second blackout,
validity gating, and reacquisition. Replace the example observation source with
calibrated camera poses. It does not drive hardware or actuators.

## Reproducible measurements

```sh
cargo run --release --locked -p apsis-core --example memory_profile
python scripts/embedded_evidence.py
```

The first command measures persistent type sizes and executes corner detection,
association, PnP, convex-boundary verification, filtering and reference guidance on a host thread
requesting a 128 KiB stack.
The workload uses fixed arrays. Thread creation and printing belong to the host
harness, not the core. The OS may round the request; successful completion is a
stack-budget smoke test, not a measured high-water mark or worst-case guarantee.

The second command writes `output/embedded-evidence.json`, recording the compiler,
host, storage sizes, and per-function stack displacement from optimized x86-64
assembly/unwind metadata with the red zone disabled. Unsupported frame-pointer or
dynamic CFI expressions are explicitly reported as unmeasured. The reported maximum
is only among measured functions; it excludes callees, interrupts, runtime overhead,
and whole-call-chain peak. It is not an ARM stack bound. The parser is tested and
restricted to `x86_64-unknown-linux-gnu`; it fails rather than guessing on another host.

The demo records host vision p50/p95/p99 and filter p95. They exclude rendering,
PNG encoding, file I/O, scheduling deadlines, camera drivers, and hardware transfer.
Vision is sampled at simulated 10 Hz; propagation is sampled at 50 Hz. Playback at
those rates is not evidence of a real-time deadline guarantee.

## Cross-builds

CI independently builds the release library with default features disabled for:

- `aarch64-unknown-linux-gnu`: AArch64 Linux.
- `aarch64-unknown-none`: bare-metal AArch64.
- `thumbv7em-none-eabihf`: Armv7E-M hard-float portability check, not a Zynq target.

These are library artifacts. No board startup code, firmware linking, device I/O,
WCET proof, power/thermal test, or radiation qualification is supplied. Zynq
UltraScale+ MPSoC remains an intended deployment target with no available hardware.

## Bounded timestamp replay

```sh
python scripts/replay_fixture.py
cargo run --release --locked -p apsis-replay -- output/replay/faults.jsonl --paced
cargo run --release --locked -p apsis-replay -- output/real-images/shirt-roe1-lightbox.jsonl
```

NDJSON packets contain `captured_s`, `delivered_s`, and `pose` (null or an object with
`position_m` and scalar-first unit `quaternion_wxyz`). Units and rotation follow the
camera convention in `apsis-math`. The adapter keeps one record at a time: at most
4096 bytes per record, 100000 packets per stream and 10 s between deliveries.
Propagation uses ≤20 ms substeps. There is no growing observation queue or timing
sample vector. The host parser/CLI uses `std` and bounded allocations; this is not
firmware or a camera driver.

Late/duplicate captures are rejected by navigation. Non-unit quaternions, malformed
packets, oversized records, reversed delivery time or excessive scheduler gaps stop
the stream with a nonzero exit. A consuming process must treat stream termination
as unavailable/invalid; it must never keep the last valid output indefinitely.
Reinitialization after transport corruption is the supervisor's responsibility.

Packets optionally include `covariance_6`, a row-major 6x6 covariance array ordered
`[x_m, y_m, z_m, attitude_x_rad, attitude_y_rad, attitude_z_rad]`. It must be finite,
symmetric and positive definite. The adapter preserves its cross terms and passes
it to `observe_with_covariance`; malformed covariance stops replay. Older packets
without this field use the documented fixed-R convenience default. Pixel consumers
now supply projection-derived covariance plus the demo discrepancy floor; see the
[adaptive uncertainty evidence](adaptive-perception.md). Learned-model packets are
quarantined with null poses until perception/integrity qualification.

`--paced` schedules delivery against a monotonic host clock and includes scheduling
lateness in measurement age. Without it, recorded delivery times advance virtually.
Reports separate scheduler lateness from propagation/filter compute time and count
20 ms update deadline misses. Parsing, file IO and image processing are excluded.
These are host measurements, not end-to-end deadlines or real-time OS guarantees.

The [151-packet host run](evidence/replay-2026-10-18.json) rejects one stale and one
duplicate capture, handles 30 missing poses, returns to tracking, and records zero
update deadline misses (maximum 549 µs; maximum scheduler lateness 279 µs).
The official SHIRT pose-or-null replay accepts zero poses and stays acquiring,
confirming the failed detector does not silently use ground truth.

Hardware bring-up still requires an exact board/part and processor allocation,
OS/firmware, camera transport and clock domain, memory/power/deadline budgets,
ARM stack high-water and timing distributions, watchdog/reset recovery, and
device/mission-specific fault and radiation assessment. No Zynq runtime measurement
or radiation tolerance follows from cross-compilation.
