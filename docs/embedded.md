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
association, PnP, convex-boundary verification, and filtering on a host thread
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
