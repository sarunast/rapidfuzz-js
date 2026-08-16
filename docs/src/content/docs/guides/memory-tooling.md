---
title: Memory tooling
description: Reachability tests, retained-byte soaks, heap snapshots, and allocation profiles.
---

The repository uses four complementary memory checks. They answer different questions, so
one is not evidence for another:

- `pnpm test:memory` counts live objects after full garbage collections. It is the blocking
  regression gate for object lifetime and runs separately from coverage.
- `pnpm bench:memory:soak` samples retained bytes after repeated indexed-matcher work. It
  catches unbounded state even when the retained objects have no useful constructor to count.
- `--snapshot` writes object-graph evidence for manual retainer-path investigation.
- Node's `--heap-prof` samples allocations and answers where bytes were allocated, including
  transient peaks that disappear before a retained-heap reading.

`heapUsed`, `arrayBuffers`, and their sum (`retained`) are the soak's asserted signals.
`heapTotal`, `external`, and `rss` are reported only for diagnosis: allocator reservation,
double-counted external memory, and resident pages make them unsuitable regression gates.

## Blocking reachability tests

```sh
pnpm test:memory
```

The tests use Node's [`v8.queryObjects()`](https://nodejs.org/api/v8.html#v8queryobjectsctor-options),
available since Node 20.13.0 and 22.0.0. It remains experimental on Node 22, the oldest
supported release, and is non-experimental from Node 24.13.1 and 25.4.0 onward. Each call
performs a full garbage collection and returns only objects created in the current execution
context. That context restriction is why the tests are sequential, isolated, and load
`QueryState` and `createIndexedMatcher` through the same Vitest module graph. It is also why
they are not part of the coverage run and cannot be reused by a future browser runner.

Every assertion calls a scoped helper before counting. A query, result, matcher, thrown error,
or sentinel left in the test driver's lexical scope can be kept alive by V8 and produce a false
failure.

## Retained-byte soaks

```sh
pnpm bench:memory:soak
pnpm bench:memory:soak -- --scenario=steady
```

Each scenario builds one deterministic 100,000-choice Dice index in an isolated
`--expose-gc` child. Every sample follows three explicit collections. The asserted growth rate
is the Theil–Sen median of all pairwise
`(retained[j] - retained[i]) / (j - i)` slopes over the final ten samples.

Committed thresholds are applied only to the canonical 100,000-choice, 5,000-operation run,
without snapshots or diagnostic instrumentation, on the environment recorded with the
thresholds. Custom sizes, snapshots, `--diagnostic` runs, and other Node/platform/architecture
combinations report their samples with `evaluated: false`, an `evaluationReason`, and
`passed: null`. Calibrate that workload and environment instead of interpreting the committed
thresholds against it. Snapshot and allocation-profile capture are forensic and never produce
a byte-threshold verdict.

Custom steady batches proportionally preserve the canonical `best`, top-5, top-100, unrelated,
and unlimited operation mix. This makes custom calibration and snapshot runs smaller versions
of the same scenario rather than a best-only workload.

The committed thresholds were calibrated on 2026-08-16 using Node v26.7.0, macOS arm64, with
20 clean isolated runs per scenario and 5,000 operations per measured batch:

| Scenario      | Clean slope range (B/batch) | Threshold (B/batch) | Clean recovery range (B) | Recovery threshold (B) | Threshold B/op |
| ------------- | --------------------------: | ------------------: | -----------------------: | ---------------------: | -------------: |
| Steady        |           2,601.78–2,615.11 |              12,288 |            45,896–46,920 |                327,680 |         2.4576 |
| Query profile |           2,593.14–2,613.71 |              12,288 |            43,688–44,696 |                327,680 |         2.4576 |
| Touched set   |               942.67–942.67 |              12,288 |            19,528–20,032 |                327,680 |         2.4576 |
| Exception     |           2,176.00–2,261.71 |              12,288 |            39,752–42,024 |                327,680 |         2.4576 |

Calibration takes the positive slope and recovery delta from each run. The slope threshold is
the next 4 KiB/batch above `max(8 KiB/batch, 3 × clean p95)`. Recovery is the next 64 KiB above
`max(256 KiB, 3 × clean p95)`. Recalibrate after changing the workload, Node/V8 generation,
or CI platform:

```sh
pnpm bench:memory:soak -- --calibrate --artifact=bench/memory/artifacts/calibration.json
```

The controls deliberately fail if retained growth is injected:

```sh
pnpm bench:memory:soak -- --scenario=steady --fixture=slope
pnpm bench:memory:soak -- --scenario=query-profile --fixture=recovery
```

The slope fixture retains one `ArrayBuffer` per 100 operations, totaling twice the scenario's
slope threshold per batch. The recovery fixture retains a buffer twice the recovery threshold
after the spike. Recovery injection is not defined for steady state, and fixtures cannot be
combined with calibration.

## Snapshots and allocation profiles

Snapshots are opt-in because writing one blocks the event loop and may require about twice the
current heap size. Never enable snapshot capture in CI:

```sh
pnpm bench:memory:soak -- --scenario=query-profile --snapshot
pnpm bench:memory:soak -- --scenario=touched-set --snapshot=/tmp/rapidfuzz-snapshots
```

With plain `--snapshot`, the gitignored output goes into one timestamped directory per run,
then one subdirectory per scenario. A steady sequence contains `post-warmup` and `final`;
spike scenarios contain `pre-spike`, `post-spike`, and `recovered`. Each scenario directory is
therefore one coherent object history from one child process. Inspect it without adding MemLab
to this repository:

```sh
pnpm dlx memlab view-heap --snapshot path/to/file.heapsnapshot
pnpm dlx memlab analyze unbound-object --snapshot-dir path/to/run/query-profile
```

For allocation sampling, invoke the child workload directly so Node owns the profile lifecycle:

```sh
pnpm build
mkdir -p bench/memory/artifacts/heap-prof
node \
  --heap-prof \
  --heap-prof-dir=bench/memory/artifacts/heap-prof \
  --expose-gc \
  bench/memory/soak.ts \
  --child \
  --diagnostic \
  --scenario=steady
```

`--diagnostic` explicitly suppresses the normal retained-byte verdict. The soak also detects
`--heap-prof` in Node's execution arguments as a safeguard for direct child invocations. Node
writes a `.heapprofile` on exit. `--heap-prof-dir`, `--heap-prof-name`, and
`--heap-prof-interval` control its location, name, and sampling interval.
