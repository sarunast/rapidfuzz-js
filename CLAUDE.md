# Project conventions

## Goal: RapidFuzz algorithms behind a JavaScript-first API

This project is a JavaScript/TypeScript implementation of
[RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) algorithms (MIT). Algorithm
results follow RapidFuzz, but the public API is intentionally JavaScript-first:
`Metric -> Scorer -> Matcher`, canonical lowercase algorithm subpaths, and no
legacy aggregate namespaces.

Practical consequences:

- **The Python implementation is the algorithm spec.** When numeric behaviour is
  unclear, read the reference `src/rapidfuzz/**/*_py.py`. Preserve its edge cases
  and adaptive fuzzy weighting while expressing them through this package's API.
- `normalizeText` follows the installed RapidFuzz `utils.default_process`
  behavior, including treating underscore as non-alphanumeric.
- **Tests are ported from RapidFuzz's own suite**, under `tests/`, with a header
  comment naming the source file. Port the assertions faithfully, including the
  regression tests named after upstream issue numbers — those numbers are the
  reason the test exists. Aim for the fullest coverage of the upstream suite we
  can express.
- **Naming is camelCase**, but public names describe this API: `similarity`,
  `partialSimilarity`, `fuzzySimilarity`, `createScorer`, and `createMatcher`.
  Preserve mathematical and functional RapidFuzz capability
  while replacing Python-specific mechanics. Do not restore legacy spellings
  such as `WRatio`, `QRatio`, or `extractOne`.
- **Correctness before speed.** Where upstream uses a bit-parallel or SIMD
  implementation, a straightforward DP that produces identical results is
  acceptable; optimise later, behind the ported tests.

## No type casting

Do not cast or assert types anywhere in this project — not in `src/`, not in
tests, not in config files. Specifically banned:

- `x as T`, including `as unknown as T`
- `<T>x` angle-bracket assertions
- `x!` non-null assertions

All of these silence the type checker without proving anything, which is exactly
the failure mode a library's public types exist to prevent. If a value does not
typecheck, fix the type: widen a parameter, add a generic, introduce a type
guard (`function isFoo(x: unknown): x is Foo`), or use `satisfies` when you want
inference plus a constraint.

**Enforced, not just documented.** `.oxlintrc.json` turns these into lint
errors — `typescript/consistent-type-assertions` at `assertionStyle: "never"`
for the first two, `typescript/no-non-null-assertion` for the third — and
`pnpm lint` is a bare `oxlint`, so it reads every file the config does not
ignore rather than `src/` alone. `pnpm check` runs it.

`as const` is **allowed**, and the rule above exempts it by design. It widens
nothing and hides nothing: it makes a type narrower than inference would, which
is the opposite of what the ban exists to prevent. A tuple table in a test is
its ordinary use.

`as` in a non-type sense is fine (`import * as ns from '...'`, `export { x as y }`).

The same rule covers the JSDoc spellings, because the tooling under `scripts/`
is plain `.mjs`: each is a shell-invoked entry point that has to run against a
checkout which may not have built anything. Those files carry `// @ts-check`
and JSDoc types, checked by `tsconfig.scripts.json`; `pnpm typecheck` runs it
alongside the main config. There:

- `/** @type {T} */ const x = …` on a declaration is an **annotation** — fine,
  and the way to type an accumulator or an empty array.
- `/** @type {T} */ (expr)` around an expression is a **cast** — banned, same
  as `as T`. This one the linter cannot see: it is a comment, and no oxlint
  rule reads it. Convention is all that holds it.
- `@ts-ignore`, `@ts-expect-error` and `@ts-nocheck` are banned for the same
  reason, and `typescript/ban-ts-comment` enforces all three.

The benchmark tooling under `bench/tooling/` is TypeScript and checked by
`tsconfig.dev.json` like everything else: Node runs those files directly by
stripping the types, so being an entry point costs it nothing. Keep it that
way — the case files are `bench/*.bench.ts` and the machinery that measures
them is `bench/tooling/`, which is the only thing separating the two at a
glance. Type stripping is on by default from Node 22.18, so that is the floor
for running the benchmarks; `engines` stays at `>=22` because it describes
what the published library needs, and the published library is plain JS.

When a runtime check proves something the checker cannot see, restructure so
the proof is visible — `filter((x) => x !== undefined)` narrows where
`some((x) => x === undefined)` followed by a throw does not.

Note: `noUncheckedIndexedAccess` is deliberately **off** in `tsconfig.json`.
With it on, every read from an array or typed array widens to `T | undefined`,
and the numeric DP loops in `src/algorithms/` would need a `!` or a `?? 0` on
every single indexed read. Turning it off removes the need for the assertion
rather than hiding it.

## Type imports go at the top of the file

Import types with a top-level `import type { Direction } from '../core/types.js'`,
never inline as `import('../core/types.js').Direction` in a signature or
annotation. Both erase to nothing, so this is about reading the file: the
top-level form puts every dependency in one place, keeps the annotation short
enough to read at a glance, and lets a rename or a moved module be fixed once.
The inline form hides a module path in the middle of a parameter list, and
repeats it at every use.

No lint rule catches this — `.oxlintrc.json` says nothing about it — so
convention is all that holds it. The `{@link import('./x.js').y}` spelling in a
JSDoc comment is a different thing and stays.

## Comments are short

Two or three lines. A comment says what the code cannot: why a bound is that
number, what a measurement found, what a non-obvious ordering protects. It does
not narrate the code beside it, restate a type, or reproduce the investigation
that produced it — a date, a version and the conclusion carry that.

**Types carry no comments unless asked for.** A type declaration, its members
and its parameters are read as code, not documentation. Behaviour a caller
needs to know goes in README.md; a rule the compiler enforces needs no prose
beside it.

## Coverage is 100%, and that is enforced

`pnpm coverage` passes `--coverage.thresholds.100` and fails below it on all
four metrics. CI runs the same script, so a laptop and a pull request refuse
for the same reason.

The number is only meaningful because unreachable code is **deleted rather than
excused**. `src/` carries **no `/* v8 ignore */` at all**, and that is the state
to keep it in: every line the coverage report counts is a line some test runs.

So a new `v8 ignore` is a claim, and the claim has to be proved before it is
written — and the audit that emptied `src/` of them suggests the proof will fail.
Forty-seven were audited once by putting a `throw` on exactly the ignored
condition and running the suite, the benchmark corpus, a randomised sweep over
every entry point and an exhaustive driver over the kernel's own precondition
grid. Three of the forty-seven turned out to be false. What that audit found, in
order of how often it applied:

- **It is a type-checker artefact, not dead logic.** Restructure so the proof
  is visible. Sixteen `pool === null` guards became four accessors in
  the algorithm-owned bitmask helpers that return the buffer instead of a
  nullable binding.
  The last one to go was `identityOrder`'s: `compareElements` narrows with
  `isObjectLike` while the runtime proof is still in scope, so the callee takes
  `object` and needs no guard of its own. Narrowing at the call site is what an
  `as object` inside the callee would have asserted, except checked.
- **It is reachable, just not cheaply.** Add a seam and test it. The two stamp
  wraps are two billion mask builds away, so `resetBitVectorScratch` and
  `resetDamerauScratch` take a starting generation — the only reason either
  takes an argument.
- **It is genuinely dead.** Delete it, and say in a comment what makes it dead.
  A branch that cannot run cannot be relied on, and one that is wrong when it
  does run is worse — `jaroOneWord`'s mask reset was both.

## Library constraints

These are load-bearing for bundle size — see README.md for the full rationale:

- `"sideEffects": false` — nothing in `src/` may do work at import time.
- Export standalone named functions, never namespace objects.
- No runtime dependencies, and no Node built-ins in `src/`.

`pnpm check:bundles` is **not** one of them, and a few bytes over budget is not
a blocker. Its budgets are `measured × 1.02`, so they drift under ordinary work,
and a different toolchain gzips the same bundle 20-30 B apart — CI has failed an
entry that passed locally. Never reshape code or drop a fix to win bytes back:
re-record all fifteen entries in `scripts/check-bundles.mjs` together, or leave
it failing and say so. Re-recording only the entry that failed leaves its
siblings a byte from the same surprise.

## Public API and dependency ownership

The root exports orchestration only: scorer creation, matching/search, batch
scoring, threshold helpers, and text normalization. Algorithms are imported
from canonical subpaths such as `rapidfuzz-js/levenshtein` and
`rapidfuzz-js/fuzz`; never re-export them from the root.

Preserve each algorithm operation's RapidFuzz scale. Fuzz similarities return
`0..100`; raw edit/count distances and similarities use native units;
normalized distances and similarities return `0..1`; Jaro-family measures are
naturally `0..1`. Generic infrastructure must not rescale between families.

Source ownership follows dependency direction:

- `core/` knows types, protocols, scorer construction, thresholds, and
  normalization. It never imports algorithms.
- Each algorithm directory owns its public metric, compilation, preparation,
  and hot kernels. Shared algorithm code is limited to proven low-level data
  structures under `algorithms/shared/`.
- `fuzz/` is split by scorer family. Basic similarity must not import token or
  adaptive-fuzzy modules.
- `search/` and `batch/` depend only on core protocols, never on named
  algorithms. They execute private prepared kernels rather than public
  `Scorer.score()` calls.
- Tests mirror these domains, with import direction and public reachability
  guarded in `tests/architecture/`.

Do not add compatibility aliases, aggregate namespaces, raw prepared
representations, or a horizontal `common`/`preparation` subsystem. Delete
unused code; do not keep it for a removed API.

The opaque `PreparedChoice` from `scorer.prepareChoice` is the sanctioned form
of a public prepared handle, and the only one. It stays opaque, keeps its
metric brand in the declarations a consumer compiles against, and is checked
at runtime for scorer compatibility — `pnpm check:consumer` guards the second
of those, which source-level type tests cannot see.

Any type the public API can *infer* into a consumer's exported signature must
be nameable without importing our internals: exported from a public
entrypoint, or built from language-native constructs — the metric brand is
the id literal itself for exactly this reason. A type that fails this rule
typechecks everywhere and breaks only in the consumer's own declaration emit,
as a deep `import("…/dist/…")`; `check-consumer.mjs` emits a consumer's
declarations to catch it, but only for what its fixture exports — so a new
inference surface gets an unannotated `export const` there in the same
change.

Prepared search must not add a per-candidate cost to the text search path.
`bench/process.bench.ts` is where that is noticed.

Shared infrastructure owns policy and metadata; an algorithm module declares
algorithm facts and nothing else. A metric names itself once, as
`BuiltInMetric<'levenshtein.distance', 'distance'>`. The name derives the
compile-time brand and nothing else — the adapter owns preparation identity,
configuration and the missing-value policy, which follow from the direction
and the metric's own options. Those names are compile-time discriminators:
nothing exposes one at runtime and no API accepts one, so they may appear in a
diagnostic but never in a call. The
converse also holds: a driver may duplicate mechanics deliberately where that
is what keeps a hot loop monomorphic, which is why `bestDistance` and
`bestSimilarity` are two literal loops and not one comparator.

## Benchmarks

The whole suite is 139 cases and about four minutes to compare. Almost nothing
needs the whole suite. Both filters — a bench file, named by any substring of
its path, and `-t` over case names — work on every script below, and they
compose:

```sh
pnpm bench:quick fuzz -t 'partialRatio'      # under a second; the edit loop
pnpm bench:compare bench/fuzz.bench.ts       # before believing a number
pnpm bench:compare:quick -t 'indelDistance'  # ~5s, ±15%: did I break it
pnpm bench:confirm -t 'partialRatio 512'     # is that 4% real
pnpm bench:baseline bench/fuzz.bench.ts      # re-record, after a real change
```

**Detect with `bench:compare:quick`; spend a full `bench:compare` only on
numbers that get written down.** A full pass is ~100s, and a comparison is two
of them plus four control runs. `--quick` is the same comparison against the
same baseline at a tenth of the window and one pass: the whole suite in ~20s, a
single filtered file in a few seconds. Its ±15% band is wide, and real findings are
usually nowhere near it — a 2.43x regression and a set of 1.6-2x wins were all
comfortably outside it. Escalate to the full run once, at the end, for the
figures that go in the commit message. Never quote `--quick` as evidence of an
improvement; it refuses to record a baseline for exactly that reason.

A filtered comparison is still anchored — `bench/control.bench.ts` brackets
every pass whatever is filtered out — so it stays comparable to a baseline
recorded from a full run. It gives up only the suite-wide move, which needs
five comparable cases and reports `n/a` below that.

The suite does not run under vitest: `bench/tooling/runner.ts` bundles each
bench file once with esbuild and measures it in bare `node --expose-gc`,
because vite's transform layer added ~2.5x to every case body and did so
asymmetrically across module layouts. One child per file, never one process
for all of them — sharing a process made the fuzz cases 1.05-1.54x slower and
their noise up to 52%. Each case is sampled adaptively: it stops once four
consecutive 50 ms blocks agree to 1%, which is why a full pass is ~100s rather
than the ~200s fixed windows cost. `compare.ts` spawns three runner children
per pass and shows only which one is running; `--verbose` streams the per-case
progress when a pass looks stuck. `node bench/tooling/compare.ts --help` lists
the rest. Six rules that are not guessable from the code:

- **Two passes, comparing and recording alike.** A third refines a spread the
  ±3% floor discards in almost every case, and costs the whole suite again — a
  full comparison is about two minutes a pass, and a run has to be cheap enough
  to sit through or it stops being run. `--repeat=N` still takes more, which is
  what a case with a genuinely wide band is worth. Two is the floor: one measures
  no spread at all, stores a zero band, and is reported as stale forever after.
  `pnpm bench:confirm` is the other direction: widened windows and four repeats
  over the one case a normal run flagged, with a ±1.5% floor.
- **`pnpm bench` and `pnpm bench:quick` are not baseline-comparable.** They
  measure one run with no controls around it, and quick mode shortens every
  window to a tenth; `--quick` widens its threshold to ±15% and refuses to
  record. They answer "did I break something", never "is this 4% faster".
- **A flag is a place to look, not a finding.** A noise band covers the spread
  within one run, not between two. Re-measure with `pnpm bench:confirm` before
  believing anything the report highlighted.
- **`bench/tooling/harness.ts`, `bench/tooling/corpus.ts` and
  `bench/tooling/runner.ts` are hashed into every one of the baseline
  entries.** Editing any of them — a comment
  counts — marks the entire baseline "definition changed" and costs a full
  `pnpm bench:baseline` to recover. Batch such edits with a re-record you were
  going to do anyway; do not touch them in passing.
- **Re-record a whole file, never a subset.** A fingerprint covers a file, so
  recording part of one leaves its siblings stored against a hash that no
  longer matches. `--record` with `-t` is refused for that reason, and
  `--record <file>` _replaces_ that file's entries rather than merging into
  them — which is what clears a case you renamed or deleted. Other files are
  untouched; a bare `--record` replaces the baseline entirely, so it also drops
  entries from a bench file that no longer exists.
- **A baseline recorded under another `MEASUREMENT_VERSION` is refused**, and
  `--allow-environment-change` does not waive it. That flag is for a machine
  that differs — Node, CPU, esbuild — not for stored numbers that mean something
  else. Bump the constant only when the anchor or the aggregation changes, and
  expect a full re-record to be part of the same commit.

Because a bench file is hashed into its own cases, run `pnpm bench:compare` on
a file _before_ editing it — afterwards there is nothing left to compare
against.
