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
- **Tests are ported from RapidFuzz's own suite**, under `tests/`, with a header
  comment naming the source file. Port the assertions faithfully, including the
  regression tests named after upstream issue numbers — those numbers are the
  reason the test exists. Aim for the fullest coverage of the upstream suite we
  can express.
- **Naming is camelCase**, but public names describe this API: `similarity`,
  `partialSimilarity`, `fuzzySimilarity`, `createScorer`, and `createMatcher`.
  Do not restore removed RapidFuzz aliases such as `WRatio`, `QRatio`, or
  `extractOne`.
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

The same rule covers the JSDoc spellings, because the tooling under `bench/`
and `scripts/` is plain `.mjs` — `compare.mjs` sets environment variables
around a child process, so it cannot be a `.ts` file needing a build step, and
it must stay outside the set it fingerprints. Those files carry `// @ts-check`
and JSDoc types, checked by `tsconfig.scripts.json`; `pnpm typecheck` runs it
alongside the main config. There:

- `/** @type {T} */ const x = …` on a declaration is an **annotation** — fine,
  and the way to type an accumulator or an empty array.
- `/** @type {T} */ (expr)` around an expression is a **cast** — banned, same
  as `as T`. This one the linter cannot see: it is a comment, and no oxlint
  rule reads it. Convention is all that holds it.
- `@ts-ignore`, `@ts-expect-error` and `@ts-nocheck` are banned for the same
  reason, and `typescript/ban-ts-comment` enforces all three.

When a runtime check proves something the checker cannot see, restructure so
the proof is visible — `filter((x) => x !== undefined)` narrows where
`some((x) => x === undefined)` followed by a throw does not.

Note: `noUncheckedIndexedAccess` is deliberately **off** in `tsconfig.json`.
With it on, every read from an array or typed array widens to `T | undefined`,
and the numeric DP loops in `src/algorithms/` would need a `!` or a `?? 0` on
every single indexed read. Turning it off removes the need for the assertion
rather than hiding it.

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

## Public API and dependency ownership

The root exports orchestration only: scorer creation, matching/search, batch
scoring, threshold helpers, and text normalization. Algorithms are imported
from canonical subpaths such as `rapidfuzz-js/levenshtein` and
`rapidfuzz-js/fuzz`; never re-export them from the root.

Preserve each algorithm's natural scale. Fuzz similarities return `0..100`,
normalized algorithm similarities and Jaro-family similarities return `0..1`,
and distances remain in native edit/count units. Generic infrastructure must
not rescale between families.

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

Do not add compatibility aliases, aggregate namespaces, public prepared
handles, or a horizontal `common`/`preparation` subsystem. Delete unused code;
do not keep it for a removed API.

## Benchmarks

The whole suite is 155 cases and about ten minutes. Almost nothing needs the
whole suite. Both filters — a bench file, named by any substring of its path,
and `-t` over case names — work on every script below, and they compose:

```sh
pnpm bench:quick fuzz -t 'partialRatio'      # under a second; the edit loop
pnpm bench:compare bench/fuzz.bench.ts       # before believing a number
pnpm bench:compare:quick -t 'indelDistance'  # ~5s, ±15%: did I break it
pnpm bench:baseline bench/fuzz.bench.ts      # re-record, after a real change
```

**Detect with `bench:compare:quick`; spend a full `bench:compare` only on
numbers that get written down.** A full suite run is 155 cases at 1.3s each,
twice — about seven minutes — and reaching for it to answer "did that move?"
burns whole stretches of a session. `--quick` is the same comparison against the
same baseline at a tenth of the window and one pass: the whole suite in ~45s, a
single filtered file in ~6s. Its ±15% band is wide, and real findings are
usually nowhere near it — a 2.43x regression and a set of 1.6-2x wins were all
comfortably outside it. Escalate to the full run once, at the end, for the
figures that go in the commit message. Never quote `--quick` as evidence of an
improvement; it refuses to record a baseline for exactly that reason.

A filtered comparison is still anchored — `bench/control.bench.ts` brackets
every pass whatever is filtered out — so it stays comparable to a baseline
recorded from a full run. It gives up only the suite-wide move, which needs
five comparable cases and reports `n/a` below that.

`compare.mjs` spawns three children per pass and shows only which one is
running; `--verbose` lets vitest's own reporter through when a pass looks
stuck. `node bench/compare.mjs --help` lists the rest. Six rules that are not
guessable from the code:

- **Two passes, comparing and recording alike.** A third refines a spread the
  ±3% floor discards in almost every case, and costs the whole suite again — a
  full comparison is about ten minutes a pass, and a run has to be cheap enough
  to sit through or it stops being run. `--repeat=N` still takes more, which is
  what a case with a genuinely wide band is worth. Two is the floor: one measures
  no spread at all, stores a zero band, and is reported as stale forever after.
- **`pnpm bench` and `pnpm bench:quick` are not baseline-comparable.** They run
  without `--expose-gc` and, quick, at a tenth of the window; `--quick` widens
  its threshold to ±15% and refuses to record. They answer "did I break
  something", never "is this 4% faster".
- **A flag is a place to look, not a finding.** A noise band covers the spread
  within one run, not between two. Re-run the file before believing anything
  the report highlighted.
- **`bench/_harness.ts`, `bench/_corpus.ts` and `vitest.config.ts` are hashed
  into every one of the 155 baseline entries.** Editing any of them — a comment
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
  that differs — Node, CPU, vitest — not for stored numbers that mean something
  else. Bump the constant only when the anchor or the aggregation changes, and
  expect a full re-record to be part of the same commit.

Because a bench file is hashed into its own cases, run `pnpm bench:compare` on
a file _before_ editing it — afterwards there is nothing left to compare
against.
