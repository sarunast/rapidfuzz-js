# Project conventions

## Goal: RapidFuzz algorithms behind a JavaScript-first API

A JavaScript/TypeScript implementation of
[RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) algorithms (MIT). Results
follow RapidFuzz; the API is JavaScript-first: `Metric -> Scorer -> Matcher`,
canonical lowercase algorithm subpaths, no legacy aggregate namespaces.

- **The Python implementation is the algorithm spec.** When numeric behaviour is
  unclear, read the installed reference at
  `.venv/lib/python3.14/site-packages/rapidfuzz/*_py.py`, and run it rather than
  reasoning about what upstream does. Preserve its edge cases and adaptive fuzzy
  weighting while expressing them through this package's API.
- `normalizeText` follows `utils.default_process`, including treating underscore
  as non-alphanumeric.
- **Tests are ported from RapidFuzz's own suite**, under `tests/`, with a header
  comment naming the source file. Port the assertions faithfully, including the
  regression tests named after upstream issue numbers — the number is the reason
  the test exists.
- **Naming is camelCase, and the algorithms use RapidFuzz's vocabulary**: the
  fuzz scorers are `ratio`, `partialRatio`, `tokenSortRatio`, `tokenSetRatio`,
  `weightedRatio`, matching what upstream computes under `fuzz.*`. The casing is
  still ours: never restore the spellings `WRatio`, `QRatio` or `extractOne`, and
  `weightedRatio` is deliberately not a transliteration of `WRatio`.
  Orchestration keeps its own names — `createScorer`, `createMatcher`,
  `bestMatch`, `search`, `normalizeText` — because it describes this API rather
  than an upstream algorithm.
- **`QRatio` is deliberately absent, and stays absent.** It is upstream's `ratio`
  with one difference — two empty strings score `0` rather than `100` — because
  its processor became opt-in in RapidFuzz v3 and left the fuzzywuzzy
  empty-string rule behind. It is a compatibility shim rather than an algorithm,
  and `fuzzball` does not ship it either. Do not re-litigate this without new
  evidence.
- **Correctness before speed.** A straightforward DP matching a bit-parallel or
  SIMD upstream kernel is acceptable; optimise later, behind the ported tests.

## No type casting

Banned everywhere — `src/`, tests, config files: `x as T` (including
`as unknown as T`), `<T>x`, and `x!`. All three silence the type checker without
proving anything, which is exactly the failure mode a library's public types
exist to prevent. Fix the type instead: widen a parameter, add a generic, write
a type guard (`function isFoo(x: unknown): x is Foo`), or use `satisfies`.
`.oxlintrc.json` turns all three into lint errors, and `pnpm lint` reads every
file the config does not ignore rather than `src/` alone.

`as const` is **allowed** and deliberately exempt: it narrows, which is the
opposite of what the ban prevents.

The tooling under `scripts/` is plain `.mjs` — each is a shell entry point that
has to run against a checkout which may not have built anything — so it carries
`// @ts-check` and JSDoc types, checked by `tsconfig.scripts.json`. There:

- `/** @type {T} */ const x = …` on a declaration is an **annotation** — fine,
  and the way to type an accumulator or an empty array.
- `/** @type {T} */ (expr)` around an expression is a **cast** — banned. No lint
  rule can read a comment, so convention is all that holds it.
- `@ts-ignore`, `@ts-expect-error` and `@ts-nocheck` are banned, and
  `typescript/ban-ts-comment` enforces all three.

When a runtime check proves something the checker cannot see, restructure so the
proof is visible — `filter((x) => x !== undefined)` narrows where
`some((x) => x === undefined)` followed by a throw does not.

`noUncheckedIndexedAccess` is deliberately **off**: with it on, every indexed
read in the numeric DP loops under `src/algorithms/` would need a `!` or a
`?? 0`. Turning it off removes the need for the assertion rather than hiding it.

## Type parameters are named, and `T`-prefixed

Every type parameter is a `T`-prefixed noun saying what it holds — `TItem`,
`TKey`, `TDirection`, `TBrand`, `TConfig`, `TKind`, `TArray`, `TMetric`,
`TScorer`. A bare letter is banned: these names surface in the published
`.d.ts`, in editor tooltips and in the generated API reference, where `<T, D, B>`
tells a consumer nothing.

## Type imports go at the top of the file

`import type { Direction } from '../core/types.js'`, never inline as
`import('../core/types.js').Direction` in a signature or annotation. Both erase
to nothing, so this is about reading the file: the top-level form puts every
dependency in one place and lets a rename be fixed once. No lint rule catches
it.

## `{@link}` names a symbol, never a module path

`{@link import('./tokenSort.js').tokenSortRatio}` does not resolve.
TypeDoc renders it verbatim, so the published reference reads
"The higher of import('./tokenSort.js').tokenSortRatio and …" — a relative
path into our source, shown to someone who imported a package.

Write `{@link name}` when the symbol is in the same file, and a plain
`` `name` `` code span across modules. A code span is worth more than a broken
link: the name is what a reader searches for, and the sidebar is one click away.

## Clean code wins everywhere except a hot loop

The default is the clearer shape: one function rather than six copies, a named
helper rather than an inlined body, a rule stated once rather than re-derived at
each call site. Duplication is what drifts, and a rule that exists in six places
is a rule that will be wrong in one of them.

Speed overrides that in exactly two cases:

- **Inside a hot loop** — a body that runs once per candidate, per cell, per
  character. `bestSimilarity` and `bestDistance` are two literal loops rather
  than one comparator for that reason, and so are `topSimilarity`/`topDistance`
  and the batch cell loops.
- **A large, measured gain.** Large means it survives `pnpm bench:confirm` and
  is worth writing in a commit message.

Everything else loses to the cleaner shape, including a few percent. A `for`
loop is not hot because it is a loop: `search/`'s option-and-threshold preamble,
the missing-query walk, and construction bookkeeping all run once per call or
once per collection, and belong in one place whatever they cost. Where a hot
loop does keep a duplicated body, say in a comment what was measured — a reader
must be able to tell a deliberate copy from an un-refactored one.

Two habits follow from this:

- **Measure before defending a duplicate, not after.** "It might be slower" is
  not a reason; a confirmed number is. Most of these turn out to be neutral —
  the shared threshold helpers moved 0 of 46 cases.
- **Measure after removing one, too**, if it sits anywhere near a scan. A shared
  walk that added one call per choice cost 5% of `createMatcher` construction,
  which was worth paying and worth knowing.

## Code speaks for itself

Write almost no internal comments. A better name, a named intermediate, or an
extracted function explains more reliably than prose beside the code does — a
comment that restates the line is one more thing that can drift out of true,
and it will.

The one exception is genuinely kernel-hard code: a bit-parallel recurrence, a
carry chain, an aliasing or ordering constraint that is invisible in the
expression. There, two or three lines saying what the reader cannot derive.
Nowhere else, and never the investigation that produced the code — measurement
narratives belong in the commit message, not beside the loop.

This governs the **inside** of a module. The public surface is the exception,
and it is a hard requirement — see below.

## Every public export carries a JSDoc block

Everything the twelve entry points export — **functions, constants, types,
interfaces, and the members of an interface** — is documented, with no
exceptions. This is the opposite rule to the one above, and deliberately: an
internal comment competes with a better name, while a `.d.ts` with no prose
leaves a consumer nothing at all. It is what a reader sees on hover, and what
the generated API reference renders.

A block earns its place by carrying what the signature cannot:

- **What it measures or does**, in one sentence a reader can act on.
- **The scale**, whenever a number comes back — `0..100`, `0..1`, or a count.
  A raw `similarity` is not a percentage and has to say so.
- **A worked example** for anything with a choice to make. Take the numbers
  from running the built package, never from memory; a wrong figure in a doc is
  worse than no figure.
- **`@throws` with the real error class.** `TypeError` and `RangeError` are not
  interchangeable here — check which one the code actually raises.
- **The trap.** Say where the export is the wrong tool: token-set containment
  scoring a flat `100`, `bestMatch` returning the least-bad candidate without a
  threshold. A reference that only lists strengths is how the wrong scorer gets
  chosen.

Two mechanical points, both of which have silently swallowed docs here:

- On an **overloaded** function the block goes above the _first_ signature —
  that is the one editors and TypeDoc read.
- An **interface with documented properties is not a documented interface**.
  Hovering the type name shows the block above `interface`, and nothing else.

New exports arrive documented. To check the whole surface, read the emitted
`dist/**/*.d.ts` rather than the sources: that is what a consumer compiles
against, and it is where a lost or misplaced block shows up.

## Coverage is 100%, and that is enforced

`pnpm coverage` fails below 100% on all four metrics, and CI runs the same
script. The number is only meaningful because unreachable code is **deleted
rather than excused**: `src/` carries **no `/* v8 ignore */` at all**, and that
is the state to keep it in.

## Library constraints

Load-bearing for bundle size — see README.md for the full rationale:

- `"sideEffects": false` — nothing in `src/` may do work at import time.
- Export standalone named functions, never namespace objects.
- No runtime dependencies, and no Node built-ins in `src/`.

Nothing measures bytes. A `check:bundles` with sixteen recorded gzip budgets was
removed because the number it produced was never one to act on: the budgets
drifted under ordinary work, a different toolchain gzipped the same bundle
20-30 B apart, and CI failed entries that passed locally. Measure with a bundler
over a real import when you want a figure, and treat it as a figure rather than
a gate.

## Public API and dependency ownership

The root exports orchestration only: scorer creation, matching/search, batch
scoring, threshold helpers, and text normalization. Algorithms are imported from
canonical subpaths such as `rapidfuzz-js/levenshtein` and `rapidfuzz-js/fuzz`;
never re-export them from the root.

Preserve each operation's RapidFuzz scale. Fuzz similarities return `0..100`;
raw edit/count distances and similarities use native units; normalized measures
and the Jaro family return `0..1`. Generic infrastructure must not rescale
between families.

Source ownership follows dependency direction, guarded in `tests/architecture/`:

- `core/` never imports algorithms, and is itself two layers. Directly under it
  are the primitives — `types`, `sequence`, `normalize`, `options` — and
  `core/scoring/` holds the machinery built on them: metric, compilation,
  choice index, scorer, prepared choice, threshold, match. The edge runs one
  way only, `core/scoring/ -> core/`, so a primitive that reached back into
  scoring would collapse the two into one layer again:

  ```text
  search / batch / algorithms
            ↓
       core/scoring
            ↓
     core primitives
  ```

- Each algorithm directory owns its public metric, compilation, preparation and
  hot kernels. Shared algorithm code is limited to proven low-level data
  structures under `algorithms/shared/`. Where one grows past a single file it
  becomes a directory of peer modules with no barrel, imported file by file —
  `shared/bitmask/`, and `shared/ngram/`, whose layers run
  `key -> packing -> profile -> compare -> kernel` with the optional inverted
  index in `ngram/inverted/` built on top and never reached back into.
  `tests/architecture/imports.test.ts` pins both directory listings and that
  direction. `shared/editops/` is the one directory there with a barrel, and
  deliberately: it is a representation facade — `Editops`, `Opcodes`, their
  types and the constructor helper are one thing to import — where `ngram/`'s
  file-by-file import _is_ its dependency architecture.
- **The algorithms are not all peers.** Three are defined on another, so three
  cross-algorithm edges exist and are allowed by name: `indel -> lcs`
  (Indel distance is `|a| + |b| - 2·LCS`), `jaroWinkler -> jaro` (Jaro plus a
  prefix bonus), and `levenshtein -> lcs` (weighted Levenshtein degenerates to
  scaled Indel when insertion and deletion cost the same positive amount and
  substitution costs at least their sum). `lcs` is the foundation the Indel and
  fuzz families are built on, not a sibling of them. `imports.test.ts` holds
  that allow-list and fails both ways — a fourth edge, or an entry no longer
  used. It scopes itself to `algorithms/`, so the `fuzz -> lcs` edges are a
  different question and are not on this list.
- `fuzz/` is split by scorer family; basic similarity must not import token or
  adaptive-fuzzy modules.
- `search/` and `batch/` depend only on core protocols, never on named
  algorithms, and execute private prepared kernels rather than public
  `Scorer.score()` calls.

Do not add compatibility aliases, aggregate namespaces, raw prepared
representations, or a horizontal `common`/`preparation` subsystem. Delete unused
code; do not keep it for a removed API.

The opaque `PreparedChoice` from `scorer.prepareChoice` is the sanctioned form
of a public prepared handle, and the only one. It stays opaque, keeps its metric
brand in the declarations a consumer compiles against, and is checked at runtime
for scorer compatibility — `pnpm check:consumer` guards that second part, which
source-level type tests cannot see.

Any type the public API can _infer_ into a consumer's exported signature must be
nameable without importing our internals: exported from a public entrypoint, or
built from language-native constructs — the metric brand is the id literal
itself for exactly this reason. A type that fails this rule typechecks
everywhere and breaks only in the consumer's own declaration emit, as a deep
`import("…/dist/…")`; `check-consumer.mjs` catches it only for what its fixture
exports, so a new inference surface gets an unannotated `export const` there in
the same change.

Prepared search must not add a per-candidate cost to the text search path.
`bench/process.bench.ts` is where that is noticed.

Shared infrastructure owns policy and metadata; an algorithm module declares
algorithm facts and nothing else. A metric names itself once, as
`BuiltInMetric<'levenshtein.distance', 'distance'>`, and the name derives the
compile-time brand and nothing else — the adapter owns preparation identity,
configuration and the missing-value policy. Those names are compile-time
discriminators: nothing exposes one at runtime and no API accepts one, so they
may appear in a diagnostic but never in a call. The converse also holds — a
driver may duplicate mechanics deliberately where that keeps a hot loop
monomorphic, which is why `bestDistance` and `bestSimilarity` are two literal
loops and not one comparator.

## Test ownership

A test's location says what owns the invariant it protects. The rule is one
line: **one owner colocates, many owners go up**.

Production TypeScript and its module-owned tests live together under `src/`,
the test named `<module>.test.ts` beside what it covers, or
`<module>.<aspect>.test.ts` where one module needs several files —
`metric.test.ts`, `metric.fastPath.test.ts`, `metric.weightedBand.test.ts`.
Never `levenshtein2.test.ts` or `…Advanced.test.ts`; a numbered name says
nothing about what broke.

Do not name a file after a module it does not own. A test that reaches an
implementation, its public metric, the scorer and search is not
`implementation.test.ts` — give it the directory's own name until it is split,
because a name the contents contradict is worse than a vague one.

Reusable test machinery lives under `testing/`, and stays small: no test cases,
no domain-specific suites, no production code, and no algorithm-specific helper
unless more than one suite shares it. Shipping source must never import it —
every relative import made by a non-test file under `src/` resolves to another
non-test file under `src/`. That rule exists because the architecture suite
filters `*.test.ts` out of the source graph, which would otherwise let a
production module import one unseen.

Only tests whose subject is the library as a whole, or which span independent
subsystems, live under `tests/`: architecture rules, public-surface and
cross-algorithm contracts, RapidFuzz parity, and their fixtures. The admission
question is not "does it touch several files" but **would moving it beside one
implementation make its ownership misleading?** If not, colocate it — `tests/`
is meant to stay small, and a mirror of `src/` under it is the shape this
layout exists to prevent.

## Benchmarks

Read the `benchmarks` skill before running any `pnpm bench*` script, before
editing a bench file or anything under `bench/tooling/`, and before quoting a
number. Two rules bite before it loads: a bench file is hashed into its own
cases, so run `pnpm bench:compare` on it _before_ editing it, and the same holds
for `bench/tooling/harness.ts`, `corpus.ts` and `runner.ts`, where any edit
invalidates the entire baseline.
