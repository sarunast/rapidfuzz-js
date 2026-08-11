# rapidfuzz-js

[![CI](https://github.com/sarunast/rapidfuzz-js/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sarunast/rapidfuzz-js/actions/workflows/ci.yml)
[![coverage](https://codecov.io/gh/sarunast/rapidfuzz-js/branch/main/graph/badge.svg)](https://codecov.io/gh/sarunast/rapidfuzz-js)
[![npm](https://img.shields.io/npm/v/rapidfuzz-js)](https://www.npmjs.com/package/rapidfuzz-js)
[![bundle size](https://img.shields.io/bundlejs/size/rapidfuzz-js@latest)](https://bundlejs.com/?q=rapidfuzz-js)
[![license](https://img.shields.io/npm/l/rapidfuzz-js)](./LICENSE)

Fast fuzzy string matching for JavaScript and TypeScript, based on
[RapidFuzz](https://github.com/rapidfuzz/RapidFuzz).

- Compatible with RapidFuzz's scoring algorithms
- Works in Node.js 22+, browsers, and edge runtimes
- Ships an ESM build with TypeScript declarations
- Has no runtime dependencies
- Supports tree-shaking

## Install

```sh
npm install rapidfuzz-js
```

## Quick start

Import individual functions from the package root:

```ts
import { extractOne, levenshteinDistance, ratio } from 'rapidfuzz-js'

ratio('this is a test', 'this is a test!')
// 96.55172413793103

levenshteinDistance('lewenstein', 'levenshtein')
// 2

extractOne('new york mets', ['new york mets', 'atlanta braves'])
// { choice: 'new york mets', score: 100, key: 0 }
```

The Python-style namespace imports are also available:

```ts
import * as Indel from 'rapidfuzz-js/distance/Indel'
import * as fuzz from 'rapidfuzz-js/fuzz'
import * as search from 'rapidfuzz-js/search'

Indel.distance('kitten', 'sitting')
fuzz.tokenSortRatio('red green blue', 'blue red green')
search.scoreMatrix(['alpha'], ['alfa'])
```

The package is ESM only. CommonJS callers reach it through dynamic `import`,
which is asynchronous — a `require` will not work, and neither will a top-level
`await` in a `.cjs` file:

```js
async function main() {
  const { ratio } = await import('rapidfuzz-js')
  ratio('this is a test', 'this is a test!')
}
```

## Usage

### Find the best matches

```ts
import { extract, extractOne } from 'rapidfuzz-js'

const choices = ['Atlanta Falcons', 'New York Jets', 'New York Giants']

extractOne('new york jet', choices)
// { choice: 'New York Jets', score: 72, key: 1 }

extract('new york', choices, { limit: 2, scoreCutoff: 60 })
// Up to two highest-scoring matches at or above 60
```

`extract`, `extractOne`, and `extractIter` accept arrays, maps, plain objects,
sets, generators, and other iterables. A bare string is rejected because it is
almost always a single choice passed by mistake.

Search results use named fields:

```ts
type ExtractResult<T, K> = {
  choice: T
  score: number
  key: K
}
```

The key is the array or iterable index, the `Map` key, or the property name of
a plain object. `extractOne` returns `undefined` when no choice meets
`scoreCutoff`.

### Build a score matrix

```ts
import { ratio, scoreMatrix } from 'rapidfuzz-js'

const matrix = scoreMatrix(['cat', 'dog'], ['cats', 'dogs'], {
  scorer: ratio,
})

matrix.rows // 2
matrix.cols // 2
matrix.at(0, 0) // score for "cat" and "cats"
matrix.data // Float64Array, stored in row-major order
matrix.toArray() // nested JavaScript arrays
```

Use `into` to choose a typed-array format:

```ts
const compact = scoreMatrix(['cat'], ['cats'], {
  scorer: ratio,
  into: 'u8',
  scoreMultiplier: 2.55,
})
```

Available formats are `'f64'`, `'f32'`, `'i32'`, `'i16'`, `'i8'`, `'u32'`,
`'u16'`, `'u8'`, and `'u8c'`. Integer formats round scores half away from zero.
`scorePairs` compares corresponding items and returns the typed array directly.

All search functions are synchronous.

### Prepare choices once for many queries

`extract*` prepares the query and streams the choices, which is right for one
call and wasteful for a run of them: the processor runs over every choice every
time, and the token scorers split, deduplicate, sort and rejoin every choice
every time. `prepareChoices` moves that to a single pass, and the resulting
index is passed to `extract*` in place of the collection it was built from:

```ts
import { extractOne, prepareChoices, tokenSortRatio } from 'rapidfuzz-js'

const index = prepareChoices(titles, {
  scorer: tokenSortRatio,
  processor: defaultProcess,
})

for (const query of queries) extractOne(query, index)
```

The scorer and the processor are baked in, because they decide what a prepared
choice may hold; naming a different one on a later call is a `TypeError`.
`scoreCutoff`, `scoreHint` and `limit` are unaffected and stay per-call, and the
results — score, `choice` and `key` alike — are identical to those of the
collection, given the two things an index has to assume:

- **The choices do not change after it is built.** Not the collection, and not
  the contents of a choice that is itself mutable — pushing to an array choice
  or rewriting an element of one leaves the prepared state describing what that
  choice used to be, while the result still hands you the array. Strings, being
  immutable, cannot hit this.
- **The processor is deterministic.** `extract` runs it once per choice per
  query and an index runs it once per choice, so a processor that counts calls
  or reads a clock sees a different sequence and is entitled to answer
  differently.

Worth it for a list queried more than once. On 2000 five-word choices with
`defaultProcess`, the default `wRatio` runs at `0.48` of its unindexed time and
`tokenSortRatio` at `0.17`; without a processor, `0.56` and `0.25`. Building the
index costs well under one query at that size. A scorer with nothing per-choice
to cache — `ratio` over plain strings and no processor — is unchanged rather
than faster.

The index holds its choices, so it keeps them alive, and it grows a little as
queries ask for derived forms. Build one per list you query repeatedly, not one
per call. It is frozen once built — `values` and `keys` are there to be read,
and choices that change need a new index rather than an edited one.

### Prepare one query or one choice

`prepareChoices` helps callers who go through `extract*`. A caller scoring pairs
directly — a custom ranker, a join, a loop over `wRatio(query, choice)` — pays
for both halves on every call, because nothing holds either. `prepareQuery` and
`prepareChoice` hand over one half each:

```ts
import { prepareChoice, prepareQuery, tokenSortRatio } from 'rapidfuzz-js'

const query = prepareQuery('new york mets', { scorer: tokenSortRatio })
const scores = titles.map((title) => query(title))

const choice = prepareChoice(title, { scorer: tokenSortRatio })
const ranked = queries.map((q) => choice(q))
```

Both return a frozen callable carrying the `scorer` and `processor` it was built
for. The two compose, and that is where they are worth the most — neither half
is prepared twice:

```ts
const prepared = titles.map((t) => prepareChoice(t, { scorer: tokenSortRatio }))
for (const q of queries) {
  const query = prepareQuery(q, { scorer: tokenSortRatio })
  for (const title of prepared) query(title)
}
```

**The operand order never changes.** `query(choice)` and `choice(query)` are
both `scorer(query, choice)` — a handle holds a side, it does not swap the two.
That matters for asymmetric scorers, such as a weighted Levenshtein whose
insertion cost differs from its deletion cost.

Only `scoreCutoff` and `scoreHint` are per call; naming a scorer or processor
there is a type error, and the two halves of a composed call must have been
prepared with the same ones. Measured against a plain `scorer(query, choice)`
loop, 40 queries over 2000 five-word choices, lower is better:

| scorer                | `query(choice)` | composed | `choice(query)` + `defaultProcess` |
| --------------------- | --------------- | -------- | ---------------------------------- |
| `ratio`               | `0.33`          | `0.41`   | `0.57`                             |
| `wRatio`              | `0.64`          | `0.37`   | `0.66`                             |
| `tokenSortRatio`      | `0.48`          | `0.14`   | `0.54`                             |
| `levenshteinDistance` | `0.80`          | `0.85`   | `0.78`                             |

Two differences from `extract*` worth knowing. A missing operand — `null`,
`undefined` or `NaN` — is refused rather than dropped, at build time and on
every call, because a single score has no "skip this one" to return. And a
third-party scorer is called exactly as the loop would have called it:
`scorer(query, choice)` with two arguments when the handle is called with no
options, and with the caller's own options object — not a rebuilt one — when it
is.

**A handle is a snapshot, and freezing the handle does not freeze the operand.**
The `Object.freeze` seals the handle's own properties; an array operand, or a
mutable sequence a processor returned, is left as it was. So the rule is the one
[`prepareChoices`](#prepare-choices-once-for-many-queries) keeps: do not mutate
an operand after preparing it — the prepared state would go on describing what
it used to be — and keep the processor deterministic. Rebuild the handle
instead.

### Configure a scorer

Use `configure` when a search operation needs scorer-specific options:

```ts
import { configure, levenshteinDistance, scoreMatrix } from 'rapidfuzz-js'

const weightedDistance = configure(levenshteinDistance, {
  weights: { insertion: 1, deletion: 1, substitution: 2 },
})

scoreMatrix(['kitten'], ['sitting'], { scorer: weightedDistance })
```

Options passed directly to the configured scorer take precedence. `scoreCutoff`
and `scoreHint` cannot be configured because they apply to each individual call.

### Test a threshold

`matchScore` and `isMatch` avoid the sentinel values returned by bounded
scorers:

```ts
import { isMatch, levenshteinDistance, matchScore, ratio } from 'rapidfuzz-js'

matchScore(ratio, 'martha', 'marhta', { threshold: 80 })
// score or undefined

isMatch(levenshteinDistance, 'kitten', 'sitting', { threshold: 3 })
// true
```

For similarity scorers, `threshold` is a minimum. For distance scorers, it is a
maximum.

### Work with edit operations

```ts
import { levenshteinEditops } from 'rapidfuzz-js'

const edits = levenshteinEditops('kitten', 'sitting')

edits.operations
edits.apply('kitten', 'sitting') // 'sitting'
edits.toOpcodes()
edits.toMatchingBlocks()
edits.inverse()
```

`Editops` and `Opcodes` are readonly results. Their `operations` arrays contain
named records rather than positional tuples.

## API overview

### Distance metrics

The package includes:

- `Indel`
- `LCSseq`
- `Levenshtein`, including weighted operations
- `DamerauLevenshtein`
- `OSA`
- `Hamming`
- `Jaro`
- `JaroWinkler`
- `Prefix`
- `Postfix`

Each metric provides `distance`, `similarity`, `normalizedDistance`, and
`normalizedSimilarity`. Standalone root exports use the metric name as a
prefix, such as `levenshteinDistance` and `jaroSimilarity`.

`Levenshtein`, `Indel`, `LCSseq`, and `Hamming` also provide `editops` and
`opcodes` functions.

### Fuzzy scorers

- `ratio`
- `partialRatio`
- `partialRatioAlignment`
- `tokenSortRatio`
- `tokenSetRatio`
- `tokenRatio`
- `partialTokenSortRatio`
- `partialTokenSetRatio`
- `partialTokenRatio`
- `wRatio`
- `qRatio`

### Search

- `extract`
- `extractOne`
- `extractIter`
- `prepareChoices`
- `prepareQuery`
- `prepareChoice`
- `scoreMatrix`
- `scorePairs`

### Other exports

- `configure`
- `matchScore`
- `isMatch`
- `defaultProcess`
- `Editops`
- `Opcodes`

## Moving from Python RapidFuzz

Function and option names use camelCase. Keyword arguments become an options
object.

| Python RapidFuzz                 | rapidfuzz-js                                              |
| -------------------------------- | --------------------------------------------------------- |
| `rapidfuzz.process`              | `rapidfuzz-js/search`                                     |
| `cdist()`                        | `scoreMatrix()`                                           |
| `cpdist()`                       | `scorePairs()`                                            |
| `dtype='int'`                    | `into: 'i32'`                                             |
| `(choice, score, key)`           | `{ choice, score, key }`                                  |
| `extract_one()` returning `None` | `extractOne()` returning `undefined`                      |
| `scorer_kwargs={...}`            | `scorer: configure(scorer, {...})`                        |
| `weights=(1, 1, 2)`              | `weights: { insertion: 1, deletion: 1, substitution: 2 }` |

Other differences to keep in mind:

- `scoreMatrix` returns a `ScoreMatrix` backed by one typed array, not a NumPy
  array. `scorePairs` returns a typed array.
- Edit operations are readonly objects with named fields.
- Sequence elements are compared with JavaScript's `===`.
- `null`, `undefined`, and `NaN` are treated as missing values. There is no
  pandas integration.
- A raw `scoreCutoff` must be finite. Fractional distance cutoffs are truncated,
  matching RapidFuzz's C++ extension.
- `prepareChoices`, `prepareQuery` and `prepareChoice` have no counterpart.
  RapidFuzz caches the query side of a scorer inside `process` and hands a
  caller neither half; these expose both — the choice side of a whole
  collection, a single query, and a single choice.
- `prepareQuery` and `prepareChoice` refuse a missing operand rather than
  treating it as a missing value, unlike every scorer and `extract*`.

The implementation is tested against RapidFuzz 3.14.5. Where RapidFuzz's C++
and pure-Python implementations disagree, this package follows the public C++
behavior unless it is clearly inconsistent with the rest of RapidFuzz. Known
exceptions cover an exact Jaro-Winkler floating-point boundary and tokenization
of U+0085 and U+00A0 whitespace.

## Performance

The main distance algorithms use bit-parallel kernels. Repeated-query search
also prepares and caches the query representation, character masks, and token
state. For symmetric matrices built from the same array, only one triangle is
calculated.

### Benchmark results

These results were recorded on an M1 Max with Node.js 26.5. Each number is
`rapidfuzz-js time / competitor time`, so a result below `1.00` means
`rapidfuzz-js` was faster. For example, `0.25` means it took one quarter of the
time.

Levenshtein distance:

| Input length     | `fastest-levenshtein` | `leven` | `js-levenshtein` | `fuzzball` |
| ---------------- | --------------------: | ------: | ---------------: | ---------: |
| 8 characters     |                  1.06 |    0.71 |             1.72 |       0.34 |
| 32 characters    |                  0.94 |    0.34 |             0.53 |      0.088 |
| 128 characters   |                  0.75 |    0.11 |             0.19 |      0.065 |
| 1,024 characters |                  0.63 |   0.054 |             0.10 |      0.051 |

For larger tasks, the same benchmark measured these speedups:

| Task                                 | Compared with                   | `rapidfuzz-js` speedup |
| ------------------------------------ | ------------------------------- | ---------------------: |
| `ratio` over 200 sentence pairs      | `fuzzball`                      |                    16× |
| `ratio` over 200 sentence pairs      | `string-similarity`             |                    27× |
| Best of 2,000 choices for 20 queries | `fuzzball`                      |                   5.1× |
| Best of 2,000 choices for 20 queries | `string-similarity`             |                    21× |
| Best of 2,000 choices for 20 queries | `fuse.js` with a prebuilt index |                    59× |

The `fuzzball` comparisons are the closest like-for-like results. The
`string-similarity` and `fuse.js` rows compare complete tasks that use different
matching algorithms.

Against RapidFuzz 3.14.5 for Python, using its C++ extension:

| Task                                        | `rapidfuzz-js time / Python time` |
| ------------------------------------------- | --------------------------------: |
| Levenshtein distance, 200 × 8 characters    |                              0.57 |
| `ratio`, 200 sentence pairs                 |                              0.90 |
| Levenshtein distance, 200 × 32 characters   |                              1.22 |
| Levenshtein distance, 200 × 128 characters  |                              1.79 |
| Levenshtein distance, 25 × 1,024 characters |                              2.20 |
| Best of 2,000 choices with `extractOne`     |                              1.84 |
| `scoreMatrix`, 50 × 200 comparisons         |                              3.47 |

Python RapidFuzz is faster on most substantial workloads, particularly when it
can keep the full operation inside C++. `rapidfuzz-js` is intended for Node.js,
browsers, and edge runtimes where using the Python extension is not practical.

The benchmark suite includes comparisons with other JavaScript libraries and
Python RapidFuzz:

```sh
pnpm build
pnpm install --dir bench/comparison
pnpm bench:libraries
```

To include Python RapidFuzz:

```sh
python3 -m venv .venv
.venv/bin/pip install rapidfuzz numpy
node bench/comparison/run.mjs --python=.venv/bin/python
```

## Development

```sh
pnpm install
pnpm test
pnpm build
pnpm lint
pnpm format
pnpm check
```

Useful benchmark commands:

```sh
pnpm bench                 # standard benchmark run
pnpm bench:quick           # shorter feedback loop
pnpm bench:compare         # compare with bench/baseline.json
pnpm bench:compare:quick   # quick comparison with wider tolerance
pnpm bench:baseline        # record a new baseline
pnpm bench:libraries       # compare with other libraries
```

Benchmark commands accept a file filter and Vitest's `-t` case-name filter:

```sh
pnpm bench:quick fuzz -t 'partialRatio'
pnpm bench:compare:quick -t 'indelDistance'
pnpm bench:compare bench/fuzz.bench.ts
```

The test suite includes RapidFuzz's upstream cases, property-based checks
against reference implementations, and boundary tests for the optimized
kernels.

### Build constraints

These choices keep browser bundles small and portable:

- `package.json` must keep `"sideEffects": false`.
- Source modules must not perform work at import time.
- The build emits one output file per source file.
- Public functions are exported individually for tree-shaking.
- Runtime code must not depend on Node.js built-ins or third-party packages.

See [CLAUDE.md](CLAUDE.md) for parity and implementation rules.

## License

MIT. See [LICENSE](LICENSE).

This project is derived from
[RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) and
[rapidfuzz-cpp](https://github.com/rapidfuzz/rapidfuzz-cpp). Their copyright and
license notices are included in `LICENSE`.
