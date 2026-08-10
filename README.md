# rapidfuzz-js

Fast fuzzy string matching for JavaScript and TypeScript, based on
[RapidFuzz](https://github.com/rapidfuzz/RapidFuzz).

- Compatible with RapidFuzz's scoring algorithms
- Works in Node.js 22+, browsers, and edge runtimes
- Ships ESM and CommonJS builds with TypeScript declarations
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

CommonJS is supported:

```js
const { ratio } = require('rapidfuzz-js')
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
