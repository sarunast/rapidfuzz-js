# rapidfuzz-js

[![CI](https://github.com/sarunast/rapidfuzz-js/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sarunast/rapidfuzz-js/actions/workflows/ci.yml)
[![coverage](https://codecov.io/gh/sarunast/rapidfuzz-js/branch/main/graph/badge.svg)](https://codecov.io/gh/sarunast/rapidfuzz-js)
[![npm](https://img.shields.io/npm/v/rapidfuzz-js)](https://www.npmjs.com/package/rapidfuzz-js)
[![bundle size](https://img.shields.io/bundlejs/size/rapidfuzz-js@latest)](https://bundlejs.com/?q=rapidfuzz-js)
[![license](https://img.shields.io/npm/l/rapidfuzz-js)](./LICENSE)

Fast fuzzy string matching for JavaScript and TypeScript, based on
[RapidFuzz](https://github.com/rapidfuzz/RapidFuzz).

- Works in Node.js 22+, browsers, and edge runtimes
- ESM with TypeScript declarations
- No runtime dependencies
- Tree-shakeable

## Install

```sh
npm install rapidfuzz-js
```

## Quick start

```ts
import { extractOne, levenshteinDistance, ratio } from 'rapidfuzz-js'

ratio('this is a test', 'this is a test!')
// 96.55172413793103

levenshteinDistance('lewenstein', 'levenshtein')
// 2

extractOne('new york mets', ['new york mets', 'atlanta braves'])
// { choice: 'new york mets', score: 100, key: 0 }
```

## Usage

### Find the best matches

```ts
import { extract, extractOne } from 'rapidfuzz-js'

const teams = ['Atlanta Falcons', 'New York Jets', 'New York Giants']

extractOne('new york jet', teams)
// Best match, or undefined when no result meets scoreCutoff

extract('new york', teams, { limit: 2, scoreCutoff: 60 })
// Up to two matches with a score of at least 60
```

Search results have the following shape:

```ts
type ExtractResult<T, K> = {
  choice: T
  score: number
  key: K
}
```

`extract`, `extractOne`, and `extractIter` accept arrays, maps, plain objects,
sets, generators, and other iterables. The result key is the item index, map
key, or object property name.

### Build a score matrix

```ts
import { ratio, scoreMatrix } from 'rapidfuzz-js'

const matrix = scoreMatrix(['cat', 'dog'], ['cats', 'dogs'], {
  scorer: ratio,
})

matrix.rows // 2
matrix.cols // 2
matrix.at(0, 0) // Score for "cat" and "cats"
matrix.data // Row-major Float64Array
matrix.toArray() // Nested JavaScript arrays
```

Use `into` to select another typed-array format: `f64`, `f32`, `i32`, `i16`,
`i8`, `u32`, `u16`, `u8`, or `u8c`. `scorePairs` compares items at matching
positions and returns a typed array directly.

All search functions are synchronous.

### Reuse prepared inputs

Prepare a collection once when running multiple queries against it:

```ts
import { defaultProcess, extractOne, prepareChoices, tokenSortRatio } from 'rapidfuzz-js'

const index = prepareChoices(titles, {
  scorer: tokenSortRatio,
  processor: defaultProcess,
})

for (const query of queries) {
  extractOne(query, index)
}
```

For custom scoring loops, use `prepareQuery` or `prepareChoice`:

```ts
import { prepareQuery, tokenSortRatio } from 'rapidfuzz-js'

const score = prepareQuery('new york mets', { scorer: tokenSortRatio })
const scores = titles.map((title) => score(title))
```

A prepared value is a snapshot. Do not mutate its inputs after preparation, and
use deterministic processors. Rebuild it when the source data changes.

### Configure a scorer

Use `configure` to bind scorer-specific options before passing a scorer to a
search function:

```ts
import { configure, levenshteinDistance, scoreMatrix } from 'rapidfuzz-js'

const weightedDistance = configure(levenshteinDistance, {
  weights: { insertion: 1, deletion: 1, substitution: 2 },
})

scoreMatrix(['kitten'], ['sitting'], { scorer: weightedDistance })
```

### Test a threshold

```ts
import { isMatch, levenshteinDistance, matchScore, ratio } from 'rapidfuzz-js'

matchScore(ratio, 'martha', 'marhta', { threshold: 80 })
// Score or undefined

isMatch(levenshteinDistance, 'kitten', 'sitting', { threshold: 3 })
// true
```

A threshold is a minimum for similarity scorers and a maximum for distance
scorers.

### Work with edit operations

```ts
import { levenshteinEditops } from 'rapidfuzz-js'

const edits = levenshteinEditops('kitten', 'sitting')

edits.apply('kitten', 'sitting') // 'sitting'
edits.toOpcodes()
edits.toMatchingBlocks()
edits.inverse()
```

`Editops` and `Opcodes` are readonly and use named records instead of tuples.

## Imports

Functions can be imported from the package root. Python-style namespaces are
also available:

```ts
import * as Indel from 'rapidfuzz-js/distance/Indel'
import * as fuzz from 'rapidfuzz-js/fuzz'
import * as search from 'rapidfuzz-js/search'

Indel.distance('kitten', 'sitting')
fuzz.tokenSortRatio('red green blue', 'blue red green')
search.scoreMatrix(['alpha'], ['alfa'])
```

This package is ESM-only. CommonJS code must use asynchronous dynamic import:

```js
async function main() {
  const { ratio } = await import('rapidfuzz-js')
  return ratio('this is a test', 'this is a test!')
}
```

## API overview

| Area             | Exports                                                                                                              |
| ---------------- | -------------------------------------------------------------------------------------------------------------------- |
| Distance metrics | `Indel`, `LCSseq`, `Levenshtein`, `DamerauLevenshtein`, `OSA`, `Hamming`, `Jaro`, `JaroWinkler`, `Prefix`, `Postfix` |
| Fuzzy scorers    | `ratio`, `partialRatio`, `tokenSortRatio`, `tokenSetRatio`, `tokenRatio`, their partial variants, `wRatio`, `qRatio` |
| Search           | `extract`, `extractOne`, `extractIter`, `scoreMatrix`, `scorePairs`                                                  |
| Preparation      | `prepareChoices`, `prepareQuery`, `prepareChoice`                                                                    |
| Utilities        | `configure`, `matchScore`, `isMatch`, `defaultProcess`, `Editops`, `Opcodes`                                         |

Each distance metric provides `distance`, `similarity`, `normalizedDistance`,
and `normalizedSimilarity`. Root exports include the metric name, such as
`levenshteinDistance` and `jaroSimilarity`. `Levenshtein`, `Indel`, `LCSseq`,
and `Hamming` also provide edit operations and opcodes.

## Moving from Python RapidFuzz

JavaScript names use camelCase, and keyword arguments become an options object.

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

Other differences:

- `scoreMatrix` returns a `ScoreMatrix` backed by a typed array, not NumPy.
- Sequence elements are compared with JavaScript's `===`.
- `null`, `undefined`, and `NaN` are treated as missing values.
- Prepared inputs are an extension provided by this package.

The implementation is tested against RapidFuzz 3.14.5 and follows its public
C++ behavior where the C++ and pure-Python implementations differ.

## Performance

Distance algorithms use bit-parallel kernels. Repeated-query search caches
prepared query data, and symmetric matrices compute only one triangle.

On the recorded M1 Max comparison, `rapidfuzz-js` was competitive with the
fastest specialized JavaScript Levenshtein libraries and substantially faster
than `fuzzball` for the measured workloads. Python RapidFuzz remains faster for
most larger workloads that stay inside its C++ extension.

See [Benchmarks](BENCHMARKS.md) for results, methodology, metrics, caveats, and
reproduction instructions.

## Development

Requires Node.js 22+ and pnpm.

```sh
pnpm install
pnpm test
pnpm build
pnpm lint
pnpm check
```

Keep source modules free of import-time work and runtime dependencies. The
package relies on per-module output and `"sideEffects": false` for
tree-shaking. See [CLAUDE.md](CLAUDE.md) for implementation and parity rules.

## License

MIT. See [LICENSE](LICENSE).

This project is derived from
[RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) and
[rapidfuzz-cpp](https://github.com/rapidfuzz/rapidfuzz-cpp). Their notices are
included in the license file.
