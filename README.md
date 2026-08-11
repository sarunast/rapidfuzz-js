# rapidfuzz-js

[![CI](https://github.com/sarunast/rapidfuzz-js/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sarunast/rapidfuzz-js/actions/workflows/ci.yml)
[![coverage](https://codecov.io/gh/sarunast/rapidfuzz-js/branch/main/graph/badge.svg)](https://codecov.io/gh/sarunast/rapidfuzz-js)
[![npm](https://img.shields.io/npm/v/rapidfuzz-js)](https://www.npmjs.com/package/rapidfuzz-js)
[![license](https://img.shields.io/npm/l/rapidfuzz-js)](./LICENSE)

Fast fuzzy matching for JavaScript and TypeScript, powered by the algorithms of
[RapidFuzz](https://github.com/rapidfuzz/RapidFuzz).

- Node.js 22+, browsers, and edge runtimes
- ESM with strict TypeScript declarations
- No runtime dependencies
- Tree-shakeable algorithm subpaths

## Install

```sh
npm install rapidfuzz-js
```

## The API in one minute

Version 0.6 follows one composition model:

```text
Metric → Scorer object → Matcher
```

Import metrics from algorithm subpaths and orchestration from the package root:

```ts
import { createMatcher, createScorer, normalizeText } from 'rapidfuzz-js'
import { tokenSortSimilarity } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(tokenSortSimilarity)

const products = [
  { title: 'Wireless mechanical keyboard' },
  { title: 'Compact wireless mouse' },
]

const matcher = createMatcher(products, {
  scorer,
  getText: (product) => product.title,
  normalize: normalizeText,
})

matcher.best('mechanical keybord', { threshold: 70 })
// { item: products[0], key: 0, score: ... }
```

Use `bestMatch` or `search` for one ranked query, `searchIter` for lazy
source-order results, and a `Matcher` when the same collection will receive
many queries.

## Metrics and score scales

Metrics are directly callable:

```ts
import { similarity as fuzzySimilarity } from 'rapidfuzz-js/fuzz'
import {
  distance as levenshteinDistance,
  normalizedSimilarity as levenshteinNormalizedSimilarity,
  similarity as levenshteinSimilarity,
} from 'rapidfuzz-js/levenshtein'

fuzzySimilarity('this is a test', 'this is a test!')
// 96.55172413793103 (0–100)

levenshteinDistance('lewenstein', 'levenshtein')
// 2 (native edit count)

levenshteinSimilarity('abc', 'axc')
// 2 (raw maximum-minus-distance similarity)

levenshteinNormalizedSimilarity('abc', 'axc')
// 0.6666666666666667
```

The library never rescales between families:

| Operation                              | Scale                  |
| -------------------------------------- | ---------------------- |
| Fuzz similarities                      | `0–100`                |
| Raw edit/count distance and similarity | Native algorithm units |
| Normalized distance and similarity     | `0–1`                  |
| Jaro and Jaro-Winkler measures         | `0–1`                  |

Available subpaths:

```text
rapidfuzz-js/fuzz
rapidfuzz-js/levenshtein
rapidfuzz-js/indel
rapidfuzz-js/lcs
rapidfuzz-js/osa
rapidfuzz-js/damerau-levenshtein
rapidfuzz-js/hamming
rapidfuzz-js/jaro
rapidfuzz-js/jaro-winkler
rapidfuzz-js/prefix
rapidfuzz-js/postfix
```

Every algorithm subpath exposes `distance`, `similarity`,
`normalizedDistance`, and `normalizedSimilarity`. Levenshtein, Indel, LCS,
and Hamming also export `editops` and `opcodes`.

## Scorer objects

`createScorer` freezes direction, bounds, symmetry, algorithm configuration,
and private preparation hooks into a reusable object:

```ts
import { createScorer } from 'rapidfuzz-js'
import { distance } from 'rapidfuzz-js/levenshtein'

const weighted = createScorer(distance, {
  weights: { insertion: 1, deletion: 1, substitution: 2 },
})

weighted.direction // 'distance'
weighted.bounds // [0, Infinity]
weighted.symmetric // true
weighted.score('kitten', 'sitting') // 5
weighted.score('kitten', 'sitting', { threshold: 3 }) // undefined
```

Similarity thresholds are minimums. Distance thresholds are maximums. A
threshold uses the scorer's own scale and must be finite.

`scoreIfMatch` provides the thresholded result as a standalone operation;
`isMatch` returns only the boolean.

## One query or many

One-shot search streams its input and does not retain the collection:

```ts
import { bestMatch, createScorer, search, searchIter } from 'rapidfuzz-js'
import { fuzzySimilarity } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(fuzzySimilarity)
const teams = ['Atlanta Falcons', 'New York Jets', 'New York Giants']

bestMatch('new york jet', teams, { scorer })
search('new york', teams, { scorer, threshold: 60, limit: 2 })

for (const match of searchIter('new york', teams, {
  scorer,
  threshold: 60,
})) {
  // qualifying matches arrive lazily in source order
}
```

A Matcher snapshots searchable sequences and prepares them once:

```ts
import { createMatcher } from 'rapidfuzz-js'

const matcher = createMatcher(teams, { scorer })

matcher.size // 3
matcher.best('new york jet')
matcher.search('new york', { limit: null }) // every result, best first
matcher.searchIter('new york', { threshold: 60 }) // lazy, source order
```

Arrays and iterables use source positions as keys. Maps retain map keys. Plain
objects retain property names. Missing source items and missing `getText`
results are skipped by default without compacting those keys; use
`missingItems: 'throw'` to reject them instead.

Strings are retained. Non-string array-like sequences are shallow-copied into
Matcher-owned storage, so later top-level mutations do not change search
scores. Returned items and nested element objects remain live references.

## Matrices and paired scoring

```ts
import { scoreMatrix, scorePairs } from 'rapidfuzz-js'

const matrix = scoreMatrix(['cat', 'dog'], ['cats', 'dogs'], { scorer })
matrix.rows
matrix.cols
matrix.at(0, 0)
matrix.data // row-major Float64Array

scorePairs(['cat', 'dog'], ['cats', 'dogs'], { scorer })
```

Set `into` to `f64`, `f32`, `i32`, `i16`, `i8`, `u32`, `u16`, `u8`, or `u8c`
to select the typed-array storage. Batch `threshold` uses the scorer's natural
unscaled domain; `scoreMultiplier` is applied afterward:

```ts
import { normalizedSimilarity } from 'rapidfuzz-js/levenshtein'

const normalized = createScorer(normalizedSimilarity)
scoreMatrix(['cat'], ['cats'], {
  scorer: normalized,
  threshold: 0.5,
  scoreMultiplier: 100,
  into: 'u8',
})
```

## RapidFuzz capability mapping

The API preserves RapidFuzz's mathematical operations while using
JavaScript-native orchestration:

| RapidFuzz                    | rapidfuzz-js                          |
| ---------------------------- | ------------------------------------- |
| `process.extractOne`         | `bestMatch`                           |
| `process.extract`            | `search`                              |
| `process.extract_iter`       | `searchIter`                          |
| `process.cdist` / `cpdist`   | `scoreMatrix` / `scorePairs`          |
| `score_cutoff`               | `threshold`                           |
| `score_multiplier`           | `scoreMultiplier`                     |
| `scorer_kwargs`              | `createScorer(metric, configuration)` |
| repeated prepared extraction | `createMatcher`                       |

Public `processor`, `scoreHint`, worker, and prepared-handle APIs are omitted.
Normalization belongs at search/batch boundaries, and preparation belongs to a
Matcher. QRatio is intentionally omitted because it only changes ratio's
empty-input compatibility result; WRatio remains available as
`fuzzySimilarity`.

## Missing and invalid values

Only `null` and `undefined` are missing. Similarity scorers return `0` for a
missing operand by default:

```ts
const strict = createScorer(fuzzySimilarity, { missing: 'throw' })
strict.score(null, 'text') // throws TypeError
```

Distance scorers always throw on missing operands. Empty sequences are valid.
Numbers (including `NaN`), booleans, and objects without a valid array-like
`length` are invalid.

## Custom metrics

Custom metrics must declare enough metadata for safe ordering and validation:

```ts
const custom = createScorer((a, b) => (a === b ? 1 : 0), {
  direction: 'similarity',
  bounds: [0, 1],
  symmetric: true,
})
```

Every custom result must be finite and inside its declared bounds. The result
is validated before thresholding, ordering, or pruning.

## Performance and package validation

The benchmark vocabulary maps directly to the public API:

| Workload                      | API                              |
| ----------------------------- | -------------------------------- |
| One best result               | `bestMatch`                      |
| Top N results                 | `search` with `limit: N`         |
| Lazy qualifying results       | `searchIter`                     |
| Prepare a reusable collection | `createMatcher` construction     |
| Repeated prepared query       | `matcher.best/search/searchIter` |

The release check runs type checking, linting, formatting, all functional
tests, the build, export-map validation, package validation, and tarball
inspection. Source maps are shipped with embedded source content; TypeScript
source files are not included in the package.

## License

[MIT](./LICENSE)
