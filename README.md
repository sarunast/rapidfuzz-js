# rapidfuzz-js

[![CI](https://github.com/sarunast/rapidfuzz-js/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sarunast/rapidfuzz-js/actions/workflows/ci.yml)
[![coverage](https://codecov.io/gh/sarunast/rapidfuzz-js/branch/main/graph/badge.svg)](https://codecov.io/gh/sarunast/rapidfuzz-js)
[![npm](https://img.shields.io/npm/v/rapidfuzz-js)](https://www.npmjs.com/package/rapidfuzz-js)
[![license](https://img.shields.io/npm/l/rapidfuzz-js)](./LICENSE)

Fast fuzzy matching for JavaScript and TypeScript, powered by the algorithms of
[RapidFuzz](https://github.com/rapidfuzz/RapidFuzz).

- Node.js 22+, browsers, and edge runtimes
- ESM with strict TypeScript declarations, TypeScript 5.4+
- No runtime dependencies, `"sideEffects": false`
- One subpath per algorithm — import Jaro, pay for Jaro

**[Documentation](https://sarunast.github.io/rapidfuzz-js/)** ·
[API reference](https://sarunast.github.io/rapidfuzz-js/api/) ·
[Benchmarks](https://sarunast.github.io/rapidfuzz-js/benchmarks/) ·
[Errors](https://sarunast.github.io/rapidfuzz-js/reference/errors/)

## Install

```sh
npm install rapidfuzz-js
```

## Quick start

Compare two strings — metrics are plain functions:

```ts
import { ratio, tokenSortRatio } from 'rapidfuzz-js/fuzz'

ratio('this is a test', 'this is a test!') // 96.55
tokenSortRatio('fuzzy wuzzy was a bear', 'wuzzy fuzzy was a bear') // 100
```

Find the best match in a list — wrap a metric in a scorer:

```ts
import { bestMatch, createScorer, normalizeText, search } from 'rapidfuzz-js'
import { weightedRatio } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(weightedRatio)
const teams = ['Atlanta Falcons', 'New York Jets', 'New York Giants']

bestMatch('new york jet', teams, { scorer, normalize: normalizeText })
// { item: 'New York Jets', key: 1, score: 96 }

search('new york', teams, { scorer, threshold: 60, limit: 2 })
// [{ item: 'New York Jets', key: 1, score: 68.4 }, { item: 'New York Giants', … }]
```

Query the same collection many times — build a `Matcher`, which prepares it
once:

```ts
import { createMatcher } from 'rapidfuzz-js'

const products = [
  { title: 'Wireless mechanical keyboard' },
  { title: 'Compact wireless mouse' },
]

const matcher = createMatcher(products, {
  scorer: createScorer(tokenSortRatio),
  getText: (product) => product.title,
  normalize: normalizeText,
})

matcher.best('mechanical keybord', { threshold: 70 })
// { item: { title: 'Wireless mechanical keyboard' }, key: 0, score: 78.26… }
```

Scoring is case-sensitive unless you pass `normalize`. Without a `threshold`,
`bestMatch` returns the least-bad candidate, however bad it is.

## Algorithms

Each algorithm is its own subpath and exports `distance`, `similarity`,
`normalizedDistance` and `normalizedSimilarity` (except `fuzz`, which exports
similarity scorers only). Scores keep RapidFuzz's scale for their family and
are never rescaled.

| Subpath                            | Measures                                               | Scale          |
| ---------------------------------- | ------------------------------------------------------ | -------------- |
| `rapidfuzz-js/fuzz`                | `ratio`, `partialRatio`, token and weighted scorers    | `0–100`        |
| `rapidfuzz-js/levenshtein`         | insertions, deletions, substitutions; optional weights | native / `0–1` |
| `rapidfuzz-js/indel`               | insertions and deletions only                          | native / `0–1` |
| `rapidfuzz-js/lcs`                 | longest common subsequence                             | native / `0–1` |
| `rapidfuzz-js/osa`                 | Levenshtein plus adjacent transpositions, restricted   | native / `0–1` |
| `rapidfuzz-js/damerau-levenshtein` | Levenshtein plus unrestricted transpositions           | native / `0–1` |
| `rapidfuzz-js/hamming`             | substitutions at equal positions                       | native / `0–1` |
| `rapidfuzz-js/prefix`              | length of the common prefix                            | native / `0–1` |
| `rapidfuzz-js/postfix`             | length of the common suffix                            | native / `0–1` |
| `rapidfuzz-js/jaro`                | Jaro similarity                                        | `0–1`          |
| `rapidfuzz-js/jaro-winkler`        | Jaro with a common-prefix bonus                        | `0–1`          |
| `rapidfuzz-js/dice`                | Sørensen-Dice over n-gram multisets                    | `0–1`          |
| `rapidfuzz-js/cosine`              | cosine of n-gram frequency vectors                     | `0–1`          |
| `rapidfuzz-js/tversky`             | asymmetric, weighted, optionally fuzzy n-gram overlap  | `0–1`          |

"Native" means edit or character counts; the `normalized*` exports are `0–1`.
Levenshtein, Indel, LCS and Hamming also export `editops` and `opcodes`.

**Which fuzz scorer?** `weightedRatio` is the general-purpose choice: it tries
substring and word-order strategies and keeps the best. Pick one strategy
directly when you know the shape of your data — `partialRatio` when one string
contains the other, `tokenSortRatio` for reordered words, `tokenSetRatio` for
overlapping word sets (containment scores a flat `100`). See the
[fuzz guide](https://sarunast.github.io/rapidfuzz-js/algorithms/fuzz/).

## API

Everything below is exported from the package root.

| Export                       | Does                                                                     |
| ---------------------------- | ------------------------------------------------------------------------ |
| `createScorer`               | binds a metric to its direction, bounds and configuration                |
| `isMatch` / `scoreIfMatch`   | one pair against a threshold                                             |
| `bestMatch`                  | the best single match in a collection                                    |
| `search`                     | ranked top matches — five by default, `limit: null` for all              |
| `searchIter`                 | lazily yields every qualifying match, in source order                    |
| `createMatcher`              | a prepared collection with `best`, `search` and `searchIter`             |
| `createIndexedMatcher`       | the same `Matcher` over an inverted n-gram index (Dice, Cosine, Tversky) |
| `scoreMatrix` / `scorePairs` | every query against every choice, or pairwise, into a typed array        |
| `normalizeText`              | lowercases, replaces non-alphanumerics with spaces, trims                |

Collections can be arrays, iterables, `Map`s or plain objects; `key` is the
position, map key or property name. Similarity thresholds are minimums,
distance thresholds maximums, both on the scorer's own scale.

### Scorer configuration

Algorithm options are scorer configuration, not call arguments:

```ts
import { createScorer } from 'rapidfuzz-js'
import { distance } from 'rapidfuzz-js/levenshtein'

const scorer = createScorer(distance, {
  weights: { insertion: 1, deletion: 1, substitution: 2 },
})

scorer.direction // 'distance'
scorer.score('kitten', 'sitting') // 5
scorer.score('kitten', 'sitting', { threshold: 3 }) // undefined
```

A custom function becomes a scorer once it declares its metadata:

```ts
const exact = createScorer((a, b) => (a === b ? 1 : 0), {
  direction: 'similarity',
  bounds: [0, 1],
  symmetric: true,
})
```

### Prepared choices

When you own the collection and filter it yourself, prepare each choice once
and hand the opaque handle back through `getPrepared`:

```ts
import { bestMatch, createScorer, normalizeText } from 'rapidfuzz-js'
import { tokenSetRatio } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(tokenSetRatio)
const rows = ['Acme Corp', 'Globex Corporation'].map((name) => ({
  name,
  prepared: scorer.prepareChoice(name, { normalize: normalizeText }),
}))

bestMatch('acme corp inc', rows, {
  scorer,
  getPrepared: (row) => row.prepared,
  normalize: normalizeText, // must be the normalizer the choices were prepared with
})
// { item: { name: 'Acme Corp', prepared: … }, key: 0, score: 100 }
```

A handle only works with a compatible scorer, and a mismatched normalizer
throws. See [prepared choices](https://sarunast.github.io/rapidfuzz-js/guides/prepared-choices/).

### Batch scoring

```ts
import { createScorer, scoreMatrix, scorePairs } from 'rapidfuzz-js'
import { ratio } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(ratio)
const matrix = scoreMatrix(['cat', 'dog'], ['cats', 'dogs'], { scorer })
matrix.at(0, 0) // 85.71
matrix.data // row-major Float64Array

scorePairs(['cat', 'dog'], ['cats', 'dogs'], { scorer }) // Float64Array [85.71, 85.71]
```

`into: 'u8'` (or `f32`, `i32`, …) picks the storage; a score the type cannot
hold throws a `RangeError` rather than wrapping.

### Missing and invalid values

`null` and `undefined` are missing: similarity scorers score them `0` (or throw
with `missing: 'throw'`), distance scorers always throw. Options objects reject
unknown keys, so a misspelt `thresold` throws instead of being ignored.

## Coming from RapidFuzz

| RapidFuzz                  | rapidfuzz-js                                       |
| -------------------------- | -------------------------------------------------- |
| `fuzz.ratio`               | `ratio`                                            |
| `fuzz.WRatio`              | `weightedRatio`                                    |
| `fuzz.QRatio`              | none — `ratio`, except two empty strings score `0` |
| `process.extractOne`       | `bestMatch`                                        |
| `process.extract`          | `search`                                           |
| `process.extract_iter`     | `searchIter`                                       |
| `process.cdist` / `cpdist` | `scoreMatrix` / `scorePairs`                       |
| `score_cutoff`             | `threshold`                                        |
| `score_multiplier`         | `scoreMultiplier`                                  |
| `scorer_kwargs`            | `createScorer(metric, configuration)`              |
| `processor`                | `normalize`                                        |
| `utils.default_process`    | `normalizeText`                                    |

Upgrading from 0.11? The `fuzz` exports were renamed in 0.12 — see the
[0.12.0 release notes](https://github.com/sarunast/rapidfuzz-js/releases/tag/v0.12.0).

## License

[MIT](./LICENSE)
