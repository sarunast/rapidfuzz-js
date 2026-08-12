# rapidfuzz-js

[![CI](https://github.com/sarunast/rapidfuzz-js/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sarunast/rapidfuzz-js/actions/workflows/ci.yml)
[![coverage](https://codecov.io/gh/sarunast/rapidfuzz-js/branch/main/graph/badge.svg)](https://codecov.io/gh/sarunast/rapidfuzz-js)
[![npm](https://img.shields.io/npm/v/rapidfuzz-js)](https://www.npmjs.com/package/rapidfuzz-js)
[![license](https://img.shields.io/npm/l/rapidfuzz-js)](./LICENSE)

Fast fuzzy matching for JavaScript and TypeScript, powered by the algorithms of
[RapidFuzz](https://github.com/rapidfuzz/RapidFuzz).

- Node.js 22+, browsers, and edge runtimes
- ESM with strict TypeScript declarations, TypeScript 5.4+
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

### Reusable prepared choices

`prepareChoice` returns an opaque handle holding one choice in the form the
scorer's kernels want. Store it beside your own data and hand it back through
`getPrepared`, and a search that would re-prepare every candidate on every
query prepares nothing:

```ts
import { createScorer, normalizeText, searchIter } from 'rapidfuzz-js'
import { tokenSetSimilarity } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(tokenSetSimilarity)
const companies = records.map((record) => ({
  record,
  prepared: scorer.prepareChoice(record.name, { normalize: normalizeText }),
}))

// The guards run before the scorer does, and only survivors are scored.
function* plausible(query: Query) {
  for (const row of companies) {
    if (row.record.country !== query.country) continue
    if (!sharesADigit(row.record.postcode, query.postcode)) continue
    yield row
  }
}

for (const match of searchIter(query.name, plausible(query), {
  scorer,
  getPrepared: (row) => row.prepared,
  // The same normalizer the choices were prepared with. Naming a different one
  // — or none — throws rather than comparing two sides made differently.
  normalize: normalizeText,
})) {
  // scored against handles prepared once, however many queries run
}
```

Normalization is either the library's job on both sides or yours on both
sides, and a handle records which. Pass `normalize` to `prepareChoice` and to
the search, and the two are checked against each other; pass it to neither, and
what you hand in is what gets scored:

```ts
// Library-managed: the handle records the normalizer and the search names the
// same one, so the two are checked against each other.
const prepared = scorer.prepareChoice(name, { normalize: normalizeText })
searchIter(query, rows, { scorer, getPrepared, normalize: normalizeText })

// Caller-managed: you normalize both sides yourself, and the search is told
// about neither.
const prepared = scorer.prepareChoice(normalizeText(name))
searchIter(normalizeText(query), rows, { scorer, getPrepared })
```

Mixing them throws. `prepareChoice(normalizeText(name))` produces the same text
as the first line but records nothing, so a search that normalizes its query
cannot confirm the choice was normalized too and refuses the pair rather than
comparing two sides made differently.

The check is by function identity — two arrows that do the same thing are two
normalizers — so name one function and pass it to both. Identity is the
compatibility token, which makes stable behaviour part of the contract: a
normalizer used with prepared choices must transform a given input the same way
at preparation and at search. One that reads mutable state outside itself
passes the check and still compares two sides made differently, and no check
can see that. Build a configured normalizer as a new function rather than
mutating what an existing one captured.

That ordering is the point: a generator decides what is worth scoring, and the
scoring pays nothing to prepare what it accepts. A handle is read only for the
candidates a guard lets through, so the check costs nothing for the ones it
rejects. `createMatcher` amortizes the
same preparation but owns the collection — it snapshots one field up front, so
there is no place to put a guard and no way to grow it. Here the collection
stays yours. `createMatcher` accepts `getPrepared` too, and resolves every
handle once at construction.

Which scorers accept a handle is decided conservatively, by identity rather
than by proving two preparations equivalent:

| Prepared by                                    | Accepted by                            |
| ---------------------------------------------- | -------------------------------------- |
| a scorer using a metric's default preparation  | any scorer of that metric using it too |
| a scorer with configuration the metric records | that scorer alone                      |
| a custom metric's scorer                       | that scorer alone                      |

So two separately created `createScorer(fuzz.similarity)` scorers share their
handles; treat a configured or custom scorer as owning the handles it made.

Anything else throws: a handle a scorer does not accept is refused as
incompatible, and a value that is not a handle at all is refused as invalid.
Built-in metrics also carry their identity in the type, so most of those
mistakes are compile errors first — spell a stored handle's type with
`PreparedChoiceOf<typeof scorer>`, and a stored scorer's with
`ScorerOf<typeof tokenSetSimilarity>`. The identity is the metric's own id
literal, so your declaration emit spells it
`Scorer<'similarity', 'fuzz.tokenSetSimilarity'>` without importing anything
of ours.
Widening a scorer to `Scorer<'similarity'>` gives that up deliberately: the
type no longer names a metric, so only the runtime check remains.

Prepared mode is strict. There is nothing to skip, so `missingItems` is not
accepted, and neither is `getText` beside it. `normalize` still applies to the
query, but never to a choice — the choice was prepared before the search saw
it, so normalize it yourself when you prepare it.

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

Arrays and iterables use source positions as keys. Maps retain map keys — any
map-shaped value is read as a map, including one typed as an iterable of
entries, since nothing at runtime can tell which of the two was meant. Plain
objects retain property names. Missing source items and missing `getText`
results are skipped by default without compacting those keys; use
`missingItems: 'throw'` to reject them instead. A single string is not a
collection of its characters and is rejected.

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

| RapidFuzz                    | rapidfuzz-js                                                  |
| ---------------------------- | ------------------------------------------------------------- |
| `process.extractOne`         | `bestMatch`                                                   |
| `process.extract`            | `search`                                                      |
| `process.extract_iter`       | `searchIter`                                                  |
| `process.cdist` / `cpdist`   | `scoreMatrix` / `scorePairs`                                  |
| `score_cutoff`               | `threshold`                                                   |
| `score_multiplier`           | `scoreMultiplier`                                             |
| `scorer_kwargs`              | `createScorer(metric, configuration)`                         |
| repeated prepared extraction | `createMatcher`, or `scorer.prepareChoice` with `getPrepared` |

Public `processor`, `scoreHint`, and worker APIs are omitted, as are raw
prepared representations — `scorer.prepareChoice` hands back an opaque handle
instead, which is the supported way to reuse preparation across queries.
Normalization belongs at search/batch boundaries. QRatio is intentionally omitted because it only changes ratio's
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

A custom scorer prepares choices like any other, and its handles belong to it
alone: two scorers built from the same function still prepare for themselves,
because nothing about a plain function says the two are interchangeable.

## Performance and package validation

The benchmark vocabulary maps directly to the public API:

| Workload                            | API                                |
| ----------------------------------- | ---------------------------------- |
| One best result                     | `bestMatch`                        |
| Top N results                       | `search` with `limit: N`           |
| Lazy qualifying results             | `searchIter`                       |
| Prepare a reusable collection       | `createMatcher` construction       |
| Repeated prepared query             | `matcher.best/search/searchIter`   |
| Repeated query, caller's collection | one-shot search with `getPrepared` |

The release check runs type checking, linting, formatting, all functional
tests, the build, export-map validation, package validation, and tarball
inspection. Source maps are shipped with embedded source content; TypeScript
source files are not included in the package.

## License

[MIT](./LICENSE)
