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

This file is the tour. The [documentation site](./docs) is the reference:
guides, per-algorithm pages, the error reference, and an API section
generated from the source.

## Install

```sh
npm install rapidfuzz-js
```

## Quick start

Comparing two values needs no setup — metrics are plain functions:

```ts
import { ratio } from 'rapidfuzz-js/fuzz'

ratio('this is a test', 'this is a test!') // 96.55…
```

The rest of the API builds on one composition model:

```text
Metric → Scorer → Matcher
```

Metrics come from algorithm subpaths; orchestration comes from the package
root:

```ts
import { createMatcher, createScorer, normalizeText } from 'rapidfuzz-js'
import { tokenSortRatio } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(tokenSortRatio)

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
// { item: { title: 'Wireless mechanical keyboard' }, key: 0, score: 78.26… }
```

A `Matcher` prepares a collection once and answers many queries. For a single
query, use the standalone `bestMatch`, `search`, or `searchIter` instead.

## Metrics and score scales

Metrics are directly callable:

```ts
import { ratio } from 'rapidfuzz-js/fuzz'
import {
  distance as levenshteinDistance,
  normalizedSimilarity as levenshteinNormalizedSimilarity,
  similarity as levenshteinSimilarity,
} from 'rapidfuzz-js/levenshtein'

ratio('this is a test', 'this is a test!')
// 96.55172413793103 (0–100)

levenshteinDistance('lewenstein', 'levenshtein')
// 2 (native edit count)

levenshteinSimilarity('abc', 'axc')
// 2 (raw maximum-minus-distance similarity)

levenshteinNormalizedSimilarity('abc', 'axc')
// 0.6666666666666667
```

Scores are never rescaled between families:

| Operation                              | Scale                  |
| -------------------------------------- | ---------------------- |
| Fuzz similarities                      | `0–100`                |
| Raw edit/count distance and similarity | Native algorithm units |
| Normalized distance and similarity     | `0–1`                  |
| Jaro and Jaro-Winkler measures         | `0–1`                  |
| Dice, Cosine and Tversky measures      | `0–1`                  |

Available subpaths:

```text
rapidfuzz-js/fuzz
rapidfuzz-js/levenshtein
rapidfuzz-js/indel
rapidfuzz-js/lcs
rapidfuzz-js/osa
rapidfuzz-js/cosine
rapidfuzz-js/damerau-levenshtein
rapidfuzz-js/dice
rapidfuzz-js/hamming
rapidfuzz-js/jaro
rapidfuzz-js/jaro-winkler
rapidfuzz-js/prefix
rapidfuzz-js/postfix
rapidfuzz-js/tversky
```

Every algorithm subpath exposes `distance`, `similarity`, `normalizedDistance`,
and `normalizedSimilarity`. Levenshtein, Indel, LCS, and Hamming also export
`editops` and `opcodes`. The `Editops` and `Opcodes` they return carry their
alignment in `operations`, a readonly array, and are themselves iterable with a
`length`, so `for (const op of editops(a, b))` and `[...editops(a, b)]` work
without reaching through it. Jaro, Jaro-Winkler, Dice, Cosine, and Tversky are
normalized by construction, so their `normalized*` exports are the same metrics
under the names the other algorithms use.

### Sørensen-Dice and Cosine

`rapidfuzz-js/dice` and `rapidfuzz-js/cosine` compare two sequences as bags of
n-grams rather than position by position, which is what you want when word
order is unreliable and `fuzz`'s tokenisation is too coarse. Both read the same
exact n-gram frequency profile and differ only in how they combine it:

```text
                2 · Σ min(a_g, b_g)                         Σ a_g · b_g
Dice(A, B) = ─────────────────────────      Cosine(A, B) = ───────────────
              gramCount(A) + gramCount(B)                   ‖A‖ · ‖B‖
```

Cosine here is the dot product of the two frequency vectors — not the
intersection-count formula (`|A ∩ B| / sqrt(|A| · |B|)`) that some libraries
ship under the same name. On `ab:3, bc:1` against `ab:2, bc:2` this one answers
`0.894`; that one answers `0.75`.

Three further choices are worth knowing, because other implementations make
them differently:

- **Multiset, not set.** A gram occurring three times on one side and twice on
  the other contributes `min(3, 2) = 2` to Dice's overlap sum — four to its
  numerator, which is twice that — and `3 · 2 = 6` to Cosine's dot product.
  Dice on `('banana', 'bananas')` is `0.909091`; a set-based Dice answers
  `0.857143`.
- **No padding.** Nothing is added at the ends, so `aba` and `bab` have the same
  bigram multiset and score `1`. Implementations that wrap each input in guard
  characters answer `0.5`.
- **Sequences shorter than `gramSize` have no grams at all**, which would make
  the ratio `0/0`. Two such sequences score `1` if they are equal and `0`
  otherwise; against a sequence that does have grams they score `0`. So at the
  default `gramSize` of 2, `('a', 'a')` is `1`, `('a', 'b')` is `0`, and
  `('a', 'ab')` is `0` — for both metrics.

`gramSize` is a scorer configuration, not a call argument — the same rule
Levenshtein's `weights` and Hamming's `pad` follow:

```ts
import { createScorer } from 'rapidfuzz-js'
import { similarity as diceSimilarity } from 'rapidfuzz-js/dice'
import { similarity as cosineSimilarity } from 'rapidfuzz-js/cosine'

diceSimilarity('night', 'nacht')
// 0.25 — `ni ig gh ht` against `na ac ch ht` shares only `ht`

cosineSimilarity('night', 'nacht')
// 0.25 — the same, because every gram here occurs once

createScorer(diceSimilarity, { gramSize: 3 }).score('night', 'nacht')
// 0
```

A scorer left at the default and one written as `{ gramSize: 2 }` prepare
interchangeable choices; a scorer at any other depth, or of the other metric,
refuses theirs.

Dice also carries an exact upper bound — `2 · min(gA, gB) / (gA + gB)`, with
the `0/0` case above standing in when neither sequence has a gram — that turns
down a candidate on gram counts alone, before either profile is built. That
makes it markedly cheaper than Cosine under a high threshold when scoring a
pair. Cosine has no such bound. Note that a `search` over raw text profiles
each candidate as it reads it, so the bound saves nothing there; prepared
choices or a `Matcher` are what let it apply.

`rapidfuzz-js/tversky` generalizes Dice over the same n-gram profiles with a
separate price on each side's unmatched grams: `alpha` for grams only the
first sequence has, `beta` for the second's. The defaults (`0.5` each) are
exactly Dice, `{ alpha: 1, beta: 1 }` is multiset Jaccard, and
`{ alpha: 1, beta: 0 }` asks how completely the second sequence contains the
first — asymmetric, so argument order matters there. With `gramSize: 1` and
token arrays it scores exact-token overlap:

```ts
import { createScorer } from 'rapidfuzz-js'
import { similarity as tverskySimilarity } from 'rapidfuzz-js/tversky'

const containment = createScorer(tverskySimilarity, {
  gramSize: 1,
  alpha: 1,
  beta: 0,
})
containment.score(['google', 'ag'], ['google', 'deepmind', 'ag'])
// 1 — every query token is covered
```

At that gram size each element can also carry its own weight, so a generic
suffix need not count as much as a name:

```ts
const company = createScorer(tverskySimilarity, {
  gramSize: 1,
  elementWeights: new Map([
    ['swisscom', 5],
    ['ag', 0.1],
  ]),
})
company.score(['swisscom', 'ag'], ['swisscom']) // 0.99 — `ag` costs little
```

Weights are per element, global to the scorer, applied per occurrence, and
snapshotted when the scorer is created — nothing is inferred from the collection
being searched.

Weighting alone does not make token matching fuzzy: `swisscom` and `swisscomm`
share no mass at all. `elementSimilarity` is what closes that gap. Exact overlap
still claims everything it can, and only the tokens it left over are scored
against each other by an inner scorer:

```ts
import { normalizedSimilarity as indel } from 'rapidfuzz-js/indel'

const fuzzy = createScorer(tverskySimilarity, {
  gramSize: 1,
  elementSimilarity: { scorer: createScorer(indel), threshold: 0.8 },
})
fuzzy.score(['swisscom', 'ag'], ['swisscomm', 'ag'])
// 0.9705882352941176 — `ag` pairs exactly, `swisscom` partially
// without elementSimilarity the same pair scores 0.5
```

Each surviving pair shares `min(firstWeight, secondWeight) × similarity` rather
than a whole occurrence, so a typo costs a little instead of everything. Four
things to know before reaching for it: only multi-character **string** tokens are
compared, since a single code point canonicalizes to a number — which also means
a plain string scores exactly what it scored before; exact pairs are reserved
first, so the answer is the best matching over what is left rather than the best
matching overall, and best only up to floating-point path arithmetic; a pair with
more than 32 distinct fuzzy-comparable leftovers on either side throws rather than
quietly becoming slow; and such a scorer reports `symmetric: false` and offers no
indexed representation, so `createIndexedMatcher` refuses it.

A `gramSize: 1` scorer — weighted or not — also carries `explain`, which reports
what a score was made of:

```ts
const evidence = company.explain(['swisscom', 'ag'], ['swisscom'])
evidence.totals.sharedMass // 5
evidence.unmatchedFirst // [{ element: 'ag', index: 1, weight: 0.1, unmatchedMass: 0.1 }]
```

Four things are worth knowing before relying on it. It exists only at
`gramSize: 1`, where a gram is a whole element a caller named — every other
scorer has no `explain` at all, so an unsupported call is a compile error rather
than a throw. It recomputes one pair from scratch and is deliberately not part
of a matcher's results: `search` answers _which candidate_, `explain` answers
_why this pair_, for the few results search already chose. An element weighing
`0` appears nowhere in the evidence, having contributed neither overlap nor
penalty. And the masses are on the scorer's own normalized scale, which is a
constant factor away from the numbers you passed — Tversky is invariant to that
factor, so no score changes, but they are not a unit quantity.

```ts
for (const match of matcher.search(query, { threshold: 0.82, limit: 5 })) {
  const evidence = company.explain(query, match.item)
  // evidence.score === company.score(query, match.item)
}
```

`explain` takes the pair as you hand it over. A matcher's `normalize` option
transforms both sides before scoring, so pass the normalized pair if you want
evidence for a normalized match's score.

The `fuzz` subpath is the exception: it exports similarity scorers only. Two
of them are easy to mix up:

- `ratio` compares the two strings exactly as given, in one pass.
- `weightedRatio` also tries substring and word-reordering comparisons and
  returns the best weighted score.

`weightedRatio` is the general-purpose choice: it stays high where
`ratio` drops — reordered words, one string contained in the other, large
length differences. Reach for the other fuzz scorers (`partialRatio`,
`tokenSortRatio`, `tokenSetRatio`, …) when you want exactly one of
those strategies.

## Scorers

`createScorer` bundles a metric with its direction, bounds, symmetry, and
configuration into a reusable object:

```ts
import { createScorer } from 'rapidfuzz-js'
import { distance } from 'rapidfuzz-js/levenshtein'

const scorer = createScorer(distance, {
  weights: { insertion: 1, deletion: 1, substitution: 2 },
})

scorer.direction // 'distance'
scorer.bounds // [0, Infinity]
scorer.symmetric // true
scorer.score('kitten', 'sitting') // 5
scorer.score('kitten', 'sitting', { threshold: 3 }) // undefined
```

Similarity thresholds are minimums, distance thresholds are maximums. A
threshold uses the scorer's own scale and must be finite.

`scoreIfMatch` returns the thresholded score as a standalone operation;
`isMatch` returns only the boolean.

## One query or many

|              |                                                             |
| ------------ | ----------------------------------------------------------- |
| `bestMatch`  | the best single match                                       |
| `search`     | ranked top matches — five by default, `limit: null` for all |
| `searchIter` | lazily yields every qualifying match, in source order       |

`search` ranks, so it buffers results; `searchIter` does not rank and buffers
nothing. Each takes only the options it defines — passing `limit` to
`bestMatch` or `searchIter` throws rather than being ignored.

One-shot search streams its input and does not retain the collection:

```ts
import { bestMatch, createScorer, search, searchIter } from 'rapidfuzz-js'
import { weightedRatio } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(weightedRatio)
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

A `Matcher` snapshots the collection and prepares it once:

```ts
import { createMatcher } from 'rapidfuzz-js'

const matcher = createMatcher(teams, { scorer })

matcher.size // 3
matcher.best('new york jet')
matcher.search('new york', { limit: null }) // every result, best first
matcher.searchIter('new york', { threshold: 60 }) // lazy, source order
```

Keys follow the collection: arrays and iterables use source positions, maps
keep their keys (anything map-shaped is read as a map, even when typed as an
iterable of entries — the runtime cannot tell them apart), and plain objects
keep property names. Missing items and missing `getText` results are skipped
by default without renumbering keys; `missingItems: 'throw'` rejects them
instead. A single string is rejected — it is not a collection of characters.

Strings are retained as-is. Other array-like sequences are shallow-copied, so
mutating the source later does not change scores. Returned items and nested
objects stay live references.

For a Dice, Cosine or Tversky scorer over a large collection, `createIndexedMatcher`
builds the same `Matcher` over one inverted n-gram index instead of a prepared
handle per choice:

```ts
import { createIndexedMatcher, createScorer } from 'rapidfuzz-js'
import { similarity as diceSimilarity } from 'rapidfuzz-js/dice'

const matcher = createIndexedMatcher(files, {
  scorer: createScorer(diceSimilarity, { gramSize: 3 }),
  getText: (file) => file.path,
})
matcher.search('src/algorthms/dice.ts', { limit: 5, threshold: 0.5 })
```

Every member behaves the same and the scores are exact. On 10,000 file paths a
query measured **11-13x faster**, construction retaining **256 bytes a choice
against 1,282** and costing about 1.2x more. Query scratch is separate from that
figure and reused between queries, and an oversized reservation is released once
query demand returns to the normal retained range. It is not uniformly
faster: a query made of grams nearly every choice shares reaches everything
anyway and measured **0.7x**. Only `dice.similarity`, `cosine.similarity` and
`tversky.similarity` offer one — anything else throws at construction, a distance scorer is a
compile error, and `searchIter` settles which choices qualify before yielding
the first rather than scoring lazily.

## Reusable prepared choices

`prepareChoice` converts one choice into the form the scorer's kernels want
and returns it as an opaque handle. Store the handle beside your own data and
return it from `getPrepared`, and searches skip preparation entirely:

```ts
import { createScorer, normalizeText, searchIter } from 'rapidfuzz-js'
import { tokenSetRatio } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(tokenSetRatio)
const companies = records.map((record) => ({
  record,
  prepared: scorer.prepareChoice(record.name, { normalize: normalizeText }),
}))

// Cheap guards run first; only survivors are scored.
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
  // Must be the normalizer the choices were prepared with; a different one,
  // or none, throws.
  normalize: normalizeText,
})) {
  // handles are prepared once, however many queries run
}
```

This pattern keeps the collection yours: a generator filters candidates before
any scoring happens. `createMatcher` amortizes the same preparation but owns
the collection, so there is no place for a guard. It accepts `getPrepared` too
and resolves every handle once at construction.

### Normalization is all-or-nothing

Either the library normalizes both sides, or you do. The handle records which:

```ts
// Library-managed: the handle records the normalizer, and the search must
// name the same one.
const prepared = scorer.prepareChoice(name, { normalize: normalizeText })
searchIter(query, rows, { scorer, getPrepared, normalize: normalizeText })

// Caller-managed: you normalize both sides yourself and tell the search
// nothing.
const prepared = scorer.prepareChoice(normalizeText(name))
searchIter(normalizeText(query), rows, { scorer, getPrepared })
```

Mixing the two throws. `prepareChoice(normalizeText(name))` produces the same
text as the first line but records no normalizer, so a search that normalizes
its query cannot verify the choice was normalized the same way, and refuses.

The check compares function identity — two arrow functions with the same body
count as different normalizers — so define one function and pass it to both
sides. This also means a normalizer must be deterministic: one that reads
mutable outside state passes the identity check but can still normalize the
two sides differently, and no check can catch that. To reconfigure a
normalizer, create a new function instead of mutating captured state.

`normalizeText` lowercases, replaces every non-alphanumeric character with a
space, and trims. Non-string sequences pass through unchanged, so it works as
the `Normalizer` for array-like choices too. Values that are not sequences at
all are still refused.

### Which scorers accept a handle

Compatibility is decided by identity, not by proving two preparations
equivalent:

| Prepared by                                    | Accepted by                            |
| ---------------------------------------------- | -------------------------------------- |
| a scorer using a metric's default preparation  | any scorer of that metric using it too |
| a scorer with configuration the metric records | that scorer alone                      |
| a custom metric's scorer                       | that scorer alone                      |

Two separately created `createScorer(fuzz.ratio)` scorers share handles;
a configured or custom scorer owns the handles it made.

Anything else throws: an incompatible handle is refused, and a value that is
not a handle at all is refused as invalid. Built-in metrics carry their
identity in the type, so most of these mistakes are compile errors first.
Spell a stored handle's type as `PreparedChoiceOf<typeof scorer>` and a stored
scorer's as `ScorerOf<typeof tokenSetRatio>`. The identity is the
metric's own id literal — declaration emit spells it
`Scorer<'similarity', 'fuzz.tokenSetRatio'>` without importing anything
from this package. Widening to `Scorer<'similarity'>` drops the metric from
the type, leaving only the runtime check.

Prepared mode is strict: `missingItems` and `getText` are not accepted —
there is nothing to skip or extract. `normalize` applies to the query only,
never to a choice; a choice is normalized when it is prepared.

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
to choose the typed-array storage. Batch `threshold` uses the scorer's
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

A score the chosen element type cannot hold is a `RangeError`, not a wrapped
number: `scoreMultiplier: 3` on a `0..100` scorer reaches `300`, which a `u8`
would otherwise store as `44`. The check costs nothing where the scorer's
bounds and multiplier prove every score fits, so a `0..100` scorer into `u8`
stays as it was, and a `Infinity`-bounded distance into `u8` is still allowed —
it is refused when a score actually arrives that does not fit, not up front.
`u8c` is the exception, and the way to ask for the lossy behaviour on purpose:
`Uint8ClampedArray` saturates to `0..255` by definition.

## RapidFuzz capability mapping

| RapidFuzz                      | rapidfuzz-js                                                  |
| ------------------------------ | ------------------------------------------------------------- |
| `fuzz.ratio`                   | `ratio`                                                       |
| `fuzz.WRatio`                  | `weightedRatio`                                               |
| `fuzz.QRatio`                  | none — see below                                              |
| `process.extractOne`           | `bestMatch`                                                   |
| `process.extract`              | `search`                                                      |
| `process.extract_iter`         | `searchIter`                                                  |
| `process.cdist` / `cpdist`     | `scoreMatrix` / `scorePairs`                                  |
| `score_cutoff`                 | `threshold`                                                   |
| `score_multiplier`             | `scoreMultiplier`                                             |
| `scorer_kwargs`                | `createScorer(metric, configuration)`                         |
| repeated prepared extraction   | `createMatcher`, or `scorer.prepareChoice` with `getPrepared` |
| large-collection n-gram search | `createIndexedMatcher` (no RapidFuzz counterpart)             |

For everything RapidFuzz spells differently — extraction, cutoffs, scorer
configuration, prepared reuse — the table above is the whole translation.

`fuzz.QRatio` is deliberately absent. It is upstream's own `fuzz.ratio` with one
difference: two empty strings score `0` rather than `100`. Its processor is
opt-in and defaults to none, so it does no normalization — the name is a
fuzzywuzzy inheritance rather than a separate algorithm, and `fuzzball`, the
JavaScript port of fuzzywuzzy, does not ship it either. Callers who want that
empty-string rule can write it in a line.

### Upgrading from 0.11

Every `rapidfuzz-js/fuzz` export was renamed in 0.12.0, so that the scorers carry
RapidFuzz's own vocabulary. Nothing else moved: the package root and the distance
subpaths are unchanged, and no score changed with the names.

| 0.11                         | 0.12                    |
| ---------------------------- | ----------------------- |
| `similarity`                 | `ratio`                 |
| `partialSimilarity`          | `partialRatio`          |
| `partialSimilarityAlignment` | `partialRatioAlignment` |
| `tokenSimilarity`            | `tokenRatio`            |
| `tokenSortSimilarity`        | `tokenSortRatio`        |
| `tokenSetSimilarity`         | `tokenSetRatio`         |
| `partialTokenSimilarity`     | `partialTokenRatio`     |
| `partialTokenSortSimilarity` | `partialTokenSortRatio` |
| `partialTokenSetSimilarity`  | `partialTokenSetRatio`  |
| `weightedSimilarity`         | `weightedRatio`         |

The metric brands moved with them, so a stored type such as
`Scorer<'similarity', 'fuzz.tokenSetSimilarity'>` becomes
`Scorer<'similarity', 'fuzz.tokenSetRatio'>`. The first argument there is the
score direction and is unrelated to the rename.

## Missing and invalid values

Only `null` and `undefined` count as missing. Similarity scorers return `0`
for a missing operand by default:

```ts
const strict = createScorer(weightedRatio, { missing: 'throw' })
strict.score(null, 'text') // throws TypeError
```

Two missing operands are also `0`, not `100`: unknown compared with unknown is
not a match, and scoring it perfect would sort every missing record to the top
of a search.

Distance scorers always throw on missing operands. Empty sequences are valid.
Numbers (including `NaN`), booleans, and objects without a valid array-like
`length` are invalid.

Empty inputs are where the fuzz scorers disagree with each other, and
deliberately so — `tokenSetRatio`, `partialTokenSetRatio` and
`weightedRatio` answer `0` for two empty inputs where `ratio` and the
sort-based scorers answer `100`. FuzzyWuzzy returns `0` there and RapidFuzz
keeps it (issue 110), so this port does too.

Whitespace-only inputs split the three: the two token-set scorers still answer
`0`, because a side that tokenizes to nothing has no set to intersect, while
`weightedRatio` sees two non-empty strings and scores identical whitespace
`100`.

Options objects — for searches, Matcher methods, `scoreMatrix`, `scorePairs`,
and `prepareChoice` — reject unknown keys:

```ts
search(query, choices, { scorer, thresold: 90 })
// TypeError: unknown search option 'thresold'
```

Without this, the misspelling would typecheck (TypeScript's excess-property
check only covers fresh object literals) and silently return unthresholded
results. The threshold argument to `score`, `isMatch`, and `scoreIfMatch` is
not enumerated: its one key is required, so a misspelling already fails with
`threshold must be finite`.

## Custom metrics

Custom metrics declare the metadata needed for ordering and validation:

```ts
const custom = createScorer((a, b) => (a === b ? 1 : 0), {
  direction: 'similarity',
  bounds: [0, 1],
  symmetric: true,
})
```

Every custom result must be finite and inside its declared bounds; results are
validated before thresholding, ordering, or pruning.

A custom scorer prepares choices like any other, and its handles belong to it
alone — two scorers built from the same function do not share handles, because
nothing about a plain function proves they are interchangeable. A built-in
scorer's handle holds a precomputed kernel representation; a custom scorer's
holds an owned snapshot of the sequence, since there is nothing else to
precompute for a plain function. Ownership, the compatibility check, and the
opaque shape are the same for both.

## Small by construction

Bundle size is a structural property here, not something measured after the
fact:

- `"sideEffects": false`, and nothing runs at import time, so bundlers can
  drop whatever a build does not reach.
- Standalone named functions, never namespace objects — touching one member of
  a namespace object keeps every member alive.
- Algorithms live on their own subpaths and are never re-exported from the
  root: importing `rapidfuzz-js/jaro` pays for Jaro and nothing else.
- No runtime dependencies and no Node built-ins, so the same build runs
  unchanged in browsers and edge runtimes.

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
inspection. Source maps ship with embedded source content; TypeScript source
files are not included in the package.

## License

[MIT](./LICENSE)
