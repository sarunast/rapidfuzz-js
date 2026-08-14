---
title: Metrics
description: How the library measures similarity — directions, scales, and choosing a metric.
---

A metric is the measuring instrument: a function that takes two sequences and
returns a number describing how alike they are.

```ts
import { similarity as fuzzSimilarity } from 'rapidfuzz-js/fuzz'
import {
  distance as levenshteinDistance,
  normalizedSimilarity as levenshteinNormalizedSimilarity,
} from 'rapidfuzz-js/levenshtein'

fuzzSimilarity('this is a test', 'this is a test!')
// 96.55172413793103 (0–100)

levenshteinDistance('lewenstein', 'levenshtein')
// 2 (edits)

levenshteinNormalizedSimilarity('abc', 'axc')
// 0.6666666666666667 (0–1)
```

Metrics accept strings, but also any array-like sequence — arrays of numbers,
typed arrays, arrays of arbitrary values compared by identity. "String
matching" is really sequence matching.

## Two directions

Every metric points one of two ways, and reading a score starts with knowing
which:

- A **similarity** answers _"how alike?"_ — higher is better, and there's a
  maximum (identical strings).
- A **distance** answers _"how far apart?"_ — lower is better, `0` means
  identical, and for most algorithms there's no upper limit.

The direction is part of the metric's TypeScript type, and everything built
on top — thresholds, sorting, search — automatically respects it.

## Raw or normalized: four names per algorithm

Direction is only half the naming. The other half is whether the number
counts something or scales to `0–1`, which is why the edit-distance subpaths
export **four** metrics apiece:

| Export                 | Direction  | Scale | `('kitten', 'sitting')` |
| ---------------------- | ---------- | ----- | ----------------------- |
| `distance`             | distance   | edits | `3`                     |
| `similarity`           | similarity | edits | `4`                     |
| `normalizedDistance`   | distance   | `0–1` | `0.4285…`               |
| `normalizedSimilarity` | similarity | `0–1` | `0.5714…`               |

The raw `similarity` is _not_ a percentage — it's the count of what the two
inputs share, `maximum − distance` in the same units the distance uses. If
you want a 0–1 score, the name you want is `normalizedSimilarity`. This is
the single most common surprise in the library, and it's inherited from
RapidFuzz, where the pairs are named the same way.

Jaro and Jaro-Winkler are the exception: their scores are 0–1 by
construction, so `similarity` and `normalizedSimilarity` are the same
function.

## Scales: why the numbers differ

Different families report on different scales, and the library never
converts between them — a score always means what its algorithm defined it
to mean:

| Family                      | Scale        | Reading                      |
| --------------------------- | ------------ | ---------------------------- |
| Fuzz similarities           | `0–100`      | Percent-like                 |
| `normalized*` edit measures | `0–1`        | Fraction of the longer input |
| Jaro and Jaro-Winkler       | `0–1`        | Its own formula              |
| Dice and Cosine             | `0–1`        | Shared n-grams               |
| `distance` and `similarity` | Native units | Edit counts, usually         |

Practical consequence: a `threshold` is always in the scorer's own scale.
`threshold: 70` makes sense for a fuzz metric; for Levenshtein
`normalizedSimilarity` you'd write `threshold: 0.7`, and for Levenshtein
`distance` `threshold: 3` ("at most 3 edits").

## The available metrics

```text
rapidfuzz-js/fuzz                  0–100 similarity family (start here)
rapidfuzz-js/levenshtein           insert + delete + substitute
rapidfuzz-js/indel                 insert + delete only
rapidfuzz-js/lcs                   longest common subsequence
rapidfuzz-js/osa                   Levenshtein + adjacent swaps (restricted)
rapidfuzz-js/damerau-levenshtein   Levenshtein + adjacent swaps (full)
rapidfuzz-js/hamming               position-by-position differences
rapidfuzz-js/dice                  shared n-grams, order ignored
rapidfuzz-js/cosine                the same n-grams as frequency vectors
rapidfuzz-js/jaro                  short-string similarity
rapidfuzz-js/jaro-winkler          Jaro with a shared-prefix bonus
rapidfuzz-js/prefix                common prefix length
rapidfuzz-js/postfix               common suffix length
```

If you're unsure, start with `weightedRatio` from `rapidfuzz-js/fuzz` and
only specialize when you can say what's wrong with its answers. The
[Algorithms](/algorithms/levenshtein/) section gives each metric a plain
explanation and a "when to use it".

Levenshtein, Indel, LCS, and Hamming can also _show their work_ — their
`editops`/`opcodes` list the exact edits behind a score
([Comparing strings](/guides/comparing-strings/#recovering-the-edits)).

## Configuring a metric

A bare metric call uses defaults. Tuning knobs — weighted edit costs,
Jaro-Winkler's prefix bonus, strictness about missing values — are applied
through [`createScorer`](/concepts/scorers/), which type-checks the options
against that specific metric:

```ts
import { createScorer } from 'rapidfuzz-js'
import { distance } from 'rapidfuzz-js/levenshtein'

const weighted = createScorer(distance, {
  weights: { insertion: 1, deletion: 1, substitution: 2 },
})
```

Passing `weights` to a metric that has none is a type error, not a silent
no-op.

## Writing your own

Any `(a, b) => number` function can join the system. Declare its direction,
its bounds, and whether `f(a, b) === f(b, a)`, and everything downstream —
thresholds, ranking, Matchers — works with it:

```ts
import { createScorer } from 'rapidfuzz-js'

const custom = createScorer((a, b) => (a === b ? 1 : 0), {
  direction: 'similarity',
  bounds: [0, 1],
  symmetric: true,
})
```

The declared metadata is a contract the library enforces: every result must
be finite and inside the bounds, checked before any thresholding or ordering
relies on it. A buggy custom metric fails loudly instead of quietly
mis-ranking your results.

All three fields are required. `missing` is accepted for a similarity and
rejected for a distance, which always throws on a missing operand. Bounds
must be an ordered numeric pair with a finite lower bound, and a
`'compatible'` similarity's bounds have to include `0` — otherwise the `0`
returned for a missing operand would itself be out of bounds.

A custom scorer prepares choices like any other, and its handles belong to it
alone: two scorers built from the _same function_ don't share handles,
because nothing about a plain function proves they're interchangeable. A
built-in scorer's handle holds a precomputed kernel representation; a custom
one holds an owned snapshot of the sequence, since there's nothing else to
precompute. Ownership, the compatibility check, and the opaque shape are
identical either way.
