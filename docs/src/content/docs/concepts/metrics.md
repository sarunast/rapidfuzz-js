---
title: Metrics
description: How the library measures similarity — directions, scales, and choosing a metric.
---

A metric is the measuring instrument: a function that takes two sequences and
returns a number describing how alike they are.

```ts
import { similarity as fuzzySimilarity } from 'rapidfuzz-js/fuzz'
import {
  distance as levenshteinDistance,
  similarity as levenshteinSimilarity,
} from 'rapidfuzz-js/levenshtein'

fuzzySimilarity('this is a test', 'this is a test!')
// 96.55172413793103 (0–100)

levenshteinDistance('lewenstein', 'levenshtein')
// 2 (edits)

levenshteinSimilarity('abc', 'axc')
// 0.6666666666666667 (0–1)
```

Metrics accept strings, but also any array-like sequence — arrays of numbers,
typed arrays, arrays of arbitrary values compared by identity. "String
matching" is really sequence matching.

## Two directions

Every metric points one of two ways, and reading a score starts with knowing
which:

- A **similarity** answers *"how alike?"* — higher is better, and there's a
  maximum (identical strings).
- A **distance** answers *"how far apart?"* — lower is better, `0` means
  identical, and for most algorithms there's no upper limit.

The direction is part of the metric's TypeScript type, and everything built
on top — thresholds, sorting, search — automatically respects it. Most
algorithm subpaths export both a `distance` and a `similarity`.

## Scales: why the numbers differ

Different families report on different scales, and the library never
converts between them — a score always means what its algorithm defined it
to mean:

| Family                       | Scale                  | Reading                    |
| ---------------------------- | ---------------------- | -------------------------- |
| Fuzz similarities            | `0–100`                | Percent-like               |
| Normalized edit similarities | `0–1`                  | Fraction of the longer input |
| Jaro and Jaro-Winkler        | `0–1`                  | Its own formula            |
| Distances                    | Native units           | Edit counts, usually       |

Practical consequence: a `threshold` is always in the scorer's own scale.
`threshold: 70` makes sense for a fuzz metric; for Levenshtein similarity
you'd write `threshold: 0.7`, and for Levenshtein distance `threshold: 3`
("at most 3 edits").

## The available metrics

```text
rapidfuzz-js/fuzz                  0–100 similarity family (start here)
rapidfuzz-js/levenshtein           insert + delete + substitute
rapidfuzz-js/indel                 insert + delete only
rapidfuzz-js/lcs                   longest common subsequence
rapidfuzz-js/osa                   Levenshtein + adjacent swaps (restricted)
rapidfuzz-js/damerau-levenshtein   Levenshtein + adjacent swaps (full)
rapidfuzz-js/hamming               position-by-position differences
rapidfuzz-js/jaro                  short-string similarity
rapidfuzz-js/jaro-winkler          Jaro with a shared-prefix bonus
rapidfuzz-js/prefix                common prefix length
rapidfuzz-js/postfix               common suffix length
```

If you're unsure, start with `fuzzySimilarity` from `rapidfuzz-js/fuzz` and
only specialize when you can say what's wrong with its answers. The
[Algorithms](/algorithms/levenshtein/) section gives each metric a plain
explanation and a "when to use it".

Levenshtein, Indel, LCS, and Hamming can also *show their work* — their
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
