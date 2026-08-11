---
title: Comparing strings
description: Score one pair, pick the right metric for your data, see the edits, and score in bulk.
---

## The simplest case

Import a metric, call it with two strings:

```ts
import { distance, similarity } from 'rapidfuzz-js/levenshtein'

distance('kitten', 'sitting') // 3 — three single-character fixes apart
similarity('kitten', 'sitting') // 0.571 — as a 0–1 score
```

Not just strings — any array-like sequence works, with elements compared by
identity:

```ts
distance([1, 2, 3], [1, 2, 4]) // 1
```

If you need the same comparison tuned or thresholded, wrap it in a
[scorer](/concepts/scorers/):

```ts
import { createScorer } from 'rapidfuzz-js'

const weighted = createScorer(distance, {
  weights: { insertion: 1, deletion: 1, substitution: 2 },
})
weighted.score('kitten', 'sitting') // 5
```

## Picking the metric that matches your data

This is the decision that matters most, because each metric has a different
opinion about what "similar" means. Match the opinion to how *your* data
goes wrong:

| Your strings differ by…                    | Reach for                                        |
| ------------------------------------------ | ------------------------------------------------ |
| Typos, small edits                         | `levenshtein`, or fuzz `similarity`              |
| Swapped adjacent letters (`teh` → `the`)   | [`osa`](/algorithms/osa/)                        |
| Word order (`"Smith, John"`)               | fuzz `tokenSortSimilarity`                       |
| One contains the other                     | fuzz `partialSimilarity`                         |
| Extra words on one side                    | fuzz `tokenSetSimilarity`                        |
| They're names of people or places          | [`jaro-winkler`](/algorithms/jaro-winkler/)      |
| A bit of everything / not sure yet         | fuzz `fuzzySimilarity`                           |

Two habits worth forming:

- **Start with `fuzzySimilarity`**, look at real mismatches, then
  specialize. Guessing the perfect metric up front rarely survives contact
  with real data.
- **Normalize first.** Half of "the score is too low" cases are case and
  punctuation, not the metric — see
  [Preprocessing](/guides/preprocessing/).

## Recovering the edits

Sometimes the score isn't enough — you want to *show* the difference
(highlighting, diffs). Levenshtein, Indel, LCS, and Hamming can report the
exact operations behind their distance:

```ts
import { editops } from 'rapidfuzz-js/levenshtein'

for (const op of editops('kitten', 'sitting')) {
  op.tag // 'replace' | 'insert' | 'delete'
  op.srcPos // position in 'kitten'
  op.destPos // position in 'sitting'
}
```

`opcodes` reports the same information as ranges — including the `equal`
stretches between edits, which is usually what a highlighter wants. Both
convert into each other and into matching blocks.

## Many pairs at once

When you have arrays on both sides, don't loop — the batch entry points
score into a single typed array with no per-score allocation:

```ts
import { createScorer, scoreMatrix, scorePairs } from 'rapidfuzz-js'
import { fuzzySimilarity } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(fuzzySimilarity)

// Element-wise: [score(cat,cats), score(dog,dogs)]
scorePairs(['cat', 'dog'], ['cats', 'dogs'], { scorer })

// Every query × every choice:
const matrix = scoreMatrix(['cat', 'dog'], ['cats', 'dogs'], { scorer })
matrix.at(0, 1) // score('cat', 'dogs')
matrix.data // row-major Float64Array
```

The `into` option picks the element type — `f64` (default), `f32`, `i32`,
`i16`, `i8`, `u32`, `u16`, `u8`, or `u8c`. A `u8` matrix holds any 0–100
fuzz score in an eighth of the memory, and the buffer can be handed to a
worker or WebAssembly without copying.

If what you actually want is "the best matches for each query", that's a
search, not a matrix — see
[Finding the best match](/guides/finding-the-best-match/).
