---
title: Comparing strings
description: Score one pair, pick the right metric for your data, see the edits, and score in bulk.
---

## The simplest case

Import a metric, call it with two strings:

```ts
import { distance, normalizedSimilarity } from 'rapidfuzz-js/levenshtein'

distance('kitten', 'sitting') // 3 — three single-character fixes apart
normalizedSimilarity('kitten', 'sitting') // 0.571 — as a 0–1 score
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
opinion about what "similar" means. Match the opinion to how _your_ data
goes wrong:

| Your strings differ by…                  | Reach for                                   |
| ---------------------------------------- | ------------------------------------------- |
| Typos, small edits                       | `levenshtein`, or fuzz `similarity`         |
| Swapped adjacent letters (`teh` → `the`) | [`osa`](/algorithms/osa/)                   |
| Word order (`"Smith, John"`)             | fuzz `tokenSortSimilarity`                  |
| One contains the other                   | fuzz `partialSimilarity`                    |
| Extra words on one side                  | fuzz `tokenSetSimilarity`                   |
| Shuffled or missing words, short text    | [`dice`](/algorithms/dice/)                 |
| They're names of people or places        | [`jaro-winkler`](/algorithms/jaro-winkler/) |
| A bit of everything / not sure yet       | fuzz `weightedSimilarity`                   |

Two habits worth forming:

- **Start with `weightedSimilarity`**, look at real mismatches, then
  specialize. Guessing the perfect metric up front rarely survives contact
  with real data.
- **Normalize first.** Half of "the score is too low" cases are case and
  punctuation, not the metric — see
  [Preprocessing](/guides/preprocessing/).

## Recovering the edits

Sometimes the score isn't enough — you want to _show_ the difference
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

The returned `Editops` is iterable and has a `length`, so `for…of`,
spreading, and `Array.from` all work on it directly — you only reach for its
`operations` array (readonly) when you want indexed access.

`opcodes` reports the same information as ranges — including the `equal`
stretches between edits, which is usually what a highlighter wants. The two
convert into each other with `toOpcodes()` and `toEditops()`, and both offer
`toMatchingBlocks()`, `inverse()`, and `apply(source, destination)` — which
replays the operations, so you can check that a set of edits really does
turn one string into the other.

## Many pairs at once

When you have arrays on both sides, don't loop — the batch entry points
score into a single typed array with no per-score allocation:

```ts
import { createScorer, scoreMatrix, scorePairs } from 'rapidfuzz-js'
import { weightedSimilarity } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(weightedSimilarity)

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

A score the element type can't hold is a `RangeError` rather than a wrapped
number — the important part, because a typed array wraps silently and `300`
stored into a `u8` reads back as `44`, which looks like a score:

```ts
scoreMatrix(['cat'], ['cat'], { scorer, into: 'u8', scoreMultiplier: 3 })
// RangeError: scoreMatrix produced the score 300, which 'u8' cannot store
```

The check is on the score, not on the type: a `0–100` scorer into `u8` is
proven safe up front and costs nothing, and an unbounded distance into `u8`
is allowed until a score actually arrives that doesn't fit. `u8c` is the
deliberate exception — `Uint8ClampedArray` saturates to `0–255` by
definition, so it's how you ask for the lossy behaviour on purpose.

If what you actually want is "the best matches for each query", that's a
search, not a matrix — see
[Finding the best match](/guides/finding-the-best-match/).
