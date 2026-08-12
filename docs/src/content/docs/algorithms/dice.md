---
title: Sørensen-Dice
description: Overlap of n-gram bags — order-insensitive similarity for short text.
---

Edit distances walk two strings in step, so moving a word wrecks the score.
Sørensen-Dice throws position away: it chops both inputs into **n-grams**
(pairs of adjacent characters by default), then asks what fraction of those
grams the two sides share.

```ts
import { similarity } from 'rapidfuzz-js/dice'

similarity('night', 'nacht') // 0.25 — `ni ig gh ht` against `na ac ch ht`
similarity('banana', 'bananas') // 0.909
similarity('new york mets', 'mets new york') // 0.833
```

Scores are 0–1, with 1 identical. That last line is the point of the metric:
[Levenshtein](/algorithms/levenshtein/) scores the same pair `0.231`, because
almost every position moved. Only the grams straddling the two words changed,
so Dice barely notices.

The formula is the shared gram count against the total, counting repeats:

```text
              2 · Σ min(a_g, b_g)
Dice(A, B) = ─────────────────────
              gramCount(A) + gramCount(B)
```

`distance` is `1 - similarity`, and `normalizedSimilarity` and
`normalizedDistance` are the same two functions under the names the other
algorithms use — Dice is normalized by construction, so there is no raw form
to normalize.

## Three choices other implementations make differently

If you are porting a threshold from another library, check these before
trusting it:

- **Multiset, not set.** A gram occurring three times on one side and twice on
  the other contributes `min(3, 2) = 2` to the overlap sum, and so four to the
  numerator above it. `('banana', 'bananas')` is `0.909091`; a set-based Dice
  answers `0.857143`.
- **No padding.** Nothing is added at the ends, so `aba` and `bab` have the
  same bigram bag and score `1`. Implementations that wrap each input in guard
  characters answer `0.5`.
- **Sequences shorter than `gramSize` have no grams at all**, which would make
  the ratio `0/0`. Two such sequences score `1` if they are equal and `0`
  otherwise; against a sequence that does have grams they score `0`. At the
  default `gramSize` of 2, `('a', 'a')` is `1`, `('a', 'b')` is `0`, and
  `('a', 'ab')` is `0`.

## Choosing the gram size

`gramSize` is scorer configuration rather than a call argument, the same rule
Levenshtein's `weights` and Hamming's `pad` follow:

```ts
import { createScorer } from 'rapidfuzz-js'
import { similarity } from 'rapidfuzz-js/dice'

const trigrams = createScorer(similarity, { gramSize: 3 })

trigrams.score('night', 'nacht') // 0 — no shared trigram
trigrams.score('banana', 'bananas') // 0.889
```

Bigrams are forgiving and generic. Larger grams demand longer runs of exact
agreement, which sharpens a long-text comparison and makes short inputs
collapse to `0` — a trigram scorer cannot see anything in common between two
five-letter words that differ in the middle. A `gramSize` below `1`, or one
that is not a _safe_ integer, is a `RangeError` — `1e300` is an integer, and a
trie that deep is not a request anyone means.

Not every sequence is text: elements are compared by identity, so
`similarity([1, 2, 3], [1, 2, 4])` is `0.5`, and astral characters are
compared as whole code points rather than as surrogate halves.

## Searching with a threshold

Dice carries an exact upper bound on its own score — `2 · min(gA, gB) / (gA +
gB)` whenever either sequence has a gram, and the `0/0` case above when
neither does — computable from the two gram _counts_ alone. Under a threshold
that rejects a candidate too long or too short to reach the cutoff **before
either bag of grams is built**, which makes it markedly cheaper than
[Cosine](/algorithms/cosine/): 100 length-skewed pairs at `threshold: 0.8`
measure 0.005 ms here against 4.2 ms through Cosine, which has no such bound
and builds both profiles every time.

That saving is a property of scoring a pair. Passing raw text to `search` does
not get it: each candidate is turned into a profile as it is read, before the
bound is in a position to reject it. Hand `search` prepared choices, or use a
`Matcher`, when the candidates are long and the threshold is high.

```ts
import { createScorer, search } from 'rapidfuzz-js'
import { similarity } from 'rapidfuzz-js/dice'

const scorer = createScorer(similarity)
const products = [
  'Wireless mechanical keyboard',
  'Compact wireless mouse',
  'Mechanical keyboard wireless',
]

search('mechanical keyboard', products, { scorer, threshold: 0.6 })
// [ { item: 'Wireless mechanical keyboard', key: 0, score: 0.8 },
//   { item: 'Mechanical keyboard wireless', key: 2, score: 0.756 } ]
```

If the same choices are searched repeatedly, prepare them once —
[Prepared choices](/guides/prepared-choices/) explains the mechanics. A
prepared choice remembers the gram size it was built at: a default scorer and
one written as `{ gramSize: 2 }` accept each other's, and any other depth, or
the other metric, is refused.

## When to use it

Short text where word order is unreliable and the fuzz family's tokenization
is too coarse: product titles, addresses, tags, song and film names,
deduplicating list entries. It is also the cheap end of a search under a high
threshold, thanks to the bound above.

The trap is the same as its strength — position is gone entirely. `aba` and
`bab` are identical to it, and a typo that breaks two grams at once costs more
than its size suggests: `similarity('recieve', 'receive')` is `0.5` where
Levenshtein `normalizedSimilarity` says `0.714`. For plain typos reach for
[Levenshtein](/algorithms/levenshtein/) or fuzz `similarity`; for reordered
_words_ specifically, fuzz `tokenSortSimilarity` scores that pair a flat
`100`.
