---
title: Tversky
description: N-gram overlap with a separate price on each side's unmatched grams.
---

[Sørensen-Dice](/algorithms/dice/) treats the two inputs the same: an unmatched
gram costs the score equally whichever side carries it. Tversky splits that
price in two — `alpha` for grams only the **first** sequence has, `beta` for
grams only the **second** has — which turns one formula into a family:
symmetric overlap, multiset Jaccard, and "how completely does the second
sequence contain the first?".

```ts
import { similarity } from 'rapidfuzz-js/tversky'

similarity('night', 'nacht') // 0.25 — the defaults are exactly Dice
```

Scores are 0–1. Identical inputs score 1, but so do non-identical inputs whose
n-gram bags are equivalent — `aba` and `bab` share the bigram bag `{ab, ba}` —
and, once the weights forgive one side entirely, any pair whose unmatched
grams all sit on the forgiven side. The formula, over bags of `gramSize`
adjacent elements with nothing padded onto either end:

```text
                          Σ min(a_g, b_g)
Tversky(A, B) = ─────────────────────────────────────────
                 shared + α · onlyIn(A) + β · onlyIn(B)
```

`distance` is `1 - similarity`, and `normalizedSimilarity` and
`normalizedDistance` are the same two functions under the names the other
algorithms use — Tversky is normalized by construction.

## One formula, three familiar metrics

The weights are scorer configuration, like Dice's `gramSize`:

```ts
import { createScorer } from 'rapidfuzz-js'
import { similarity } from 'rapidfuzz-js/tversky'

// alpha 0.5, beta 0.5 — the default — is exactly Dice.
const dice = createScorer(similarity)

// alpha 1, beta 1 is multiset Jaccard.
const jaccard = createScorer(similarity, { alpha: 1, beta: 1 })
jaccard.score('banana', 'bananas') // 0.833 — Dice says 0.909

// alpha 1, beta 0 asks how completely the second contains the first.
const containment = createScorer(similarity, { alpha: 1, beta: 0 })
containment.score('bana', 'banana') // 1 — every query bigram is covered
containment.score('banana', 'bana') // 0.6 — two query bigrams are not
```

These are equivalences of one formula, not separate modes: any finite
non-negative pair is valid — `{ alpha: 1, beta: 0.1 }` is containment that
still mildly prefers shorter candidates — except both weights at `0`, which is
a `RangeError` because the ratio it defines answers no question at all.

## Argument order matters

The moment `alpha` and `beta` differ the metric is **asymmetric**: swapping
the arguments changes the score, as the containment example above shows. Keep
the query first and the candidate second — `alpha` prices what the query has
that the candidate lacks, `beta` the reverse. Swapping the two weights is
exactly equivalent to swapping the two arguments. A compiled scorer reports
this: `createScorer(similarity, { alpha: 1, beta: 0 }).symmetric` is `false`.

The trap is containment's generosity: `{ alpha: 1, beta: 0 }` scores a flat
`1` for _any_ candidate that covers the query's grams, however much else it
carries. Give `beta` a small positive value when "shortest covering candidate"
should win.

## Tokens instead of characters

Every sequence the library accepts works here, and `gramSize: 1` turns the
metric into plain element overlap. Hand it token arrays and it scores
exact-token containment with no substring credit at all:

```ts
const tokens = createScorer(similarity, { gramSize: 1, alpha: 1, beta: 0 })

tokens.score(['google', 'ag'], ['google', 'deepmind', 'ag']) // 1
tokens.score(['swisscom'], ['swisscomm']) // 0 — different tokens entirely
```

The same pair as character bigrams gets partial fuzzy credit instead — the
representation chooses the granularity, the metric never changes. Sequences
shorter than one gram follow the n-gram family's rule: they score `1` against
an equal input and `0` against anything else, whatever the weights say.

## Weighing tokens against each other

At `gramSize: 1` each element can carry its own weight, which is what entity
matching needs: a company suffix should not count as much as a name.
`elementWeights` makes `shared` and each side's unmatched remainder **weighted
masses** rather than counts, leaving the formula exactly as it was.

```ts
const company = createScorer(similarity, {
  gramSize: 1,
  elementWeights: new Map([
    ['swisscom', 5],
    ['ag', 0.1],
  ]),
})

company.score(['swisscom', 'ag'], ['swisscom']) // 0.99 — `ag` costs little
company.score(['swisscom', 'ag'], ['ag']) // 0.0385 — the name is missing
```

The rules are short:

- Weights are **per element**, and **global to the scorer** — nothing is derived
  from the collection being searched, so a pair scores the same through
  `score`, `createMatcher` and `createIndexedMatcher`.
- They apply **per occurrence**: `['react', 'react']` at `3` carries `6`.
- An element the map does not name weighs `defaultElementWeight`, which defaults
  to `1`. Set it to `0` to score only the vocabulary you listed.
- A weight of `0` drops an element from the comparison entirely. Two sequences
  made of nothing but ignored elements score `1` only when their multisets are
  equal — ignored suffixes alone never make a perfect match.
- Weights need `gramSize: 1`. A shingle of several elements has no single weight
  to carry, and no rule for combining its elements is more right than another.
- The map is **snapshotted** when the scorer is created, so mutating it
  afterwards changes nothing. Build a new scorer to change a weight.
- `'a'` and `97` are the same element, so naming both with different weights is
  a `RangeError` rather than a race between them.
- A weighting where **everything weighs the same positive amount** prices
  nothing — one constant factor cancels from the ratio — so the scorer drops it
  and scores as plain unigram Tversky. `defaultElementWeight: 0` is not that
  case: ignoring every element has its own rules, above.

Weighted scorers keep their own inverted index, at every weight pair — Dice's
knows nothing about element weights — and indexed weighted search over 10,000
token records runs about **8.7x** faster than the exhaustive path.

The representation is not free where the weights really differ: one weight group
costs about **4.3x** the unweighted index and three tiers about **5.5x**, with
`scorer.score` at about **3.1x**, so reach for weights when tokens differ in
importance rather than by default. Weights that are all equal cost nothing at all
— 0.98x indexed and 1.01x on `scorer.score`, which is the unweighted path inside
measurement noise — because the scorer detects them and never builds the
weighted representation. Nothing about the weights is decided per pair, so a
large vocabulary is never walked while scoring: 20,000 entries no query or
candidate mentions, beside the same real ones, measured **1.11x** — the cost of
looking into a bigger map, not of its size.

The trap:

> Element weighting does not make token matching fuzzy.

`swisscom` and `swisscomm` share no mass at all, whatever their weights. Making
near-matching tokens count is a different feature.

## Searching with a threshold

Like Dice, Tversky's score has an exact upper bound computable from the two
gram counts alone. For direct pair scoring that rejects a hopeless pair before
either bag of grams is built; with prepared choices or a `Matcher` the
profiles already exist, so the bound instead skips the overlap walk. The
bound follows the weights: with `{ alpha: 1, beta: 0 }` a short query can
still reach `1` against an arbitrarily long candidate, while the reversed
orientation is bounded low and rejected on the counts — orientation prunes,
not length as such.

Prepared choices behave as Dice's do: the profile a choice is prepared into
depends only on the gram size — the weights are applied at scoring time.
Sharing the handles is nonetheless per scorer configuration today: the
all-default scorer and one written as `{ gramSize: 2, alpha: 0.5, beta: 0.5 }`
accept each other's, and any other configured scorer owns the choices it
prepares.

## Searching a large collection

Tversky joins Dice and Cosine in
[`createIndexedMatcher`](/concepts/matchers/#indexed-matchers), which builds
one inverted n-gram index over the collection instead of preparing each
choice. The same Matcher, the same exact scores, and the weights ride along:

```ts
import { createIndexedMatcher, createScorer } from 'rapidfuzz-js'
import { similarity } from 'rapidfuzz-js/tversky'

const scorer = createScorer(similarity, { alpha: 1, beta: 0.1 })
const matcher = createIndexedMatcher(titles, { scorer })
matcher.search('new york mets', { limit: 5, threshold: 0.5 })
```

At the default weights the scorer shares Dice's index outright — the same
index hot loop, score arithmetic, and retained index representation — since
the two are the same metric there. Token arrays like the `['google', 'ag']`
example above index too: arbitrary elements are keyed by ordinal, so exact-token
search over 10,000 token records runs **14–36x** faster than the exhaustive
path, depending on the operation — and word shingles, whose grams are rarer
still, far more than that. Element weights are a separate index with its own
figure, above.

## When to use it

Everything on the [Dice page](/algorithms/dice/) about short, order-unreliable
text applies at the default weights. Reach for Tversky specifically when the
two sides are not peers: a short query against long titles, a tag list against
a document's keywords, deduplicating where one record is an abbreviation of
the other. When the weights are exactly `0.5` each, use
[Dice](/algorithms/dice/) directly; at `1` each, Tversky is multiset Jaccard;
other equal weights remain symmetric Tversky variants in their own right. For
typo tolerance between peers, an edit distance like
[Levenshtein](/algorithms/levenshtein/) remains the better tool.
