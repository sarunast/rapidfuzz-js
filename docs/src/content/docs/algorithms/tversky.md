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
candidate mentions, beside the same real ones, measured **1.11x** — a constant
factor for reaching into a larger map, not a cost that follows its size.

The trap:

> Element weighting does not make token matching fuzzy.

`swisscom` and `swisscomm` share no mass at all, whatever their weights. Making
near-matching tokens count is the next section.

## Matching tokens that are not quite equal

`elementSimilarity` hands the tokens exact matching could not pair to an inner
scorer, and lets each surviving pair share part of its mass:

```ts
import { normalizedSimilarity as indel } from 'rapidfuzz-js/indel'

const fuzzy = createScorer(similarity, {
  gramSize: 1,
  elementSimilarity: { scorer: createScorer(indel), threshold: 0.8 },
})

fuzzy.score(['swisscom', 'ag'], ['swisscomm', 'ag']) // 0.9705882352941176
```

The same pair scores `0.5` without it: `ag` matches, `swisscom` does not, and
half the mass on each side goes unmatched.

**Exact overlap still decides everything it can.** Only what it leaves over is
offered to the inner scorer, and the pairs it reserved are never reconsidered.
`['google', 'deepmind']` against `['google', 'deepmindd']` always pairs `google`
with `google`, even in the rare case where pairing it elsewhere would have scored
higher. The documented answer is the best matching over what exact matching left,
not the best matching overall — and "best" there means the maximum-share matching
subject to floating-point path arithmetic, so two matchings whose totals differ in
the last bit are not told apart.

Each pair shares `min(firstWeight, secondWeight) × similarity`, so mass is
conserved on each side and the arithmetic reduces to today's when the similarity
is `1` and the weights agree. Fuzzy matching therefore cannot lower the exact
score. That is a statement about the mathematics rather than about the last bit:
once any pair is matched across, the components are folded per element rather
than per weight group, so the two can differ by an ulp in either direction. Where
nothing is matched across, the guarantee is exact — see below. The matching is
one-to-one: `swisscom` and `swisscoma` cannot both claim the same `swisscomm`.

`threshold` is on a `0..1` scale whatever the inner scorer's own range — a scorer
bounded `0..100` is rescaled for you, so the fuzz scorers work unadapted. There
is deliberately no default: a useful threshold is a property of your data, and
`0` is refused because it admits arbitrarily weak pairings.

Four traps, and the first is the one that surprises people:

> Only multi-character **string** tokens are compared.

A single code point canonicalizes to a number — `'a'` becomes `97`, and `'😀'`
becomes `128512` — so one-character tokens are exact-only, as are numbers,
objects and array-valued tokens. A plain string at `gramSize: 1` is a sequence of
code points, so `elementSimilarity` changes nothing about it:

```ts
fuzzy.score('swisscom', 'swisscomm') // exactly what it scored before
```

This is a feature for arrays of word tokens. The other three:

- **It costs up to `n × m` element comparisons** on the distinct unmatched
  elements, then a matching over them. Past 32 distinct _fuzzy-comparable_
  leftovers on either side it throws a `RangeError` rather than quietly becoming
  slow. The limit is per side rather than on the product because the matching,
  not the comparing, is what a long sequence against a short one makes
  expensive. Repeats do not count towards it — `['react', 'react', 'react']`
  against `['reakt', 'reakt']` is one comparison, not six, and one element a
  side, not three — but they are not entirely free either: a second `RangeError`
  refuses a pair whose occurrence counts are skewed enough to need more than 512
  augmenting paths to match. Nothing measured has come close.
- **`symmetric` is `false`**, even at `alpha === beta`. The optimum itself is
  symmetric there; the tie-breaking and the order masses are folded in are not,
  so the last bit may differ. `scoreMatrix` therefore scores both halves of a
  pair rather than mirroring one.
- **Indexed search depends on the inner scorer.** `createIndexedMatcher` accepts
  a soft similarity scorer whose `elementSimilarity.scorer` can shortlist
  candidates — normalized Indel does, and so does any exact indexed similarity.
  Jaro, Jaro-Winkler and `fuzz.ratio` cannot, and are still refused, as is soft
  Tversky in the distance direction. The index only chooses _what to rescore_:
  every score it returns comes from this same soft kernel, so it agrees with
  `createMatcher` to the bit.

A configuration whose threshold nothing reaches scores bit-for-bit what the same
configuration without `elementSimilarity` scores, on every path — so the feature
can be switched on and tuned down without moving any existing number.

What it does not fix is a **different tokenization**. `['google', 'deepmind']`
against `['google', 'deep', 'mind']` is not a typo, and no element threshold
pairs one token with two; that is a job for the preprocessing that produced the
tokens, or for a whole-string scorer.

## Explaining a match

A `gramSize: 1` scorer — weighted or not — also carries `explain`, which reports
what a score was made of rather than only what it came to:

```ts
const evidence = company.explain(['swisscom', 'ag'], ['swisscom'])

evidence.score // 0.9900990099009901, the same number `score` gives
evidence.totals.sharedMass // 5
evidence.matches // [{ first: 'swisscom', second: 'swisscom', sharedMass: 5, … }]
evidence.unmatchedFirst // [{ element: 'ag', index: 1, weight: 0.1, unmatchedMass: 0.1 }]
```

`matches` pairs occurrences in input order, so `['react', 'react']` against
`['react']` leaves the _second_ `react` unmatched. Elements are reported as you
passed them while equality is decided on canonical elements, so `'a'` matches
`97` and each is shown as its own side held it; a string is walked by code
point, so `'😀'` is one occurrence at one index.

Under `elementSimilarity` a row can be a fuzzy pair, and `exact` says which:

```ts
const evidence = fuzzy.explain(['swisscom', 'ag'], ['swisscomm', 'ag'])

evidence.matches[0] // { first: 'swisscom', second: 'swisscomm', exact: false,
//                       similarity: 0.9411764705882353,
//                       sharedMass: 0.9411764705882353,
//                       firstUnmatchedMass: 0.05882352941176472, … }
evidence.matches[1] // { first: 'ag', second: 'ag', exact: true, similarity: 1, … }
evidence.totals.sharedMass // 1.9411764705882353
```

Read `exact` rather than `similarity === 1` — an inner scorer is free to call two
different elements identical. A partially matched occurrence appears in `matches`
with a positive `firstUnmatchedMass`, not in `unmatchedFirst`; those arrays hold
only the occurrences with no partner at all.

Four things are worth knowing:

- It exists **only at `gramSize: 1`**, where a gram is a whole element somebody
  named. Every other scorer has no `explain` member at all, so an unsupported
  call is a compile error rather than a throw. The capability is read off the
  configuration _literal_, so hoisting one into a variable widens `gramSize` to
  `number` and gives back an ordinary scorer unless you name what it satisfies:
  `satisfies TverskyExplainConfiguration`, or
  `satisfies TverskySimilarityExplainConfiguration` where the configuration also
  carries `missing` — the same division as
  `TverskyDistanceConfiguration`/`TverskySimilarityConfiguration`, since a
  distance metric refuses `missing`.
- It is **not part of search**. A matcher answers _which candidate_; `explain`
  answers _why this pair_, recomputed cold for the few results search chose.
  Nothing is retained between calls, and no index stores evidence.
- An element weighing `0` **appears nowhere** — it contributed neither overlap
  nor penalty. So with every element ignored you get empty evidence and zero
  totals, and the score comes from the zero-mass rule instead: `1` for equal
  multisets, `0` otherwise. `firstMass === 0 && secondMass === 0` identifies it.
- Masses are on the scorer's own **normalized** scale, a constant factor away
  from the numbers you passed. Tversky is invariant to that factor so no score
  changes, but a mass is not a unit quantity, and only `totals` is
  authoritative — the per-occurrence masses are for reading, not for re-deriving
  the totals.

The intended shape is search first, explain after:

```ts
for (const match of matcher.search(query, { threshold: 0.82, limit: 5 })) {
  const evidence = company.explain(query, match.item)
}
```

`explain` takes the pair as given. A matcher's `normalize` runs over both sides
before scoring, so pass the normalized pair to explain a normalized score.

Explaining a weighted pair costs about **1.31x** scoring it, and unweighted
evidence is cheaper still — about a fifth of the weighted figure. The cost
follows the pair rather than the table it reads: the same comparisons, at the
same weights and against a weight table grown from 7 entries to 20,007,
measured **1.06x**, showing that explanation does not walk the table. Direct
scoring shows the same shape of effect — **1.11x** for its own 20,000-entry
control above — so what remains is a constant factor for reaching into a larger
map, not a cost that follows the vocabulary.

## Searching with a threshold

Like Dice, Tversky's score has an exact upper bound computable from the two
gram counts alone. For direct pair scoring that rejects a hopeless pair before
either bag of grams is built; with prepared choices or a `Matcher` the
profiles already exist, so the bound instead skips the overlap walk. The
bound follows the weights: with `{ alpha: 1, beta: 0 }` a short query can
still reach `1` against an arbitrarily long candidate, while the reversed
orientation is bounded low and rejected on the counts — orientation prunes,
not length as such.

A soft scorer prunes nothing on the counts. The bound above says a pair cannot
share more than the shorter side holds, which stays true, but a soft score is
folded by a different route than the bound is computed by, and the two need not
agree in the last bit — so `elementSimilarity` skips the bound rather than risk
rejecting a candidate that qualifies.

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
other equal weights remain symmetric Tversky variants in their own right.

For typo tolerance between two whole strings, an edit distance like
[Levenshtein](/algorithms/levenshtein/) remains the better tool. Reach for
`elementSimilarity` when you want both at once — per-token pricing _and_
tolerance of a typo inside a token — which is the shape entity deduplication
takes.

### Indexing a soft scorer

`createIndexedMatcher` accepts a soft similarity scorer whose inner scorer can
shortlist candidates — normalized Indel does. It indexes two channels: exact
tokens by their canonical element, and the distinct fuzzy vocabulary through a
q-gram index over the inner scorer. A query unions both, and every choice that
survives is then scored by the ordinary soft kernel, so the numbers that come
back are the ones `createMatcher` returns.

Over 50,000 three-token records — 150,000 token occurrences over about 116,700
distinct tokens, including the `ag` and `gmbh` suffixes a third of the corpus
shares each — timing `best`:

| query                                     | exhaustive |    indexed |                 |
| ----------------------------------------- | ---------: | ---------: | --------------: |
| a typo, two exact tokens left             |    52.45ms |     2.11ms |      25x faster |
| a typo, reachable only through vocabulary |    57.43ms |     2.08ms |      28x faster |
| a typo, behind a shared `ag` suffix       |    45.40ms |    13.33ms |     3.4x faster |
| a typo, with element weights              |    84.62ms |     2.15ms |      39x faster |
| no token seen before                      |    58.03ms |     1.29ms |      45x faster |
| no token within the inner threshold       |    52.66ms |     0.09ms |     601x faster |
| a token a third of the corpus shares      |    45.38ms |    12.62ms |     3.6x faster |
| **`best` on an exact early record**       | **0.38ms** | **1.99ms** | **5.2x slower** |

Two rows say most of it. The second is the path the feature exists for — the
other two query tokens are unseen, so only the q-gram vocabulary index can reach
the answer, and it is the _best_ case rather than a discount on the first. The
third is what an application usually looks like: once the only exact token left
is a suffix a third of the corpus shares, correctness drags every `ag` record
into the candidate union, and the gain falls to the same 3–4x that any common
token buys.

The last row is the shape to know about. `best` stops at the first score of `1`,
so an exhaustive scan that finds an exact match 398 records in never looks at the
other 49,602 — while the index still pays its vocabulary lookup. Lowering
`elementSimilarity.threshold` narrows the gain the same way, to about 1.8x at
`0.5`, because a loose inner threshold shortlists most of the vocabulary.

It is not free to build. The index retains about **410 bytes per distinct
vocabulary token** — 3.7x a `createMatcher` over a corpus where every token is
distinct, but only 1.37x over one drawing its three tokens from a 20,000-token
vocabulary, since the cost follows the vocabulary rather than the corpus.
Construction goes from 37ms to 284ms over 150,000 distinct tokens, and from 21ms
to 78ms over 20,000. Index when you will run many queries against one
collection; scan when you will run few.

If the inner scorer cannot shortlist — Jaro, Jaro-Winkler, `fuzz.ratio` — or you
are scoring in the distance direction, the matcher is refused and a scan is what
is left. Narrowing the corpus yourself is the way around that, but **mind what an
exact token scorer does to the cutoff you can then use**. `Swisscom AG` against
`Swisscomm AG`
shares one token of two, so an exact Tversky blocker scores that pair `0.5`, and
any cutoff above `0.5` throws away the very pair `elementSimilarity` exists to
catch — with the recall gone before the soft scorer is ever called. Whether a
cutoff that low leaves you a corpus small enough to rescore is a question about
your corpus.

Block on something a typo cannot break instead. Character bigrams over the joined
record survive one, and [Dice](/algorithms/dice/) indexes them:

```ts
// `records` is an array of token arrays; the blocker searches them as text.
const blocker = createIndexedMatcher(
  records.map((tokens) => tokens.join(' ')),
  { scorer: createScorer(diceSimilarity) },
)

const nearby = blocker.search(query.join(' '), { threshold: 0.5, limit: 200 })
const rescored = nearby.map(({ key }) => ({
  record: records[key],
  score: fuzzy.score(query, records[key]),
}))
```

`['swisscomm', 'ag']` survives that blocker at about **0.95** against
`['swisscom', 'ag']`, because one doubled letter costs only a handful of
character bigrams — where an exact token scorer gives that same pair `0.5`, and
any cutoff above it loses the match.

Whatever you block with, pick its threshold by checking it against pairs you know
should match. A blocking step that drops one of them is a recall bug no later
threshold can undo.
