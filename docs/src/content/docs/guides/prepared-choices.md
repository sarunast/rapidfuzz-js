---
title: Prepared choices
description: Prepare candidates once, keep the collection yours — one-shot search speed without handing your data to a Matcher.
---

A [Matcher](/concepts/matchers/) amortizes preparation by _owning the
collection_: it snapshots one text field at construction, and from then on
you query what it remembers. That's the right trade for a search box — and
the wrong one when the collection is really yours: rows you filter
differently per query, records that carry more than one searchable field,
data that lives in your own index.

`scorer.prepareChoice` splits the difference. It returns an opaque handle
holding one candidate in the form the scorer's kernels want. Store it beside
your own data, hand it back through `getPrepared`, and a one-shot search that
would re-prepare every candidate on every query prepares nothing:

```ts
import { createScorer, normalizeText, searchIter } from 'rapidfuzz-js'
import { tokenSetSimilarity } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(tokenSetSimilarity)
const companies = records.map((record) => ({
  record,
  prepared: scorer.prepareChoice(record.name, { normalize: normalizeText }),
}))

// Your guards run before the scorer does — only survivors get scored.
function* plausible(query: Query) {
  for (const row of companies) {
    if (row.record.country !== query.country) continue
    yield row
  }
}

for (const match of searchIter(query.name, plausible(query), {
  scorer,
  getPrepared: (row) => row.prepared,
  normalize: normalizeText,
})) {
  // scored against handles prepared once, however many queries run
}
```

That ordering is the point: a generator decides what is worth scoring, and
the scoring pays nothing to prepare what it accepts. `bestMatch`, `search`,
and `searchIter` all take `getPrepared`; so does `createMatcher`, which
resolves every handle once at construction.

`searchIter` pulls lazily, so a generator's guards run only for candidates
actually consumed — and filtering renumbers the `key` a match reports.
[Matching records](/guides/matching-records/#guards-first-in-a-generator)
covers both, with a worked multi-field example.

## Normalizing: hand the function over, don't apply it yourself

If you clean your choices — and you usually should — the query has to be
cleaned the same way, or the two sides are not being compared alike. So
`prepareChoice` takes the normalizer as an **option** rather than expecting
already-clean text, and the search checks that it got the same one:

```ts
prepared: scorer.prepareChoice(record.name, { normalize: normalizeText })
normalize: normalizeText // the same function, at search time
```

The check is by function identity, and it is strict in both directions: a
handle prepared without a normalizer refuses a search that normalizes, a
handle prepared with one refuses a search that doesn't, and two different
cleaning functions are refused even if they'd agree on every input. Passing
`prepareChoice(normalizeText(record.name))` — normalizing by hand and then
telling the search to normalize too — throws for exactly this reason,
because the handle would report itself as un-normalized.

The other item options don't apply in prepared mode: `getText` has nothing
to extract from, and `missingItems` has nothing to skip — a prepared row is
either resolvable or an error, never a gap.

## Which scorers accept a handle

Compatibility is decided conservatively, by identity rather than by proving
two preparations equivalent:

| Prepared by                                    | Accepted by                            |
| ---------------------------------------------- | -------------------------------------- |
| a scorer using a metric's default preparation  | any scorer of that metric using it too |
| a scorer with configuration the metric records | that scorer alone                      |
| a custom metric's scorer                       | that scorer alone                      |

So two separately created `createScorer(weightedSimilarity)` scorers share
their handles; treat a configured or custom scorer as owning the handles it
made. Anything else throws — a handle from the wrong scorer is refused as
incompatible, a value that never was a handle as invalid.

[Dice](/algorithms/dice/) and [Cosine](/algorithms/cosine/) are the one place
where writing the default out still shares: their preparation depends only on
`gramSize`, so a default scorer and one written as `{ gramSize: 2 }` accept
each other's handles, and any other depth does not. Levenshtein's `weights`
are not like that — spelling out the default costs the sharing, because the
handle records the scorer rather than the values.

Most of those mistakes never reach runtime: a handle's type carries which
metric made it, so a Levenshtein handle doesn't typecheck where a Jaro
scorer's is expected. Name a stored handle's type with
`PreparedChoiceOf<typeof scorer>`:

```ts
import type { PreparedChoiceOf } from 'rapidfuzz-js'

interface Row {
  record: CompanyRecord
  prepared: PreparedChoiceOf<typeof scorer>
}
```

Widening a scorer to `Scorer<'similarity'>` gives that up deliberately — the
type no longer names a metric, so only the runtime check remains.

## A handle owns its text

`prepareChoice` copies what it keeps. Mutating the source array or typed
array afterwards does not reach through the handle — the same snapshot rule
a [Matcher follows](/concepts/matchers/#what-a-matcher-remembers), so the
two modes never disagree about what they scored.

## When to reach for which

| Workload                                              | Use                               |
| ----------------------------------------------------- | --------------------------------- |
| One query, once                                       | `bestMatch` / `search`, text mode |
| Same collection, many queries, text is one field      | `createMatcher`                   |
| Many queries, but _you_ own, filter, or grow the rows | `prepareChoice` + `getPrepared`   |

What preparation is worth depends entirely on how much setup your scorer
does: the recorded `Matcher` figures — which remove the same per-query work
by the same means — range from **1.44×** on plain `similarity` to **6.63×**
on `tokenSortSimilarity`, where tokenizing and sorting every choice is the
bulk of the cost ([Benchmarks](/benchmarks/)). Expect the token scorers to
gain most and the cheap character kernels least. See
[Performance](/guides/performance/) for where this sits among the other
habits.
