---
title: Prepared choices
description: Prepare candidates once, keep the collection yours — one-shot search speed without handing your data to a Matcher.
---

A [Matcher](/concepts/matchers/) amortizes preparation by *owning the
collection*: it snapshots one text field at construction, and from then on
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
  prepared: scorer.prepareChoice(normalizeText(record.name)),
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

## Normalize before preparing, and tell the search

A handle holds the text exactly as you prepared it. If you cleaned the
choices first — and you usually should — the query has to be cleaned the
same way, or the two sides are not being compared alike:

```ts
prepared: scorer.prepareChoice(normalizeText(record.name)) // choices: at prepare time
normalize: normalizeText // query: at search time
```

In prepared mode `normalize` applies to the **query only** — the choices
were prepared before the search ever saw them. `getText` and `missingItems`
don't apply either: a prepared row is either resolvable or an error, never a
gap to skip.

## Which scorers accept a handle

Compatibility is decided conservatively, by identity rather than by proving
two preparations equivalent:

| Prepared by                                   | Accepted by                            |
| --------------------------------------------- | -------------------------------------- |
| a scorer using a metric's default preparation | any scorer of that metric using it too |
| a scorer with configuration the metric records | that scorer alone                     |
| a custom metric's scorer                      | that scorer alone                      |

So two separately created `createScorer(fuzzySimilarity)` scorers share
their handles; treat a configured or custom scorer as owning the handles it
made. Anything else throws — a handle from the wrong scorer is refused as
incompatible, a value that never was a handle as invalid.

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

| Workload                                        | Use                                    |
| ----------------------------------------------- | -------------------------------------- |
| One query, once                                 | `bestMatch` / `search`, text mode      |
| Same collection, many queries, text is one field | `createMatcher`                       |
| Many queries, but *you* own, filter, or grow the rows | `prepareChoice` + `getPrepared`  |

Preparation pays for itself from the first query — the per-query cost of a
prepared search measured **1.2–6× lower** than text mode, depending on the
scorer. See [Performance](/guides/performance/) for where this sits among
the other habits.
