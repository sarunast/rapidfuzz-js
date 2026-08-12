---
title: Matchers
description: Prepare a collection once, query it many times — the workhorse for search boxes and lookups.
---

Fuzzy-scoring a collection has a hidden cost: before an algorithm can score
a pair, it builds internal structures from the text. A one-shot search
rebuilds them for every item on **every call** — fine once, wasteful in a
search box firing on each keystroke.

A **Matcher** splits the work: pay the preparation once, at construction;
every query after that only pays for itself.

```ts
import { createMatcher, createScorer } from 'rapidfuzz-js'
import { weightedSimilarity } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(weightedSimilarity)
const teams = ['Atlanta Falcons', 'New York Jets', 'New York Giants']

const matcher = createMatcher(teams, { scorer }) // preparation happens here

matcher.best('new york jet')
// { item: 'New York Jets', key: 1, score: ... }
matcher.search('new york', { limit: null }) // every result, best first
matcher.size // 3
```

The rule of thumb: **more than one query over the same collection → build a
Matcher.** For a genuine one-off, `bestMatch` and `search` from the package
root take the same options and retain nothing
([Searching collections](/guides/searching-collections/)).

## Every result says where it came from

Results are `{ item, key, score }` — the original item, its address in your
collection, and its score. The `key` depends on what you passed in:

| You pass     | `key` is          |
| ------------ | ----------------- |
| Array        | The index         |
| Iterable     | The position      |
| Map          | The map key       |
| Plain object | The property name |

So a Matcher over a `Map<ProductId, Product>` hands back real product IDs,
not positions you have to translate.

## Searching objects, not strings

Collections are rarely raw strings. Two options bridge the gap:

```ts
import { createMatcher, normalizeText } from 'rapidfuzz-js'

const products = [
  { title: 'Wireless mechanical keyboard' },
  { title: 'Compact wireless mouse' },
]

const matcher = createMatcher(products, {
  scorer,
  getText: (product) => product.title, // what to search
  normalize: normalizeText, // how to clean it first
})
```

`getText` picks the searchable text out of each item; `normalize` cleans it
(and every query) so case and punctuation stop mattering — see
[Preprocessing](/guides/preprocessing/). Results always return your
original objects, never the extracted text.

## Gaps in the data

Items that are `null`/`undefined` — or whose `getText` returns
`null`/`undefined` — are **skipped, keeping their keys**: item 3 stays item
3 even if item 2 was a gap. Your keys keep pointing at the right things.

If a gap would mean a bug, `missingItems: 'throw'` rejects the collection at
construction instead of quietly searching around the hole.

## What a Matcher remembers

A Matcher snapshots _what it scores_, not _what it returns_:

- Strings are kept as-is; non-string sequences are shallow-copied at
  construction. Pushing to the source array later does **not** change search
  results — the Matcher answers about the collection as it stood.
- Returned items are live references to your original objects — mutating
  `product.price` is visible in results, because the Matcher never copied
  your objects, only the text it scores.

If the collection itself changes, build a new Matcher — construction is the
cheap part compared to re-preparing on every query.

When rebuilding doesn't fit — the collection is filtered differently per
query, or preparation is worth keeping _across_ rebuilds — prepare each
candidate yourself and hand the handles to a one-shot search (or to
`createMatcher` via `getPrepared`, which then resolves them once instead of
preparing at all): [Prepared choices](/guides/prepared-choices/).
