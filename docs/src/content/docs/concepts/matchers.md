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
import { weightedRatio } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(weightedRatio)
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

## Indexed Matchers, for Dice and Cosine

`createIndexedMatcher` builds the **same Matcher over a different
representation**. Instead of preparing every choice and scoring them one at a
time, it turns the whole collection into a single inverted n-gram index that
answers a query without visiting choices that share nothing with it.

```ts
import { createIndexedMatcher, createScorer } from 'rapidfuzz-js'
import { similarity as diceSimilarity } from 'rapidfuzz-js/dice'

const matcher = createIndexedMatcher(files, {
  scorer: createScorer(diceSimilarity, { gramSize: 3 }),
  getText: (file) => file.path,
})
matcher.search('src/algorthms/dice.ts', { limit: 5, threshold: 0.5 })
```

Everything after construction is identical — `best`, `search`, `searchIter`,
`size`, `scorer`, the same `Match` objects, the same order and the same tie
breaks — so swapping the constructor is the whole change at a call site. The
results are exact, not approximate: it returns what scoring every choice would
have returned, to the bit.

On 10,000 `node_modules` file paths at `gramSize: 3`:

| Query                     | Indexed | Matcher |      |
| ------------------------- | ------: | ------: | ---: |
| a path that exists        | 0.20 ms | 2.23 ms |  11x |
| the same path with a typo | 0.20 ms | 2.52 ms |  13x |
| `'node_modules/'`         | 0.16 ms | 0.11 ms | 0.7x |
| a rare fragment           | 0.11 ms | 0.11 ms | 1.0x |

Retained memory is **256 bytes a choice against 1,282** — 5x less — against
construction costing about **1.2x** more, since a prepared collection packs each
choice's grams into two typed arrays where the index has a corpus-wide
structure to assemble. Both figures used to be far larger: a prepared choice
held a trie of `Map`s and cost 18,049 bytes, which made the index look 57x
faster and 77x smaller than it is against today's representation.

### When not to reach for it

- **Only `dice.similarity` and `cosine.similarity` have one.** Any other scorer
  throws at construction, and a distance scorer is a compile error.
- **The win is selectivity, not size.** It comes from a query's grams naming few
  choices. The `'node_modules/'` row above is the adverse case in miniature: a
  query made of grams nearly every choice shares reaches everything anyway, and
  Dice's exhaustive path prunes it with a length bound the index has no use for.
  A two-letter alphabet loses outright. Cosine keeps its lead on that row only
  because its exhaustive path has no such bound to prune with.
- **`searchIter` is no longer lazy.** It yields the same values in the same
  collection order, but accumulation is the work an index does, so it settles
  which choices qualify before yielding the first. Breaking out early still
  skips building the results you never asked for — of 100,000 qualifying
  matches, taking one measured 0.06x against materializing them all — but it
  cannot skip the scoring, which is already done.
- **Choices must be text, or sequences of integers.** Code points qualify;
  an array of objects does not, and is refused at construction.
- **`getPrepared` is not an option.** A prepared handle is the per-choice
  representation an index replaces.
