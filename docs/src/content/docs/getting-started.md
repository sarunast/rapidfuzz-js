---
title: Getting started
description: Install rapidfuzz-js and get your first fuzzy matches in a minute.
---

## Install

```sh
npm install rapidfuzz-js
```

Works in Node.js 22+, browsers, and edge runtimes. ESM-only, no runtime
dependencies, strict TypeScript types included.

## Compare two strings

Import a metric and call it. This one scores similarity from 0 (nothing in
common) to 100 (identical):

```ts
import { ratio } from 'rapidfuzz-js/fuzz'

ratio('this is a test', 'this is a test!')
// 96.55 — nearly identical, one extra character
```

Other metrics count instead of scoring — Levenshtein reports how many
single-character edits separate two strings:

```ts
import { distance } from 'rapidfuzz-js/levenshtein'

distance('recieve', 'receive') // 2
```

Different metrics, different scales — that's deliberate, and
[Metrics](/concepts/metrics/) explains the system. For now: fuzz metrics are
0–100, higher is better.

## Find the best match in a list

This is the "did you mean?" pattern. Wrap a metric in a scorer, then ask:

```ts
import { bestMatch, createScorer } from 'rapidfuzz-js'
import { weightedRatio } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(weightedRatio)
const teams = ['Atlanta Falcons', 'New York Jets', 'New York Giants']

bestMatch('new york jet', teams, { scorer })
// { item: 'New York Jets', key: 1, score: ... }
```

`weightedRatio` is the "just make it work" metric — it tries several
strategies per pair and reports the best. Every result carries the matched
`item`, its position (`key`), and its `score`.

Want a ranked list instead of one winner? Use `search`, with a `threshold`
so weak matches don't pad the results:

```ts
import { search } from 'rapidfuzz-js'

search('new york', teams, { scorer, threshold: 60, limit: 2 })
// the two best matches scoring at least 60, best first
```

## Search the same collection many times

If your collection answers many queries — a search box over a product
catalog, say — build a **Matcher** once. It prepares every item up front, so
each query only pays for itself:

```ts
import { createMatcher, createScorer, normalizeText } from 'rapidfuzz-js'
import { tokenSortRatio } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(tokenSortRatio)

const products = [
  { title: 'Wireless mechanical keyboard' },
  { title: 'Compact wireless mouse' },
]

const matcher = createMatcher(products, {
  scorer,
  getText: (product) => product.title, // where the searchable text lives
  normalize: normalizeText, // lowercase, strip punctuation
})

matcher.best('mechanical keybord', { threshold: 70 })
// { item: products[0], key: 0, score: ... } — typo and all
```

Three things happened there worth naming:

- `getText` — the items are objects, so we point at the text to search.
- `normalize: normalizeText` — case and punctuation stop mattering.
- `tokenSortRatio` — word order stops mattering too.

Each is optional, and each is explained where it belongs:
[Matchers](/concepts/matchers/), [Preprocessing](/guides/preprocessing/),
and [Fuzz](/algorithms/fuzz/).

## Where to go next

- Results not what you expected? [Metrics](/concepts/metrics/) explains what
  the numbers mean; [Algorithms](/algorithms/levenshtein/) explains which
  algorithm suits which data.
- Building a real search? [Searching collections](/guides/searching-collections/)
  and [Performance](/guides/performance/).
