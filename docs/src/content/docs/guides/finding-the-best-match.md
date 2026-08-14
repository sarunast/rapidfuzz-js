---
title: Finding the best match
description: The "did you mean?" pattern — one winner from a collection, or honestly nothing.
---

The most common fuzzy-matching task: a user typed something, you have a list
of valid values, find the one they meant.

```ts
import { bestMatch, createScorer } from 'rapidfuzz-js'
import { weightedRatio } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(weightedRatio)
const teams = ['Atlanta Falcons', 'New York Jets', 'New York Giants']

bestMatch('new york jet', teams, { scorer })
// { item: 'New York Jets', key: 1, score: ... }
```

The result is the matched `item`, its `key` (position, map key, or property
name — see [Matchers](/concepts/matchers/#every-result-says-where-it-came-from)),
and its `score`.

## Always set a threshold

"Best" only means _best available_. Ask for the best match to `'quokka'` and
you'll get one — some team name that's marginally less unlike it than the
others. That's how "did you mean?" features end up suggesting nonsense.

A threshold makes "nothing was close enough" an honest answer:

```ts
bestMatch('quokka', teams, { scorer, threshold: 80 })
// undefined
```

For a similarity scorer the threshold is a minimum; for a distance scorer, a
maximum — always in the scorer's own units.

There's no universal right number. Log real queries with their scores for a
day, look at where good and bad matches separate, and put the line there.
As a starting point for `weightedRatio`, `70` is conservative and `60`
permissive.

## Answering many queries

`bestMatch` scans and prepares the whole collection on **every call**. Fine
for validating one CSV row; wasteful when the same list answers query after
query. Build a [Matcher](/concepts/matchers/) once and ask it instead:

```ts
import { createMatcher } from 'rapidfuzz-js'

const matcher = createMatcher(teams, { scorer })

matcher.best('new york jet')
matcher.best('atlanta', { threshold: 70 })
```

Same options, same results — the preparation just happened once, at
construction. [Performance](/guides/performance/) shows why this is the
single most effective optimization in the library.

## Matching against objects

Real collections are objects, not strings. `getText` selects the text,
`normalize` cleans it:

```ts
import { normalizeText } from 'rapidfuzz-js'

const products = [{ title: 'Wireless mechanical keyboard' }]

bestMatch('keybord', products, {
  scorer,
  getText: (p) => p.title,
  normalize: normalizeText,
})
```

The result's `item` is your original object, not the extracted text. See
[Preprocessing](/guides/preprocessing/) for what `normalizeText` does.
