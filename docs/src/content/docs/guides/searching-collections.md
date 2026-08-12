---
title: Searching collections
description: Ranked result lists — thresholds, limits, and searching Maps and objects.
---

Where [`bestMatch`](/guides/finding-the-best-match/) returns one winner,
`search` returns a ranked list — the shape behind autocomplete dropdowns and
search results pages:

```ts
import { createScorer, search } from 'rapidfuzz-js'
import { weightedSimilarity } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(weightedSimilarity)
const teams = ['Atlanta Falcons', 'New York Jets', 'New York Giants']

search('new york', teams, { scorer, threshold: 60, limit: 2 })
// [ { item: 'New York Jets', key: 1, score: ... },
//   { item: 'New York Giants', key: 2, score: ... } ]
```

Results come best first. Two options shape the list:

- **`threshold`** — the quality bar. Results below it (similarity) or above
  it (distance) are dropped entirely. Without one, a search for `'zzz'`
  still returns the least-bad five — set a threshold so irrelevant results
  don't dress up as answers.
- **`limit`** — the count cap, default `5`. Pass `limit: null` for
  everything that clears the threshold.

The two compose naturally: _threshold_ decides what deserves to appear,
_limit_ decides how many you show.

## More than arrays

Anything list-like is searchable — arrays, iterables, Maps, plain objects —
and each result's `key` tells you where the item lives in _your_ structure:

```ts
const byId = new Map([
  ['fal', 'Atlanta Falcons'],
  ['jets', 'New York Jets'],
])

search('jets', byId, { scorer, threshold: 60 })
// [ { item: 'New York Jets', key: 'jets', score: ... } ]
```

A Map gives you your own IDs back — no index-to-ID translation layer.
Objects with text buried inside them use `getText`; messy text uses
`normalize` ([Preprocessing](/guides/preprocessing/)).

Gaps (`null`/`undefined` items, or `getText` returning one) are skipped
without disturbing the other keys; `missingItems: 'throw'` makes them an
error instead ([Matchers](/concepts/matchers/#gaps-in-the-data)).

## The one-shot vs Matcher decision

`search` streams the collection and remembers nothing — every call prepares
every item again. That's the right trade for one-off questions, and the
wrong one for a search box.

```ts
import { createMatcher } from 'rapidfuzz-js'

const matcher = createMatcher(teams, { scorer }) // prepare once

matcher.search('new york', { limit: null }) // fast
matcher.search('falcons', { threshold: 70 }) // fast
```

The heuristic is simple: **can you name a second query? Build a
[Matcher](/concepts/matchers/).**

There's a third position between the two: if the collection has to stay
yours — filtered differently per query, growing, carrying more than one
searchable field — prepare each candidate yourself with
`scorer.prepareChoice` and hand the handles back through `getPrepared`.
One-shot flexibility, Matcher-grade speed:
[Prepared choices](/guides/prepared-choices/).

## When the items have more than one field

Everything above searches one piece of text per item. When a record has
several fields that should each be judged on their own terms — a title _and_
a company, a name _and_ an address — don't concatenate them into one string:
[Matching records](/guides/matching-records/) shows why that inflates scores
on pairs that aren't duplicates, and what to do instead.

## When you want scores, not rankings

If the question is "score everything against everything" rather than "what
matches best", that's `scoreMatrix`/`scorePairs` —
[Comparing strings](/guides/comparing-strings/#many-pairs-at-once).
